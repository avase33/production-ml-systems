"""
MLOps Pipeline Orchestrator
============================
Orchestrates the complete ML lifecycle using AWS Step Functions:
  data validation → feature engineering → training → evaluation →
  approval gate → shadow deployment → A/B test → promote or rollback

Author: Akhil Vase | Senior AI/ML Engineer
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

import boto3

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Enums + Config
# ---------------------------------------------------------------------------

class ModelStage(str, Enum):
    NONE = "None"
    STAGING = "Staging"
    PRODUCTION = "Production"
    ARCHIVED = "Archived"


class DeploymentStrategy(str, Enum):
    SHADOW = "shadow"          # 0% user traffic, full monitoring
    CANARY = "canary"          # configurable % traffic (e.g. 10%)
    BLUE_GREEN = "blue_green"  # instant full swap after validation


@dataclass
class PipelineConfig:
    model_name: str
    s3_data_path: str
    s3_artifact_bucket: str
    sagemaker_role_arn: str
    training_instance: str = "ml.m5.4xlarge"
    training_instance_count: int = 1
    evaluation_threshold: dict[str, float] = None   # e.g. {"auc_roc": 0.95}
    ab_test_duration_hours: int = 48
    ab_test_traffic_pct: int = 20   # % to challenger
    deployment_strategy: DeploymentStrategy = DeploymentStrategy.BLUE_GREEN
    require_human_approval: bool = False
    slack_webhook_url: str | None = None

    def __post_init__(self):
        if self.evaluation_threshold is None:
            self.evaluation_threshold = {"auc_roc": 0.93, "precision": 0.90}


# ---------------------------------------------------------------------------
# Step Functions State Machine Definition
# ---------------------------------------------------------------------------

def build_state_machine_definition(config: PipelineConfig) -> dict[str, Any]:
    """
    Returns the ASL (Amazon States Language) definition for the MLOps pipeline.
    Deploy this via Terraform or boto3 create_state_machine().
    """
    return {
        "Comment": f"MLOps pipeline for {config.model_name}",
        "StartAt": "ValidateData",
        "States": {
            "ValidateData": {
                "Type": "Task",
                "Resource": "arn:aws:states:::lambda:invoke",
                "Parameters": {
                    "FunctionName": "mlops-data-validator",
                    "Payload": {
                        "model_name": config.model_name,
                        "s3_data_path": config.s3_data_path,
                    }
                },
                "Catch": [{"ErrorEquals": ["DataValidationFailed"], "Next": "NotifyFailure"}],
                "Next": "FeatureEngineering"
            },
            "FeatureEngineering": {
                "Type": "Task",
                "Resource": "arn:aws:states:::sagemaker:createProcessingJob.sync",
                "Parameters": {
                    "ProcessingJobName.$": "States.Format('feature-eng-{}-{}', $.model_name, $$.Execution.Name)",
                    "ProcessingResources": {
                        "ClusterConfig": {
                            "InstanceCount": 2,
                            "InstanceType": "ml.m5.4xlarge",
                            "VolumeSizeInGB": 100
                        }
                    },
                    "AppSpecification": {
                        "ImageUri": "683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:1.2-1",
                        "ContainerEntrypoint": ["python3", "/opt/ml/processing/feature_engineering.py"]
                    },
                    "RoleArn": config.sagemaker_role_arn,
                },
                "Next": "TrainModel"
            },
            "TrainModel": {
                "Type": "Task",
                "Resource": "arn:aws:states:::sagemaker:createTrainingJob.sync",
                "Parameters": {
                    "TrainingJobName.$": "States.Format('train-{}-{}', $.model_name, $$.Execution.Name)",
                    "AlgorithmSpecification": {
                        "TrainingImage": "683313688378.dkr.ecr.us-east-1.amazonaws.com/xgboost:1.7-1",
                        "TrainingInputMode": "File"
                    },
                    "ResourceConfig": {
                        "InstanceCount": config.training_instance_count,
                        "InstanceType": config.training_instance,
                        "VolumeSizeInGB": 50
                    },
                    "RoleArn": config.sagemaker_role_arn,
                    "OutputDataConfig": {
                        "S3OutputPath.$": f"s3://{config.s3_artifact_bucket}/models"
                    },
                    "StoppingCondition": {"MaxRuntimeInSeconds": 7200}
                },
                "Next": "EvaluateModel"
            },
            "EvaluateModel": {
                "Type": "Task",
                "Resource": "arn:aws:states:::lambda:invoke",
                "Parameters": {
                    "FunctionName": "mlops-model-evaluator",
                    "Payload": {
                        "model_name": config.model_name,
                        "thresholds": config.evaluation_threshold,
                    }
                },
                "Next": "EvaluationGate"
            },
            "EvaluationGate": {
                "Type": "Choice",
                "Choices": [
                    {
                        "Variable": "$.evaluation_passed",
                        "BooleanEquals": True,
                        "Next": "HumanApprovalGate" if config.require_human_approval else "DeployToStaging"
                    }
                ],
                "Default": "NotifyFailure"
            },
            "HumanApprovalGate": {
                "Type": "Task",
                "Resource": "arn:aws:states:::sqs:sendMessage.waitForTaskToken",
                "Parameters": {
                    "QueueUrl": "https://sqs.us-east-1.amazonaws.com/ACCOUNT/mlops-approvals",
                    "MessageBody": {
                        "model_name": config.model_name,
                        "metrics.$": "$.evaluation_metrics",
                        "task_token.$": "$$.Task.Token"
                    }
                },
                "TimeoutSeconds": 86400,
                "Next": "DeployToStaging"
            },
            "DeployToStaging": {
                "Type": "Task",
                "Resource": "arn:aws:states:::lambda:invoke",
                "Parameters": {
                    "FunctionName": "mlops-deploy-staging",
                    "Payload": {
                        "model_name": config.model_name,
                        "strategy": config.deployment_strategy.value,
                        "traffic_pct": config.ab_test_traffic_pct,
                    }
                },
                "Next": "WaitForABTest"
            },
            "WaitForABTest": {
                "Type": "Wait",
                "Seconds": config.ab_test_duration_hours * 3600,
                "Next": "EvaluateABTest"
            },
            "EvaluateABTest": {
                "Type": "Task",
                "Resource": "arn:aws:states:::lambda:invoke",
                "Parameters": {
                    "FunctionName": "mlops-ab-evaluator",
                    "Payload": {"model_name": config.model_name}
                },
                "Next": "ABTestGate"
            },
            "ABTestGate": {
                "Type": "Choice",
                "Choices": [
                    {
                        "Variable": "$.challenger_wins",
                        "BooleanEquals": True,
                        "Next": "PromoteToProduction"
                    }
                ],
                "Default": "RollbackChallenger"
            },
            "PromoteToProduction": {
                "Type": "Task",
                "Resource": "arn:aws:states:::lambda:invoke",
                "Parameters": {
                    "FunctionName": "mlops-promote-production",
                    "Payload": {"model_name": config.model_name}
                },
                "Next": "NotifySuccess"
            },
            "RollbackChallenger": {
                "Type": "Task",
                "Resource": "arn:aws:states:::lambda:invoke",
                "Parameters": {
                    "FunctionName": "mlops-rollback",
                    "Payload": {"model_name": config.model_name, "reason": "AB test failed"}
                },
                "Next": "NotifyRollback"
            },
            "NotifySuccess": {
                "Type": "Task",
                "Resource": "arn:aws:states:::sns:publish",
                "Parameters": {
                    "TopicArn": "arn:aws:sns:us-east-1:ACCOUNT:mlops-notifications",
                    "Message.$": "States.Format('✅ {} promoted to production', $.model_name)"
                },
                "End": True
            },
            "NotifyRollback": {
                "Type": "Task",
                "Resource": "arn:aws:states:::sns:publish",
                "Parameters": {
                    "TopicArn": "arn:aws:sns:us-east-1:ACCOUNT:mlops-notifications",
                    "Message.$": "States.Format('⚠️ {} challenger rolled back after A/B test', $.model_name)"
                },
                "End": True
            },
            "NotifyFailure": {
                "Type": "Task",
                "Resource": "arn:aws:states:::sns:publish",
                "Parameters": {
                    "TopicArn": "arn:aws:sns:us-east-1:ACCOUNT:mlops-notifications",
                    "Message.$": "States.Format('❌ {} pipeline failed', $.model_name)"
                },
                "End": True
            }
        }
    }


# ---------------------------------------------------------------------------
# Pipeline Runner
# ---------------------------------------------------------------------------

class MLOpsPipelineRunner:
    """
    Triggers and monitors an MLOps pipeline execution via Step Functions.
    """

    def __init__(self, state_machine_arn: str, region: str = "us-east-1"):
        self.sfn = boto3.client("stepfunctions", region_name=region)
        self.state_machine_arn = state_machine_arn

    def trigger(self, config: PipelineConfig) -> str:
        """Start a pipeline execution. Returns execution ARN."""
        execution_name = f"{config.model_name}-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
        response = self.sfn.start_execution(
            stateMachineArn=self.state_machine_arn,
            name=execution_name,
            input=json.dumps({
                "model_name": config.model_name,
                "s3_data_path": config.s3_data_path,
                "trigger_time": datetime.utcnow().isoformat(),
            }),
        )
        log.info("Started execution: %s", response["executionArn"])
        return response["executionArn"]

    def wait_for_completion(self, execution_arn: str, poll_interval: int = 60) -> str:
        """Poll until execution completes. Returns final status."""
        while True:
            response = self.sfn.describe_execution(executionArn=execution_arn)
            status = response["status"]
            if status in ("SUCCEEDED", "FAILED", "TIMED_OUT", "ABORTED"):
                log.info("Execution %s: %s", execution_arn.split(":")[-1], status)
                return status
            log.info("Execution running … status=%s", status)
            time.sleep(poll_interval)

    def get_execution_history(self, execution_arn: str) -> list[dict]:
        """Retrieve step-by-step history for debugging."""
        response = self.sfn.get_execution_history(
            executionArn=execution_arn,
            includeExecutionData=False,
        )
        return response["events"]

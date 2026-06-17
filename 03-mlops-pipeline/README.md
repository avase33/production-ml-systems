# 🟡 End-to-End MLOps Pipeline

> Full ML lifecycle automation across **20+ production models** — retraining, versioning, A/B testing, blue-green deployments, and drift-triggered rollbacks. Zero manual intervention.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python)](https://python.org)
[![SageMaker](https://img.shields.io/badge/AWS_SageMaker-Latest-FF9900?style=flat-square&logo=amazon-aws)](https://aws.amazon.com/sagemaker/)
[![MLflow](https://img.shields.io/badge/MLflow-2.x-0194E2?style=flat-square&logo=mlflow)](https://mlflow.org)
[![Terraform](https://img.shields.io/badge/Terraform-IaC-7B42BC?style=flat-square&logo=terraform)](https://terraform.io)

---

## 📋 Overview

This project implements **production MLOps infrastructure** that automates the complete ML model lifecycle. Built at Citibank and Baker Hughes to manage 20+ production models with zero-touch retraining and controlled deployment pipelines.

**Capabilities:**
- 🔄 **Zero-touch retraining** triggered by data drift or scheduled cron
- 📊 **Champion-challenger A/B testing** with configurable traffic splits
- 🚀 **Blue-green deployments** with automatic rollback on metric regression
- 📈 **MLflow model registry** with full lineage: data → training → production
- 🔔 **Slack / PagerDuty alerting** on drift, latency, or prediction quality degradation

---

## 🏗️ Pipeline Architecture

```
         ┌──────────────────────────────────────────────┐
         │            Trigger Layer                      │
         │  • Cron (weekly retraining)                   │
         │  • Evidently AI drift alert (webhook)         │
         │  • Data volume threshold (CloudWatch alarm)   │
         └──────────────────┬───────────────────────────┘
                            │ EventBridge / Lambda trigger
         ┌──────────────────▼───────────────────────────┐
         │         AWS Step Functions Orchestrator       │
         │                                               │
         │  1. Data Validation (Great Expectations)      │
         │  2. Feature Engineering (Spark / EMR)         │
         │  3. Model Training (SageMaker Training Job)   │
         │  4. Model Evaluation (SageMaker Processing)   │
         │  5. Human Approval Gate (optional)            │
         │  6. Shadow Deployment (10% traffic)           │
         │  7. A/B Testing (champion vs challenger)      │
         │  8. Full Promotion or Rollback                │
         └──────────────────┬───────────────────────────┘
                            │
         ┌──────────────────▼───────────────────────────┐
         │           MLflow Model Registry               │
         │  Staging → Production lifecycle               │
         │  Full artifact + metric lineage               │
         └──────────────────┬───────────────────────────┘
                            │
         ┌──────────────────▼───────────────────────────┐
         │       SageMaker Real-Time Endpoint            │
         │  Blue-Green swap | Auto-scaling | Multi-AZ   │
         └──────────────────┬───────────────────────────┘
                            │
         ┌──────────────────▼───────────────────────────┐
         │     Observability (Datadog + Evidently AI)    │
         │  Prediction quality | Drift | Latency | SHAP  │
         └──────────────────────────────────────────────┘
```

---

## 📂 Project Structure

```
03-mlops-pipeline/
├── README.md
├── requirements.txt
├── mlops_pipeline.py          # Step Functions state machine definition
├── sagemaker_pipeline.py      # SageMaker Pipeline steps
├── drift_monitor.py           # Evidently AI drift detection + alerting
└── model_registry.py          # MLflow registry promotion workflow
```

---

## 🚀 Quick Start

```bash
pip install -r requirements.txt

# Deploy Step Functions state machine (Terraform)
cd terraform && terraform init && terraform apply

# Trigger a retraining run manually
python mlops_pipeline.py --model fraud-detection-v3 --trigger manual

# Check model registry status
python model_registry.py --list-models --stage production

# Run drift check on current production model
python drift_monitor.py --model-name fraud-detection-v3 --reference-window 7d
```

---

## 📊 Operational Metrics

| Metric | Value |
|--------|-------|
| Models Under Management | 20+ |
| Avg Retraining Cycle Time | 4.2 hours (end-to-end) |
| Pipeline Maintenance Reduction | 45% |
| Blue-Green Swap Downtime | 0ms (SageMaker in-place) |
| Mean Time to Detect Drift | <15 minutes |
| A/B Test Duration | 24–72h (configurable) |

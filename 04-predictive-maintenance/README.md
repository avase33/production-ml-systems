# 🟢 Predictive Maintenance — IoT Sensor ML

> ML system for oilfield equipment failure prediction on live IoT sensor data. **85% failure prediction accuracy** · **25% reduction** in unplanned downtime · **200+ active wells**.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch)](https://pytorch.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-337AB7?style=flat-square)](https://xgboost.readthedocs.io)
[![Kafka](https://img.shields.io/badge/Apache_Kafka-Streaming-231F20?style=flat-square&logo=apachekafka)](https://kafka.apache.org)
[![Grafana](https://img.shields.io/badge/Grafana-Dashboard-F46800?style=flat-square&logo=grafana)](https://grafana.com)

---

## 📋 Overview

Built for **Baker Hughes' Leucipa** automated production optimization platform. This system ingests live IoT sensor streams from drilling and production equipment, detects anomalies, and predicts failures before they cause well shutdowns.

In oilfield operations, **a single unplanned shutdown costs $100K–$500K per day**. Getting this right matters.

**Results:**
- ✅ **85% failure prediction accuracy** (48h prediction horizon)
- 📉 **25% reduction** in unplanned equipment downtime
- 🏭 **200+ active wells** monitored in real time
- ⚡ Sensor data latency: batch windows of **hours → under 60 seconds** (Kafka)
- 📊 False positive alerts reduced by **30%** (domain-feature collaboration with petroleum engineers)

---

## 🏗️ Architecture

```
┌──────────────────────────────────────┐
│   IoT Sensors (200+ Wells)           │
│   Pressure · Temp · Vibration · Flow  │
└──────────────┬───────────────────────┘
               │  MQTT / OPC-UA
┌──────────────▼───────────────────────┐
│   Apache Kafka Topics                │
│   well-sensor-raw (10K msgs/sec)     │
└──────────────┬───────────────────────┘
               │
┌──────────────▼───────────────────────┐
│   Feature Pipeline (Spark Streaming) │
│   • Rolling stats (1m, 5m, 1h, 24h) │
│   • Rate-of-change features          │
│   • Cross-sensor correlation         │
└──────────────┬───────────────────────┘
               │
      ┌────────┴─────────┐
      │                  │
┌─────▼──────┐   ┌───────▼───────┐
│  LSTM      │   │ Isolation     │
│  Failure   │   │ Forest        │
│  Predictor │   │ Anomaly Det.  │
└─────┬──────┘   └───────┬───────┘
      │                  │
┌─────▼──────────────────▼───────┐
│   Decision Engine              │
│   • Ensemble prediction        │
│   • Severity scoring           │
│   • Alert deduplication        │
└─────────────┬──────────────────┘
              │
┌─────────────▼──────────────────┐
│   Grafana Dashboard            │
│   Real-time well health view   │
│   PagerDuty alerts             │
└────────────────────────────────┘
```

---

## 📂 Project Structure

```
04-predictive-maintenance/
├── README.md
├── requirements.txt
├── lstm_model.py              # PyTorch LSTM for time-series failure prediction
├── anomaly_detection.py       # Isolation Forest + Autoencoder anomaly detection
└── sensor_feature_pipeline.py # Spark Streaming feature computation
```

---

## 🚀 Quick Start

```bash
pip install -r requirements.txt

# Train LSTM on historical sensor data (CSV or Parquet)
python lstm_model.py --data-path ./data/sensor_logs.parquet --well-id WELL_001

# Run anomaly detection on a sensor window
python anomaly_detection.py --input ./data/sensor_window.csv --threshold 0.05

# Start streaming feature pipeline (requires Kafka)
python sensor_feature_pipeline.py \
  --kafka-brokers localhost:9092 \
  --topic well-sensor-raw \
  --output-topic well-features
```

---

## 📊 Model Performance

| Model | Metric | Value |
|-------|--------|-------|
| LSTM | Failure Prediction Accuracy | 85% |
| LSTM | False Positive Rate | 12% |
| LSTM | Prediction Horizon | 48 hours |
| Isolation Forest | Anomaly Recall | 91% |
| Isolation Forest | Precision | 78% |
| Ensemble | AUC-ROC | 0.924 |

---

## 🔑 Domain Context

**Why LSTM for failure prediction?**
Equipment failure follows temporal patterns — vibration signatures degrade over hours/days before mechanical failure. LSTM networks capture long-range sequential dependencies that tree models miss.

**Why Isolation Forest for anomaly detection?**
Oilfield sensor data has no reliable "normal" baseline — operating conditions shift with reservoir pressure, fluid composition, and seasonal temperatures. Unsupervised anomaly detection avoids labeled data dependency.

**Feature collaboration with domain experts:**
Petroleum engineers provided operational knowledge (pressure differential thresholds, expected vibration signatures for each equipment type) that translated into 12 high-signal features, reducing false positive alerts by 30%.

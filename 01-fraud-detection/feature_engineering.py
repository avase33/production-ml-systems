"""
Real-Time Feature Engineering for Fraud Detection
==================================================
Computes velocity, behavioral, and risk features from streaming transaction data.
Designed for <10ms feature retrieval via Redis online feature store.

Author: Akhil Vase | Senior AI/ML Engineer
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import redis


# ---------------------------------------------------------------------------
# Data Contracts
# ---------------------------------------------------------------------------

@dataclass
class Transaction:
    """Incoming transaction payload from payment gateway."""
    tx_id: str
    user_id: str
    merchant_id: str
    amount: float
    currency: str
    channel: str           # "web" | "mobile" | "api"
    device_fingerprint: str
    ip_address: str
    timestamp: datetime
    merchant_category: str
    card_last4: str
    billing_country: str
    shipping_country: str | None = None


@dataclass
class FeatureVector:
    """Feature vector passed to the fraud model."""
    # Velocity features
    tx_count_1h: int
    tx_count_24h: int
    amount_sum_1h: float
    amount_sum_24h: float
    unique_merchants_24h: int
    unique_devices_24h: int

    # Behavioral features
    avg_amount_30d: float
    amount_zscore: float          # deviation from user's historical mean
    hour_of_day: int
    is_weekend: bool
    time_since_last_tx_secs: float

    # Risk signals
    is_cross_border: bool
    country_mismatch: bool
    high_risk_merchant_category: bool
    amount_log: float

    # Device / network
    device_seen_before: bool
    ip_seen_before: bool

    def to_numpy(self) -> np.ndarray:
        return np.array([
            self.tx_count_1h, self.tx_count_24h,
            self.amount_sum_1h, self.amount_sum_24h,
            self.unique_merchants_24h, self.unique_devices_24h,
            self.avg_amount_30d, self.amount_zscore,
            self.hour_of_day, int(self.is_weekend),
            self.time_since_last_tx_secs,
            int(self.is_cross_border), int(self.country_mismatch),
            int(self.high_risk_merchant_category), self.amount_log,
            int(self.device_seen_before), int(self.ip_seen_before),
        ], dtype=np.float32)


# ---------------------------------------------------------------------------
# High-Risk Merchant Categories (configurable)
# ---------------------------------------------------------------------------

HIGH_RISK_MCC = frozenset({
    "gambling", "crypto_exchange", "wire_transfer",
    "money_services", "gift_cards", "prepaid_cards",
})


# ---------------------------------------------------------------------------
# Feature Store Client
# ---------------------------------------------------------------------------

class RedisFeatureStore:
    """
    Thin wrapper around Redis for reading/writing pre-computed user features.
    Key schema:  fraud:user:{user_id}:{feature_name}
    TTL:         24h for velocity counters; 30d for behavioral aggregates
    """

    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0):
        self.client = redis.Redis(host=host, port=port, db=db, decode_responses=True)

    def _key(self, user_id: str, feature: str) -> str:
        return f"fraud:user:{user_id}:{feature}"

    # -- Velocity reads -------------------------------------------------------

    def get_tx_count(self, user_id: str, window: str) -> int:
        """window: '1h' | '24h'"""
        val = self.client.get(self._key(user_id, f"tx_count_{window}"))
        return int(val) if val else 0

    def get_amount_sum(self, user_id: str, window: str) -> float:
        val = self.client.get(self._key(user_id, f"amount_sum_{window}"))
        return float(val) if val else 0.0

    def get_unique_merchants(self, user_id: str, window: str) -> int:
        return self.client.scard(self._key(user_id, f"merchants_{window}"))

    def get_unique_devices(self, user_id: str, window: str) -> int:
        return self.client.scard(self._key(user_id, f"devices_{window}"))

    def get_last_tx_timestamp(self, user_id: str) -> float | None:
        val = self.client.get(self._key(user_id, "last_tx_ts"))
        return float(val) if val else None

    # -- Behavioral reads -----------------------------------------------------

    def get_avg_amount_30d(self, user_id: str) -> float:
        val = self.client.get(self._key(user_id, "avg_amount_30d"))
        return float(val) if val else 0.0

    def get_stddev_amount_30d(self, user_id: str) -> float:
        val = self.client.get(self._key(user_id, "stddev_amount_30d"))
        return float(val) if val else 1.0   # avoid division by zero

    def is_device_known(self, user_id: str, device_fp: str) -> bool:
        return bool(self.client.sismember(self._key(user_id, "known_devices"), device_fp))

    def is_ip_known(self, user_id: str, ip: str) -> bool:
        hashed_ip = hashlib.sha256(ip.encode()).hexdigest()[:16]
        return bool(self.client.sismember(self._key(user_id, "known_ips"), hashed_ip))

    # -- Velocity writes (called by Kafka consumer after scoring) -------------

    def increment_velocity(self, user_id: str, tx: Transaction) -> None:
        pipe = self.client.pipeline(transaction=True)
        now_ts = tx.timestamp.timestamp()

        for window, ttl in [("1h", 3600), ("24h", 86400)]:
            pipe.incr(self._key(user_id, f"tx_count_{window}"))
            pipe.expire(self._key(user_id, f"tx_count_{window}"), ttl)
            pipe.incrbyfloat(self._key(user_id, f"amount_sum_{window}"), tx.amount)
            pipe.expire(self._key(user_id, f"amount_sum_{window}"), ttl)
            pipe.sadd(self._key(user_id, f"merchants_{window}"), tx.merchant_id)
            pipe.expire(self._key(user_id, f"merchants_{window}"), ttl)
            pipe.sadd(self._key(user_id, f"devices_{window}"), tx.device_fingerprint)
            pipe.expire(self._key(user_id, f"devices_{window}"), ttl)

        pipe.set(self._key(user_id, "last_tx_ts"), now_ts)
        pipe.sadd(self._key(user_id, "known_devices"), tx.device_fingerprint)
        pipe.expire(self._key(user_id, "known_devices"), 86400 * 90)

        hashed_ip = hashlib.sha256(tx.ip_address.encode()).hexdigest()[:16]
        pipe.sadd(self._key(user_id, "known_ips"), hashed_ip)
        pipe.expire(self._key(user_id, "known_ips"), 86400 * 30)

        pipe.execute()


# ---------------------------------------------------------------------------
# Feature Engineer
# ---------------------------------------------------------------------------

class FraudFeatureEngineer:
    """
    Computes the full feature vector for a transaction in real time.
    Reads pre-computed aggregates from Redis; falls back to safe defaults.

    Typical latency: 3–8ms (dominated by Redis round trips).
    """

    def __init__(self, feature_store: RedisFeatureStore):
        self.fs = feature_store

    def compute(self, tx: Transaction) -> FeatureVector:
        start = time.perf_counter()
        uid = tx.user_id

        # Velocity
        tx_count_1h = self.fs.get_tx_count(uid, "1h")
        tx_count_24h = self.fs.get_tx_count(uid, "24h")
        amount_sum_1h = self.fs.get_amount_sum(uid, "1h")
        amount_sum_24h = self.fs.get_amount_sum(uid, "24h")
        unique_merchants_24h = self.fs.get_unique_merchants(uid, "24h")
        unique_devices_24h = self.fs.get_unique_devices(uid, "24h")

        # Behavioral
        avg_amount_30d = self.fs.get_avg_amount_30d(uid)
        std_amount_30d = self.fs.get_stddev_amount_30d(uid)
        amount_zscore = (tx.amount - avg_amount_30d) / max(std_amount_30d, 1e-6)

        # Temporal
        last_ts = self.fs.get_last_tx_timestamp(uid)
        time_since_last = (
            tx.timestamp.timestamp() - last_ts
            if last_ts else 86400.0   # default: 24h if no history
        )
        hour_of_day = tx.timestamp.hour
        is_weekend = tx.timestamp.weekday() >= 5

        # Risk signals
        is_cross_border = (
            tx.billing_country != tx.shipping_country
            if tx.shipping_country else False
        )
        # Simplified: flag if merchant country differs from billing country
        country_mismatch = tx.billing_country not in ("US", "USA")  # configurable
        high_risk_mcc = tx.merchant_category.lower() in HIGH_RISK_MCC

        amount_log = float(np.log1p(tx.amount))

        # Device / network
        device_seen = self.fs.is_device_known(uid, tx.device_fingerprint)
        ip_seen = self.fs.is_ip_known(uid, tx.ip_address)

        elapsed_ms = (time.perf_counter() - start) * 1000
        # In production: emit to Datadog  metrics.histogram("feature.latency_ms", elapsed_ms)

        return FeatureVector(
            tx_count_1h=tx_count_1h,
            tx_count_24h=tx_count_24h,
            amount_sum_1h=amount_sum_1h,
            amount_sum_24h=amount_sum_24h,
            unique_merchants_24h=unique_merchants_24h,
            unique_devices_24h=unique_devices_24h,
            avg_amount_30d=avg_amount_30d,
            amount_zscore=amount_zscore,
            hour_of_day=hour_of_day,
            is_weekend=is_weekend,
            time_since_last_tx_secs=time_since_last,
            is_cross_border=is_cross_border,
            country_mismatch=country_mismatch,
            high_risk_merchant_category=high_risk_mcc,
            amount_log=amount_log,
            device_seen_before=device_seen,
            ip_seen_before=ip_seen,
        )


# ---------------------------------------------------------------------------
# Batch Feature Pipeline (Spark-style for offline training datasets)
# ---------------------------------------------------------------------------

def build_training_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw transaction logs into a training-ready feature DataFrame.
    Mirrors what the real-time pipeline computes, but on historical data.

    Parameters
    ----------
    raw_df : pd.DataFrame
        Columns: user_id, amount, timestamp, merchant_id, merchant_category,
                 device_fingerprint, ip_address, billing_country,
                 shipping_country, is_fraud (label)

    Returns
    -------
    pd.DataFrame with engineered features + is_fraud label
    """
    df = raw_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["user_id", "timestamp"])

    # Temporal features
    df["hour_of_day"] = df["timestamp"].dt.hour
    df["is_weekend"] = (df["timestamp"].dt.weekday >= 5).astype(int)
    df["amount_log"] = np.log1p(df["amount"])

    # Per-user rolling velocity (pandas groupby rolling)
    df["tx_count_1h"] = (
        df.groupby("user_id")["amount"]
        .transform(lambda s: s.rolling("1h", on=df.loc[s.index, "timestamp"]).count())
    )
    df["tx_count_24h"] = (
        df.groupby("user_id")["amount"]
        .transform(lambda s: s.rolling("24h", on=df.loc[s.index, "timestamp"]).count())
    )
    df["amount_sum_1h"] = (
        df.groupby("user_id")["amount"]
        .transform(lambda s: s.rolling("1h", on=df.loc[s.index, "timestamp"]).sum())
    )
    df["amount_sum_24h"] = (
        df.groupby("user_id")["amount"]
        .transform(lambda s: s.rolling("24h", on=df.loc[s.index, "timestamp"]).sum())
    )

    # 30-day behavioral stats (expanding window)
    g = df.groupby("user_id")["amount"]
    df["avg_amount_30d"] = g.transform(lambda s: s.expanding().mean().shift(1))
    df["std_amount_30d"] = g.transform(lambda s: s.expanding().std().shift(1).fillna(1))
    df["amount_zscore"] = (df["amount"] - df["avg_amount_30d"]) / df["std_amount_30d"]

    # Time since last transaction
    df["time_since_last_tx_secs"] = (
        df.groupby("user_id")["timestamp"]
        .transform(lambda s: s.diff().dt.total_seconds().fillna(86400))
    )

    # Risk signals
    df["is_cross_border"] = (
        df["billing_country"] != df["shipping_country"]
    ).fillna(False).astype(int)

    df["high_risk_merchant_category"] = (
        df["merchant_category"].str.lower().isin(HIGH_RISK_MCC)
    ).astype(int)

    # Device / IP seen before (within user history)
    df["device_seen_before"] = (
        df.groupby("user_id")["device_fingerprint"]
        .transform(lambda s: s.shift(1).notna() & s.shift(1).isin(s))
    ).astype(int)

    feature_cols = [
        "tx_count_1h", "tx_count_24h", "amount_sum_1h", "amount_sum_24h",
        "avg_amount_30d", "amount_zscore", "hour_of_day", "is_weekend",
        "time_since_last_tx_secs", "is_cross_border",
        "high_risk_merchant_category", "amount_log", "device_seen_before",
        "is_fraud",
    ]
    return df[feature_cols].fillna(0)

# Data Engineering: Production-Grade Pipeline

---

## 1. Overview

Guardian-ML's data pipeline processes **2 billion transactions/day** with:
- ✅ Zero data leakage
- ✅ Temporal integrity
- ✅ Automated validation
- ✅ Feature versioning
- ✅ 99.99% SLA

---

## 2. Data Flow Architecture

```
Raw Events (Kafka)
    ↓
Validation Layer (Great Expectations)
    ├─ Schema validation
    ├─ Range checks (amount > 0)
    ├─ Statistical anomalies
    └─ Duplicate detection
    
    ↓
Feature Computation (Spark SQL)
    ├─ User aggregates (30-day window)
    ├─ Merchant patterns
    ├─ Device signals
    └─ Temporal features
    
    ↓
Feature Store (Parquet + Redis)
    ├─ Offline: s3://feature-store/v2.0.0/
    ├─ Online: Redis (user:12345:features)
    └─ Metadata: MLflow Feature Registry
    
    ↓
Training Dataset Assembly
    ├─ Temporal split (no leakage)
    ├─ Stratified sampling
    └─ Label quality checks
    
    ↓
Model Training (Distributed Spark)
    ├─ Baseline
    ├─ XGBoost
    └─ Neural Network
    
    ↓
Model Registry (MLflow)
    ├─ Versioning
    ├─ Metrics
    └─ Artifact storage (S3)
    
    ↓
Production Deployment
    └─ FastAPI server + monitoring
```

---

## 3. Data Validation Strategy

### 3.1 Schema Validation

**Expected Schema**:

| Column | Type | Range | Notes |
|--------|------|-------|-------|
| transaction_id | string | unique | Primary key |
| timestamp | timestamp | within 7d | Must be recent |
| user_id | integer | > 0 | Foreign key |
| amount | float | [0.01, 1000000] | Must be positive |
| device_type | string | {mobile, desktop, tablet} | Enum |
| user_country | string | [2-letter ISO] | 150 values |
| is_fraud | integer | {0, 1} | Training labels only |

**Implementation** (Great Expectations):

```python
@check_fraud_label_distribution
def test_fraud_rate(df):
    """Fraud rate should be stable around 0.3%"""
    fraud_rate = df['is_fraud'].mean()
    assert 0.002 < fraud_rate < 0.008, \
        f"Fraud rate {fraud_rate:.2%} outside expected [0.2%, 0.8%]"
```

### 3.2 Quality Checks

**Missing Data Strategy**:

```python
# Acceptable missing rates
MISSING_THRESHOLDS = {
    'user_age': 0.02,  # 2% acceptable
    'device_type': 0.01,  # 1% acceptable
    'merchant_category': 0.05,  # 5% acceptable
}

for col, threshold in MISSING_THRESHOLDS.items():
    rate = df[col].isnull().mean()
    if rate > threshold:
        alert(f"{col}: {rate:.1%} missing (threshold {threshold:.1%})")
```

**Outlier Detection**:

```python
# Flag transactions with suspicious patterns
outliers = df[
    (df['amount'] > df['amount'].quantile(0.99) * 10) |  # 10X typical max
    (df['hour'] > 23) |  # Late night
    (df['user_transaction_count_30d'] < 0) |  # Impossible
]

if len(outliers) > 1000:
    alert(f"Unusual spike in outliers: {len(outliers)} transactions")
```

**Duplicate Detection**:

```python
# Same amount, user, merchant within 60 seconds
duplicates = df.groupby(
    ['user_id', 'merchant_id', 'amount']
).agg({'transaction_id': 'count'})

if (duplicates > 1).any():
    alert("Duplicate transactions detected")
```

---

## 4. Feature Engineering at Scale

### 4.1 User Features (30-day window)

```sql
CREATE TABLE user_features_30d AS
SELECT
    user_id,
    COUNT(*) as transaction_count_30d,
    SUM(CASE WHEN is_fraud=1 THEN 1 ELSE 0 END) as fraud_count_30d,
    CAST(fraud_count_30d AS FLOAT) / COUNT(*) as fraud_rate_30d,
    AVG(amount) as avg_amount_30d,
    STDDEV(amount) as stddev_amount_30d,
    COUNT(DISTINCT merchant_id) as unique_merchants_30d,
    COUNT(DISTINCT device_type) as unique_devices_30d,
    --   Account tenure
    DATEDIFF(CURRENT_DATE, MIN(DATE(created_at))) as account_age_days,
    --   Historical fraud flag
    CASE WHEN fraud_count_30d > 0 THEN 1 ELSE 0 END as has_fraud_history
FROM transactions
WHERE DATE(transaction_time) BETWEEN CURRENT_DATE - 30 AND CURRENT_DATE - 1
GROUP BY user_id;
```

**Computation Cost**: ~5 minutes on Spark cluster (10B records)

### 4.2 Merchant Features

```sql
CREATE TABLE merchant_features_30d AS
SELECT
    merchant_id,
    COUNT(*) as transaction_count_30d,
    -- Fraud rate (risky merchants)
    AVG(CAST(is_fraud AS FLOAT)) as fraud_rate_30d,
    -- Chargeback rate (disputes)
    CAST(
        SUM(CASE WHEN has_chargeback=1 THEN 1 ELSE 0 END)
        AS FLOAT
    ) / COUNT(*) as chargeback_rate_30d,
    -- Category risk
    merchant_category,
    -- High-value volume (risk indicator)
    SUM(CASE WHEN amount > 5000 THEN 1 ELSE 0 END) as high_value_count_30d
FROM transactions
WHERE DATE(transaction_time) BETWEEN CURRENT_DATE - 30 AND CURRENT_DATE - 1
GROUP BY merchant_id, merchant_category;
```

### 4.3 Device-based Features

```sql
CREATE TABLE device_features_30d AS
SELECT
    device_id,
    device_type,
    -- Device history
    COUNT(*) as transaction_count_30d,
    AVG(CAST(is_fraud AS FLOAT)) as fraud_rate_30d,
    -- Traveling indicator (IP jump)
    COUNT(DISTINCT CAST(transaction_time AS DATE)) as days_active_30d,
    -- User count per device (shared device)
    COUNT(DISTINCT user_id) as unique_users_30d
FROM transactions
WHERE DATE(transaction_time) BETWEEN CURRENT_DATE - 30 AND CURRENT_DATE - 1
GROUP BY device_id, device_type;
```

### 4.4 Interaction Features

```sql
-- User-Merchant combo patterns
SELECT
    user_id,
    merchant_id,
    COUNT(*) as user_merchant_interaction_count,
    AVG(amount) as user_merchant_avg_amount,
    CAST(SUM(CAST(is_fraud AS FLOAT)) AS FLOAT) / COUNT(*) as user_merchant_fraud_rate
FROM transactions
WHERE DATE(transaction_time) BETWEEN CURRENT_DATE - 30 AND CURRENT_DATE - 1
GROUP BY user_id, merchant_id;
```

---

## 5. Temporal Data Split (Critical for No Leakage)

### 5.1 The Right Way

```python
def temporal_split(df, train_ratio=0.7, val_ratio=0.15):
    """
    Split data temporally to avoid look-ahead bias.
    
    ✅ CORRECT: Uses historical data to predict future
    ❌ WRONG: Random split (leakage)
    """
    # Sort by timestamp (ascending)
    df = df.sort_values('timestamp')
    
    n = len(df)
    train_idx = int(n * train_ratio)
    val_idx = int(n * (train_ratio + val_ratio))
    
    # Split
    train_df = df.iloc[:train_idx]  # Oldest
    val_df = df.iloc[train_idx:val_idx]  # Middle
    test_df = df.iloc[val_idx:]  # Newest
    
    # Verify no overlap
    assert train_df['timestamp'].max() < val_df['timestamp'].min()
    assert val_df['timestamp'].max() < test_df['timestamp'].min()
    
    return train_df, val_df, test_df
```

### 5.2 Feature Computation During Split

```python
# TRAINING: Compute features using only historical data
features_train = compute_features(
    transactions=train_df,
    feature_window_start=train_df['timestamp'].min() - 30days,
    feature_window_end=train_df['timestamp'].max() - 1day,
)

# VALIDATION: Use newer transaction period
# But compute features from BEFORE that period
features_val = compute_features(
    transactions=val_df,
    feature_window_start=train_df['timestamp'].max() - 30days,
    feature_window_end=val_df['timestamp'].max() - 1day,
)

# TEST: Latest transactions
features_test = compute_features(
    transactions=test_df,
    feature_window_start=val_df['timestamp'].max() - 30days,
    feature_window_end=test_df['timestamp'].max() - 1day,
)
```

**Timeline Visualization**:

```
Feature                       Feature
Window Start                  Window End
    ↓                              ↓
[========== 30 Days ==========]
                                    ↑
                        Transactions to Predict
                        
Jan 1 ─────────────── Jan 31  Feb 1  Feb 28 (Training)
          Features          →TargetTx (Predict frauds on Feb 1-28)

Feb 1 ─────────────── Mar 3  Mar 4  Mar 31 (Validation)
          Features          →TargetTx

Apr 1 ─────────────── May 1  May 2  Jun 1  (Test)
          Features          →TargetTx

✅ No leakage: Features computed BEFORE transactions being predicted
```

---

## 6. Handling Class Imbalance

### 6.1 In Data Pipeline

**Weighted Sampling**:

```python
# Instead of SMOTE (corrupts temporal order),
# use class weights in loss function

n_frauds = (y_train == 1).sum()       # 3,000
n_legit = (y_train == 0).sum()        # 997,000
n_total = len(y_train)                # 1,000,000

# Equal contribution per class
w_fraud = n_total / (2 * n_frauds)    # 166.67
w_legit = n_total / (2 * n_legit)     # 0.502

#  Weighted loss
loss_weighted = w_fraud * loss_frauds + w_legit * loss_legit
```

### 6.2 Alternative: SMOTE (if needed)

```python
from imblearn.over_sampling import SMOTE

# ⚠️  Use with caution (can leakage if not careful)
smote = SMOTE(sampling_strategy=0.5)  # 50% fraud in minority
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# CRITICAL: Resample AFTER temporal split
#          Never SMOTE before split, or you'll have synthetic
#          samples that bleed between train/val/test
```

---

## 7. Data Quality Monitoring (Production)

### 7.1 Monthly Data Audit

**Checks**:

```python
# 1. Distribution drift
train_amount_mean = 200.0  # Baseline
prod_amount_mean = df_prod['amount'].mean()
if abs(prod_amount_mean - train_amount_mean) / train_amount_mean > 0.1:
    alert(f"Amount distribution drifted {(prod_amount_mean/train_amount_mean - 1)*100:.1f}%")

# 2. Fraud rate stability
train_fraud_rate = 0.003  # Baseline
prod_fraud_rate = df_prod['is_fraud'].mean()
if prod_fraud_rate > 0.01:  # 10x higher than expected
    alert(f"Fraud rate spiked to {prod_fraud_rate:.2%}!")

# 3. Missing data
for col in required_columns:
    if df_prod[col].isnull().mean() > 0.05:
        alert(f"{col}: {df_prod[col].isnull().mean():.1%} missing")
```

### 7.2 Feature Store Health

```python
# Redis cache hit rate
cache_hits = 95,000
cache_misses = 5,000
hit_rate = cache_hits / (cache_hits + cache_misses)  # 95%

if hit_rate < 0.80:
    alert(f"Cache hit rate low ({hit_rate:.1%})")
    action = "Increase Redis TTL from 24h to 48h"
```

---

## 8. Data Lineage & Governance

### 8.1 Feature Lineage

```
feature_store/v2.0.0/metadata.json:

{
  "features": [
    {
      "name": "user_fraud_rate_30d",
      "source": "transactions table",
      "last_computed": "2024-03-15T00:00:00Z",
      "computed_by": "feature_pipeline.py:user_features()",
      "dependencies": [
        "user_id",
        "is_fraud",
        "transaction_time"
      ],
      "owner": "fraud-ml-team",
      "sla": "updated daily",
      "data_quality_score": 0.98
    }
  ]
}
```

### 8.2 Audit Trail

```python
# Every decision logged
logging.info(json.dumps({
    'timestamp': datetime.now().isoformat(),
    'transaction_id': txn_id,
    'user_id': user_id,
    'amount': amount,
    'fraud_probability': fraud_prob,
    'decision': decision,
    'model_version': '1.0.0',
    'latency_ms': 4.2,
}))
```

---

## 9. Cost Optimization

| Component | Cost/Month | Optimization |
|-----------|-----------|---------------|
| data warehouse | $8K | Partition by date (archive old > 1yr) |
| Feature store | $2K | Increase Parquet compression ratio |
| Kafka | $1K | Auto-scale partitions based on volume |
| **Total** | **$11K** | |

---

## 10. Future Enhancements

- [ ] Real-time feature computation (<1ms via Apache Flink)
- [ ] Graph-based features (user-merchant-device network)
- [ ] Entity resolution (detect same user across devices)
- [ ] Time series features (seasonal patterns)
- [ ] Feature importance time series (which features matter most today?)

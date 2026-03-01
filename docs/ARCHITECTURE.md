# System Architecture: Real-Time Fraud Detection

**Version**: 1.0  
**Last Updated**: March 2026  

---

## 1. High-Level Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│                        TRANSACTION FLOW (< 10ms)                            │
└────────────────────────────────────────────────────────────────────────────┘

     Transaction
     Request
         ↓
    ┌────────────────────┐
    │ Feature Retrieval  │  (5ms)
    │ - User history     │  - Cache hits: ~500μs
    │ - Merchant profile │  - DB fallback: ~50ms (rare)
    │ - Device signals   │
    └────────────────────┘
         ↓
    ┌────────────────────┐
    │ Model Inference    │  (3ms)
    │ - XGBoost scoring  │  - Latency: consistent
    │ - Probability      │  - GPU-accelerated (optional)
    └────────────────────┘
         ↓
    ┌────────────────────┐
    │ Decision Logic     │  (1ms)
    │ - Threshold check  │
    │ - Approval Rules   │
    └────────────────────┘
         ↓
    ┌────────────────────┐
    │ Response + Log     │  (1ms)
    │ - Decision         │
    │ - Confidence       │
    └────────────────────┘

         ↓
    Result to Gateway
    (Approve/Review/Block)


┌────────────────────────────────────────────────────────────────────────────┐
│                     OFFLINE TRAINING PIPELINE                              │
└────────────────────────────────────────────────────────────────────────────┘

    Schedule: Weekly (Monday midnight)
    
    Day 1-30: Training Data Window
         ↓
    ┌──────────────────────────────┐
    │ Data Loading (Spark)         │
    │ - Read from Data Warehouse   │
    │ - Partition by day           │
    │ - Validate schema            │
    └──────────────────────────────┘
         ↓
    ┌──────────────────────────────┐
    │ Feature Computation          │
    │ - User 30-day aggregates     │
    │ - Merchant patterns          │
    │ - Device history             │
    │ - Temporal features          │
    └──────────────────────────────┘
         ↓
    ┌──────────────────────────────┐
    │ Data Validation              │
    │ - Great Expectations         │
    │ - Statistical checks         │
    │ - Quality assurance          │
    └──────────────────────────────┘
         ↓
    ┌──────────────────────────────┐
    │ Feature Engineering          │
    │ - Scaling (RobustScaler)     │
    │ - Encoding (Label)           │
    │ - Selection (SelectKBest)    │
    │ - Store in Feature Store v2  │
    └──────────────────────────────┘
         ↓
    ┌──────────────────────────────┐
    │ Train/Val/Test Split         │
    │ - Temporal split (no leakage) │
    │ - 70% / 15% / 15%           │
    │ - Equal fraud rate           │
    └──────────────────────────────┘
         ↓
    ┌──────────────────────────────┐
    │ Model Training               │
    │ - Baseline (LogReg)          │
    │ - Intermediate (XGBoost)     │
    │ - SotA (Neural Net)          │
    │ - MLflow tracking            │
    └──────────────────────────────┘
         ↓
    ┌──────────────────────────────┐
    │ Evaluation & Testing         │
    │ - Bootstrap CI               │
    │ - Statistical tests          │
    │ - Fairness analysis          │
    │ - Comparison vs. production  │
    └──────────────────────────────┘
         ↓
    ┌──────────────────────────────┐
    │ Champion/Challenger Logic    │
    │ - New AUC > old by 0.5%?    │
    │ - Fairness metrics OK?       │
    │ - Business metrics positive? │
    └──────────────────────────────┘
            ↓
         YES → Model Registry (MLflow)
            ↓
         Staging → Canary (5% traffic) → Full Rollout
            ↓
         NO → Alert + Investigation
```

---

## 2. System Components

### 2.1 Data Ingestion Layer

**Input**: Transaction events from payment gateway

**Components**:
- **Kafka Topic**: `transactions-raw`
  - Partitions: 32 (for parallelism)
  - Replication factor: 3 (for durability)
  - Retention: 7 days
  - Format: Avro (schema evolution)

- **Message Schema**:
  ```json
  {
    "transaction_id": "txn_abc123",
    "timestamp": "2024-01-15T14:32:45Z",
    "user_id": 12345,
    "merchant_id": 67890,
    "amount": 150.00,
    "device_type": "mobile",
    "user_country": "US",
    "merchant_category": "retail"
  }
  ```

- **SLA**: <1ms latency, 99.99% availability

### 2.2 Feature Engineering Layer

**Real-time Feature Computation** (Streaming)

```python
# Kafka Streams topology
stream = (
    kafka_stream
    .filter(lambda x: validate_schema(x))
    .branch(
        (lambda x: x['amount'] > 10000, lambda x: handle_high_value(x)),
        (lambda x: is_repeat_user(x), lambda x: compute_user_history(x)),
    )
    .aggregate(
        initializer=lambda: {},
        aggregator=lambda state, x: update_state(state, x),
        window_size='5-minute-tumbling-window'
    )
    .map(enrich_with_merchant_data)
)
```

**Feature Store** (Batch + Real-time)

```
Feature Store Architecture:

┌─────────────────────────────────────────────┐
│         Feature Store Layer                  │
├─────────────────────────────────────────────┤
│                                              │
│  Online Store (Low Latency)                  │
│  ┌──────────────────────────────────────┐   │
│  │ Redis Cache                          │   │
│  │ - User features (30-day window)      │   │
│  │ - Merchant patterns                  │   │
│  │ - Device history                     │   │
│  │ TTL: 24 hours                        │   │
│  └──────────────────────────────────────┘   │
│                                              │
│  Offline Store (Batch)                       │
│  ┌──────────────────────────────────────┐   │
│  │ Parquet files (S3/HDFS)              │   │
│  │ - Historical snapshots               │   │
│  │ - Versioned (v1.0, v1.1, v2.0)      │   │
│  │ - Lineage tracking                   │   │
│  └──────────────────────────────────────┘   │
│                                              │
│  Metadata Layer                              │
│  ┌──────────────────────────────────────┐   │
│  │ Feature Registry (MLflow)            │   │
│  │ - Feature definitions                │   │
│  │ - Data types & ranges                │   │
│  │ - Ownership & SLA                    │   │
│  └──────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
```

### 2.3 Model Serving Layer

**Infrastructure**:

```
┌─────────────────────────────────────────────┐
│      FastAPI Inference Server               │
│      (gunicorn + uvicorn)                   │
├─────────────────────────────────────────────┤
│                                              │
│  Worker Pool: 8 workers                      │
│  - Each handles up to 1,000 req/s           │
│  - Total capacity: ~8,000 req/s             │
│                                              │
│  Request Processing:                         │
│  1. Validate input (200μs)                   │
│  2. Fetch features from cache (5ms)          │
│  3. Model inference (3ms)                    │
│  4. Apply decision rules (1ms)               │
│  5. Log + respond (1ms)                      │
│                                              │
│  Caching:                                    │
│  - Redis: User/merchant/device features     │
│  - Local: Model loaded in memory            │
│  - CDN: Static configs                       │
│                                              │
└─────────────────────────────────────────────┘
   ↓
┌─────────────────────────────────────────────┐
│  Load Balancer (NGINX / AWS ELB)            │
│  - Round-robin across 4 server instances    │
│  - Health checks every 5 seconds            │
│  - Auto-scale: +1 instance if >70% CPU      │
└─────────────────────────────────────────────┘
```

**Latency Budget Analysis**:

| Component | Budget | Typical | P99 |
|-----------|--------|---------|-----|
| Network | 1ms | 0.5ms | 2ms |
| Feature fetch | 5ms | 2ms | 8ms |
| Model inference | 3ms | 2ms | 5ms |
| Decision logic | 1ms | 0.3ms | 1ms |
| **Total** | **10ms** | **4.8ms** | **16ms** |

→ P99 exceeds 10ms in rare cases (DB fallback for uncached features)

### 2.4 Monitoring Layer

**Metrics Collection**:

```
Real-time Metrics
├── Throughput: TPS
├── Latency: P50, P95, P99
├── Error rate: Network errors, model errors
├── Cache hit rate: % of features from cache
├── Model drift: PSI of prediction distribution
├── Data drift: PSI of input features
└── Business metrics: False positive rate, fraud detection rate

Timeseries DB: Prometheus
Visualization: Grafana
Alerting: AlertManager

Alert Examples:
- P99 latency > 15ms → Page on-call
- Cache hit rate < 80% → Increase TTL or capacity
- PSI > 0.1 → Schedule drift investigation
- False positive rate > 2.5% → Review threshold
```

**Logging**:

```json
{
  "timestamp": "2024-01-15T14:32:45.123Z",
  "transaction_id": "txn_abc123",
  "user_id": 12345,
  "amount": 150.00,
  "features_latency_ms": 2.1,
  "model_inference_latency_ms": 2.9,
  "fraud_probability": 0.185,
  "decision": "approve",
  "model_version": "1.0.0",
  "request_id": "req_xyz789"
}
```

### 2.5 Model Registry & Versioning

**MLflow Setup**:

```
MLflow Server
├── Backend Store (Database)
│   └── Experiments table
│       ├── experiment_id
│       ├── run_id
│       ├── params (hyperparameters)
│       ├── metrics (AUC, precision, recall)
│       └── tags (prod, staging, etc.)
│
└── Artifact Store (S3)
    └── artifacts/
        ├── v1.0.0/
        │   ├── xgboost_model/
        │   ├── feature_importances.json
        │   ├── evaluation_report.html
        │   └── metrics.json
        ├── v1.1.0/
        └── v2.0.0/
```

**Model Promotion**:

```
Development → Staging → Production
   ↓            ↓            ↓
Local eval  Canary test  5% → 100% rollout
```

---

## 3. Scalability Considerations

### 3.1 Horizontal Scaling

**Current Setup** (handles 2B transactions/day):

- 4 server instances (can handle 10K req/s each)
- 1 Redis cluster (3 nodes, 100GB memory each)
- Kafka: 32 partitions
- Training: 1 Spark cluster (20 nodes)

**Growth Path**:

```
2B → 10B transactions/day:
├── Servers: 4 → 20 (10X)
├── Redis: 3 nodes → 10 nodes (more features cached)
├── Kafka: 32 → 128 partitions
├── Models: 1-day retraining → 6-hour retraining
└── Cost increase: ~5X (economy of scale helps)
```

### 3.2 Feature Computation Optimization

```
Current: ~5ms feature fetch

Optimization Options:
1. Increase Redis memory (TTL 24h → 48h)
   - Cost: +50GB memory
   - Benefit: +10% cache hit rate

2. Pre-compute more aggregations
   - Cost: +storage, +computation
   - Benefit: Eliminate DB lookups entirely

3. Use GPU for model inference
   - Cost: $10K/month GPU instances
   - Benefit: <1ms inference vs. 3ms (but overkill for current load)
```

---

## 4. Disaster Recovery & High Availability

### 4.1 Redundancy

```
Production Deployment:

Multi-region (3 regions):
├── Primary (us-east-1): 40% traffic
├── Secondary (eu-west-1): 40% traffic
└── Backup (ap-southeast-1): 20% traffic

Cross-region feature store replication:
├── Every 5 minutes
├── Latency: <500ms
└── Consistency: eventual

Database failover:
├── Standby DB auto-promoted (RTO <5min)
├── Continuous replication
└── Weekly failover test
```

### 4.2 Rollback Strategy

```
If production model breaks:
1. Immediate: Route 100% to previous model (blue-green)
2. Dashboard alert: notify ML team
3. Investigation: compare metrics, data drift
4. Solution: either fix + redeploy OR wait for next training cycle

Rollback time: < 2 minutes (infrastructure SLA)
Data loss: 0 (all requests logged)
```

---

## 5. Security & Compliance

### 5.1 Data Privacy

- **PII Handling**: User/merchant IDs hashed before logging
- **Encryption**: TLS 1.3 in transit, AES-256 at rest
- **Access Control**: Role-based (engineers, data scientists, on-call)
- **Audit Logging**: All model decisions logged for compliance

### 5.2 Model Explanation (GDPR/CCPA)

```
On fraud block:
├── User can request explanation
├── SHAP values computed post-hoc
├── Top 3 contributing features returned
└── Appeal process: manual review within 24h
```

---

## 6. Cost Breakdown (Monthly)

| Component | Cost | Notes |
|-----------|------|-------|
| Compute (EC2) | $5,000 | 4 large instances |
| Redis | $3,000 | 3-node cluster |
| Data Storage | $2,000 | S3 for training data + models |
| Kafka/Messaging | $1,000 | Managed Kafka service |
| MLflow/Tracking | $500 | Experiment tracking |
| Monitoring (DataDog) | $2,000 | Logs + metrics |
| **Total** | **$13,500** | **Per month** |

**ROI**: Fraud prevented (~$85M/year) vs. cost ($162K/year) = **500:1 ROI**

---

## 7. Future Architecture Enhancements

- [ ] Real-time feature pipeline (<1ms) using Apache Flink
- [ ] Graph-based fraud detection (user-merchant-device network)
- [ ] Causal inference for policy recommendations
- [ ] Multi-armed bandit for dynamic threshold optimization
- [ ] Federated learning for privacy-preserving improvements

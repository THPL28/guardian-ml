# Problem Formulation: Real-Time Fraud Detection at Scale

**Version**: 1.0  
**Date**: March 2026  
**Status**: Finalized  

---

## Executive Summary

We formalize a **binary classification problem** to detect fraudulent transactions in a real-time payment system processing 2 billion transactions daily.

**Key Challenge**: Extreme class imbalance (0.3% fraud rate) combined with strict latency requirements (<10ms) and high business impact.

---

## 1. Business Context

### 1.1 The Problem

A global payments platform processes **2 billion transactions/day** across 150+ countries.

**Fraud Impact**:
- **Annual loss**: $1+ billion USD (undetected fraud)
- **0.3-0.5% of transactions**: Fraudulent (3-5 million/day)
- **False positive cost**: $5-10 per transaction (user friction, support overhead)
- **False negative cost**: $500-2,000 per undetected fraud (chargeback + investigation)

### 1.2 Business Metrics (North Star)

| Metric | Target | Rationale |
|--------|--------|-----------|
| **Fraud Detection Rate (Recall)** | > 85% | Prevent $850M+ in annual loss |
| **False Positive Rate** | < 2% | Keep user experience friction acceptable |
| **ROI (Fraud Prevented / Cost of False Alarms)** | > 50:1 | Financial sustainability |
| **Latency (P99)** | < 10ms | Real-time transaction approvals |
| **Model AUC-ROC** | > 0.95 | Excellent separation between fraud and legitimate |

### 1.3 Scale & Technical Constraints

| Constraint | Value | Impact |
|-----------|-------|--------|
| **Daily Transactions** | 2 billion | Distributed system required |
| **Transactions/Second** | ~23,000 TPS | Real-time inference infrastructure |
| **Feature Computation Budget** | 5ms | Eliminate complexity to meet latency |
| **Model Inference Budget** | 3ms | Can't use slow models (e.g., ensemble chains) |
| **Total Latency Budget** | 10ms | P99 requirement |
| **Availability Target** | 99.99% | Can afford ~4s downtime/month |
| **Fairness Requirement** | No disparate impact | Must not discriminate by geography/demographics |

### 1.4 Trade-offs

```
RECALL vs PRECISION
  ↑ Recall (catch more fraud)    ↔ Fewer false alarms, better UX
  ↓ Precision (fewer alarms)     ↔ More undetected fraud

LATENCY vs ACCURACY
  ↓ Latency (fast inference)     ↔ Simple model, lower AUC
  ↑ Accuracy (complex model)     ↔ Slower inference

COST(FN) vs COST(FP)
  Minimize Cost(FN)              ↔ Block more legitimate transactions
  Minimize Cost(FP)              ↔ Miss more fraud
```

**Solution**: Operate at **optimal point** where:
```
ΔCost = Cost(FN) × Δrecall - Cost(FP) × ΔFP_rate = 0
```

---

## 2. Mathematical Formulation

### 2.1 Problem Definition

**Given**:
- Feature vector: $\mathbf{x} \in \mathbb{R}^d$ (transaction features)
- Binary label: $y \in \{0, 1\}$ where 1 = fraud, 0 = legitimate
- Training data: $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^{n}$
  - $n = 10^7$ (10 million samples) in training
  - $\sum_i \mathbb{1}[y_i=1] / n \approx 0.003$ (0.3% fraud rate)

**Objective**: Learn a function $f_\theta: \mathbb{R}^d \to [0,1]$ that:
1. Predicts fraud probability: $\hat{p}_i = f_\theta(\mathbf{x}_i)$
2. Maximizes recall (catch fraud)
3. Minimizes false positive rate (preserve UX)
4. Respects latency budget

### 2.2 Probabilistic Formulation (Bayesian)

Model the posterior probability of fraud given features:

$$P(y=1|\mathbf{x}, \theta) = \sigma(\mathbf{w}^T \mathbf{x} + b)$$

where:
- $\sigma(z) = \frac{1}{1 + e^{-z}}$ is the logistic sigmoid function
- $\theta = (\mathbf{w}, b)$ are learnable parameters
- $\mathbf{w} \in \mathbb{R}^d$ is the weight vector
- $b \in \mathbb{R}$ is the bias term

**Intuition**: Combines feature information linearly and maps to probability via sigmoid.

### 2.3 Likelihood Function (Maximum Likelihood Estimation)

For dataset $\mathcal{D}$, the **likelihood** under Bernoulli distribution:

$$L(\theta | \mathcal{D}) = \prod_{i=1}^{n} P(y_i|\mathbf{x}_i, \theta)^{y_i} \cdot (1-P(y_i|\mathbf{x}_i, \theta))^{1-y_i}$$

Expand:
$$L(\theta | \mathcal{D}) = \prod_{i=1}^{n} \hat{p}_i^{y_i} (1-\hat{p}_i)^{1-y_i}$$

where $\hat{p}_i = P(y_i=1|\mathbf{x}_i, \theta)$

**Numerical issue**: This product is tiny (approaches 0). Take **negative log-likelihood**:

$$\ell(\theta) = -\log L(\theta) = -\sum_{i=1}^{n} [y_i \log \hat{p}_i + (1-y_i) \log(1-\hat{p}_i)]$$

This is the **binary cross-entropy loss**.

### 2.4 Loss Function with Class Weighting

To handle extreme class imbalance (99.7% negative, 0.3% positive), use **weighted cross-entropy**:

$$\mathcal{L}_{weighted}(\theta) = -\frac{1}{n} \sum_{i=1}^{n} w_{y_i} [y_i \log \hat{p}_i + (1-y_i) \log(1-\hat{p}_i)]$$

where class weights:

$$w_+ = \frac{n}{2 n_+}, \quad w_- = \frac{n}{2 n_-}$$

**Derivation**:
- $n_+ = \sum_i \mathbb{1}[y_i=1]$ = count of positive samples (frauds)
- $n_- = n - n_+$ = count of negative samples (legitimate)
- Each class contributes equally to loss

**Numerical Example** ($n=10^6$, fraud rate 0.3%):
- $n_+ = 3,000$, $n_- = 997,000$
- $w_+ = \frac{10^6}{2 \times 3,000} = 166.67$
- $w_- = \frac{10^6}{2 \times 997,000} = 0.502$

**Effect**: Fraud examples weighted 333× more heavily than legitimate examples, ensuring model doesn't ignore them.

### 2.5 Regularization (L2 Penalty)

To prevent overfitting on training data, add **L2 regularization**:

$$\mathcal{L}_{total}(\theta) = \mathcal{L}_{weighted}(\theta) + \frac{\lambda}{2} ||\mathbf{w}||_2^2$$

$$= -\frac{1}{n} \sum_{i=1}^{n} w_{y_i} [y_i \log \hat{p}_i + (1-y_i) \log(1-\hat{p}_i)] + \frac{\lambda}{2} \sum_{j=1}^{d} w_j^2$$

**Intuition**:
- Penalizes large weights (prevents overfitting to noise)
- Encourages smooth, generalizable decision boundaries
- $\lambda$ (hyperparameter) controls regularization strength

**Bias-Variance Trade-off**:
- Small $\lambda$: Low bias, high variance (overfitting)
- Large $\lambda$: High bias, low variance (underfitting)
- Optimal $\lambda$: Found via cross-validation

### 2.6 Gradient Descent Optimization

To minimize $\mathcal{L}_{total}(\theta)$, use **stochastic gradient descent**:

$$\theta^{(t+1)} = \theta^{(t)} - \alpha \nabla_\theta \mathcal{L}_{total}(\theta^{(t)})$$

where $\alpha$ is the learning rate (step size).

**Gradient derivation** for a single sample $i$:

$$\frac{\partial \mathcal{L}}{\partial w_j} = w_{y_i} (\hat{p}_i - y_i) x_{ij} + \lambda w_j$$

$$\frac{\partial \mathcal{L}}{\partial b} = w_{y_i} (\hat{p}_i - y_i)$$

**Interpretation**:
- Error term $(\hat{p}_i - y_i)$ drives update direction
- If $\hat{p}_i > y_i$ (overestimating), reduce weights
- If $\hat{p}_i < y_i$ (underestimating), increase weights
- Feature magnitude $x_{ij}$ scales the update
- L2 term pulls weights toward zero

**Batch update** for all samples:

$$\mathbf{w} := \mathbf{w} - \alpha \left[ \frac{1}{n} \sum_{i=1}^{n} w_{y_i} (\hat{p}_i - y_i) \mathbf{x}_i + \lambda \mathbf{w} \right]$$

### 2.7 Key Assumptions

1. **Conditional Independence**: Transactions independent given features
   - Violated when user's devices/patterns are related
   - Mitigation: Add temporal features to capture dependencies

2. **Feature Signal**: Features contain predictive information
   - Verify via feature importance / correlation analysis
   - Sanity check: AUC > 0.6 with simple features

3. **Stationarity**: Fraud patterns don't drift over time
   - Violated: New fraud techniques emerge
   - Mitigation: Regular model retraining (weekly/monthly)

4. **No Perfect Separation**: No single feature perfectly predicts fraud
   - If violated: Model may have > 99% AUC (suspect)
   - Check via domain knowledge

5. **Causal Features**: Features are not consequences of fraud
   - Bad: Use "chargeback_received" (happens after fraud detected)
   - Good: Use "user_historical_chargeback_rate" (before transaction)

6. **Label Quality**: Training labels are accurate
   - Fraud labels from investigations/customer reports (may be noisy)
   - Mitigation: Label audit / confidence weighting

---

## 3. Data Engineering Strategy

### 3.1 Data Pipeline Architecture

```
Real-time Stream (Kafka)
│
├─────→ Validation (Great Expectations)
│       - Schema checks
│       - Statistical anomaly detection
│
├─────→ Feature Computation (Spark Streaming)
│       - User aggregations (30-day windows)
│       - Merchant patterns
│       - Device behavior
│
├─────→ Feature Store (Versioned)
│       - v1.0.0: Initial features
│       - v1.1.0: With interactions
│       - v2.0.0: Enhanced signals
│
├─────→ Model Inference
│       - Low-latency serving
│       - Redis caching
│
└─────→ Scoring Pipeline
        - Fraud probability
        - Decision (approve/review/block)
```

### 3.2 Feature Categories

**User Features**:
- `user_age_group`: Age demographic
- `user_account_age_days`: Account tenure
- `user_transaction_count_30d`: Recent activity
- `user_fraud_report_count`: Fraud history

**Transaction Features**:
- `amount_log`: Log-transformed amount (handle skew)
- `time_of_day`: Hour of transaction
- `is_night`: 10pm-6am (higher fraud)
- `day_of_week`: Temporal pattern

**Merchant Features**:
- `merchant_category`: Merchant type (merchant_category_risk varies)
- `merchant_fraud_rate_30d`: Merchant trustworthiness
- `merchant_chargeback_rate`: Dispute history

**Device Features**:
- `device_type`: Mobile/desktop/tablet
- `device_fraud_rate_30d`: Device history
- `known_device_to_user`: User-device relationship

**Relationship Features**:
- `user_merchant_interaction_count`: Interaction history
- `user_country_merchant_country_match`: Geographic match

### 3.3 Data Leakage Prevention

**Temporal Integrity Rules**:

| ❌ WRONG | ✅ CORRECT |
|---------|-----------|
| `fraud_reported_today` | `fraud_count_historical` |
| `chargeback_received` | `chargeback_rate_30d` |
| `support_contacted` | `support_contact_count_30d` |
| `account_frozen` | `account_age` |

**Implementation**:
```python
# Training: day D uses data from [D-30, D-1]
features_day_D = compute_aggregate_features(
    data_start=D-30,
    data_end=D-1,
    target_date=D
)

# Serving: current time T uses data before T
features_now = compute_aggregate_features(
    data_start=now()-30days,
    data_end=now()-1minute,
    target_date=now()
)
```

### 3.4 Class Imbalance Handling

**Strategy**: Weighted loss + threshold optimization

1. **In data pipeline**:
   ```
   Loss_weighted = w_pos * Loss_fraud + w_neg * Loss_legitimate
   
   w_pos = n / (2 * n_frauds) = 10^7 / (2 * 30,000) = 166
   w_neg = n / (2 * n_legit) = 10^7 / (2 * 9.97M) = 0.5
   ```

2. **In decision boundary**:
   ```
   Default threshold = 0.5
   Optimal threshold = argmin_t [FN(t) * cost_fn + FP(t) * cost_fp]
   
   Example: optimal_threshold ≈ 0.1 (lower bar for fraud detection)
   ```

3. **Alternative** (if needed):
   ```
   SMOTE: Generate synthetic fraudDroid examples
   Undersampling: Sample subset of legitimate
   ```

---

## 4. Model Architecture Progression

### Model 1: Logistic Regression (Baseline)

**Complexity**: O(d) parameters  
**Interpretability**: High (see coefficients)  
**Latency**: <1ms  

### Model 2: XGBoost (Intermediate)

**Complexity**: O(n_trees × d²)  
**Interpretability**: Medium (feature importance)  
**Latency**: 3-5ms  

### Model 3: Neural Network (SotA)

**Complexity**: O(d × h × l) where h=hidden size, l=layers  
**Interpretability**: Low (black box)  
**Latency**: 5-10ms  

---

## 5. Evaluation Framework

### 5.1 Metrics

**Primary**:
- AUC-ROC: Overall discrimination (0.5 baseline, 1.0 perfect)
- AUC-PR: Focus on minority class

**Secondary**:
- Precision: When we predict fraud, how often right?
- Recall: Of actual frauds, what % do we catch?
- F1: Harmonic mean (balanced)

### 5.2 Statistical Rigor

**Confidence Intervals**: Bootstrap (1,000 resamples)
- If AUC = 0.92 [0.910, 0.930], we're 95% confident true AUC ∈ [0.910, 0.930]

**Hypothesis Testing**: Permutation test
- H0: Model A and Model B equally good
- H1: They differ
- p-value < 0.05 → statistically significant difference

### 5.3 Fairness Analysis

Compare performance across demographics:
- Precision by country
- Recall by device type
- False positive rate by user age
- Check for disparate impact (legal requirement)

---

## 6. Expected Results

| Model | AUC-ROC | Latency | Interpretability |
|-------|---------|---------|------------------|
| Logistic Regression | 0.78 | <1ms | High |
| XGBoost | 0.92 | 3-5ms | Medium |
| Neural Network | 0.95 | 5-10ms | Low |

---

## 7. Deployment Considerations

### 7.1 Real-time Serving

```
Request → Feature Computation (5ms) → Model Inference (3ms) → Decision (2ms)
          └─────────────────────────────────────────────────────┘
                          Total: <10ms (P99)
```

### 7.2 Monitoring

- **Data drift**: Monthly drift test on feature distributions
- **Model drift**: Weekly performance check on holdout test set
- **Fairness drift**: Monthly fairness audit by geography

### 7.3 Retraining

- **Trigger**: Weekly retraining with 4 weeks of data
- **Validation**: Must beat production model on holdout before deploy
- **Rollback**: Keep previous model for safety

---

## 8. Conclusion

This formulation provides:
✅ Clear business objectives (fraud detection, latency, fairness)  
✅ Rigorous mathematics (Bayesian, MLE, gradient descent)  
✅ Scale-aware engineering (100M+ features, <10ms latency)  
✅ Evaluation rigor (CI, hypothesis testing, fairness)  
✅ Production readiness (monitoring, retraining, alerting)  

This is a complex, realistic, Big Tech-level problem suitable for senior interviewing.

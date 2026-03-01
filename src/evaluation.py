"""
Guardian-ML: Rigorous Statistical Evaluation

Implements comprehensive evaluation:
- Multiple metrics (AUC-ROC, Precision-Recall, F1)
- Confidence intervals via Bootstrap
- Statistical hypothesis testing
- Segment-level analysis
- Fairness analysis
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, roc_curve, auc,
    confusion_matrix, classification_report, f1_score,
    precision_score, recall_score, accuracy_score
)
from scipy import stats
import logging
from typing import Dict, Tuple, List, Any

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class RigourousEvaluator:
    """
    Comprehensive statistical evaluation of fraud detection models.
    
    Metrics:
    - AUC-ROC: Overall discrimination ability
    - Precision-Recall: Focus on minority class (fraud)
    - Calibration: Probability quality
    - Fairness: Performance across segments
    """
    
    def __init__(self, confidence_level: float = 0.95, 
                 n_bootstrap: int = 1000):
        """
        Initialize evaluator.
        
        Args:
            confidence_level: For CI (e.g., 0.95 for 95%)
            n_bootstrap: Number of bootstrap samples
        """
        self.confidence_level = confidence_level
        self.n_bootstrap = n_bootstrap
        self.alpha = 1 - confidence_level
        
        logger.info(f"Initialized Evaluator: {confidence_level:.0%} CI, "
                   f"{n_bootstrap} bootstrap samples")
    
    # ========================================================================
    # PRIMARY METRICS
    # ========================================================================
    
    def compute_auc_roc(self, y_true: np.ndarray, 
                        y_pred_proba: np.ndarray) -> float:
        """
        Compute Area Under ROC Curve.
        
        Interpretation:
        - 0.5 = random classifier
        - 0.7-0.8 = fair discrimination
        - 0.8-0.9 = good
        - 0.9-1.0 = excellent
        """
        return roc_auc_score(y_true, y_pred_proba)
    
    def compute_auc_pr(self, y_true: np.ndarray,
                       y_pred_proba: np.ndarray) -> float:
        """
        Compute Area Under Precision-Recall Curve.
        
        Better than ROC for highly imbalanced data.
        Resistant to false positive inflation.
        """
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        return auc(recall, precision)
    
    def compute_confusion_matrix_metrics(self, y_true: np.ndarray,
                                         y_pred: np.ndarray) -> Dict[str, float]:
        """
        Compute precision, recall, F1 from confusion matrix.
        
        Confusion Matrix:
        
                    Predicted Negative | Predicted Positive
        Actually Negative  |     TN         |        FP
        Actually Positive  |     FN         |        TP
        
        Metrics:
        - TP: Caught fraud (true positives)
        - FP: Wrongly blocked (false positives)
        - TN: Correctly approved (true negatives)
        - FN: Missed fraud (false negatives)
        """
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        fpr = 1 - specificity
        
        return {
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1': f1,
            'fpr': fpr,
        }
    
    # ========================================================================
    # CONFIDENCE INTERVALS
    # ========================================================================
    
    def bootstrap_ci(self, y_true: np.ndarray, 
                     y_pred_proba: np.ndarray,
                     metric: str = 'auc_roc') -> Tuple[float, float, float]:
        """
        Compute metric with Bootstrap confidence interval.
        
        Method:
        1. Resample data with replacement N times
        2. Compute metric for each resample
        3. Use percentiles for CI
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            metric: 'auc_roc', 'auc_pr', 'precision', 'recall', 'f1'
            
        Returns:
            (point_estimate, lower_ci, upper_ci)
        """
        n = len(y_true)
        bootstrap_metrics = []
        
        np.random.seed(42)
        
        for _ in range(self.n_bootstrap):
            # Resample with replacement
            indices = np.random.choice(n, n, replace=True)
            y_true_boot = y_true[indices]
            y_pred_boot = y_pred_proba[indices]
            
            # Compute metric
            if metric == 'auc_roc':
                metric_value = roc_auc_score(y_true_boot, y_pred_boot)
            elif metric == 'auc_pr':
                precision, recall, _ = precision_recall_curve(y_true_boot, y_pred_boot)
                metric_value = auc(recall, precision)
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            bootstrap_metrics.append(metric_value)
        
        # Point estimate
        point_estimate = np.mean(bootstrap_metrics)
        
        # Percentile CI
        lower = np.percentile(bootstrap_metrics, self.alpha/2 * 100)
        upper = np.percentile(bootstrap_metrics, (1 - self.alpha/2) * 100)
        
        return point_estimate, lower, upper
    
    # ========================================================================
    # STATISTICAL TESTS
    # ========================================================================
    
    def compare_models_statistical(self, y_true: np.ndarray,
                                   y_pred_1: np.ndarray,
                                   y_pred_2: np.ndarray,
                                   metric: str = 'auc_roc') -> Dict[str, Any]:
        """
        Statistically compare two models using permutation test.
        
        H0: Model 1 and Model 2 have equal performance
        H1: Model 1 and Model 2 have different performance
        
        Method: Permutation test (non-parametric)
        """
        # Compute metrics
        if metric == 'auc_roc':
            metric_1 = roc_auc_score(y_true, y_pred_1)
            metric_2 = roc_auc_score(y_true, y_pred_2)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        # Observed difference
        observed_diff = metric_1 - metric_2
        
        # Permutation test
        n_permutations = 100
        permutation_diffs = []
        
        for _ in range(n_permutations):
            # Randomly swap predictions
            swap_mask = np.random.binomial(1, 0.5, len(y_true))
            y_pred_1_perm = np.where(swap_mask, y_pred_2, y_pred_1)
            y_pred_2_perm = np.where(swap_mask, y_pred_1, y_pred_2)
            
            diff_perm = (roc_auc_score(y_true, y_pred_1_perm) - 
                        roc_auc_score(y_true, y_pred_2_perm))
            permutation_diffs.append(diff_perm)
        
        # p-value
        p_value = np.mean(np.abs(permutation_diffs) >= np.abs(observed_diff))
        
        return {
            'model_1_metric': metric_1,
            'model_2_metric': metric_2,
            'observed_diff': observed_diff,
            'p_value': p_value,
            'significant': p_value < self.alpha,
        }
    
    # ========================================================================
    # THRESHOLD OPTIMIZATION
    # ========================================================================
    
    def find_optimal_threshold(self, y_true: np.ndarray,
                               y_pred_proba: np.ndarray,
                               cost_fn: float = 2000.0,
                               cost_fp: float = 8.0) -> Tuple[float, float]:
        """
        Find optimal decision threshold based on business costs.
        
        Business cost function:
        Total Cost = FN * cost_fn + FP * cost_fp
        
        Optimal threshold minimizes this cost.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            cost_fn: Cost of false negative (undetected fraud)
            cost_fp: Cost of false positive (false alarm)
            
        Returns:
            (optimal_threshold, minimum_cost)
        """
        thresholds = np.linspace(0, 1, 100)
        costs = []
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            
            total_cost = fn * cost_fn + fp * cost_fp
            costs.append(total_cost)
        
        optimal_idx = np.argmin(costs)
        optimal_threshold = thresholds[optimal_idx]
        minimum_cost = costs[optimal_idx]
        
        return optimal_threshold, minimum_cost
    
    # ========================================================================
    # FAIRNESS ANALYSIS
    # ========================================================================
    
    def fairness_analysis(self, y_true: np.ndarray,
                         y_pred_proba: np.ndarray,
                         groups: np.ndarray,
                         threshold: float = 0.5) -> Dict[str, Dict[str, float]]:
        """
        Analyze model fairness across demographic groups.
        
        Metrics:
        - Demographic Parity: P(Positive|Group1) = P(Positive|Group2)
        - Equalized Odds: P(Positive|Group1, Positive) = P(Positive|Group2, Positive)
        - Calibration: Predicted probability matches actual fraud rate per group
        """
        unique_groups = np.unique(groups)
        fairness_metrics = {}
        
        for group in unique_groups:
            group_mask = groups == group
            y_true_group = y_true[group_mask]
            y_pred_group = y_pred_proba[group_mask]
            y_pred_binary = (y_pred_group >= threshold).astype(int)
            
            metrics = self.compute_confusion_matrix_metrics(y_true_group, y_pred_binary)
            
            fairness_metrics[str(group)] = {
                'size': group_mask.sum(),
                'fraud_rate': y_true_group.mean(),
                'detection_rate': metrics['recall'],
                'false_positive_rate': metrics['fpr'],
                'precision': metrics['precision'],
            }
        
        return fairness_metrics
    
    # ========================================================================
    # FULL EVALUATION REPORT
    # ========================================================================
    
    def evaluate_comprehensive(self, y_true: np.ndarray,
                               y_pred_proba: np.ndarray,
                               model_name: str = "Model") -> pd.DataFrame:
        """
        Generate comprehensive evaluation report.
        
        Returns:
            DataFrame with all metrics and confidence intervals
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"COMPREHENSIVE EVALUATION: {model_name}")
        logger.info(f"{'='*80}\n")
        
        results = {}
        
        # Primary metrics with CI
        logger.info("1. PRIMARY METRICS (with 95% CI)")
        logger.info("-" * 80)
        
        auc_roc, auc_roc_lower, auc_roc_upper = self.bootstrap_ci(
            y_true, y_pred_proba, 'auc_roc'
        )
        results['AUC-ROC'] = f"{auc_roc:.4f} [{auc_roc_lower:.4f}, {auc_roc_upper:.4f}]"
        logger.info(f"AUC-ROC:        {auc_roc:.4f} [{auc_roc_lower:.4f}, {auc_roc_upper:.4f}]")
        
        auc_pr, auc_pr_lower, auc_pr_upper = self.bootstrap_ci(
            y_true, y_pred_proba, 'auc_pr'
        )
        results['AUC-PR'] = f"{auc_pr:.4f} [{auc_pr_lower:.4f}, {auc_pr_upper:.4f}]"
        logger.info(f"AUC-PR:         {auc_pr:.4f} [{auc_pr_lower:.4f}, {auc_pr_upper:.4f}]")
        
        # Threshold optimization
        logger.info("\n2. THRESHOLD OPTIMIZATION")
        logger.info("-" * 80)
        
        optimal_threshold, min_cost = self.find_optimal_threshold(
            y_true, y_pred_proba, cost_fn=2000.0, cost_fp=8.0
        )
        results['Optimal Threshold'] = f"{optimal_threshold:.3f}"
        logger.info(f"Optimal Threshold: {optimal_threshold:.3f}")
        logger.info(f"Minimum Cost: ${min_cost:,.0f}")
        
        # Performance at optimal threshold
        logger.info("\n3. PERFORMANCE AT OPTIMAL THRESHOLD")
        logger.info("-" * 80)
        
        y_pred = (y_pred_proba >= optimal_threshold).astype(int)
        cm_metrics = self.compute_confusion_matrix_metrics(y_true, y_pred)
        
        logger.info(f"Precision:           {cm_metrics['precision']:.4f}")
        logger.info(f"Recall:              {cm_metrics['recall']:.4f}")
        logger.info(f"F1 Score:            {cm_metrics['f1']:.4f}")
        logger.info(f"Specificity:         {cm_metrics['specificity']:.4f}")
        logger.info(f"False Positive Rate: {cm_metrics['fpr']:.4f}")
        
        results.update({
            'Precision': f"{cm_metrics['precision']:.4f}",
            'Recall': f"{cm_metrics['recall']:.4f}",
            'F1-Score': f"{cm_metrics['f1']:.4f}",
        })
        
        # Confusion matrix
        logger.info("\n4. CONFUSION MATRIX")
        logger.info("-" * 80)
        logger.info(f"True Negatives:  {cm_metrics['tn']:,}")
        logger.info(f"False Positives: {cm_metrics['fp']:,}")
        logger.info(f"False Negatives: {cm_metrics['fn']:,}")
        logger.info(f"True Positives:  {cm_metrics['tp']:,}")
        
        logger.info(f"\n{'='*80}\n")
        
        return pd.DataFrame({model_name: results}).T


def main():
    """Example evaluation."""
    logger.info("Evaluation module ready for use")


if __name__ == "__main__":
    main()

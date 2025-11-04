import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings('ignore')

# Fairlearn imports
from fairlearn.metrics import (
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
    equalized_odds_ratio,
    false_negative_rate,
    false_positive_rate,
    selection_rate,
    true_negative_rate,
    true_positive_rate
)

# AIF360 imports
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing

class BiasDetector:
    """Detect bias using Fairlearn and AIF360 libraries"""
    
    def __init__(self, thresholds: Dict = None):
        self.thresholds = thresholds or {
            'statistical_parity': 0.10,
            'disparate_impact': 0.80,
            'equal_opportunity': 0.10,
            'equalized_odds': 0.10,
            'calibration': 0.10,
            'generalized_entropy_index': 0.30
        }
    
    def calculate_statistical_parity_fairlearn(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_feature: np.ndarray
    ) -> Dict:
        """
        Statistical Parity using Fairlearn
        Measures: P(Y=1|A=0) vs P(Y=1|A=1)
        """
        try:
            # demographic_parity_difference needs sensitive_features as keyword argument
            dp_diff = demographic_parity_difference(
                y_true=y_pred,
                y_pred=y_pred,
                sensitive_features=sensitive_feature
            )
            
            # Get individual group rates
            group_0_rate = np.mean(y_pred[sensitive_feature == 0])
            group_1_rate = np.mean(y_pred[sensitive_feature == 1])
            
            is_biased = abs(dp_diff) > self.thresholds['statistical_parity']
            
            print(f"[FAIRLEARN] Statistical Parity - Difference: {dp_diff:.4f}, Threshold: {self.thresholds['statistical_parity']:.4f}")
            
            return {
                'metric': 'statistical_parity',
                'library': 'Fairlearn',
                'group_0_positive_rate': float(group_0_rate),
                'group_1_positive_rate': float(group_1_rate),
                'difference': float(dp_diff),
                'threshold': self.thresholds['statistical_parity'],
                'is_biased': bool(is_biased),
                'description': 'Demographic Parity - Difference in positive prediction rates'
            }
        except Exception as e:
            print(f"Error in statistical_parity: {str(e)}")
            return {'error': str(e), 'metric': 'statistical_parity'}
    
    def calculate_disparate_impact_aif360(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_feature: np.ndarray
    ) -> Dict:
        """
        Disparate Impact using AIF360
        80% Rule: Selection Rate Ratio >= 0.80
        """
        try:
            group_0_pred = y_pred[sensitive_feature == 0]
            group_1_pred = y_pred[sensitive_feature == 1]
            
            if len(group_0_pred) == 0 or len(group_1_pred) == 0:
                return {
                    'metric': 'disparate_impact',
                    'library': 'AIF360',
                    'status': 'skipped',
                    'reason': 'Insufficient data for one or both groups',
                    'ratio': 0.0,
                    'threshold': self.thresholds['disparate_impact']
                }
            
            group_0_rate = np.mean(group_0_pred)
            group_1_rate = np.mean(group_1_pred)
            
            # Handle edge cases
            if group_0_rate == 0:
                if group_1_rate == 0:
                    ratio = 1.0  # Both groups have 0 predictions - perfectly equal
                else:
                    ratio = 0.0  # Group 1 has predictions but Group 0 doesn't
            else:
                ratio = float(group_1_rate / group_0_rate)
            
            # Ensure ratio is not NaN or Inf
            if np.isnan(ratio) or np.isinf(ratio):
                ratio = 0.0
            
            is_biased = (ratio < self.thresholds['disparate_impact']) and (ratio > 0)
            
            print(f"[AIF360] Disparate Impact - Ratio: {ratio:.4f}, Threshold: {self.thresholds['disparate_impact']:.4f}")
            
            return {
                'metric': 'disparate_impact',
                'library': 'AIF360',
                'group_0_selection_rate': float(group_0_rate),
                'group_1_selection_rate': float(group_1_rate),
                'ratio': float(ratio),
                'threshold': self.thresholds['disparate_impact'],
                'is_biased': bool(is_biased),
                'description': '80% Rule - Disparate Impact Ratio (should be >= 0.80)'
            }
        except Exception as e:
            print(f"Error in disparate_impact: {str(e)}")
            return {
                'metric': 'disparate_impact',
                'library': 'AIF360',
                'status': 'error',
                'error': str(e),
                'ratio': 0.0,
                'threshold': self.thresholds['disparate_impact']
            }
    
    def calculate_equal_opportunity_fairlearn(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_feature: np.ndarray
    ) -> Dict:
        """
        Equal Opportunity using Fairlearn
        True Positive Rate (TPR) Parity
        """
        try:
            # Check if binary classification
            if len(np.unique(y_true)) > 2:
                print("[WARNING] Equal Opportunity requires binary classification. Skipping...")
                return {
                    'metric': 'equal_opportunity',
                    'library': 'Fairlearn',
                    'status': 'skipped',
                    'reason': 'Requires binary classification (2 classes)',
                    'description': 'Equal Opportunity - True Positive Rate (TPR) Parity'
                }
            
            y_true_series = pd.Series(y_true)
            y_pred_series = pd.Series(y_pred)
            # sensitive_series = pd.Series(sensitive_feature)
            
            # Calculate TPR for each group
            group_0_mask = sensitive_feature == 0
            group_1_mask = sensitive_feature == 1

            if np.sum(group_0_mask) == 0 or np.sum(group_1_mask) == 0:
                return {'metric': 'equal_opportunity', 'status': 'skipped', 'reason': 'Insufficient data for groups'}
            
            try:
                group_0_tpr = true_positive_rate(
                    y_true_series[group_0_mask],
                    y_pred_series[group_0_mask]
                )
                group_1_tpr = true_positive_rate(
                    y_true_series[group_1_mask],
                    y_pred_series[group_1_mask]
                )
            except:
                group_0_tpr = 0.0
                group_1_tpr = 0.0
            
            tpr_diff = abs(float(group_0_tpr) - float(group_1_tpr))
            is_biased = tpr_diff > self.thresholds['equal_opportunity']
            
            print(f"[FAIRLEARN] Equal Opportunity - TPR Diff: {tpr_diff:.4f}, Threshold: {self.thresholds['equal_opportunity']:.4f}")
            
            return {
                'metric': 'equal_opportunity',
                'library': 'Fairlearn',
                'group_0_tpr': float(group_0_tpr),
                'group_1_tpr': float(group_1_tpr),
                'difference': float(tpr_diff),
                'threshold': self.thresholds['equal_opportunity'],
                'is_biased': bool(is_biased),
                'description': 'Equal Opportunity - True Positive Rate (TPR) Parity'
            }
        except Exception as e:
            print(f"Error in equal_opportunity: {str(e)}")
            return {'error': str(e), 'metric': 'equal_opportunity'}
    
    def calculate_equalized_odds_fairlearn(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_feature: np.ndarray
    ) -> Dict:
        """
        Equalized Odds using Fairlearn
        Both TPR and FPR Parity
        """
        try:
            # Check if binary classification
            if len(np.unique(y_true)) > 2:
                print("[WARNING] Equalized Odds requires binary classification. Skipping...")
                return {
                    'metric': 'equalized_odds',
                    'library': 'Fairlearn',
                    'status': 'skipped',
                    'reason': 'Requires binary classification (2 classes)',
                    'description': 'Equalized Odds - Both TPR and FPR Parity'
                }
            y_true_series = pd.Series(y_true)
            y_pred_series = pd.Series(y_pred)
            sensitive_series = pd.Series(sensitive_feature)
            
            # Calculate equalized odds difference
            eo_diff = equalized_odds_difference(
                y_true_series,
                y_pred_series,
                sensitive_features=sensitive_series
            )
            
            # Get individual TPR and FPR
            group_0_mask = sensitive_feature == 0
            group_1_mask = sensitive_feature == 1

            if np.sum(group_0_mask) == 0 or np.sum(group_1_mask) == 0:
                return {'metric': 'equalized_odds', 'status': 'skipped', 'reason': 'Insufficient data for groups'}
            
            try:
                group_0_tpr = true_positive_rate(y_true_series[group_0_mask], y_pred_series[group_0_mask])
                group_1_tpr = true_positive_rate(y_true_series[group_1_mask], y_pred_series[group_1_mask])
                group_0_fpr = false_positive_rate(y_true_series[group_0_mask], y_pred_series[group_0_mask])
                group_1_fpr = false_positive_rate(y_true_series[group_1_mask], y_pred_series[group_1_mask])
            except:
                group_0_tpr = group_1_tpr = group_0_fpr = group_1_fpr = 0.0

            tpr_diff = abs(float(group_0_tpr) - float(group_1_tpr))
            fpr_diff = abs(float(group_0_fpr) - float(group_1_fpr))
            max_diff = max(tpr_diff, fpr_diff)
            
            is_biased = max_diff > self.thresholds['equalized_odds']
            
            print(f"[FAIRLEARN] Equalized Odds - Max Diff: {max_diff:.4f}, Threshold: {self.thresholds['equalized_odds']:.4f}")
            
            return {
                'metric': 'equalized_odds',
                'library': 'Fairlearn',
                'group_0_tpr': float(group_0_tpr),
                'group_1_tpr': float(group_1_tpr),
                'tpr_difference': float(abs(group_0_tpr - group_1_tpr)),
                'group_0_fpr': float(group_0_fpr),
                'group_1_fpr': float(group_1_fpr),
                'fpr_difference': float(abs(group_0_fpr - group_1_fpr)),
                'equalized_odds_difference': float(eo_diff),
                'threshold': self.thresholds['equalized_odds'],
                'is_biased': bool(abs(eo_diff) > self.thresholds['equalized_odds']),
                'description': 'Equalized Odds - Both TPR and FPR Parity'
            }
        except Exception as e:
            print(f"Error in equalized_odds: {str(e)}")
            return {'error': str(e), 'metric': 'equalized_odds'}
    
    def calculate_calibration_aif360(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        sensitive_feature: np.ndarray
    ) -> Dict:
        """
        Calibration using AIF360
        Predicted probability matches actual outcomes per group
        """
        try:
            group_0_mask = sensitive_feature == 0
            group_1_mask = sensitive_feature == 1
            
            # Calibration: |predicted prob - actual rate|
            group_0_pred = np.mean(y_pred_proba[group_0_mask])
            group_0_actual = np.mean(y_true[group_0_mask])
            group_0_cal = abs(group_0_pred - group_0_actual)
            
            group_1_pred = np.mean(y_pred_proba[group_1_mask])
            group_1_actual = np.mean(y_true[group_1_mask])
            group_1_cal = abs(group_1_pred - group_1_actual)
            
            max_cal = max(group_0_cal, group_1_cal)
            is_biased = max_cal > self.thresholds['calibration']
            
            print(f"[AIF360] Calibration - Max Error: {max_cal:.4f}, Threshold: {self.thresholds['calibration']:.4f}")
            
            return {
                'metric': 'calibration',
                'library': 'AIF360',
                'group_0_predicted_prob': float(group_0_pred),
                'group_0_actual_rate': float(group_0_actual),
                'group_0_calibration_error': float(group_0_cal),
                'group_1_predicted_prob': float(group_1_pred),
                'group_1_actual_rate': float(group_1_actual),
                'group_1_calibration_error': float(group_1_cal),
                'max_calibration_error': float(max_cal),
                'threshold': self.thresholds['calibration'],
                'is_biased': bool(is_biased),
                'description': 'Calibration - Predicted probabilities should match actual outcomes per group'
            }
        except Exception as e:
            print(f"Error in calibration: {str(e)}")
            return {'error': str(e), 'metric': 'calibration'}
    
    def calculate_generalized_entropy_index_aif360(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict:
        """
        Generalized Entropy Index using AIF360
        Measures overall inequality in predictions (Theil Index)
        """
        try:
            from aif360.sklearn.metrics import generalized_entropy_error
        
            # Calculate generalized entropy error
            # b_i = y_pred_i - y_true_i + 1
            gei = generalized_entropy_error(y_true, y_pred, alpha=2)
            
            is_biased = gei > self.thresholds['generalized_entropy_index']
            
            print(f"[AIF360] Generalized Entropy Index: {gei:.4f}, Threshold: {self.thresholds['generalized_entropy_index']:.4f}")
            
            return {
                'metric': 'generalized_entropy_index',
                'library': 'AIF360',
                'entropy_index': float(gei),
                'threshold': self.thresholds['generalized_entropy_index'],
                'is_biased': bool(is_biased),
                'description': 'Generalized Entropy Index (Theil Index) - Measures overall inequality'
            }
        except Exception as e:
            print(f"Error in generalized_entropy_index: {str(e)}")
            return {'error': str(e), 'metric': 'generalized_entropy_index'}
    
    def run_all_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_feature: np.ndarray,
        y_pred_proba: np.ndarray = None
    ) -> Dict:
        """
        Run all bias detection metrics using Fairlearn and AIF360
        """
        print("\n=== Running Bias Detection ===")
        
        results = {}
        
        # Fairlearn metrics
        print("\n[Using Fairlearn Library]")
        results['statistical_parity'] = self.calculate_statistical_parity_fairlearn(
            y_true, y_pred, sensitive_feature
        )
        results['equal_opportunity'] = self.calculate_equal_opportunity_fairlearn(
            y_true, y_pred, sensitive_feature
        )
        results['equalized_odds'] = self.calculate_equalized_odds_fairlearn(
            y_true, y_pred, sensitive_feature
        )
        
        # AIF360 metrics
        print("\n[Using AIF360 Library]")
        results['disparate_impact'] = self.calculate_disparate_impact_aif360(
            y_true, y_pred, sensitive_feature
        )
        results['generalized_entropy_index'] = self.calculate_generalized_entropy_index_aif360(
            y_true, y_pred
        )
        
        if y_pred_proba is not None:
            results['calibration'] = self.calculate_calibration_aif360(
                y_true, y_pred_proba, sensitive_feature
            )
        
        # Summary
        biased_metrics = [
            k for k, v in results.items() 
            if isinstance(v, dict) and v.get('is_biased', False)
        ]
        
        results['summary'] = {
            'total_metrics': len(results),
            'biased_metrics_count': len(biased_metrics),
            'biased_metrics': biased_metrics,
            'overall_bias_status': 'BIASED' if len(biased_metrics) > 0 else 'FAIR',
            'libraries_used': ['Fairlearn', 'AIF360']
        }
        
        print(f"\n=== Bias Detection Complete ===")
        print(f"Overall Status: {results['summary']['overall_bias_status']}")
        print(f"Biased Metrics: {len(biased_metrics)}/{len(results)-1}")
        
        return results
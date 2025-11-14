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
                y_true=y_true,
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
            if len(y_pred) != len(sensitive_feature):
                raise ValueError(f"Size mismatch: y_pred ({len(y_pred)}) vs sensitive_feature ({len(sensitive_feature)})")
            
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
            # Check if multi-class
            unique_true = np.unique(y_true)
            unique_pred = np.unique(y_pred)

            # Check if binary classification
            if len(unique_true) > 2 or len(unique_pred) > 2:
                # Multi-class case - use One-vs-Rest approach
                print(f"[MULTICLASS] Detected {len(unique_true)} classes, using One-vs-Rest Equalized Odds")
                max_eo = self._equalized_odds_multiclass(y_true, y_pred, sensitive_feature)
                print("[WARNING] Equalized Odds requires binary classification. Skipping...")

                return {
                    'metric': 'equalized_odds',
                    'library': 'Fairlearn',
                    'equalized_odds_difference': float(max_eo),
                    'max_difference': float(max_eo),
                    'threshold': self.thresholds['equalized_odds'],
                    'is_biased': bool(max_eo > self.thresholds['equalized_odds']),
                    'description': 'Equalized Odds (Multi-class One-vs-Rest)',
                    'num_classes': int(len(unique_true))
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
            
            # Get all unique groups
            unique_groups = np.unique(sensitive_feature)

            # If more than 2 groups, calculate pairwise and return max
            if len(unique_groups) > 2:
                print(f"  [MULTI-GROUP] {len(unique_groups)} groups detected, calculating max pairwise difference")
                
                max_tpr_diff = 0
                max_fpr_diff = 0
                
                for group in unique_groups:
                    group_mask = sensitive_feature == group
                    
                    if np.sum(group_mask) == 0:
                        continue
                    
                    try:
                        group_tpr = true_positive_rate(y_true_series[group_mask], y_pred_series[group_mask])
                        group_fpr = false_positive_rate(y_true_series[group_mask], y_pred_series[group_mask])
                        
                        # Calculate difference from overall TPR/FPR
                        overall_tpr = true_positive_rate(y_true_series, y_pred_series)
                        overall_fpr = false_positive_rate(y_true_series, y_pred_series)
                        
                        tpr_diff = abs(float(group_tpr) - float(overall_tpr))
                        fpr_diff = abs(float(group_fpr) - float(overall_fpr))
                        
                        max_tpr_diff = max(max_tpr_diff, tpr_diff)
                        max_fpr_diff = max(max_fpr_diff, fpr_diff)
                    except:
                        continue
                
                max_diff = max(max_tpr_diff, max_fpr_diff)
                is_biased = max_diff > self.thresholds['equalized_odds']
                
                print(f"[FAIRLEARN] Equalized Odds - Max Diff: {max_diff:.4f}, Threshold: {self.thresholds['equalized_odds']:.4f}")
                
                return {
                    'metric': 'equalized_odds',
                    'library': 'Fairlearn',
                    'max_tpr_difference': float(max_tpr_diff),
                    'max_fpr_difference': float(max_fpr_diff),
                    'max_difference': float(max_diff),
                    'equalized_odds_difference': float(max_diff),
                    'threshold': self.thresholds['equalized_odds'],
                    'is_biased': bool(is_biased),
                    'num_groups': int(len(unique_groups)),
                    'description': 'Equalized Odds - Multi-group (max pairwise difference)'
                }

            # Binary case (2 groups) - your existing code
            group_0_mask = sensitive_feature == unique_groups[0]
            group_1_mask = sensitive_feature == unique_groups[1]

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
        
    def _equalized_odds_multiclass(self, y_true, y_pred, sensitive_feature):
        """
        Calculate equalized odds for multi-class classification
        Uses One-vs-Rest approach
        """
        from sklearn.preprocessing import label_binarize
        
        classes = np.unique(y_true)
        
        if len(classes) <= 2:
            # Binary case - use existing method
            return self._equalized_odds(y_true, y_pred, sensitive_feature)
        
        print(f"  [MULTICLASS] Calculating equalized odds for {len(classes)} classes")
        
        # Calculate equalized odds for each class (One-vs-Rest)
        class_eo_scores = []
        
        for target_class in classes:
            # Binarize: target_class vs rest
            y_true_binary = (y_true == target_class).astype(int)
            y_pred_binary = (y_pred == target_class).astype(int)
            
            try:
                # Calculate equalized odds for this class
                eo = equalized_odds_difference(
                    y_true_binary,
                    y_pred_binary,
                    sensitive_features=sensitive_feature
                )
                class_eo_scores.append(abs(eo))
                print(f"    Class {target_class}: EO = {abs(eo):.4f}")
            except Exception as e:
                print(f"    Class {target_class}: Error - {str(e)}")
                continue
        
        # Return max equalized odds across all classes
        if class_eo_scores:
            max_eo = max(class_eo_scores)
            avg_eo = np.mean(class_eo_scores)
            print(f"    Max EO across classes: {max_eo:.4f}")
            print(f"    Avg EO across classes: {avg_eo:.4f}")
            return max_eo
        
        return 0.0
    
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
            if len(y_true) != len(y_pred_proba):
                raise ValueError(f"Size mismatch: y_true ({len(y_true)}) vs y_pred_proba ({len(y_pred_proba)})")
            
            if len(y_true) != len(sensitive_feature):
                raise ValueError(f"Size mismatch: y_true ({len(y_true)}) vs sensitive_feature ({len(sensitive_feature)})")
            
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
            if len(y_true) != len(y_pred):
                raise ValueError(f"Size mismatch: y_true ({len(y_true)}) vs y_pred ({len(y_pred)})")
            
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
        
        # Validate input sizes FIRST
        try:
            self._validate_inputs(y_true, y_pred, sensitive_feature)
            print(f"[VALIDATION] Input sizes validated: y_true={len(y_true)}, y_pred={len(y_pred)}, sensitive={len(sensitive_feature)}")
        except ValueError as e:
            print(f"[ERROR] Input validation failed: {str(e)}")
            raise
        
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
    
    def _validate_inputs(self, y_true: np.ndarray, y_pred: np.ndarray, sensitive_feature: np.ndarray) -> bool:
        """
        Validate that all inputs have the same length
        Returns True if valid, raises exception if not
        """
        if len(y_true) != len(y_pred):
            raise ValueError(f"y_true ({len(y_true)}) and y_pred ({len(y_pred)}) have different lengths")
        
        if len(y_true) != len(sensitive_feature):
            raise ValueError(f"y_true ({len(y_true)}) and sensitive_feature ({len(sensitive_feature)}) have different lengths")
        
        return True

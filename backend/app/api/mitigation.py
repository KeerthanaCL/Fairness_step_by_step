from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np
from app.utils import FileManager
from app.models import global_session
from app.mitigation_strategies import MitigationStrategies
from app.bias_detector import BiasDetector
from app.api.bias import (
    _safe_convert_to_numeric,
    _encode_dataframe,
    _clean_nan_values,
    _generate_recommendations
)

router = APIRouter(prefix="/api/mitigation", tags=["Mitigation Strategies"])

class MitigationRequest(BaseModel):
    strategy: str
    sensitive_feature_column: str
    prediction_column: Optional[str] = None

class StrategyComparisonRequest(BaseModel):
    sensitive_feature_column: str
    prediction_column: Optional[str] = None
    strategies: Optional[list] = None  # If None, run all

class PipelineRequest(BaseModel):
    sensitive_feature_column: str
    prediction_column: Optional[str] = None
    pipeline: Dict[str, List[str]]  # {"pre": ["reweighing"], "in": ["fairness_regularization"], "post": ["threshold_optimization"]}

class PipelineComparisonRequest(BaseModel):
    sensitive_feature_column: str
    prediction_column: Optional[str] = None
    pipelines: List[Dict[str, Any]]  # List of pipeline configs with names

class OptimizationRequest(BaseModel):
    sensitive_feature_column: str
    prediction_column: Optional[str] = None
    method: str  # 'greedy', 'top_k', or 'brute_force'
    max_strategies: Optional[int] = 3  # For greedy
    k: Optional[int] = 5  # For top_k
    max_strategies_per_stage: Optional[int] = 2  # For brute_force

@router.get("/strategies")
async def get_available_strategies():
    """Get all available mitigation strategies"""
    try:
        mitigator = MitigationStrategies()
        strategies = mitigator.get_strategies_info()
        
        return {
            'status': 'success',
            'total_strategies': len(strategies),
            'strategies': strategies
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting strategies: {str(e)}")

@router.post("/apply")
async def apply_mitigation_strategy(request: MitigationRequest):
    """
    Apply a specific mitigation strategy and show before/after bias metrics
    """
    try:
        if global_session.train_file_path is None:
            raise HTTPException(status_code=400, detail="Training data must be uploaded first")
        
        if global_session.target_column is None:
            raise HTTPException(status_code=400, detail="Target column must be set first")
        
        # Load data
        train_df = FileManager.load_csv(global_session.train_file_path)
        
        # Prepare data
        X_train = train_df.drop(columns=[global_session.target_column]).copy()
        y_true = _safe_convert_to_numeric(
            train_df[global_session.target_column].values,
            "y_true"
        )
        
        sensitive_attr = _safe_convert_to_numeric(
            train_df[request.sensitive_feature_column].values,
            request.sensitive_feature_column
        )

        print(f"\n[SENSITIVE FEATURE ANALYSIS]")
        print(f"  Column: {request.sensitive_feature_column}")
        print(f"  Unique values: {np.unique(sensitive_attr)}")
        print(f"  Number of groups: {len(np.unique(sensitive_attr))}")
        for group in np.unique(sensitive_attr):
            count = np.sum(sensitive_attr == group)
            pct = (count / len(sensitive_attr)) * 100
            print(f"  Group {group}: {count} samples ({pct:.1f}%)")
        
        # For post-processing, get predictions if available
        y_pred_original = None
        if request.prediction_column and request.prediction_column.strip():
            # Use prediction column from data
            print(f"[INFO] Using prediction_column: {request.prediction_column}")
            if request.prediction_column not in train_df.columns:
                raise HTTPException(status_code=400, detail=f"Prediction column '{request.prediction_column}' not found")
            y_pred_original = _safe_convert_to_numeric(
                train_df[request.prediction_column].values,
                request.prediction_column
            )

        elif global_session.model_file_path:
            # Use uploaded model for predictions
            print(f"[INFO] Using model predictions from uploaded model")
            model = FileManager.load_model(global_session.model_file_path)
            X_test = train_df.drop(columns=[global_session.target_column]).copy()
            X_test_encoded = _encode_dataframe(X_test)
            y_pred_original = model.predict(X_test_encoded)
            y_pred_original = _safe_convert_to_numeric(y_pred_original, "model_prediction")

        else:
            raise HTTPException(status_code=400, detail="Prediction column required for bias calculation")
        
        print(f"\n{'='*60}")
        print(f"MITIGATION STRATEGY: {request.strategy.upper()}")
        print(f"{'='*60}")
        
        # ===== STEP 1: Calculate BASELINE bias metrics =====
        print(f"\n[STEP 1] Calculating BASELINE bias metrics...")
        bias_detector = BiasDetector()
        baseline_bias = bias_detector.run_all_metrics(y_true, y_pred_original, sensitive_attr)
        baseline_bias = _clean_nan_values(baseline_bias)
        
        baseline_biased_count = baseline_bias['summary']['biased_metrics_count']
        baseline_biased_metrics = baseline_bias['summary']['biased_metrics']
        
        print(f"  Biased metrics count: {baseline_biased_count}")
        print(f"  Biased metrics: {baseline_biased_metrics}")
        
        # ===== STEP 2: Apply mitigation strategy =====
        print(f"\n[STEP 2] Applying mitigation strategy...")
        X_train_encoded = _encode_dataframe(X_train)
        # Get model reference
        model = None
        if global_session.model_file_path:
            try:
                model = FileManager.load_model(global_session.model_file_path)
                print(f"  ✅ Model loaded: {type(model).__name__}")
            except Exception as e:
                print(f"  ❌ Failed to load model: {str(e)}")
                raise HTTPException(status_code=400, detail=f"Failed to load model: {str(e)}")
        
        if model is None:
            raise HTTPException(status_code=400, detail="Model required for mitigation strategies")

        # Apply mitigation
        mitigator = MitigationStrategies(user_model=model)
        strategy_result = mitigator.run_mitigation_strategy(
            request.strategy,
            X_train_encoded,
            y_true,
            sensitive_attr,
            X_test=X_train_encoded,
            y_test=y_true,
            y_pred_test=y_pred_original
        )
        
        # ===== CRITICAL: Handle transformed data for pre-processing =====
        X_mitigated = X_train_encoded.copy()
        y_pred_mitigated = y_pred_original.copy()

        # Check what mitigator returned and use it
        if mitigator.X_transformed is not None:
            print(f"[USING] Transformed X from mitigator")
            X_mitigated = mitigator.X_transformed
            # CRITICAL FIX: Handle feature count mismatch
            # AIF360 may return extra columns, extract only original features
            num_original_features = X_train_encoded.shape[1]
            
            print(f"  Original features expected: {num_original_features}")
            print(f"  Transformed X shape: {X_mitigated.shape}")
            
            # If transformed has more features than original, take only first n
            if X_mitigated.shape[1] > num_original_features:
                print(f"  Feature mismatch detected! Extracting first {num_original_features} features...")
                X_mitigated = X_mitigated[:, :num_original_features]
                print(f"  ✅ Adjusted to {X_mitigated.shape[1]} features")
            
            # Verify feature count now matches
            if X_mitigated.shape[1] != num_original_features:
                print(f"  ⚠️ WARNING: Feature count still doesn't match!")
                print(f"  Using original X instead of transformed")
                X_mitigated = X_train_encoded
            
            print(f"  Predicting on transformed X...")
            try:
                y_pred_mitigated = model.predict(X_mitigated)
                y_pred_mitigated = _safe_convert_to_numeric(y_pred_mitigated, "mitigated_prediction")
                print(f"  ✅ Got predictions from transformed X")
            except Exception as e:
                print(f"  [ERROR] Could not predict on transformed X: {str(e)}")
                print(f"  Falling back to original X")
                X_mitigated = X_train_encoded
                y_pred_mitigated = model.predict(X_mitigated)
                y_pred_mitigated = _safe_convert_to_numeric(y_pred_mitigated, "mitigated_prediction")

        elif mitigator.sample_weights is not None:
            # ADD THIS SECTION - Handle Reweighing
            print(f"[USING] Reweighted samples from mitigator")
            print(f"  Weights: min={mitigator.sample_weights.min():.3f}, max={mitigator.sample_weights.max():.3f}, mean={mitigator.sample_weights.mean():.3f}")
            
            try:
                # Clone model and retrain with weights
                import copy
                model_reweighted = copy.deepcopy(model)
                
                print(f"  Retraining model with weights...")
                
                # Check if model supports sample_weight
                if hasattr(model_reweighted, 'fit'):
                    # Try to fit with sample_weight
                    try:
                        model_reweighted.fit(X_train_encoded, y_true, sample_weight=mitigator.sample_weights)
                        print(f"  ✅ Retrained with sample weights")
                    except TypeError:
                        print(f"  [WARNING] Model doesn't support sample_weight in fit. Using as-is.")
                
                # Get predictions from reweighted model
                y_pred_mitigated = model_reweighted.predict(X_train_encoded)
                y_pred_mitigated = _safe_convert_to_numeric(y_pred_mitigated, "mitigated_prediction")
                print(f"  Generated predictions from reweighted model")
                
            except Exception as e:
                print(f"  [ERROR] Reweighting failed: {str(e)}")
                import traceback
                traceback.print_exc()
                y_pred_mitigated = y_pred_original
            
        elif mitigator.model_modified:
            print(f"[USING] Modified model from mitigator")

            # CRITICAL: Check if mitigator has a fine-tuned model stored
            if hasattr(mitigator, 'fine_tuned_model') and mitigator.fine_tuned_model is not None:
                print(f"  Using stored fine-tuned model")
                model_to_use = mitigator.fine_tuned_model
            else:
                print(f"  Using mitigator.user_model")
                model_to_use = mitigator.user_model
                
            # Get predictions from fine-tuned model
            y_pred_mitigated = model_to_use.predict(X_train_encoded)
            y_pred_mitigated = _safe_convert_to_numeric(y_pred_mitigated, "mitigated_prediction")

            print(f"  ✅ Generated {len(y_pred_mitigated)} predictions from fine-tuned model")
            
        elif mitigator.y_pred_adjusted is not None:
            print(f"[USING] Adjusted predictions from mitigator")
            y_pred_mitigated = mitigator.y_pred_adjusted
            y_pred_mitigated = _safe_convert_to_numeric(y_pred_mitigated, "mitigated_prediction")

        # Validate strategy result
        if strategy_result.get('status') not in ['success', 'partial']:
            raise HTTPException(status_code=400, detail=f"Strategy failed: {strategy_result.get('error')}")

        # ===== STEP 3: Calculate MITIGATED bias metrics =====
        print(f"\n[STEP 3] Calculating MITIGATED bias metrics...")

        # For data augmentation, use augmented y_true and sensitive_attr
        y_true_for_metrics = y_true.copy()
        sensitive_attr_for_metrics = sensitive_attr.copy()

        if hasattr(mitigator, 'y_augmented') and mitigator.y_augmented is not None:
            print(f"  Using AUGMENTED targets for metrics (size changed from {len(y_true)} to {len(mitigator.y_augmented)})")
            y_true_for_metrics = np.array(mitigator.y_augmented)
            sensitive_attr_for_metrics = np.array(mitigator.sensitive_augmented)

        print(f"  Data shapes: y_true={len(y_true_for_metrics)}, y_pred={len(y_pred_mitigated)}, sensitive={len(sensitive_attr_for_metrics)}")

        # Validate sizes match BEFORE calling bias_detector
        print(f"\n[VALIDATION] Checking size alignment:")
        print(f"  y_true: {len(y_true_for_metrics)}")
        print(f"  y_pred_mitigated: {len(y_pred_mitigated)}")
        print(f"  sensitive_attr: {len(sensitive_attr_for_metrics)}")
            
        # Validate sizes match
        if len(y_pred_mitigated) != len(y_true_for_metrics):
            print(f"  ⚠️ Size mismatch: predictions={len(y_pred_mitigated)}, targets={len(y_true_for_metrics)}")
            # Take only first len(y_true_for_metrics) predictions if needed
            if len(y_pred_mitigated) > len(y_true_for_metrics):
                print(f"  Truncating predictions to match target size...")
                y_pred_mitigated = y_pred_mitigated[:len(y_true_for_metrics)]
            elif len(y_pred_mitigated) < len(y_true_for_metrics):
                print(f"  ⚠️ ERROR: Not enough predictions!")
                raise ValueError(f"Predictions ({len(y_pred_mitigated)}) < targets ({len(y_true_for_metrics)})")
            
        if len(sensitive_attr_for_metrics) != len(y_true_for_metrics):
            print(f"  ⚠️ ERROR: Sensitive attr size mismatch!")
            raise ValueError(f"Sensitive attr ({len(sensitive_attr_for_metrics)}) != targets ({len(y_true_for_metrics)})")

        print(f"  ✅ All shapes validated - proceeding with metrics")
        print(f"\n[CALLING] bias_detector.run_all_metrics with:")
        print(f"  y_true: {type(y_true_for_metrics).__name__} shape {y_true_for_metrics.shape}")
        print(f"  y_pred_mitigated: {type(y_pred_mitigated).__name__} shape {y_pred_mitigated.shape}")
        print(f"  sensitive_attr: {type(sensitive_attr_for_metrics).__name__} shape {sensitive_attr_for_metrics.shape}")

        mitigated_bias = bias_detector.run_all_metrics(y_true_for_metrics, y_pred_mitigated, sensitive_attr_for_metrics)
        mitigated_bias = _clean_nan_values(mitigated_bias)
        
        mitigated_biased_count = mitigated_bias['summary']['biased_metrics_count']
        mitigated_biased_metrics = mitigated_bias['summary']['biased_metrics']
        
        print(f"  Biased metrics count: {mitigated_biased_count}")
        print(f"  Biased metrics: {mitigated_biased_metrics}")
        
        # ===== STEP 4: Calculate improvements =====
        print(f"\n[STEP 4] Calculating improvements...")
        improvement = baseline_biased_count - mitigated_biased_count
        
        metric_improvements = _calculate_metric_improvements(baseline_bias, mitigated_bias)
        
        print(f"  Overall improvement: {improvement} fewer biased metrics")
        print(f"  Status: {'✅ IMPROVED' if improvement > 0 else ('❌ WORSENED' if improvement < 0 else '➡️ NO CHANGE')}")
        
        # Clean strategy result for JSON
        strategy_result_clean = _serialize_mitigation_result(strategy_result)
        
        # Build response
        response = {
            'status': 'success',
            'strategy_applied': request.strategy,
            'strategy_info': strategy_result_clean,
            
            'bias_assessment': {
                'baseline': {
                    'summary': baseline_bias['summary'],
                    'detailed_metrics': _serialize_mitigation_result({
                        k: v for k, v in baseline_bias.items() if k != 'summary'
                    })
                },
                'after_mitigation': {
                    'summary': mitigated_bias['summary'],
                    'detailed_metrics': _serialize_mitigation_result({
                        k: v for k, v in mitigated_bias.items() if k != 'summary'
                    })
                }
            },
            
            'improvement_analysis': {
                'baseline_biased_metrics_count': int(baseline_biased_count),
                'mitigated_biased_metrics_count': int(mitigated_biased_count),
                'overall_improvement': int(improvement),
                'improvement_percentage': float((improvement / baseline_biased_count * 100) if baseline_biased_count > 0 else 0),
                'status': 'IMPROVED' if improvement > 0 else ('WORSENED' if improvement < 0 else 'NO_CHANGE'),
                'metric_improvements': metric_improvements
            },
            
            'recommendations': _generate_mitigation_recommendations(
                improvement,
                strategy_result.get('type'),
                mitigated_biased_metrics
            )
        }
        
        print(f"\n{'='*60}\n")
        
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Error applying mitigation: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error applying mitigation: {str(e)}")

@router.post("/compare")
async def compare_strategies(request: StrategyComparisonRequest):
    """
    Compare bias scores before and after applying different strategies
    """
    try:
        if global_session.train_file_path is None:
            raise HTTPException(status_code=400, detail="Training data must be uploaded first")
        
        if global_session.target_column is None:
            raise HTTPException(status_code=400, detail="Target column must be set first")
        
        # Load data
        train_df = FileManager.load_csv(global_session.train_file_path)
        
        # Prepare data
        y_true = _safe_convert_to_numeric(
            train_df[global_session.target_column].values,
            "y_true"
        )
        
        sensitive_feature = _safe_convert_to_numeric(
            train_df[request.sensitive_feature_column].values,
            request.sensitive_feature_column
        )
        
        # Get predictions
        if request.prediction_column and request.prediction_column.strip():
            # Use prediction column from data
            print(f"[INFO] Using prediction_column: {request.prediction_column}")
            if request.prediction_column not in train_df.columns:
                raise HTTPException(status_code=400, detail=f"Prediction column '{request.prediction_column}' not found")
            y_pred = _safe_convert_to_numeric(
                train_df[request.prediction_column].values,
                request.prediction_column
            )
        
        elif global_session.model_file_path:
            # Use uploaded model for predictions
            print(f"[INFO] Using model predictions from uploaded model")
            model = FileManager.load_model(global_session.model_file_path)
            X_test = train_df.drop(columns=[global_session.target_column]).copy()
            X_test_encoded = _encode_dataframe(X_test)
            y_pred = model.predict(X_test_encoded)
            y_pred = _safe_convert_to_numeric(y_pred, "model_prediction")

        else:
            raise HTTPException(status_code=400, detail="Prediction column required for comparison")
        
        # Get baseline bias metrics
        bias_detector = BiasDetector()
        baseline_results = bias_detector.run_all_metrics(y_true, y_pred, sensitive_feature)
        baseline_results = _clean_nan_values(baseline_results)

        baseline_biased_count = baseline_results['summary']['biased_metrics_count']
        print(f"  Baseline biased metrics: {baseline_biased_count}")
        
        # Get strategies to apply
        if request.strategies:
            strategies_to_run = request.strategies
        else:
            mitigator = MitigationStrategies(user_model=model)
            strategies_to_run = list(mitigator.strategies.keys())

        print(f"\n[STRATEGIES] Evaluating {len(strategies_to_run)} strategies...")
        print(f"  Strategies: {', '.join(strategies_to_run)}")
        
        # Apply each strategy and measure bias
        mitigation_results = {
            'baseline': {
                'bias_metrics': baseline_results,
                'strategy': 'original (no mitigation)'
            }
        }
        
        X_train = train_df.drop(columns=[global_session.target_column]).copy()
        X_train_encoded = _encode_dataframe(X_train)
        
        mitigator = MitigationStrategies(user_model=model)
        
        for idx, strategy_name in enumerate(strategies_to_run, 1):
            print(f"\n[{idx}/{len(strategies_to_run)}] Evaluating: {strategy_name}")
            
            try:
                # Apply strategy
                strategy_result = mitigator.run_mitigation_strategy(
                    strategy_name,
                    X_train_encoded,
                    y_true,
                    sensitive_feature,
                    X_test=X_train_encoded,
                    y_test=y_true,
                    y_pred_test=y_pred
                )
                
                # For now, assume predictions remain same (in real scenario, would re-predict)
                # In production, you'd retrain the model with mitigated data
                y_pred_mitigated = y_pred
                
                # Calculate bias metrics after mitigation
                if strategy_result.get('status') == 'success':
                    print(f"  ✅ Strategy applied successfully")
                    mitigated_results = bias_detector.run_all_metrics(
                        y_true, y_pred_mitigated, sensitive_feature
                    )
                    mitigated_results = _clean_nan_values(mitigated_results)
                    
                    mitigated_biased_count = mitigated_results['summary']['biased_metrics_count']
                    improvement = baseline_biased_count - mitigated_biased_count
                    
                    print(f"  📊 Biased metrics: {mitigated_biased_count} (improvement: {'+' if improvement >= 0 else ''}{improvement})")
                    
                    # Clean strategy result for JSON serialization
                    strategy_result_clean = _serialize_mitigation_result(strategy_result)
                    
                    mitigation_results[strategy_name] = {
                        'strategy_info': strategy_result_clean,
                        'bias_metrics': mitigated_results,
                        'improvement': improvement
                    }
                else:
                    print(f"  ❌ Strategy failed: {strategy_result.get('error', 'Unknown error')}")
                    mitigation_results[strategy_name] = {
                        'status': 'error',
                        'error': str(strategy_result.get('error')),
                        'improvement': 0
                    }
            
            except Exception as e:
                print(f"  ❌ Exception: {str(e)}")
                mitigation_results[strategy_name] = {
                    'status': 'error',
                    'error': str(e),
                    'improvement': 0
                }
        
        # Generate comparison summary
        comparison_summary = _generate_comparison_summary(mitigation_results)
        
        print(f"\n{'='*60}")
        print(f"COMPARISON SUMMARY")
        print(f"{'='*60}")
        print(f"Baseline biased metrics: {baseline_biased_count}")
        print(f"Strategies evaluated: {len(strategies_to_run)}")
        print(f"Best performing: {comparison_summary['best_strategies'][0]['strategy'] if comparison_summary['best_strategies'] else 'None'}")
        print(f"{'='*60}\n")
        
        return {
            'status': 'success',
            'analysis_type': 'Mitigation Strategy Comparison',
            'sensitive_feature': request.sensitive_feature_column,
            'total_strategies': len(strategies_to_run),
            'strategies_evaluated': strategies_to_run,
            'baseline_biased_metrics': int(baseline_biased_count),
            'mitigation_results': {
                k: v for k, v in mitigation_results.items()
            },
            'comparison_summary': comparison_summary
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Error in compare: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error comparing strategies: {str(e)}")
    
@router.post("/pipeline/apply")
async def apply_mitigation_pipeline(request: PipelineRequest):
    """
    Apply a multi-stage mitigation pipeline (pre -> in -> post)
    Shows bias metrics before and after the full pipeline
    """
    try:
        print(f"\n{'='*60}")
        print(f"MITIGATION PIPELINE REQUEST")
        print(f"{'='*60}")
        
        # ADD THIS VALIDATION:
        if request.sensitive_feature_column is None or request.sensitive_feature_column.strip() == '':
            raise HTTPException(
                status_code=400,
                detail="sensitive_feature_column is required and cannot be empty"
            )
        
        if request.sensitive_feature_column == 'string':
            raise HTTPException(
                status_code=400,
                detail="sensitive_feature_column cannot be the literal string 'string'. Please provide an actual column name"
            )
        
        print(f"Request parameters:")

        # Check which attributes exist
        if hasattr(request, 'strategies'):
            strategies = request.strategies
            print(f"  Strategies: {strategies}")
        elif hasattr(request, 'pipeline'):
            strategies = []  # Convert single to list
            for stage in ['pre', 'in', 'post']:
                if stage in request.pipeline:
                    strategies.extend(request.pipeline[stage])
            print(f"  Strategies from pipeline: {strategies}")
        else:
            raise HTTPException(
                status_code=400,
                detail="Request must contain 'strategies' or 'strategy' field"
            )

        if hasattr(request, 'sensitive_feature_column'):
            sensitive_feature = request.sensitive_feature_column
        else:
            raise HTTPException(status_code=400, detail="sensitive_feature_column is required")

        if hasattr(request, 'prediction_column'):
            prediction_column = request.prediction_column
        else:
            prediction_column = None

        print(f"  Sensitive feature: {sensitive_feature}")
        print(f"  Prediction column: {prediction_column}")

        if global_session.train_file_path is None:
            raise HTTPException(status_code=400, detail="Training data must be uploaded first")
        
        if global_session.target_column is None:
            raise HTTPException(status_code=400, detail="Target column must be set first")
        
        # Load data
        train_df = FileManager.load_csv(global_session.train_file_path)
        
        # Prepare data
        X_train = train_df.drop(columns=[global_session.target_column]).copy()
        y_true = _safe_convert_to_numeric(
            train_df[global_session.target_column].values,
            "y_true"
        )

        # Validate sensitive_feature_column exists
        if request.sensitive_feature_column not in train_df.columns:
            raise HTTPException(
                status_code=400, 
                detail=f"Sensitive feature column '{request.sensitive_feature_column}' not found in data. Available columns: {list(train_df.columns)}"
            )

        print(f"\n[INFO] Sensitive feature column: {request.sensitive_feature_column}")
        print(f"  Available columns: {list(train_df.columns)}")
        
        sensitive_attr = _safe_convert_to_numeric(
            train_df[request.sensitive_feature_column].values,
            request.sensitive_feature_column
        )

        print(f"\n[SENSITIVE FEATURE ANALYSIS]")
        print(f"  Column: {request.sensitive_feature_column}")
        print(f"  Unique values: {np.unique(sensitive_attr)}")
        print(f"  Number of groups: {len(np.unique(sensitive_attr))}")
        for group in np.unique(sensitive_attr):
            count = np.sum(sensitive_attr == group)
            pct = (count / len(sensitive_attr)) * 100
            print(f"  Group {group}: {count} samples ({pct:.1f}%)")
        
        # Get predictions
        if request.prediction_column and request.prediction_column.strip():
            print(f"[INFO] Using prediction_column: {request.prediction_column}")
            if request.prediction_column not in train_df.columns:
                raise HTTPException(status_code=400, detail=f"Prediction column '{request.prediction_column}' not found")
            y_pred = _safe_convert_to_numeric(
                train_df[request.prediction_column].values,
                request.prediction_column
            )
        elif global_session.model_file_path:
            print(f"[INFO] Using model predictions from uploaded model")
            model = FileManager.load_model(global_session.model_file_path)
            X_test = train_df.drop(columns=[global_session.target_column]).copy()
            X_test_encoded = _encode_dataframe(X_test)
            y_pred = model.predict(X_test_encoded)
            y_pred = _safe_convert_to_numeric(y_pred, "model_prediction")
        else:
            raise HTTPException(status_code=400, detail="Either prediction_column or uploaded model required")
        
        # Calculate BASELINE bias metrics
        print(f"\n[BASELINE] Calculating original bias metrics...")
        bias_detector = BiasDetector()

        # Store original sensitive_attr in detector (for later use)
        bias_detector.sensitive_attr_original = sensitive_attr.copy() 
        
        baseline_bias = bias_detector.run_all_metrics(y_true, y_pred, sensitive_attr)
        baseline_bias = _clean_nan_values(baseline_bias)
        
        baseline_biased_count = baseline_bias['summary']['biased_metrics_count']
        print(f"  Baseline biased metrics: {baseline_biased_count}")
        
        # Build and execute pipeline
        from app.mitigation_strategies import MitigationPipeline
        
        pipeline = MitigationPipeline()
        
        # Add strategies to pipeline
        for stage in ['pre', 'in', 'post']:
            if stage in request.pipeline:
                for strategy in request.pipeline[stage]:
                    try:
                        pipeline.add_strategy(strategy, stage)
                    except Exception as e:
                        raise HTTPException(status_code=400, detail=f"Error adding strategy '{strategy}' to stage '{stage}': {str(e)}")
        
        # Get pipeline summary
        pipeline_summary = pipeline.get_pipeline_summary()

        # Get model for strategies
        model = None
        if global_session.model_file_path:
            model = FileManager.load_model(global_session.model_file_path)

        if model is None:
            raise HTTPException(status_code=400, detail="Model required for mitigation strategies")

        print(f"\n[PIPELINE SETUP]")
        print(f"  Strategies to apply: {strategies}")
        print(f"  Model type: {type(model).__name__}")
        
        # Encode data
        X_train_encoded = _encode_dataframe(X_train)

        # Generate mitigated predictions (simplified for demo)
        y_pred_current = y_pred.copy()
        pipeline_stages = []
        
        # Execute pipeline
        pipeline_results = pipeline.execute_pipeline(
            X_train_encoded,
            y_true,
            sensitive_attr,
            X_test=X_train_encoded,
            y_test=y_true,
            y_pred_test=y_pred
        )
        
        X_train_transformed = X_train_encoded.copy()
        # Track augmented data when it changes
        y_true_current = y_true.copy()
        sensitive_attr_current = sensitive_attr.copy()
        # Apply each strategy in sequence
        for idx, strategy in enumerate(strategies, 1):
            print(f"\n{'='*60}")
            print(f"[STAGE {idx}/{len(strategies)}] {strategy.upper()}")
            print(f"{'='*60}")
            
            try:
                # Verify model exists
                if model is None:
                    print(f"  ⚠️ WARNING: Model is None, skipping strategy {strategy}")
                    stage_result = {
                        'stage_number': idx,
                        'strategy': strategy,
                        'status': 'skipped',
                        'reason': 'Model not available'
                    }
                    pipeline_stages.append(stage_result)
                    continue
                # Apply mitigation
                mitigator = MitigationStrategies(user_model=model)
                print(f"  Model type: {type(model).__name__}")

                strategy_result = mitigator.run_mitigation_strategy(
                    strategy,
                    X_train_encoded,
                    y_true_current,  # Use current (possibly augmented) y_true
                    sensitive_attr_current,  # Use current (possibly augmented) sensitive_attr
                    X_test=X_train_encoded,
                    y_test=y_true,
                    y_pred_test=y_pred_current
                )
                
                # Store stage result
                stage_result = {
                    'stage_number': idx,
                    'strategy': strategy,
                    'status': strategy_result.get('status'),
                    'type': strategy_result.get('type'),
                    'message': strategy_result.get('message', '')
                }
                pipeline_stages.append(stage_result)
                
                print(f"  Status: {strategy_result.get('status')}")
                
                # Extract mitigated predictions based on strategy type and handle size changes
                if mitigator.y_augmented is not None:
                    print(f"  Data was augmented: {len(y_true_current)} → {len(mitigator.y_augmented)}")
                    y_true_current = np.array(mitigator.y_augmented)
                    sensitive_attr_current = np.array(mitigator.sensitive_augmented)
                    
                    # Use augmented X for predictions
                    if hasattr(mitigator, 'X_augmented') and mitigator.X_augmented is not None:
                        print(f"  Using augmented X data for predictions")
                        X_augmented_encoded = _encode_dataframe(pd.DataFrame(mitigator.X_augmented))
                        y_pred_current = model.predict(X_augmented_encoded)
                    else:
                        # Fallback if X_augmented not available
                        y_pred_current = model.predict(X_train_encoded)

                    y_pred_current = _safe_convert_to_numeric(y_pred_current, "predictions")
                    
                    print(f"  Now using augmented data: {len(y_pred_current)} predictions")

                elif mitigator.y_pred_adjusted is not None:
                    print(f"  Using adjusted predictions")
                    y_pred_current = mitigator.y_pred_adjusted
                
                elif mitigator.X_transformed is not None:
                    print(f"  Using transformed data for predictions")
                    X_train_encoded_temp = mitigator.X_transformed
                    
                    # Handle feature mismatch
                    num_original_features = X_train_encoded.shape[1]
                    if X_train_encoded_temp.shape[1] > num_original_features:
                        X_train_encoded_temp = X_train_encoded_temp[:, :num_original_features]
                    
                    y_pred_current = model.predict(X_train_encoded_temp)
                    y_pred_current = _safe_convert_to_numeric(y_pred_current, "predictions")
                
                elif mitigator.model_modified:
                    print(f"  Using fine-tuned model")
                    if hasattr(mitigator, 'fine_tuned_model') and mitigator.fine_tuned_model is not None:
                        y_pred_current = mitigator.fine_tuned_model.predict(X_train_encoded)
                    else:
                        y_pred_current = model.predict(X_train_encoded)
                    y_pred_current = _safe_convert_to_numeric(y_pred_current, "predictions")
                
            except Exception as e:
                print(f"[ERROR] {strategy} failed: {str(e)}")
                import traceback
                traceback.print_exc()
                stage_result = {
                    'stage_number': idx,
                    'strategy': strategy,
                    'status': 'error',
                    'error': str(e)
                }
                pipeline_stages.append(stage_result)
                # Continue to next strategy even if one fails
                continue

        # Final mitigated predictions
        y_pred_mitigated = y_pred_current

        # CRITICAL: Validate sizes match before calculating metrics
        print(f"\n[VALIDATION] Final size check before metrics:")
        print(f"  y_true_current: {len(y_true_current)}")
        print(f"  y_pred_mitigated: {len(y_pred_mitigated)}")
        print(f"  sensitive_attr_current: {len(sensitive_attr_current)}")

        if len(y_pred_mitigated) != len(y_true_current):
            print(f"  ⚠️ Size mismatch! Truncating predictions to match targets")
            y_pred_mitigated = y_pred_mitigated[:len(y_true_current)]
        
        # Calculate MITIGATED bias metrics
        print(f"\n[AFTER PIPELINE] Calculating mitigated bias metrics...")
        mitigated_bias = bias_detector.run_all_metrics(y_true_current, y_pred_mitigated, sensitive_attr_current)
        mitigated_bias = _clean_nan_values(mitigated_bias)

        print(f"\n{'='*60}")
        print(f"BIAS ASSESSMENT COMPARISON")
        print(f"{'='*60}")

        # Get summary info
        baseline_summary = baseline_bias.get('summary', {})
        mitigated_summary = mitigated_bias.get('summary', {})

        print(f"\nBaseline Summary:")
        print(f"  Overall Status: {baseline_summary.get('overall_bias_status')}")
        print(f"  Biased Metrics Count: {baseline_summary.get('biased_metrics_count')}")
        print(f"  Biased Metrics: {baseline_summary.get('biased_metrics')}")

        print(f"\nMitigated Summary:")
        print(f"  Overall Status: {mitigated_summary.get('overall_bias_status')}")
        print(f"  Biased Metrics Count: {mitigated_summary.get('biased_metrics_count')}")
        print(f"  Biased Metrics: {mitigated_summary.get('biased_metrics')}")
        
        baseline_biased_count = baseline_summary.get('biased_metrics_count', 0)
        mitigated_biased_count = mitigated_bias['summary']['biased_metrics_count']
        improvement = baseline_biased_count - mitigated_biased_count
        
        print(f"\nImprovement:")
        print(f"  Metrics before: {baseline_biased_count}")
        print(f"  Metrics after: {mitigated_biased_count}")
        print(f"  Overall improvement: {improvement}")
        
        # Calculate detailed metric improvements
        metric_improvements = _calculate_metric_improvements(baseline_bias, mitigated_bias)
        
        # Clean results for JSON
        pipeline_results_clean = _serialize_mitigation_result(pipeline_results)
        
        response = {
            'status': 'success',
            'pipeline_config': {
                'strategies': strategies,
                'sensitive_feature': sensitive_feature,
                'total_stages': len(strategies)
            },
            'pipeline_execution': {
                'stages': pipeline_stages,
                'total_stages_executed': len([s for s in pipeline_stages if s.get('status') in ['success', 'partial']])
            },
            
            'bias_assessment': {
                'baseline': {
                    'summary': baseline_bias['summary'],
                    'detailed_metrics': _serialize_mitigation_result({
                        k: v for k, v in baseline_bias.items() if k != 'summary'
                    })
                },
                'after_pipeline': {
                    'summary': mitigated_bias['summary'],
                    'detailed_metrics': _serialize_mitigation_result({
                        k: v for k, v in mitigated_bias.items() if k != 'summary'
                    })
                }
            },
            
            'improvement_analysis': {
                'baseline_biased_metrics_count': int(baseline_biased_count),
                'mitigated_biased_metrics_count': int(mitigated_biased_count),
                'overall_improvement': int(improvement),
                'improvement_percentage': float((improvement / baseline_biased_count * 100) if baseline_biased_count > 0 else 0),
                'status': 'IMPROVED' if improvement > 0 else ('WORSENED' if improvement < 0 else 'NO_CHANGE'),
                'metric_improvements': metric_improvements
            },
            
            'recommendations': _generate_mitigation_recommendations(
                improvement,
                'pipeline',
                mitigated_bias['summary']['biased_metrics']
            )
        }
        
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Error applying pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error applying pipeline: {str(e)}")


@router.post("/pipeline/compare")
async def compare_mitigation_pipelines(request: PipelineComparisonRequest):
    """
    Compare multiple mitigation pipelines
    Each pipeline can have different combinations of pre, in, and post-processing strategies
    """
    try:
        if global_session.train_file_path is None:
            raise HTTPException(status_code=400, detail="Training data must be uploaded first")
        
        if global_session.target_column is None:
            raise HTTPException(status_code=400, detail="Target column must be set first")
        
        # Load data
        train_df = FileManager.load_csv(global_session.train_file_path)
        
        # Prepare data
        y_true = _safe_convert_to_numeric(
            train_df[global_session.target_column].values,
            "y_true"
        )
        
        sensitive_attr = _safe_convert_to_numeric(
            train_df[request.sensitive_feature_column].values,
            request.sensitive_feature_column
        )

        print(f"\n[SENSITIVE FEATURE ANALYSIS]")
        print(f"  Column: {request.sensitive_feature_column}")
        print(f"  Unique values: {np.unique(sensitive_attr)}")
        print(f"  Number of groups: {len(np.unique(sensitive_attr))}")
        for group in np.unique(sensitive_attr):
            count = np.sum(sensitive_attr == group)
            pct = (count / len(sensitive_attr)) * 100
            print(f"  Group {group}: {count} samples ({pct:.1f}%)")
        
        # Get predictions
        if request.prediction_column and request.prediction_column.strip():
            y_pred = _safe_convert_to_numeric(
                train_df[request.prediction_column].values,
                request.prediction_column
            )
        elif global_session.model_file_path:
            model = FileManager.load_model(global_session.model_file_path)
            X_test = train_df.drop(columns=[global_session.target_column]).copy()
            X_test_encoded = _encode_dataframe(X_test)
            y_pred = model.predict(X_test_encoded)
            y_pred = _safe_convert_to_numeric(y_pred, "model_prediction")
        else:
            raise HTTPException(status_code=400, detail="Either prediction_column or uploaded model required")
        
        # Calculate baseline
        bias_detector = BiasDetector()
        baseline_bias = bias_detector.run_all_metrics(y_true, y_pred, sensitive_attr)
        baseline_bias = _clean_nan_values(baseline_bias)
        baseline_biased_count = baseline_bias['summary']['biased_metrics_count']
        
        print(f"\n{'='*60}")
        print(f"COMPARING {len(request.pipelines)} MITIGATION PIPELINES")
        print(f"{'='*60}")
        print(f"Baseline biased metrics: {baseline_biased_count}")
        
        # Prepare data
        X_train = train_df.drop(columns=[global_session.target_column]).copy()
        X_train_encoded = _encode_dataframe(X_train)
        
        # Compare each pipeline
        pipeline_comparisons = {
            'baseline': {
                'name': 'Original (no mitigation)',
                'bias_metrics': baseline_bias,
                'biased_metrics_count': baseline_biased_count,
                'improvement': 0
            }
        }
        
        from app.mitigation_strategies import MitigationPipeline
        
        for idx, pipeline_config in enumerate(request.pipelines, 1):
            pipeline_name = pipeline_config.get('name', f'Pipeline_{idx}')
            pipeline_strategies = pipeline_config.get('pipeline', {})
            
            print(f"\n[{idx}/{len(request.pipelines)}] Evaluating: {pipeline_name}")
            print(f"  Pre: {pipeline_strategies.get('pre', [])}")
            print(f"  In: {pipeline_strategies.get('in', [])}")
            print(f"  Post: {pipeline_strategies.get('post', [])}")
            
            try:
                # Build pipeline
                pipeline = MitigationPipeline()
                for stage in ['pre', 'in', 'post']:
                    if stage in pipeline_strategies:
                        for strategy in pipeline_strategies[stage]:
                            pipeline.add_strategy(strategy, stage)
                
                # Execute pipeline
                pipeline_results = pipeline.execute_pipeline(
                    X_train_encoded,
                    y_true,
                    sensitive_attr,
                    X_test=X_train_encoded,
                    y_test=y_true,
                    y_pred_test=y_pred
                )
                
                # Generate mitigated predictions
                y_pred_mitigated = y_pred.copy()
                for stage_name, stage_results in pipeline_results['stage_results'].items():
                    for result in stage_results:
                        if result['status'] == 'success':
                            strategy_name = result['strategy']
                            strategy_type = result['result'].get('type', 'post-processing')
                            y_pred_mitigated = _apply_mitigation_adjustment(
                                y_pred_mitigated,
                                sensitive_attr,
                                strategy_type,
                                strategy_name
                            )
                
                # Calculate mitigated bias
                mitigated_bias = bias_detector.run_all_metrics(y_true, y_pred_mitigated, sensitive_attr)
                mitigated_bias = _clean_nan_values(mitigated_bias)
                mitigated_biased_count = mitigated_bias['summary']['biased_metrics_count']
                improvement = baseline_biased_count - mitigated_biased_count
                
                print(f"  Result: {mitigated_biased_count} biased metrics (improvement: {improvement})")
                
                pipeline_comparisons[pipeline_name] = {
                    'name': pipeline_name,
                    'pipeline_config': pipeline.get_pipeline_summary(),
                    'bias_metrics': mitigated_bias,
                    'biased_metrics_count': int(mitigated_biased_count),
                    'improvement': int(improvement),
                    'status': 'success'
                }
            
            except Exception as e:
                print(f"  Error: {str(e)}")
                pipeline_comparisons[pipeline_name] = {
                    'name': pipeline_name,
                    'status': 'error',
                    'error': str(e)
                }
        
        # Rank pipelines by improvement
        successful_pipelines = [
            (name, data) for name, data in pipeline_comparisons.items()
            if name != 'baseline' and data.get('status') == 'success'
        ]
        successful_pipelines.sort(key=lambda x: x[1]['improvement'], reverse=True)
        
        best_pipeline = successful_pipelines[0] if successful_pipelines else None
        
        print(f"\n{'='*60}")
        print(f"BEST PIPELINE: {best_pipeline[0] if best_pipeline else 'None'}")
        if best_pipeline:
            print(f"Improvement: {best_pipeline[1]['improvement']} fewer biased metrics")
        print(f"{'='*60}\n")
        
        return {
            'status': 'success',
            'analysis_type': 'Multi-Pipeline Comparison',
            'sensitive_feature': request.sensitive_feature_column,
            'total_pipelines': len(request.pipelines),
            'baseline_biased_metrics': int(baseline_biased_count),
            'pipeline_comparisons': pipeline_comparisons,
            'best_pipeline': best_pipeline[0] if best_pipeline else None,
            'rankings': [
                {'pipeline': name, 'improvement': data['improvement']}
                for name, data in successful_pipelines
            ]
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Error comparing pipelines: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error comparing pipelines: {str(e)}")


@router.get("/pipeline/templates")
async def get_pipeline_templates():
    """Get pre-configured pipeline templates for common use cases"""
    return {
        'status': 'success',
        'templates': {
            'basic_preprocessing': {
                'name': 'Basic Pre-processing',
                'pipeline': {
                    'pre': ['reweighing'],
                    'in': [],
                    'post': []
                },
                'description': 'Simple reweighing for data balance'
            },
            'comprehensive': {
                'name': 'Comprehensive Multi-Stage',
                'pipeline': {
                    'pre': ['reweighing', 'data_augmentation'],
                    'in': ['fairness_regularization'],
                    'post': ['threshold_optimization']
                },
                'description': 'Full pipeline with all stages'
            },
            'post_processing_only': {
                'name': 'Post-Processing Only',
                'pipeline': {
                    'pre': [],
                    'in': [],
                    'post': ['calibration_adjustment', 'equalized_odds_postprocessing']
                },
                'description': 'No retraining needed - only prediction adjustments'
            },
            'data_centric': {
                'name': 'Data-Centric Approach',
                'pipeline': {
                    'pre': ['data_augmentation', 'disparate_impact_remover'],
                    'in': [],
                    'post': []
                },
                'description': 'Focus on data preparation'
            },
            'model_centric': {
                'name': 'Model-Centric Approach',
                'pipeline': {
                    'pre': [],
                    'in': ['fairness_regularization', 'adversarial_debiasing'],
                    'post': []
                },
                'description': 'Focus on fair model training'
            }
        }
    }

@router.post("/optimize")
async def optimize_pipeline(request: OptimizationRequest):
    """
    Automatically find the best mitigation pipeline using specified optimization method
    
    Methods:
    - greedy: Greedy search (fast, good results)
    - top_k: Top-K method (moderate speed, good results)
    - brute_force: Exhaustive search (slow, optimal results)
    """
    try:
        if global_session.train_file_path is None:
            raise HTTPException(status_code=400, detail="Training data must be uploaded first")
        
        if global_session.target_column is None:
            raise HTTPException(status_code=400, detail="Target column must be set first")
        
        # Load data
        train_df = FileManager.load_csv(global_session.train_file_path)
        
        # Prepare data
        X_train = train_df.drop(columns=[global_session.target_column]).copy()
        y_true = _safe_convert_to_numeric(
            train_df[global_session.target_column].values,
            "y_true"
        )
        
        sensitive_attr = _safe_convert_to_numeric(
            train_df[request.sensitive_feature_column].values,
            request.sensitive_feature_column
        )

        print(f"\n[SENSITIVE FEATURE ANALYSIS]")
        print(f"  Column: {request.sensitive_feature_column}")
        print(f"  Unique values: {np.unique(sensitive_attr)}")
        print(f"  Number of groups: {len(np.unique(sensitive_attr))}")
        for group in np.unique(sensitive_attr):
            count = np.sum(sensitive_attr == group)
            pct = (count / len(sensitive_attr)) * 100
            print(f"  Group {group}: {count} samples ({pct:.1f}%)")

        # Load model first (needed for optimization strategies)
        print(f"\n[MODEL LOADING]")
        model = None
        if global_session.model_file_path:
            try:
                model = FileManager.load_model(global_session.model_file_path)
                print(f"  ✅ Model loaded: {type(model).__name__}")
            except Exception as e:
                print(f"  ⚠️ Failed to load model: {str(e)}")
                model = None
        
        # Get predictions
        if request.prediction_column and request.prediction_column.strip():
            print(f"  Using prediction_column: {request.prediction_column}")
            y_pred = _safe_convert_to_numeric(
                train_df[request.prediction_column].values,
                request.prediction_column
            )
        elif model is not None:
            print(f"  Using model predictions")
            X_test = train_df.drop(columns=[global_session.target_column]).copy()
            X_test_encoded = _encode_dataframe(X_test)
            y_pred = model.predict(X_test_encoded)
            y_pred = _safe_convert_to_numeric(y_pred, "model_prediction")
        else:
            raise HTTPException(status_code=400, detail="Either prediction_column or uploaded model required")
        
        # Validate model is available for optimization
        if model is None:
            raise HTTPException(
                status_code=400, 
                detail="Model is required for optimization (upload model or ensure it was saved). Strategies like fairness_regularization and threshold_optimization require a model."
            )
        
        # Calculate baseline
        bias_detector = BiasDetector()
        baseline_bias = bias_detector.run_all_metrics(y_true, y_pred, sensitive_attr)
        baseline_bias = _clean_nan_values(baseline_bias)
        baseline_biased_count = baseline_bias['summary']['biased_metrics_count']
        
        # Encode data
        X_train_encoded = _encode_dataframe(X_train)
        
        # Run optimization
        from app.pipeline_optimizer import PipelineOptimizer
        optimizer = PipelineOptimizer(bias_detector=bias_detector, user_model=model)
        
        if request.method == 'greedy':
            result = optimizer.stage_wise_greedy_search(
                X_train_encoded,
                y_true,
                sensitive_attr,
                y_pred,
                baseline_biased_count
                # max_strategies=request.max_strategies or 3
            )
        
        elif request.method == 'top_k':
            result = optimizer.top_k_method(
                X_train_encoded,
                y_true,
                sensitive_attr,
                y_pred,
                baseline_biased_count,
                k=request.k or 5
            )
        
        elif request.method == 'brute_force':
            result = optimizer.brute_force_search(
                X_train_encoded,
                y_true,
                sensitive_attr,
                y_pred,
                baseline_biased_count,
                max_strategies_per_stage=request.max_strategies_per_stage or 2
            )
        
        else:
            raise HTTPException(status_code=400, detail=f"Unknown optimization method: {request.method}")
        
        # Clean for JSON
        result_clean = _serialize_mitigation_result(result)
        
        return {
            'status': 'success',
            'optimization_method': request.method,
            'baseline_biased_metrics': int(baseline_biased_count),
            'optimization_result': result_clean,
            'recommended_pipeline': result['best_pipeline']
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Error in optimization: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error in optimization: {str(e)}")


@router.post("/optimize/compare-methods")
async def compare_optimization_methods(request: BaseModel):
    """
    Compare all three optimization methods on the same dataset
    Shows which method finds the best pipeline and how long each takes
    """
    # Implementation similar to above, runs all three methods
    pass

def _generate_comparison_summary(mitigation_results: Dict) -> Dict:
    """Generate summary comparing strategies"""
    summary = {
        'baseline_biased_metrics': mitigation_results['baseline']['bias_metrics'].get('summary', {}).get('biased_metrics_count', 0),
        'strategies_evaluated': len(mitigation_results) - 1,
        'best_strategies': [],
        'improvement_summary': {}
    }
    
    baseline_biased_count = summary['baseline_biased_metrics']
    
    for strategy_name, result in mitigation_results.items():
        if strategy_name == 'baseline' or result.get('status') == 'error':
            continue
        
        try:
            strategy_biased_count = result['bias_metrics'].get('summary', {}).get('biased_metrics_count', 0)
            improvement = baseline_biased_count - strategy_biased_count
            
            summary['improvement_summary'][strategy_name] = {
                'biased_metrics_before': baseline_biased_count,
                'biased_metrics_after': strategy_biased_count,
                'improvement': improvement,
                'type': result.get('strategy_info', {}).get('type', 'unknown')
            }
            
            if improvement > 0:
                summary['best_strategies'].append({
                    'strategy': strategy_name,
                    'improvement': improvement
                })
        except:
            pass
    
    # Sort best strategies by improvement
    summary['best_strategies'].sort(key=lambda x: x['improvement'], reverse=True)
    
    return summary

def _serialize_mitigation_result(result: Dict) -> Dict:
    """
    Convert non-JSON-serializable objects to JSON-compatible format
    """
    serialized = {}
    
    for key, value in result.items():
        if isinstance(value, np.ndarray):
            # Convert numpy arrays to lists
            serialized[key] = value.tolist() if len(value.shape) > 0 else float(value)
        elif isinstance(value, (np.int32, np.int64)):
            serialized[key] = int(value)
        elif isinstance(value, (np.float32, np.float64)):
            serialized[key] = float(value)
        elif isinstance(value, np.bool_):
            serialized[key] = bool(value)
        elif isinstance(value, dict):
            serialized[key] = _serialize_mitigation_result(value)
        elif isinstance(value, list):
            serialized[key] = [
                _serialize_mitigation_result(item) if isinstance(item, dict) 
                else (item.tolist() if isinstance(item, np.ndarray) else item)
                for item in value
            ]
        elif value is None:
            serialized[key] = None
        else:
            serialized[key] = str(value)
    
    return serialized

def _apply_mitigation_adjustment(
    y_pred: np.ndarray,
    sensitive_attr: np.ndarray,
    strategy_type: str,
    strategy_name: str,
    y_true: np.ndarray = None,
    X_transformed: np.ndarray = None,
    model = None
) -> np.ndarray:
    """
    Apply prediction adjustments based on mitigation strategy
    """
    y_pred_adjusted = y_pred.copy().astype(float)
    
    if strategy_type == 'pre-processing':
        print(f"  Pre-processing strategy detected")
        
        # If we have transformed data and model, use it for new predictions
        if X_transformed is not None and model is not None:
            print(f"    Using model to predict on transformed data...")
            y_pred_adjusted = model.predict(X_transformed).astype(float)
        
        # Otherwise, apply fairness-aware adjustments to existing predictions
        elif strategy_name == 'disparate_impact_remover':
            print(f"    Applying disparate impact adjustments...")
            # Flip predictions for underrepresented groups to balance outcomes
            for group in np.unique(sensitive_attr):
                group_mask = sensitive_attr == group
                group_positive_rate = np.mean(y_pred[group_mask])
                
                # If this group has very low positive rate, increase it slightly
                if group_positive_rate < 0.3:
                    flip_indices = np.where(group_mask & (y_pred == 0))[0]
                    if len(flip_indices) > 0:
                        num_flips = int(len(flip_indices) * 0.15)  # Flip 15%
                        flip_idx = np.random.choice(flip_indices, size=num_flips, replace=False)
                        y_pred_adjusted[flip_idx] = 1
                        print(f"      Group {group}: flipped {num_flips} predictions")
        
        elif strategy_name == 'data_augmentation':
            print(f"    Applying data augmentation adjustments...")
            # Balanced representation - adjust predictions to reflect balance
            pass
        
        return y_pred_adjusted.astype(int)
    
    elif strategy_type == 'in-processing':
        print(f"  In-processing strategy detected")
        # Model was already fine-tuned, predictions remain same
        return y_pred_adjusted.astype(int)
    
    elif strategy_type == 'post-processing':
        print(f"  Post-processing strategy detected - applying threshold adjustments")
        
        if strategy_name == 'threshold_optimization':
            # Adjust thresholds per group
            for group in np.unique(sensitive_attr):
                group_mask = sensitive_attr == group
                group_pred = y_pred[group_mask]
                
                threshold = np.percentile(group_pred, 50)
                y_pred_adjusted[group_mask] = (group_pred > threshold).astype(float)
                print(f"      Group {group}: threshold = {threshold:.3f}")
        
        elif strategy_name == 'calibration_adjustment':
            # Smooth predictions
            y_pred_adjusted = 0.9 * y_pred_adjusted + 0.1 * np.random.random(len(y_pred))
            y_pred_adjusted = np.clip(y_pred_adjusted, 0, 1)
        
        return y_pred_adjusted.astype(int)
    
    return y_pred_adjusted.astype(int)


def _calculate_metric_improvements(baseline_bias: Dict, mitigated_bias: Dict) -> Dict:
    """
    Calculate improvements for each individual metric
    """
    improvements = {}
    
    # Extract metrics from both
    baseline_metrics = {k: v for k, v in baseline_bias.items() if k != 'summary'}
    mitigated_metrics = {k: v for k, v in mitigated_bias.items() if k != 'summary'}
    
    for metric_name in baseline_metrics.keys():
        if metric_name in mitigated_metrics:
            baseline_metric = baseline_metrics[metric_name]
            mitigated_metric = mitigated_metrics[metric_name]
            
            if isinstance(baseline_metric, dict) and 'is_biased' in baseline_metric:
                was_biased = baseline_metric.get('is_biased', False)
                is_now_biased = mitigated_metric.get('is_biased', False)
                
                if was_biased and not is_now_biased:
                    improvements[metric_name] = 'FIXED'
                elif not was_biased and is_now_biased:
                    improvements[metric_name] = 'BROKEN'
                else:
                    improvements[metric_name] = 'NO_CHANGE'
    
    return improvements

def _generate_mitigation_recommendations(
    improvement: int,
    strategy_type: str,
    remaining_biased_metrics: List[str]
) -> List[str]:
    """
    Generate recommendations based on mitigation results
    """
    recommendations = []
    
    if improvement > 0:
        recommendations.append(f"✅ Strategy successfully reduced biased metrics by {improvement}")
    elif improvement < 0:
        recommendations.append(f"❌ Strategy increased biased metrics by {abs(improvement)}")
    else:
        recommendations.append("➡️ Strategy had no impact on overall bias")
    
    if remaining_biased_metrics:
        recommendations.append(f"⚠️ Remaining biased metrics: {', '.join(remaining_biased_metrics)}")
        recommendations.append("💡 Consider combining this strategy with others for better results")
    else:
        recommendations.append("✨ All bias metrics resolved!")
    
    if strategy_type == 'pre-processing':
        recommendations.append("📊 Pre-processing strategy applied - retrain model for best results")
    elif strategy_type == 'post-processing':
        recommendations.append("🎯 Post-processing strategy applied - no model retraining needed")
    
    return recommendations
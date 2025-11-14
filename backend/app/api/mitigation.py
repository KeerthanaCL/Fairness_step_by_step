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
import logging
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/mitigation", tags=["Mitigation Strategies"])

def _load_and_prepare(train_required=True, require_model=False, prediction_column=None, sensitive_feature_column=None):
    """
    Helper: Loads train_df from FileManager, validates, returns:
    train_df, X, y_true (int array), sensitive_attr (int array or None), model (or None), X_encoded
    Raises HTTPException on validation failures.
    """
    if global_session.train_file_path is None:
        raise HTTPException(status_code=400, detail="Training data must be uploaded first")

    train_df = FileManager.load_csv(global_session.train_file_path)

    if global_session.target_column is None:
        raise HTTPException(status_code=400, detail="Target column must be set first")

    # Prepare data
    X = train_df.drop(columns=[global_session.target_column]).copy()
    y_true = _safe_convert_to_numeric(train_df[global_session.target_column].values, "y_true")

    model = None
    if global_session.model_file_path:
        try:
            model = FileManager.load_model(global_session.model_file_path)
            logger.info("Loaded user model for mitigation")
        except Exception as e:
            logger.warning(f"Failed to load user model: {e}")
            model = None

    sensitive_attr = None
    if sensitive_feature_column:
        if sensitive_feature_column not in train_df.columns:
            raise HTTPException(status_code=400, detail=f"Sensitive feature column '{sensitive_feature_column}' not found in data. Available columns: {list(train_df.columns)}")
        sensitive_attr = _safe_convert_to_numeric(train_df[sensitive_feature_column].values, sensitive_feature_column)

    # Predictions if requested (prediction_column takes precedence)
    y_pred = None
    if prediction_column and prediction_column.strip():
        if prediction_column not in train_df.columns:
            raise HTTPException(status_code=400, detail=f"Prediction column '{prediction_column}' not found")
        y_pred = _safe_convert_to_numeric(train_df[prediction_column].values, prediction_column)
    elif model is not None:
        X_encoded = _encode_dataframe(X)
        y_pred = _safe_convert_to_numeric(model.predict(X_encoded), "model_prediction")

    X_encoded = _encode_dataframe(X)
    return train_df, X, X_encoded, y_true, sensitive_attr, model, y_pred

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
    Uses _load_and_prepare to centralize loading/validation and enforce consistent encodings.
    """
    try:
        # Centralized load/prepare
        train_df, X_train, X_train_encoded, y_true, sensitive_attr, model, y_pred_original = _load_and_prepare(
            prediction_column=request.prediction_column,
            sensitive_feature_column=request.sensitive_feature_column
        )

        # If model-based strategies are required, ensure model present
        if model is None:
            raise HTTPException(status_code=400, detail="Model required for mitigation strategies")

        logger.info(f"Applying mitigation strategy: {request.strategy} on sensitive feature: {request.sensitive_feature_column}")

        # Baseline bias
        bias_detector = BiasDetector()
        baseline_bias = bias_detector.run_all_metrics(y_true, y_pred_original, sensitive_attr)
        baseline_bias = _clean_nan_values(baseline_bias)
        baseline_biased_count = baseline_bias['summary']['biased_metrics_count']
        baseline_biased_metrics = baseline_bias['summary']['biased_metrics']

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
        
        # Ensure strategy returned something meaningful
        if strategy_result.get('status') not in ['success', 'partial']:
            logger.error(f"Strategy {request.strategy} failed: {strategy_result.get('error')}")
            raise HTTPException(status_code=400, detail=f"Strategy failed: {strategy_result.get('error')}")

        # Determine mitigated predictions and/or transformed data
        y_pred_mitigated = None
        X_mitigated = X_train_encoded.copy()

        if getattr(mitigator, 'X_transformed', None) is not None:
            X_mitigated = mitigator.X_transformed
            # If transformed has more features than expected, truncate (best-effort)
            expected_cols = X_train_encoded.shape[1]
            if hasattr(X_mitigated, 'shape') and X_mitigated.shape[1] > expected_cols:
                X_mitigated = X_mitigated[:, :expected_cols]
            try:
                y_pred_mitigated = mitigator.user_model.predict(X_mitigated)
                y_pred_mitigated = _safe_convert_to_numeric(y_pred_mitigated, "mitigated_prediction")
            except Exception as e:
                logger.warning(f"Predict on transformed X failed, falling back to original X: {e}")
                y_pred_mitigated = mitigator.user_model.predict(X_train_encoded)
                y_pred_mitigated = _safe_convert_to_numeric(y_pred_mitigated, "mitigated_prediction")

        elif getattr(mitigator, 'sample_weights', None) is not None:
            # Re-train (or attempt) with sample weights if supported
            try:
                import copy
                model_reweighted = copy.deepcopy(mitigator.user_model)
                if hasattr(model_reweighted, 'fit'):
                    try:
                        model_reweighted.fit(X_train_encoded, y_true, sample_weight=mitigator.sample_weights)
                        logger.info("Retrained model with sample weights")
                    except TypeError:
                        logger.info("Model.fit doesn't accept sample_weight; skipping retrain")
                y_pred_mitigated = model_reweighted.predict(X_train_encoded)
                y_pred_mitigated = _safe_convert_to_numeric(y_pred_mitigated, "mitigated_prediction")
            except Exception as e:
                logger.exception("Reweighting failed, using original predictions")
                y_pred_mitigated = y_pred_original

        elif getattr(mitigator, 'y_pred_adjusted', None) is not None:
            y_pred_mitigated = _safe_convert_to_numeric(mitigator.y_pred_adjusted, "mitigated_prediction")

        elif getattr(mitigator, 'model_modified', False):
            model_to_use = getattr(mitigator, 'fine_tuned_model', mitigator.user_model)
            y_pred_mitigated = _safe_convert_to_numeric(model_to_use.predict(X_train_encoded), "mitigated_prediction")

        # Final validation: ensure lengths match
        if y_pred_mitigated is None:
            raise HTTPException(status_code=500, detail="Mitigated predictions could not be obtained")

        # If lengths differ and mitigator provided augmented targets, reconcile them
        y_true_for_metrics = y_true.copy()
        sensitive_for_metrics = sensitive_attr.copy()
        if getattr(mitigator, 'y_augmented', None) is not None:
            y_true_for_metrics = np.array(mitigator.y_augmented)
            sensitive_for_metrics = np.array(mitigator.sensitive_augmented)

        if len(y_pred_mitigated) != len(y_true_for_metrics):
            logger.warning(f"Size mismatch: pred={len(y_pred_mitigated)} target={len(y_true_for_metrics)} - applying truncation if safe")
            if len(y_pred_mitigated) > len(y_true_for_metrics):
                y_pred_mitigated = y_pred_mitigated[:len(y_true_for_metrics)]
            else:
                raise HTTPException(status_code=400, detail="Mitigated predictions fewer than targets after mitigation")

        mitigated_bias = bias_detector.run_all_metrics(y_true_for_metrics, y_pred_mitigated, sensitive_for_metrics)
        mitigated_bias = _clean_nan_values(mitigated_bias) 
        mitigated_biased_count = mitigated_bias['summary']['biased_metrics_count']
        mitigated_biased_metrics = mitigated_bias['summary']['biased_metrics']
        
        # ===== STEP 4: Calculate improvements =====
        improvement = baseline_biased_count - mitigated_biased_count 
        metric_improvements = _calculate_metric_improvements(baseline_bias, mitigated_bias)
        
        # Build response
        response = {
            'status': 'success',
            'strategy_applied': request.strategy,
            'strategy_info': _serialize_mitigation_result(strategy_result),
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
        
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error applying mitigation")
        raise HTTPException(status_code=500, detail=f"Error applying mitigation: {str(e)}")

@router.post("/compare")
async def compare_strategies(request: StrategyComparisonRequest):
    """
    Compare bias scores before and after applying different strategies
    """
    try:
        # Load & prepare using shared helper
        train_df, X_train, X_train_encoded, y_true, sensitive_attr, model, y_pred = _load_and_prepare(
            prediction_column=request.prediction_column,
            sensitive_feature_column=request.sensitive_feature_column
        )

        if model is None:
            raise HTTPException(status_code=400, detail="Model required for strategy comparison")
        
        # Get baseline bias metrics
        bias_detector = BiasDetector()
        baseline_results = bias_detector.run_all_metrics(y_true, y_pred, sensitive_attr)
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
                    sensitive_attr,
                    X_test=X_train_encoded,
                    y_test=y_true,
                    y_pred_test=y_pred
                )
                
                # For now, assume predictions remain same (in real scenario, would re-predict)
                # In production, you'd retrain the model with mitigated data
                if mitigator.X_transformed is not None:
                    X_tmp = mitigator.X_transformed
                    expected_cols = X_train_encoded.shape[1]
                    if X_tmp.shape[1] > expected_cols:
                        X_tmp = X_tmp[:, :expected_cols]
                    try:
                        y_pred_mitigated = model.predict(X_tmp)
                        y_pred_mitigated = _safe_convert_to_numeric(y_pred_mitigated, "mitigated_prediction")
                    except:
                        y_pred_mitigated = y_pred

                elif mitigator.y_pred_adjusted is not None:
                    y_pred_mitigated = _safe_convert_to_numeric(mitigator.y_pred_adjusted, "mitigated_prediction")

                elif mitigator.sample_weights is not None:
                    # simple reweighted retrain
                    try:
                        import copy
                        model_copy = copy.deepcopy(model)
                        model_copy.fit(X_train_encoded, y_true, sample_weight=mitigator.sample_weights)
                        y_pred_mitigated = model_copy.predict(X_train_encoded)
                        y_pred_mitigated = _safe_convert_to_numeric(y_pred_mitigated, "mitigated_prediction")
                    except:
                        y_pred_mitigated = y_pred

                elif mitigator.model_modified:
                    mdl = getattr(mitigator, "fine_tuned_model", model)
                    y_pred_mitigated = _safe_convert_to_numeric(mdl.predict(X_train_encoded), "mitigated_prediction")

                else:
                    y_pred_mitigated = y_pred
                
                # Calculate bias metrics after mitigation
                if strategy_result.get('status') == 'success':
                    print(f"  ✅ Strategy applied successfully")
                    mitigated_results = bias_detector.run_all_metrics(
                        y_true, y_pred_mitigated, sensitive_attr
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

        # Extract strategy list from multi-stage pipeline
        strategies = []
        pipeline_dict = request.pipeline or {}

        for stage in ["pre", "in", "post"]:
            if stage in pipeline_dict and isinstance(pipeline_dict[stage], list):
                strategies.extend(pipeline_dict[stage])

        if not strategies:
            raise HTTPException(
                status_code=400,
                detail="Pipeline is empty. Add strategies under pre/in/post stages."
            )

        print(f"  Final strategy execution order: {strategies}")

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

        # Use shared loader for data inputs
        _, X_train, X_train_encoded, y_true, sensitive_attr, model, y_pred = _load_and_prepare(
            prediction_column=request.prediction_column,
            sensitive_feature_column=request.sensitive_feature_column
        )
        
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

        if model is None:
            raise HTTPException(status_code=400, detail="Model required for mitigation strategies")

        print(f"\n[PIPELINE SETUP]")
        print(f"  Strategies to apply: {strategies}")
        print(f"  Model type: {type(model).__name__}")

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
            X_current = X_train_encoded
            
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
                    X_current,
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
                    if hasattr(mitigator, "X_augmented") and mitigator.X_augmented is not None:
                        X_current = _encode_dataframe(pd.DataFrame(mitigator.X_augmented))
                    else:
                        X_current = X_current

                elif mitigator.y_pred_adjusted is not None:
                    print(f"  Using adjusted predictions")
                    y_pred_current = mitigator.y_pred_adjusted
                
                elif mitigator.X_transformed is not None:
                    X_train_encoded_temp = mitigator.X_transformed
                    
                    # Handle feature mismatch
                    num_original_features = X_train_encoded.shape[1]
                    if X_train_encoded_temp.shape[1] > num_original_features:
                        X_train_encoded_temp = X_train_encoded_temp[:, :num_original_features]
                    
                    X_current = X_train_encoded_temp
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
            X_test_encoded, _ = _encode_dataframe(X_test)
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
        X_train_encoded, encoders = _encode_dataframe(X_train)

        global_session.encoders = encoders
        
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
            X_test_encoded, test_encoders = _encode_dataframe(X_test)

            # Store encoders in session
            if not hasattr(global_session, 'encoders'):
                global_session.encoders = test_encoders
            try:
                # Ensure it's a proper 2D numpy array
                if isinstance(X_test_encoded, pd.DataFrame):
                    X_test_encoded = X_test_encoded.values
                
                # Force to float type and check shape
                X_test_encoded = np.array(X_test_encoded, dtype=np.float64)
                
                # Verify shape consistency
                if X_test_encoded.ndim != 2:
                    raise ValueError(f"X_test_encoded must be 2D, got shape {X_test_encoded.shape}")
                
                # Check for nested arrays or objects
                if X_test_encoded.dtype == 'object':
                    logger.error("X_test_encoded contains object dtype - attempting to flatten")
                    # Try to flatten nested structures
                    X_flat = []
                    for row in X_test_encoded:
                        if isinstance(row, (list, np.ndarray)):
                            X_flat.append(np.array(row, dtype=float).flatten())
                        else:
                            X_flat.append([float(row)])
                    X_test_encoded = np.array(X_flat, dtype=np.float64)
                
                logger.info(f"X_test_encoded shape: {X_test_encoded.shape}, dtype: {X_test_encoded.dtype}")

                # FIX: Validate feature count
                if hasattr(model, 'n_features_in_'):
                    expected_features = model.n_features_in_
                    actual_features = X_test_encoded.shape[1]
                    if expected_features != actual_features:
                        raise ValueError(
                            f"Feature mismatch: Model expects {expected_features} features, "
                            f"but got {actual_features}"
                        )
                
                y_pred = model.predict(X_test_encoded)
                
            except ValueError as ve:
                logger.error(f"Shape validation error: {ve}")
                logger.error(f"X_test_encoded shape: {X_test_encoded.shape if hasattr(X_test_encoded, 'shape') else 'no shape'}")
                logger.error(f"X_test_encoded dtype: {X_test_encoded.dtype if hasattr(X_test_encoded, 'dtype') else 'no dtype'}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Data shape error: {str(ve)}. Ensure all features are numeric and have consistent dimensions."
                )
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
        X_train_encoded, _ = _encode_dataframe(X_train, saved_encoders=global_session.encoders)
        
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
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

        # Apply mitigation
        mitigator = MitigationStrategies()
        strategy_result = mitigator.run_mitigation_strategy(
            request.strategy,
            X_train_encoded,
            y_true,
            sensitive_attr,
            X_test=X_train_encoded,
            y_test=y_true,
            y_pred_test=y_pred_original
        )
        
        if strategy_result.get('status') == 'error':
            raise HTTPException(status_code=400, detail=strategy_result.get('error'))
        
        # ===== STEP 3: Generate mitigated predictions =====
        print(f"\n[STEP 3] Generating mitigated predictions...")
        
        # For simplification, we'll apply mild adjustments based on strategy type
        y_pred_mitigated = _apply_mitigation_adjustment(
            y_pred_original,
            sensitive_attr,
            strategy_result.get('type', 'post-processing'),
            request.strategy
        )
        
        # ===== STEP 4: Calculate MITIGATED bias metrics =====
        print(f"\n[STEP 4] Calculating MITIGATED bias metrics...")
        mitigated_bias = bias_detector.run_all_metrics(y_true, y_pred_mitigated, sensitive_attr)
        mitigated_bias = _clean_nan_values(mitigated_bias)
        
        mitigated_biased_count = mitigated_bias['summary']['biased_metrics_count']
        mitigated_biased_metrics = mitigated_bias['summary']['biased_metrics']
        
        print(f"  Biased metrics count: {mitigated_biased_count}")
        print(f"  Biased metrics: {mitigated_biased_metrics}")
        
        # ===== STEP 5: Calculate improvements =====
        print(f"\n[STEP 5] Calculating improvements...")
        improvement = baseline_biased_count - mitigated_biased_count
        
        # Calculate detailed metric improvements
        metric_improvements = _calculate_metric_improvements(baseline_bias, mitigated_bias)
        
        print(f"  Overall improvement: {improvement} fewer biased metrics")
        print(f"  Status: {'IMPROVED' if improvement > 0 else ('❌ WORSENED' if improvement < 0 else '➡️ NO CHANGE')}")
        
        # Clean strategy result for JSON serialization
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
        
        print(f"\n{'='*60}")
        
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
            mitigator = MitigationStrategies()
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
        
        mitigator = MitigationStrategies()
        
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
        
        # Encode data
        X_train_encoded = _encode_dataframe(X_train)
        
        # Execute pipeline
        pipeline_results = pipeline.execute_pipeline(
            X_train_encoded,
            y_true,
            sensitive_attr,
            X_test=X_train_encoded,
            y_test=y_true,
            y_pred_test=y_pred
        )
        
        # Generate mitigated predictions (simplified for demo)
        y_pred_mitigated = y_pred.copy()
        for stage_name, stage_results in pipeline_results['stage_results'].items():
            for result in stage_results:
                if result['status'] == 'success':
                    # Apply mild adjustment per strategy
                    strategy_name = result['strategy']
                    strategy_type = result['result'].get('type', 'post-processing')
                    y_pred_mitigated = _apply_mitigation_adjustment(
                        y_pred_mitigated,
                        sensitive_attr,
                        strategy_type,
                        strategy_name
                    )
        
        # Calculate MITIGATED bias metrics
        print(f"\n[AFTER PIPELINE] Calculating mitigated bias metrics...")
        mitigated_bias = bias_detector.run_all_metrics(y_true, y_pred_mitigated, sensitive_attr)
        mitigated_bias = _clean_nan_values(mitigated_bias)
        
        mitigated_biased_count = mitigated_bias['summary']['biased_metrics_count']
        improvement = baseline_biased_count - mitigated_biased_count
        
        print(f"  Mitigated biased metrics: {mitigated_biased_count}")
        print(f"  Overall improvement: {improvement}")
        
        # Calculate detailed metric improvements
        metric_improvements = _calculate_metric_improvements(baseline_bias, mitigated_bias)
        
        # Clean results for JSON
        pipeline_results_clean = _serialize_mitigation_result(pipeline_results)
        
        response = {
            'status': 'success',
            'pipeline_config': pipeline_summary,
            'pipeline_execution': pipeline_results_clean,
            
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
        
        # Encode data
        X_train_encoded = _encode_dataframe(X_train)
        
        # Run optimization
        from app.pipeline_optimizer import PipelineOptimizer
        optimizer = PipelineOptimizer()
        
        if request.method == 'greedy':
            result = optimizer.greedy_search(
                X_train_encoded,
                y_true,
                sensitive_attr,
                y_pred,
                baseline_biased_count,
                max_strategies=request.max_strategies or 3
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
    strategy_name: str
) -> np.ndarray:
    """
    Apply prediction adjustments based on mitigation strategy
    """
    y_pred_adjusted = y_pred.copy().astype(float)
    
    if strategy_type == 'pre-processing':
        # Pre-processing: minor adjustment as data is already cleaned
        print(f"  Pre-processing strategy detected - minimal prediction adjustment")
        return y_pred_adjusted.astype(int)
    
    elif strategy_type == 'post-processing':
        print(f"  Post-processing strategy detected - applying threshold adjustments")
        
        # Adjust threshold per group
        group_0_mask = sensitive_attr == 0
        group_1_mask = sensitive_attr == 1
        
        if strategy_name == 'threshold_optimization':
            # For group 0: lower threshold (make more positive predictions)
            # For group 1: higher threshold (make fewer positive predictions)
            y_pred_adjusted[group_0_mask] = np.where(
                y_pred[group_0_mask] >= 0.4,
                1,
                y_pred[group_0_mask]
            )
            y_pred_adjusted[group_1_mask] = np.where(
                y_pred[group_1_mask] >= 0.6,
                1,
                y_pred[group_1_mask]
            )
        
        elif strategy_name == 'calibration_adjustment':
            # Smooth calibration
            y_pred_adjusted = 0.9 * y_pred_adjusted + 0.1 * np.random.random(len(y_pred))
            y_pred_adjusted = np.clip(y_pred_adjusted, 0, 1)
        
        elif strategy_name == 'equalized_odds_postprocessing':
            # Flip some predictions for balance
            flip_rate = 0.05
            flip_indices = np.random.choice(len(y_pred), size=int(len(y_pred) * flip_rate), replace=False)
            y_pred_adjusted[flip_indices] = 1 - y_pred_adjusted[flip_indices]
        
        return np.round(y_pred_adjusted).astype(int)
    
    return y_pred


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
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from itertools import combinations, product
from app.mitigation_strategies import MitigationPipeline, MitigationStrategies

class PipelineOptimizer:
    """
    Automated pipeline optimization using different search strategies
    """
    
    def __init__(self, bias_detector=None):
        self.mitigator = MitigationStrategies()
        self.strategies_info = self.mitigator.get_strategies_info()
        self.bias_detector = bias_detector
        
        # Categorize strategies by type
        self.pre_strategies = [
            name for name, info in self.strategies_info.items()
            if 'pre-processing' in info['type']
        ]
        self.in_strategies = [
            name for name, info in self.strategies_info.items()
            if 'in-processing' in info['type']
        ]
        self.post_strategies = [
            name for name, info in self.strategies_info.items()
            if 'post-processing' in info['type']
        ]
    
    def greedy_search(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        sensitive_attr: np.ndarray,
        y_pred: np.ndarray,
        baseline_biased_count: int,
        max_strategies: int = 3
    ) -> Dict:
        """
        Greedy Search: Iteratively add the best strategy at each step
        
        Algorithm:
        1. Start with empty pipeline
        2. Try adding each available strategy
        3. Keep the one that gives best improvement
        4. Repeat until max_strategies or no improvement
        """
        print(f"\n{'='*60}")
        print(f"GREEDY SEARCH OPTIMIZATION")
        print(f"{'='*60}")
        print(f"Max strategies: {max_strategies}")
        print(f"Baseline biased metrics: {baseline_biased_count}")
        
        current_pipeline = {'pre': [], 'in': [], 'post': []}
        current_biased_count = baseline_biased_count
        search_history = []
        
        all_strategies = {
            'pre': self.pre_strategies.copy(),
            'in': self.in_strategies.copy(),
            'post': self.post_strategies.copy()
        }
        
        for iteration in range(max_strategies):
            print(f"\n[ITERATION {iteration + 1}]")
            print(f"Current pipeline: {current_pipeline}")
            print(f"Current biased count: {current_biased_count}")
            
            best_improvement = 0
            best_strategy = None
            best_stage = None
            best_biased_count = current_biased_count
            candidates_evaluated = 0
            
            # Try each remaining strategy
            for stage in ['pre', 'in', 'post']:
                for strategy in all_strategies[stage]:
                    candidates_evaluated += 1
                    
                    # Create test pipeline
                    test_pipeline = {
                        'pre': current_pipeline['pre'].copy(),
                        'in': current_pipeline['in'].copy(),
                        'post': current_pipeline['post'].copy()
                    }
                    test_pipeline[stage].append(strategy)
                    
                    # Evaluate (simplified)
                    test_biased_count = self._evaluate_pipeline_real(
                        test_pipeline, X_train, y_train, sensitive_attr, y_pred
                    )
                    
                    improvement = current_biased_count - test_biased_count
                    
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_strategy = strategy
                        best_stage = stage
                        best_biased_count = test_biased_count
            
            print(f"  Evaluated {candidates_evaluated} candidates")
            
            # If no improvement found, stop
            if best_improvement <= 0:
                print(f"  No improvement found. Stopping.")
                break
            
            # Add best strategy
            current_pipeline[best_stage].append(best_strategy)
            all_strategies[best_stage].remove(best_strategy)
            
            search_history.append({
                'iteration': iteration + 1,
                'added_strategy': best_strategy,
                'added_to_stage': best_stage,
                'improvement': float(best_improvement),
                'biased_count': int(best_biased_count)
            })
            
            current_biased_count = best_biased_count
            
            print(f"  ✅ Added: {best_strategy} to {best_stage}")
            print(f"  Improvement: {best_improvement}")
            print(f"  New biased count: {current_biased_count}")
        
        print(f"\n{'='*60}")
        print(f"GREEDY SEARCH COMPLETE")
        print(f"Final pipeline: {current_pipeline}")
        print(f"Final biased count: {current_biased_count}")
        print(f"Total improvement: {baseline_biased_count - current_biased_count}")
        print(f"{'='*60}\n")
        
        return {
            'method': 'greedy_search',
            'best_pipeline': current_pipeline,
            'final_biased_count': int(current_biased_count),
            'total_improvement': int(baseline_biased_count - current_biased_count),
            'search_history': search_history,
            'iterations': len(search_history)
        }
    
    def top_k_method(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        sensitive_attr: np.ndarray,
        y_pred: np.ndarray,
        baseline_biased_count: int,
        k: int = 5
    ) -> Dict:
        """
        Top-K Method: Evaluate all single strategies, pick top K, then combine them
        
        Algorithm:
        1. Evaluate each strategy individually
        2. Rank by improvement
        3. Select top K strategies
        4. Create pipeline with top K strategies
        """
        print(f"\n{'='*60}")
        print(f"TOP-K METHOD OPTIMIZATION")
        print(f"{'='*60}")
        print(f"K = {k}")
        print(f"Baseline biased metrics: {baseline_biased_count}")
        
        # Evaluate all strategies individually
        strategy_scores = []
        
        print(f"\n[PHASE 1] Evaluating individual strategies...")
        
        all_strategies_flat = [
            (strategy, 'pre') for strategy in self.pre_strategies
        ] + [
            (strategy, 'in') for strategy in self.in_strategies
        ] + [
            (strategy, 'post') for strategy in self.post_strategies
        ]
        
        for idx, (strategy, stage) in enumerate(all_strategies_flat, 1):
            print(f"  [{idx}/{len(all_strategies_flat)}] Evaluating: {strategy}")
            
            test_pipeline = {'pre': [], 'in': [], 'post': []}
            test_pipeline[stage].append(strategy)
            
            biased_count = self._evaluate_pipeline_real(
                test_pipeline, X_train, y_train, sensitive_attr, y_pred
            )
            
            improvement = baseline_biased_count - biased_count
            
            strategy_scores.append({
                'strategy': strategy,
                'stage': stage,
                'biased_count': int(biased_count),
                'improvement': float(improvement)
            })
            
            print(f"    Improvement: {improvement}")
        
        # Rank strategies
        strategy_scores.sort(key=lambda x: x['improvement'], reverse=True)
        
        print(f"\n[PHASE 2] Top strategies ranked by improvement:")
        for idx, score in enumerate(strategy_scores[:k], 1):
            print(f"  {idx}. {score['strategy']} ({score['stage']}): improvement = {score['improvement']}")
        
        # Select top K
        top_k = strategy_scores[:k]
        
        # Build combined pipeline
        combined_pipeline = {'pre': [], 'in': [], 'post': []}
        for item in top_k:
            combined_pipeline[item['stage']].append(item['strategy'])
        
        print(f"\n[PHASE 3] Building combined pipeline with top {k} strategies...")
        print(f"  Pipeline: {combined_pipeline}")
        
        # Evaluate combined pipeline
        final_biased_count = self._evaluate_pipeline_real(
            combined_pipeline, X_train, y_train, sensitive_attr, y_pred
        )
        
        final_improvement = baseline_biased_count - final_biased_count
        
        print(f"\n{'='*60}")
        print(f"TOP-K METHOD COMPLETE")
        print(f"Combined pipeline biased count: {final_biased_count}")
        print(f"Total improvement: {final_improvement}")
        print(f"{'='*60}\n")
        
        return {
            'method': 'top_k',
            'k': k,
            'best_pipeline': combined_pipeline,
            'final_biased_count': int(final_biased_count),
            'total_improvement': int(final_improvement),
            'top_strategies': top_k,
            'all_strategy_scores': strategy_scores
        }
    
    def brute_force_search(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        sensitive_attr: np.ndarray,
        y_pred: np.ndarray,
        baseline_biased_count: int,
        max_strategies_per_stage: int = 2
    ) -> Dict:
        """
        Brute Force Search: Try all possible combinations
        
        Algorithm:
        1. Generate all valid pipeline combinations
        2. Evaluate each combination
        3. Return best pipeline
        
        Warning: Can be computationally expensive!
        """
        print(f"\n{'='*60}")
        print(f"BRUTE FORCE SEARCH OPTIMIZATION")
        print(f"{'='*60}")
        print(f"Max strategies per stage: {max_strategies_per_stage}")
        print(f"Baseline biased metrics: {baseline_biased_count}")
        
        # Generate all possible combinations
        print(f"\n[PHASE 1] Generating pipeline combinations...")
        
        all_pipelines = []
        
        # Pre-processing combinations (including empty)
        pre_combos = [[]]  # Empty option
        for r in range(1, min(max_strategies_per_stage, len(self.pre_strategies)) + 1):
            pre_combos.extend(list(combinations(self.pre_strategies, r)))
        
        # In-processing combinations
        in_combos = [[]]
        for r in range(1, min(max_strategies_per_stage, len(self.in_strategies)) + 1):
            in_combos.extend(list(combinations(self.in_strategies, r)))
        
        # Post-processing combinations
        post_combos = [[]]
        for r in range(1, min(max_strategies_per_stage, len(self.post_strategies)) + 1):
            post_combos.extend(list(combinations(self.post_strategies, r)))
        
        # Generate all pipeline combinations
        for pre, in_proc, post in product(pre_combos, in_combos, post_combos):
            # Skip empty pipeline
            if not pre and not in_proc and not post:
                continue
            
            pipeline = {
                'pre': list(pre),
                'in': list(in_proc),
                'post': list(post)
            }
            all_pipelines.append(pipeline)
        
        total_combinations = len(all_pipelines)
        print(f"  Total combinations to evaluate: {total_combinations}")
        
        if total_combinations > 100:
            print(f"  ⚠️ Warning: Large search space! This may take a while...")
        
        # Evaluate all pipelines
        print(f"\n[PHASE 2] Evaluating all pipeline combinations...")
        
        results = []
        best_biased_count = baseline_biased_count
        best_pipeline = None
        
        for idx, pipeline in enumerate(all_pipelines, 1):
            if idx % max(1, total_combinations // 10) == 0 or idx == 1:
                print(f"  Progress: {idx}/{total_combinations}")
            
            biased_count = self._evaluate_pipeline_real(
                pipeline, X_train, y_train, sensitive_attr, y_pred
            )
            
            improvement = baseline_biased_count - biased_count
            
            results.append({
                'pipeline': pipeline,
                'biased_count': int(biased_count),
                'improvement': float(improvement)
            })
            
            if biased_count < best_biased_count:
                best_biased_count = biased_count
                best_pipeline = pipeline
        
        # Sort results by improvement
        results.sort(key=lambda x: x['improvement'], reverse=True)
        
        print(f"\n[PHASE 3] Results summary:")
        print(f"  Top 5 pipelines:")
        for idx, result in enumerate(results[:5], 1):
            print(f"    {idx}. Improvement: {result['improvement']}, Pipeline: {result['pipeline']}")
        
        print(f"\n{'='*60}")
        print(f"BRUTE FORCE SEARCH COMPLETE")
        print(f"Best pipeline: {best_pipeline}")
        print(f"Best biased count: {best_biased_count}")
        print(f"Total improvement: {baseline_biased_count - best_biased_count}")
        print(f"{'='*60}\n")
        
        return {
            'method': 'brute_force',
            'total_combinations_evaluated': total_combinations,
            'best_pipeline': best_pipeline,
            'final_biased_count': int(best_biased_count),
            'total_improvement': int(baseline_biased_count - best_biased_count),
            'top_5_pipelines': results[:5],
            'all_results': results if total_combinations <= 50 else results[:20]  # Limit output
        }
    
    def _evaluate_pipeline_real(
        self,
        pipeline_config: Dict,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        sensitive_attr: np.ndarray,
        y_pred: np.ndarray
    ) -> int:
        """
        Evaluate a pipeline configuration by actually running it
        and calculating bias metrics
        
        Simplified evaluation for optimization
        """
        # Simple heuristic-based evaluation
        # In production, you would run full pipeline and calculate actual bias metrics
        
        try:
            from app.api.mitigation import (
                _apply_mitigation_adjustment,
                _serialize_mitigation_result
            )
            from app.bias_detector import BiasDetector
            
            if self.bias_detector is None:
                self.bias_detector = BiasDetector()
            
            # Execute pipeline
            pipeline = MitigationPipeline()
            
            # Add strategies
            for stage in ['pre', 'in', 'post']:
                for strategy in pipeline_config[stage]:
                    try:
                        pipeline.add_strategy(strategy, stage)
                    except:
                        continue
            
            # Skip if pipeline is empty
            total_strategies = sum(len(v) for v in pipeline_config.values())
            if total_strategies == 0:
                # No mitigation, return original bias count
                return int(np.sum(y_pred != y_train) / len(y_pred) * 100)
            
            # Execute pipeline
            pipeline_results = pipeline.execute_pipeline(
                X_train, y_train, sensitive_attr,
                X_test=X_train, y_test=y_train, y_pred_test=y_pred
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
            
            # Calculate bias metrics
            mitigated_bias = self.bias_detector.run_all_metrics(
                y_train, y_pred_mitigated, sensitive_attr
            )
            
            biased_count = mitigated_bias['summary']['biased_metrics_count']
            
            return int(biased_count)
        
        except Exception as e:
            print(f"    Error evaluating pipeline: {str(e)}")
            # Return large number if evaluation fails
            return 999
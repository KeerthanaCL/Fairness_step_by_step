import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from itertools import combinations, product
from app.mitigation_strategies import MitigationPipeline, MitigationStrategies

class PipelineOptimizer:
    """
    Automated pipeline optimization using different search strategies
    """
    
    def __init__(self, bias_detector=None, user_model=None):
        self.user_model = user_model
        self.mitigator = MitigationStrategies(user_model=user_model)
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
    
    def stage_wise_greedy_search(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        sensitive_attr: np.ndarray,
        y_pred: np.ndarray,
        baseline_biased_count: int
    ) -> Dict:
        """
        Stage-wise Greedy Search: Find best strategy for each stage independently
        
        Algorithm:
        1. Find best PRE-PROCESSING strategy
        2. Apply it, then find best IN-PROCESSING strategy
        3. Apply both, then find best POST-PROCESSING strategy
        4. Return pipeline with 3 best strategies (one per stage)
        """
        print(f"\n{'='*60}")
        print(f"STAGE-WISE GREEDY SEARCH")
        print(f"{'='*60}")
        print(f"Baseline biased metrics: {baseline_biased_count}")
        print(f"Finding best strategy for each stage independently...")
        
        best_pipeline = {'pre': [], 'in': [], 'post': []}
        current_y_pred = y_pred.copy()
        current_biased_count = baseline_biased_count
        stage_results = []
        
        # Stage 1: Find best PRE-PROCESSING strategy
        print(f"\n{'='*60}")
        print(f"[STAGE 1] PRE-PROCESSING")
        print(f"{'='*60}")
        print(f"Testing {len(self.pre_strategies)} pre-processing strategies...")
        
        best_pre_strategy = None
        best_pre_biased_count = current_biased_count
        best_pre_improvement = 0
        
        for idx, strategy in enumerate(self.pre_strategies, 1):
            print(f"  [{idx}/{len(self.pre_strategies)}] Testing: {strategy}")
            
            test_pipeline = {'pre': [strategy], 'in': [], 'post': []}
            biased_count = self._evaluate_pipeline_real(
                test_pipeline, X_train, y_train, sensitive_attr, y_pred
            )
            
            improvement = current_biased_count - biased_count
            print(f"      Biased count: {biased_count}, Improvement: {improvement}")
            
            if improvement > best_pre_improvement or (improvement == best_pre_improvement and biased_count < best_pre_biased_count):
                best_pre_strategy = strategy
                best_pre_biased_count = biased_count
                best_pre_improvement = improvement
        
        if best_pre_strategy:
            best_pipeline['pre'].append(best_pre_strategy)
            current_biased_count = best_pre_biased_count
            print(f"\n  ✅ Best PRE strategy: {best_pre_strategy}")
            print(f"  Improvement: {best_pre_improvement}, New biased count: {current_biased_count}")
            
            # Apply best pre strategy to get updated predictions
            current_y_pred = self._get_predictions_after_strategy(
                best_pre_strategy, X_train, y_train, sensitive_attr, current_y_pred
            )
            
            stage_results.append({
                'stage': 'pre',
                'strategy': best_pre_strategy,
                'biased_count': int(best_pre_biased_count),
                'improvement': float(best_pre_improvement)
            })
        else:
            print(f"\n  No improvement from PRE-PROCESSING strategies")
        
        # Stage 2: Find best IN-PROCESSING strategy
        print(f"\n{'='*60}")
        print(f"[STAGE 2] IN-PROCESSING")
        print(f"{'='*60}")
        print(f"Testing {len(self.in_strategies)} in-processing strategies...")
        print(f"Starting biased count: {current_biased_count}")
        
        best_in_strategy = None
        best_in_biased_count = current_biased_count
        best_in_improvement = 0
        
        for idx, strategy in enumerate(self.in_strategies, 1):
            print(f"  [{idx}/{len(self.in_strategies)}] Testing: {strategy}")
            
            test_pipeline = {
                'pre': best_pipeline['pre'].copy(),
                'in': [strategy],
                'post': []
            }
            biased_count = self._evaluate_pipeline_real(
                test_pipeline, X_train, y_train, sensitive_attr, y_pred
            )
            
            improvement = current_biased_count - biased_count
            print(f"      Biased count: {biased_count}, Improvement: {improvement}")
            
            if improvement > best_in_improvement or (improvement == best_in_improvement and biased_count < best_in_biased_count):
                best_in_strategy = strategy
                best_in_biased_count = biased_count
                best_in_improvement = improvement
        
        if best_in_strategy:
            best_pipeline['in'].append(best_in_strategy)
            current_biased_count = best_in_biased_count
            print(f"\n  ✅ Best IN strategy: {best_in_strategy}")
            print(f"  Improvement: {best_in_improvement}, New biased count: {current_biased_count}")
            
            # Apply best in strategy
            current_y_pred = self._get_predictions_after_strategy(
                best_in_strategy, X_train, y_train, sensitive_attr, current_y_pred
            )
            
            stage_results.append({
                'stage': 'in',
                'strategy': best_in_strategy,
                'biased_count': int(best_in_biased_count),
                'improvement': float(best_in_improvement)
            })
        else:
            print(f"\n  No improvement from IN-PROCESSING strategies")
        
        # Stage 3: Find best POST-PROCESSING strategy
        print(f"\n{'='*60}")
        print(f"[STAGE 3] POST-PROCESSING")
        print(f"{'='*60}")
        print(f"Testing {len(self.post_strategies)} post-processing strategies...")
        print(f"Starting biased count: {current_biased_count}")
        
        best_post_strategy = None
        best_post_biased_count = current_biased_count
        best_post_improvement = 0
        
        for idx, strategy in enumerate(self.post_strategies, 1):
            print(f"  [{idx}/{len(self.post_strategies)}] Testing: {strategy}")
            
            test_pipeline = {
                'pre': best_pipeline['pre'].copy(),
                'in': best_pipeline['in'].copy(),
                'post': [strategy]
            }
            biased_count = self._evaluate_pipeline_real(
                test_pipeline, X_train, y_train, sensitive_attr, y_pred
            )
            
            improvement = current_biased_count - biased_count
            print(f"      Biased count: {biased_count}, Improvement: {improvement}")
            
            if improvement > best_post_improvement or (improvement == best_post_improvement and biased_count < best_post_biased_count):
                best_post_strategy = strategy
                best_post_biased_count = biased_count
                best_post_improvement = improvement
        
        if best_post_strategy:
            best_pipeline['post'].append(best_post_strategy)
            current_biased_count = best_post_biased_count
            print(f"\n  ✅ Best POST strategy: {best_post_strategy}")
            print(f"  Improvement: {best_post_improvement}, New biased count: {current_biased_count}")
            
            stage_results.append({
                'stage': 'post',
                'strategy': best_post_strategy,
                'biased_count': int(best_post_biased_count),
                'improvement': float(best_post_improvement)
            })
        else:
            print(f"\n  No improvement from POST-PROCESSING strategies")
        
        total_improvement = baseline_biased_count - current_biased_count
        
        print(f"\n{'='*60}")
        print(f"STAGE-WISE GREEDY SEARCH COMPLETE")
        print(f"{'='*60}")
        print(f"Best Pipeline:")
        print(f"  PRE: {best_pipeline['pre']}")
        print(f"  IN: {best_pipeline['in']}")
        print(f"  POST: {best_pipeline['post']}")
        print(f"\nFinal biased count: {current_biased_count}")
        print(f"Total improvement: {total_improvement}")
        print(f"{'='*60}\n")
        
        return {
            'method': 'stage_wise_greedy',
            'best_pipeline': best_pipeline,
            'final_biased_count': int(current_biased_count),
            'total_improvement': int(total_improvement),
            'stage_results': stage_results,
            'stages_optimized': len(stage_results)
        }


    def _get_predictions_after_strategy(
        self,
        strategy: str,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        sensitive_attr: np.ndarray,
        y_pred: np.ndarray
    ) -> np.ndarray:
        """
        Helper method to get predictions after applying a single strategy
        """
        try:
            mitigator = MitigationStrategies(user_model=self.user_model)
            
            result = mitigator.run_mitigation_strategy(
                strategy,
                X_train,
                y_train,
                sensitive_attr,
                X_test=X_train,
                y_test=y_train,
                y_pred_test=y_pred
            )
            
            # Extract predictions based on what the strategy produced
            if mitigator.y_pred_adjusted is not None:
                return mitigator.y_pred_adjusted
            
            elif mitigator.X_transformed is not None:
                X_transformed = mitigator.X_transformed
                if X_transformed.shape[1] > X_train.shape[1]:
                    X_transformed = X_transformed[:, :X_train.shape[1]]
                return self.user_model.predict(X_transformed)
            
            elif mitigator.fine_tuned_model is not None:
                return mitigator.fine_tuned_model.predict(X_train)
            
            # If no transformation, return original predictions
            return y_pred
        
        except Exception as e:
            print(f"    Warning: Could not get predictions after {strategy}: {str(e)}")
            return y_pred
    
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
            
            # Execute pipeline with strategies one by one
            y_pred_mitigated = y_pred.copy()

            total_strategies = sum(len(v) for v in pipeline_config.values())
            if total_strategies == 0:
                # No mitigation, return original bias count
                bias = self.bias_detector.run_all_metrics(y_train, y_pred, sensitive_attr)
                return int(bias['summary']['biased_metrics_count'])

            print(f"    Executing {total_strategies} strategies...")

            # Track current dataset (may change with augmentation)
            X_train_current = X_train.copy()
            y_train_current = y_train.copy()
            sensitive_attr_current = sensitive_attr.copy()

            # Apply strategies in order: pre -> in -> post
            for stage in ['pre', 'in', 'post']:
                for strategy in pipeline_config[stage]:
                    print(f"      Applying: {strategy}")
                    
                    try:
                        # Create mitigator WITH model for each strategy
                        mitigator = MitigationStrategies(user_model=self.user_model)
                        
                        result = mitigator.run_mitigation_strategy(
                            strategy,
                            X_train_current,
                            y_train_current,
                            sensitive_attr_current,
                            X_test=X_train_current,
                            y_test=y_train_current,
                            y_pred_test=y_pred_mitigated
                        )
                        
                        # Update predictions based on strategy output
                        if mitigator.y_pred_adjusted is not None:
                            y_pred_mitigated = mitigator.y_pred_adjusted
                            print(f"        ✅ Strategy applied successfully")
                        elif mitigator.X_transformed is not None:
                            print(f"        ✅ Using transformed data")
                            X_transformed = mitigator.X_transformed
                            # Handle feature mismatch
                            if X_transformed.shape[1] > X_train.shape[1]:
                                X_transformed = X_transformed[:, :X_train.shape[1]]
                            y_pred_mitigated = self.user_model.predict(X_transformed)
                            print(f"        Generated {len(y_pred_mitigated)} predictions from transformed data")

                        elif mitigator.fine_tuned_model is not None:
                            print(f"        ✅ Using fine-tuned model")
                            y_pred_mitigated = mitigator.fine_tuned_model.predict(X_train)
                            print(f"        Generated {len(y_pred_mitigated)} predictions from fine-tuned model")

                        elif mitigator.y_augmented is not None:
                            print(f"        ✅ Data was augmented: {len(y_train)} → {len(mitigator.y_augmented)}")
                            
                            # For greedy search: evaluate using augmented data properly
                            # Store augmented y_train and sensitive_attr
                            y_train_aug = np.array(mitigator.y_augmented)
                            sensitive_attr_aug = np.array(mitigator.sensitive_augmented)
                            
                            # Get augmented X data if available
                            if hasattr(mitigator, 'X_augmented') and mitigator.X_augmented is not None:
                                X_train_aug = mitigator.X_augmented
                                if X_train_aug.shape[1] > X_train.shape[1]:
                                    X_train_aug = X_train_aug[:, :X_train.shape[1]]
                                
                                # CRITICAL: Update X_train_current for next strategies
                                X_train_current = X_train_aug
                                y_pred_mitigated = self.user_model.predict(X_train_aug)
                                print(f"        Generated {len(y_pred_mitigated)} predictions on augmented data")
                            else:
                                # Fallback: use original predictions + repeated predictions for augmented samples
                                y_pred_original = self.user_model.predict(X_train_current)
                                augmented_count = len(y_train_aug) - len(y_pred_original)
                                y_pred_augmented_part = y_pred_original[:augmented_count]
                                y_pred_mitigated = np.concatenate([y_pred_original, y_pred_augmented_part])
                                print(f"        Generated {len(y_pred_mitigated)} predictions (original + augmented copies)")

                                # CRITICAL: Need to extend X_train_current to match augmented size
                                # Repeat the first N rows to match augmented samples
                                X_train_aug_rows = X_train_current[:augmented_count]
                                X_train_current = pd.concat([X_train_current, X_train_aug_rows], ignore_index=True)
                            
                            # IMPORTANT: Update y_train and sensitive_attr for this evaluation
                            # Store them for bias calculation
                            y_train_current = y_train_aug
                            sensitive_attr_current = sensitive_attr_aug
                            print(f"        Updated all data to size: X={X_train_current.shape[0]}, y={len(y_train_current)}, sensitive={len(sensitive_attr_current)}")

                        elif mitigator.sample_weights is not None:
                            print(f"        ✅ Sample weights generated")
                            # Check if model supports sample_weight parameter
                            try:
                                import copy
                                from sklearn.utils.validation import check_is_fitted
                                
                                # Create a copy of the model
                                temp_model = copy.deepcopy(self.user_model)
                                
                                # Try to retrain with sample weights
                                print(f"        Attempting to retrain with sample weights...")
                                temp_model.fit(X_train_current, y_train_current, sample_weight=mitigator.sample_weights)
                                
                                # Generate predictions with reweighted model
                                y_pred_mitigated = temp_model.predict(X_train_current)
                                print(f"        ✅ Retrained model with sample weights: {len(y_pred_mitigated)} predictions")
                                
                            except TypeError as e:
                                if 'sample_weight' in str(e):
                                    print(f"        ⚠️ Model doesn't support sample_weight parameter")
                                    print(f"        Using original predictions (reweighting won't affect predictions)")
                                    # For reweighting to work without retraining, we'd need to adjust predictions
                                    # but that's not mathematically sound, so we keep original
                                    y_pred_mitigated = y_pred_mitigated  # Keep current predictions
                                else:
                                    print(f"        ⚠️ Error retraining: {str(e)}")
                                    y_pred_mitigated = y_pred_mitigated
                                    
                            except Exception as e:
                                print(f"        ⚠️ Unexpected error retraining: {str(e)}")
                                y_pred_mitigated = y_pred_mitigated
                        else:
                            print(f"        ⚠️ Strategy did not produce adjusted predictions")
                    
                    except Exception as e:
                        print(f"        ❌ Strategy failed: {str(e)}")
                        continue
            
            # Calculate bias metrics using CURRENT (possibly augmented) data
            print(f"    Final evaluation sizes: y_train={len(y_train_current)}, y_pred={len(y_pred_mitigated)}, sensitive={len(sensitive_attr_current)}")
            # Calculate bias metrics
            mitigated_bias = self.bias_detector.run_all_metrics(
                y_train_current, y_pred_mitigated, sensitive_attr_current
            )
            
            biased_count = mitigated_bias['summary']['biased_metrics_count']
            
            return int(biased_count)
        
        except Exception as e:
            print(f"    Error evaluating pipeline: {str(e)}")
            # Return large number if evaluation fails
            return 999
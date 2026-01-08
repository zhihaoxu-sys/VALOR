"""
Strategy search module.
Implements a two-stage search: BOHB global search + greedy local refinement.
"""

import numpy as np
import optuna
from typing import List, Dict, Tuple, Optional, Any, Union
import json
import time
from dataclasses import dataclass

from model_config import ModelConfig
from onnx_info_extractor import LayerInfo
from strategy_generator import OptimizationStrategy, StrategyType, RVVAwareStrategyGenerator
from mse_evaluator import MSEAccuracyEstimator


@dataclass
class SearchResult:
    """Search result data structure."""
    strategies: List[OptimizationStrategy]
    accuracy_loss: float
    estimated_latency_improvement: float
    search_time_seconds: float
    total_evaluations: int


class BudgetAllocator:
    """Budget allocator."""
    
    def __init__(self, global_threshold: float):
        self.global_threshold = global_threshold
        self.layer_budgets = {}
        self.remaining_pool = 0.0
        self.locked_layers = set()
    
    def allocate_initial_budgets(self, layer_infos: List[LayerInfo]) -> Dict[str, float]:
        """Allocate initial budgets."""
        # 1. Total MACs
        total_mac = sum(layer.mac_count for layer in layer_infos if layer.has_weights)
        if total_mac == 0:
            return {layer.name: 0.0 for layer in layer_infos}
        
        # 2. Allocate proportionally by MACs
        initial_budgets = {}
        for layer in layer_infos:
            if layer.has_weights:
                initial_budgets[layer.name] = self.global_threshold * (layer.mac_count / total_mac)
            else:
                initial_budgets[layer.name] = 0.0
        
        # 3. Identify locked layers (first/last heuristic)
        self._identify_locked_layers(layer_infos)
        
        # 4. Reallocate locked layer budgets
        locked_budget = sum(initial_budgets[layer.name] for layer in layer_infos 
                           if layer.name in self.locked_layers)
        
        remaining_budget = self.global_threshold - locked_budget
        remaining_mac = sum(layer.mac_count for layer in layer_infos 
                           if layer.has_weights and layer.name not in self.locked_layers)
        
        final_budgets = {}
        for layer in layer_infos:
            if layer.name in self.locked_layers:
                final_budgets[layer.name] = 0.0
            elif layer.has_weights and remaining_mac > 0:
                final_budgets[layer.name] = remaining_budget * (layer.mac_count / remaining_mac)
            else:
                final_budgets[layer.name] = 0.0
        
        self.layer_budgets = final_budgets
        self.remaining_pool = 0.0
        
        return final_budgets
    
    def _identify_locked_layers(self, layer_infos: List[LayerInfo]):
        """Identify sensitive layers to lock."""
        # Heuristic: lock the first and last weighted layer
        if layer_infos:
            # First weighted layer
            for layer in layer_infos:
                if layer.has_weights:
                    self.locked_layers.add(layer.name)
                    break
            
            # Last weighted layer
            for layer in reversed(layer_infos):
                if layer.has_weights:
                    self.locked_layers.add(layer.name)
                    break
    
    def update_remaining_pool(self, layer_name: str, actual_loss: float):
        """Update the remaining budget pool."""
        allocated_budget = self.layer_budgets.get(layer_name, 0.0)
        if actual_loss < allocated_budget:
            saved_budget = allocated_budget - actual_loss
            self.remaining_pool += saved_budget
            self.layer_budgets[layer_name] = actual_loss
    
    def can_borrow_budget(self, layer_name: str, requested_budget: float) -> bool:
        """Check whether the layer can borrow from the pool."""
        current_budget = self.layer_budgets.get(layer_name, 0.0)
        return (current_budget + self.remaining_pool) >= requested_budget
    
    def borrow_budget(self, layer_name: str, requested_budget: float) -> bool:
        """Borrow budget from the pool."""
        current_budget = self.layer_budgets.get(layer_name, 0.0)
        shortfall = requested_budget - current_budget
        
        if shortfall <= self.remaining_pool:
            self.remaining_pool -= shortfall
            self.layer_budgets[layer_name] = requested_budget
            return True
        return False


class GreedyStrategySearcher:
    """Strategy searcher."""
    
    def __init__(self, config: ModelConfig, evaluator: MSEAccuracyEstimator):
        self.config = config
        self.evaluator = evaluator
        self.budget_allocator = BudgetAllocator(config.accuracy_threshold)
        
        # BOHB search params
        self.bohb_trials = 300
        self.early_stop_multiplier = 1.2
        self.min_samples = 2
        self.max_samples = 32
        self.sample_progression = [2, 4, 8, 16, 32]
        
        # Search stats
        self.total_evaluations = 0
        self.search_start_time = 0
    
    def search_optimal_strategies(self, layer_infos: List[LayerInfo], 
                                strategies_per_layer: Dict[str, List[OptimizationStrategy]]) -> SearchResult:
        """Main search workflow."""
        self.search_start_time = time.time()
        
        # Allocate budgets
        layer_budgets = self.budget_allocator.allocate_initial_budgets(layer_infos)
        
        print(f"Starting optimization with {len(layer_infos)} layers, global threshold: {self.config.accuracy_threshold}")
        print(f"Layer budgets: {layer_budgets}")
        
        # Phase A: BOHB global search
        print("Phase A: Global search with BOHB...")
        pareto_candidates = self.global_search_bohb(layer_infos, strategies_per_layer, layer_budgets)
        
        if not pareto_candidates:
            print("Warning: No feasible candidates found in global search")
            # Return original strategies
            original_strategies = [OptimizationStrategy(
                layer_name=layer.name,
                strategy_type=StrategyType.ORIGINAL,
                parameters={},
                target="none"
            ) for layer in layer_infos]
            
            return SearchResult(
                strategies=original_strategies,
                accuracy_loss=0.0,
                estimated_latency_improvement=1.0,
                search_time_seconds=time.time() - self.search_start_time,
                total_evaluations=self.total_evaluations
            )
        
        # Phase B: greedy local refinement
        print(f"Phase B: Local refinement with {len(pareto_candidates)} candidates...")
        best_result = self.local_greedy_refine(pareto_candidates, layer_infos, strategies_per_layer)
        
        search_time = time.time() - self.search_start_time
        print(f"Search completed in {search_time:.2f} seconds with {self.total_evaluations} evaluations")
        
        return SearchResult(
            strategies=best_result["strategies"],
            accuracy_loss=best_result["accuracy_loss"],
            estimated_latency_improvement=best_result["latency_improvement"],
            search_time_seconds=search_time,
            total_evaluations=self.total_evaluations
        )
    
    def global_search_bohb(self, layer_infos: List[LayerInfo], 
                          strategies_per_layer: Dict[str, List[OptimizationStrategy]],
                          layer_budgets: Dict[str, float]) -> List[Dict[str, Any]]:
        """BOHB global search."""
        
        # Prepare test data
        from onnx_info_extractor import ONNXNodeInfoExtractor
        extractor = ONNXNodeInfoExtractor(self.config)
        test_data = extractor.generate_test_data(num_samples=self.max_samples)
        
        # Create Optuna study
        study = optuna.create_study(
            direction="minimize",  # Minimize latency
            pruner=optuna.pruners.HyperbandPruner(
                min_resource=self.min_samples,
                max_resource=self.max_samples,
                reduction_factor=2
            ),
            sampler=optuna.samplers.TPESampler(n_startup_trials=20)
        )
        
        def objective(trial):
            return self._bohb_objective(trial, layer_infos, strategies_per_layer, 
                                      layer_budgets, test_data)
        
        # Run optimization
        study.optimize(objective, n_trials=self.bohb_trials, 
                      callbacks=[self._trial_callback])
        
        # Extract Pareto candidates
        pareto_candidates = self._extract_pareto_candidates(study)
        
        print(f"BOHB completed: {len(study.trials)} trials, {len(pareto_candidates)} Pareto candidates")
        
        return pareto_candidates
    
    def _bohb_objective(self, trial, layer_infos: List[LayerInfo], 
                       strategies_per_layer: Dict[str, List[OptimizationStrategy]],
                       layer_budgets: Dict[str, float], 
                       test_data: List[np.ndarray]) -> float:
        """BOHB objective."""
        try:
            # Select strategies per layer
            selected_strategies = []
            for layer in layer_infos:
                if layer.name not in strategies_per_layer:
                    continue
                
                available_strategies = strategies_per_layer[layer.name]
                if not available_strategies:
                    continue
                
                # Strategy selection parameter
                strategy_names = [f"{s.strategy_type.value}_{hash(str(s.parameters))}" 
                                for s in available_strategies]
                
                chosen_idx = trial.suggest_categorical(f"strategy_{layer.name}", 
                                                     list(range(len(available_strategies))))
                selected_strategies.append(available_strategies[chosen_idx])

            predicted_loss = self.evaluator.predict_accuracy_loss(selected_strategies, layer_infos)
            if predicted_loss > self.config.accuracy_threshold * self.early_stop_multiplier:
                raise optuna.TrialPruned()
            
            # Sample budget for this trial
            n_samples = trial.suggest_categorical("n_samples", self.sample_progression)
            current_test_data = test_data[:n_samples]
            
            # Early stopping: quick accuracy check
            if n_samples == self.min_samples:
                if predicted_loss > self.config.accuracy_threshold * self.early_stop_multiplier:
                    raise optuna.TrialPruned()
            
            # Full evaluation
            mse_loss = self._evaluate_strategies_mse(selected_strategies, current_test_data)
            estimated_latency = self._estimate_latency_improvement(selected_strategies)
            
            # Check accuracy constraint
            if mse_loss > self.config.accuracy_threshold:
                # Return penalty value; later budgets might still recover
                return 1000.0 + mse_loss
            
            # Record trial info for Pareto extraction
            trial.set_user_attr("strategies", selected_strategies)
            trial.set_user_attr("mse_loss", mse_loss)
            trial.set_user_attr("latency_improvement", estimated_latency)
            
            # Return latency objective (smaller is better)
            return 1.0 / estimated_latency  # Minimize reciprocal = maximize speedup
        
        except optuna.TrialPruned:
            raise
        except Exception as e:
            print(f"Trial failed: {e}")
            return float('inf')
    
    def _evaluate_strategies_mse(self, strategies: List[OptimizationStrategy], 
                               test_data: List[np.ndarray]) -> float:
        """Evaluate MSE loss for a strategy set."""
        try:
            # Apply strategies to the model
            modified_model = self.evaluator.apply_strategies_to_onnx(strategies)
            
            # Compute MSE
            mse = self.evaluator.evaluate_mse(self.evaluator.original_model, 
                                            modified_model, test_data)
            
            self.total_evaluations += 1
            return mse
        
        except Exception as e:
            print(f"MSE evaluation failed: {e}")
            return float('inf')
    
    def _estimate_latency_improvement(self, strategies: List[OptimizationStrategy]) -> float:
        """Estimate latency improvement from expected speedups."""
        total_speedup = 1.0
        for strategy in strategies:
            if hasattr(strategy, 'expected_speedup'):
                total_speedup *= strategy.expected_speedup
        
        return total_speedup
    
    def _trial_callback(self, study, trial):
        """Trial callback for progress reporting."""
        if trial.number % 50 == 0:
            print(f"Trial {trial.number}: Best value so far: {study.best_value}")
    
    def _extract_pareto_candidates(self, study) -> List[Dict[str, Any]]:
        """Extract Pareto candidates from the study."""
        candidates = []
        
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                user_attrs = trial.user_attrs
                if "strategies" in user_attrs and "mse_loss" in user_attrs:
                    mse_loss = user_attrs["mse_loss"]
                    latency_improvement = user_attrs["latency_improvement"]
                    
                    # Keep only candidates that meet accuracy constraints
                    if mse_loss <= self.config.accuracy_threshold:
                        candidates.append({
                            "strategies": user_attrs["strategies"],
                            "mse_loss": mse_loss,
                            "latency_improvement": latency_improvement,
                            "trial_number": trial.number
                        })
        
        # Pareto sort: prefer larger speedups and lower loss
        candidates.sort(key=lambda x: (-x["latency_improvement"], x["mse_loss"]))
        
        # Return top-K candidates to limit downstream cost
        return candidates[:10]
    
    def local_greedy_refine(self, pareto_candidates: List[Dict[str, Any]], 
                           layer_infos: List[LayerInfo],
                           strategies_per_layer: Dict[str, List[OptimizationStrategy]]) -> Dict[str, Any]:
        """Greedy local refinement."""
        
        # Start from the best candidate
        best_candidate = pareto_candidates[0]  # Already sorted by latency improvement
        current_strategies = best_candidate["strategies"].copy()
        current_mse = best_candidate["mse_loss"]
        current_latency = best_candidate["latency_improvement"]
        
        print(f"Starting local refinement from candidate with latency improvement: {current_latency:.2f}x")
        
        # Sort layers by MACs (optimize high-value layers first)
        sorted_layers = sorted([layer for layer in layer_infos if layer.has_weights], 
                             key=lambda x: x.mac_count, reverse=True)
        
        # Prepare small batch test data for quick evaluation
        from onnx_info_extractor import ONNXNodeInfoExtractor
        extractor = ONNXNodeInfoExtractor(self.config)
        quick_test_data = extractor.generate_test_data(num_samples=8)
        
        improvements_made = 0
        
        for layer in sorted_layers:
            if layer.name not in strategies_per_layer:
                continue
            
            available_strategies = strategies_per_layer[layer.name]
            current_strategy_for_layer = None
            
            # Find current strategy for the layer
            for strategy in current_strategies:
                if strategy.layer_name == layer.name:
                    current_strategy_for_layer = strategy
                    break
            
            if not current_strategy_for_layer:
                continue
            
            # Try more aggressive strategies
            better_strategies = self._get_more_aggressive_strategies(
                current_strategy_for_layer, available_strategies)
            
            for candidate_strategy in better_strategies:
                # Build a temporary strategy set
                temp_strategies = current_strategies.copy()
                for i, s in enumerate(temp_strategies):
                    if s.layer_name == layer.name:
                        temp_strategies[i] = candidate_strategy
                        break
                
                # Quick evaluation
                temp_mse = self._evaluate_strategies_mse(temp_strategies, quick_test_data)
                temp_latency = self._estimate_latency_improvement(temp_strategies)
                
                # Check for improvement (better latency, meets accuracy)
                if (temp_latency > current_latency and 
                    temp_mse <= self.config.accuracy_threshold):
                    
                    print(f"Layer {layer.name}: {current_strategy_for_layer.strategy_type.value} -> "
                          f"{candidate_strategy.strategy_type.value}, "
                          f"latency: {current_latency:.2f}x -> {temp_latency:.2f}x")
                    
                    current_strategies = temp_strategies
                    current_mse = temp_mse
                    current_latency = temp_latency
                    improvements_made += 1
                    break  # Lock in improvement and stop for this layer
        
        print(f"Local refinement completed: {improvements_made} layers improved")
        
        return {
            "strategies": current_strategies,
            "accuracy_loss": current_mse,
            "latency_improvement": current_latency
        }
    
    def _get_more_aggressive_strategies(self, current_strategy: OptimizationStrategy, 
                                      available_strategies: List[OptimizationStrategy]) -> List[OptimizationStrategy]:
        """Get strategies that are more aggressive than the current one."""
        more_aggressive = []
        
        # Aggressiveness ordering
        aggressiveness_order = {
            StrategyType.ORIGINAL: 0,
            StrategyType.WEIGHT_QUANTIZATION: 1,
            StrategyType.ACTIVATION_QUANTIZATION: 2,
            StrategyType.SPLIT_CONSTRUCTION: 3,
            StrategyType.LOW_RANK: 4,
            StrategyType.MIXED: 5
        }
        
        current_aggressiveness = aggressiveness_order.get(current_strategy.strategy_type, 0)
        
        for strategy in available_strategies:
            strategy_aggressiveness = aggressiveness_order.get(strategy.strategy_type, 0)
            
            # More aggressive strategy types
            if strategy_aggressiveness > current_aggressiveness:
                more_aggressive.append(strategy)
            
            # Same type but more aggressive parameters
            elif (strategy_aggressiveness == current_aggressiveness and 
                  self._is_more_aggressive_params(current_strategy, strategy)):
                more_aggressive.append(strategy)
        
        # Sort by expected speedup
        more_aggressive.sort(key=lambda x: getattr(x, 'expected_speedup', 1.0), reverse=True)
        
        return more_aggressive
    
    def _is_more_aggressive_params(self, current: OptimizationStrategy, 
                                 candidate: OptimizationStrategy) -> bool:
        """Check if candidate parameters are more aggressive."""
        if current.strategy_type != candidate.strategy_type:
            return False
        
        if current.strategy_type == StrategyType.WEIGHT_QUANTIZATION:
            current_bits = current.parameters.get("bits", 32)
            candidate_bits = candidate.parameters.get("bits", 32)
            return candidate_bits < current_bits
        
        elif current.strategy_type == StrategyType.LOW_RANK:
            current_rank = current.parameters.get("rank", float('inf'))
            candidate_rank = candidate.parameters.get("rank", float('inf'))
            return candidate_rank < current_rank

        elif current.strategy_type == StrategyType.SPLIT_CONSTRUCTION:
            current_rank = current.parameters.get("d_mid", float('inf'))
            candidate_rank = candidate.parameters.get("d_mid", float('inf'))
            return candidate_rank < current_rank
        
        elif current.strategy_type == StrategyType.MIXED:
            # Mixed strategies consider both rank and quantization
            current_rank = current.parameters.get("rank", float('inf'))
            current_bits = current.parameters.get("quantization_bits", 32)
            candidate_rank = candidate.parameters.get("rank", float('inf'))
            candidate_bits = candidate.parameters.get("quantization_bits", 32)
            
            return (candidate_rank < current_rank or 
                   (candidate_rank == current_rank and candidate_bits < current_bits))
        
        return False


if __name__ == "__main__":
    # Test code
    from model_config import create_default_config
    from onnx_info_extractor import ONNXNodeInfoExtractor
    from strategy_generator import RVVAwareStrategyGenerator
    from mse_evaluator import MSEAccuracyEstimator
    
    try:
        # Create config
        config = create_default_config(
            onnx_path="test_model.onnx",
            layers_json_path="test_layers.json",
            input_shape=(1, 3, 224, 224)
        )
        
        # Create components
        extractor = ONNXNodeInfoExtractor(config)
        generator = RVVAwareStrategyGenerator(config.rvv_length)
        evaluator = MSEAccuracyEstimator(config)
        searcher = GreedyStrategySearcher(config, evaluator)
        
        # Extract layer info
        layer_infos = extractor.extract_layer_info()
        
        # Generate strategies
        strategies_per_layer = {}
        for layer in layer_infos:
            strategies_per_layer[layer.name] = generator.generate_strategies(layer)
        
        print(f"Generated strategies for {len(layer_infos)} layers")
        
        # Run search (simulated)
        print("Starting strategy search...")
        # result = searcher.search_optimal_strategies(layer_infos, strategies_per_layer)
        # print(f"Search result: {result}")
        
    except Exception as e:
        print(f"Test failed: {e}")

"""
Strategy generation module.
Builds optimization strategy candidates based on RVV length and layer info.
"""

import math
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from enum import Enum
from dataclasses import dataclass

from model_config import ModelConfig
from onnx_info_extractor import LayerInfo


class StrategyType(Enum):
    """Strategy type enum."""
    ORIGINAL = "original"
    WEIGHT_QUANTIZATION = "weight_quantization"
    ACTIVATION_QUANTIZATION = "activation_quantization"
    LOW_RANK = "low_rank"
    SPLIT_CONSTRUCTION = "split_construction"
    MIXED = "mixed"


@dataclass
class OptimizationStrategy:
    """Optimization strategy data structure."""
    layer_name: str
    strategy_type: StrategyType
    parameters: Dict[str, Any]
    target: str  # "weight", "activation", "both"
    expected_speedup: float = 1.0
    
    def __str__(self):
        return f"{self.strategy_type.value}({self.target}): {self.parameters}"


class RVVAwareStrategyGenerator:
    """RVV-aware strategy generator."""
    
    def __init__(self, rvv_length: int = 128):
        self.rvv_length = rvv_length
        self.w = rvv_length // 32  # Vector elements for FP32
        
        # Quantization strategy config
        self.weight_quantization_bits = [8, 4]
        self.activation_quantization_bits = [8]
        
        # Low-rank strategy config
        self.rank_divisors = [4, 8]  # K/4, K/8
        self.min_rank = 32
        self.max_rank = 128
        self.rank_candidates = [32, 64, 128]
    
    def generate_strategies(self, layer_info: LayerInfo) -> List[OptimizationStrategy]:
        """Generate all available optimization strategies for a layer."""
        strategies = []
        
        # 1. Original strategy (baseline)
        strategies.append(OptimizationStrategy(
            layer_name=layer_info.name,
            strategy_type=StrategyType.ORIGINAL,
            parameters={},
            target="none"
        ))
        
        # 2. If no weights, return only the original strategy
        if not layer_info.has_weights or layer_info.weight_shape is None:
            return strategies
        
        # 3. Quantization strategies
        strategies.extend(self._generate_quantization_strategies(layer_info))

        # 4. Split-construction strategies (clogging nodes)
        strategies.extend(self._generate_split_strategies(layer_info))

        # 5. Low-rank strategies (certain layer types only)
        if self._supports_low_rank(layer_info):
            strategies.extend(self._generate_low_rank_strategies(layer_info))

        # 6. Mixed strategies (low-rank + quantization)
        if self._supports_low_rank(layer_info):
            strategies.extend(self._generate_mixed_strategies(layer_info))
        
        return strategies
    
    def _generate_quantization_strategies(self, layer_info: LayerInfo) -> List[OptimizationStrategy]:
        """Generate quantization strategies."""
        strategies = []
        
        # Weight quantization strategies
        for bits in self.weight_quantization_bits:
            strategies.append(OptimizationStrategy(
                layer_name=layer_info.name,
                strategy_type=StrategyType.WEIGHT_QUANTIZATION,
                parameters={
                    "bits": bits,
                    "quantization_type": "symmetric" if bits == 8 else "symmetric",
                    "per_channel": True if bits == 8 else False
                },
                target="weight",
                expected_speedup=self._estimate_quantization_speedup(bits)
            ))
        
        # Activation quantization strategies (supported ops only)
        if layer_info.op_type in ["Conv", "MatMul", "Gemm"]:
            for bits in self.activation_quantization_bits:
                strategies.append(OptimizationStrategy(
                    layer_name=layer_info.name,
                    strategy_type=StrategyType.ACTIVATION_QUANTIZATION,
                    parameters={
                        "bits": bits,
                        "quantization_type": "asymmetric",
                        "per_tensor": True
                    },
                    target="activation",
                    expected_speedup=self._estimate_quantization_speedup(bits, is_activation=True)
                ))
        
        return strategies

    def _generate_split_strategies(self, layer_info: LayerInfo) -> List[OptimizationStrategy]:
        """Generate split-construction strategies."""
        if layer_info.op_type not in ["MatMul", "Gemm"]:
            return []
        if not layer_info.weight_shape or len(layer_info.weight_shape) < 2:
            return []

        K, M = self._get_matrix_dimensions(layer_info.weight_shape, layer_info.op_type)
        eta = self.w

        if eta <= 0 or K % eta == 0:
            return []

        max_d_mid = self._calculate_d_mid_upper_bound(K, M, eta)
        max_d_mid = min(max_d_mid, K, M)
        if max_d_mid < eta:
            return []

        candidates = []
        for d_mid in range(eta, max_d_mid + 1, eta):
            if self._calculate_clogging_level(K, M, d_mid, eta) < 0:
                candidates.append(d_mid)

        strategies = []
        for d_mid in candidates:
            strategies.append(OptimizationStrategy(
                layer_name=layer_info.name,
                strategy_type=StrategyType.SPLIT_CONSTRUCTION,
                parameters={
                    "d_mid": d_mid,
                    "eta": eta
                },
                target="weight",
                expected_speedup=self._estimate_split_speedup(K, M, d_mid, eta)
            ))

        return strategies
    
    def _generate_low_rank_strategies(self, layer_info: LayerInfo) -> List[OptimizationStrategy]:
        """Generate low-rank decomposition strategies."""
        strategies = []
        weight_shape = layer_info.weight_shape
        
        # Compute candidate ranks
        valid_ranks = self._calculate_valid_ranks(weight_shape, layer_info.op_type)
        
        for rank in valid_ranks:
            strategies.append(OptimizationStrategy(
                layer_name=layer_info.name,
                strategy_type=StrategyType.LOW_RANK,
                parameters={
                    "rank": rank,
                    "decomposition_method": "svd"
                },
                target="weight",
                expected_speedup=self._estimate_low_rank_speedup(weight_shape, rank)
            ))
        
        return strategies
    
    def _generate_mixed_strategies(self, layer_info: LayerInfo) -> List[OptimizationStrategy]:
        """Generate mixed strategies (low-rank + quantization)."""
        strategies = []
        weight_shape = layer_info.weight_shape
        
        # Get valid ranks
        valid_ranks = self._calculate_valid_ranks(weight_shape, layer_info.op_type)
        
        # Mixed strategies: low-rank + weight quantization
        for rank in valid_ranks:
            for bits in self.weight_quantization_bits:
                strategies.append(OptimizationStrategy(
                    layer_name=layer_info.name,
                    strategy_type=StrategyType.MIXED,
                    parameters={
                        "rank": rank,
                        "quantization_bits": bits,
                        "quantization_type": "symmetric",
                        "per_channel": False  # Use per-tensor quantization in mixed strategies
                    },
                    target="weight",
                    expected_speedup=self._estimate_mixed_speedup(weight_shape, rank, bits)
                ))
        
        return strategies
    
    def _supports_low_rank(self, layer_info: LayerInfo) -> bool:
        """Check whether the layer supports low-rank decomposition."""
        # Supported ops for low-rank decomposition
        supported_ops = ["Conv", "MatMul", "Gemm"]
        if layer_info.op_type not in supported_ops:
            return False
        
        # Check weight shape suitability
        weight_shape = layer_info.weight_shape
        if not weight_shape or len(weight_shape) < 2:
            return False
        
        # Get matrix dimensions for RVV rules
        K, M = self._get_matrix_dimensions(weight_shape, layer_info.op_type)
        
        # Rule: if max(K, M) < 4*w or K % w == 0, prefer quantization
        if max(K, M) < 4 * self.w or K % self.w == 0:
            return False  # Prefer quantization in this case
        
        # Check if matrix is large enough to decompose
        min_size_for_decomposition = 64
        return min(K, M) >= min_size_for_decomposition
    
    def _get_matrix_dimensions(self, weight_shape: Tuple[int, ...], op_type: str) -> Tuple[int, int]:
        """Get matrix K and M dimensions."""
        if op_type == "Conv":
            # Conv weights: (out_c, in_c, h, w)
            out_c, in_c, h, w = weight_shape
            K = in_c * h * w  # Kernel length
            M = out_c
        elif op_type in ["MatMul", "Gemm"]:
            # FC weights: (out_dim, in_dim) or (in_dim, out_dim)
            # Assume (out_dim, in_dim)
            M, K = weight_shape
        else:
            # Fallback
            K, M = weight_shape[-1], weight_shape[0]
        
        return K, M

    def _calculate_d_mid_upper_bound(self, K: int, M: int, eta: int) -> int:
        """Compute d_mid upper bound that keeps CL<0."""
        if eta <= 0:
            return 0

        k_tiles = math.ceil(K / eta)
        denom = k_tiles + (M / eta)
        if denom <= 0:
            return 0

        upper = (M * k_tiles) / denom
        upper_int = max(0, math.floor(upper - 1e-9))
        return (upper_int // eta) * eta

    def _calculate_clogging_level(self, K: int, M: int, d_mid: int, eta: int) -> float:
        """Compute clogging level (I(G') - I(G))."""
        original = self._instruction_count(K, M, eta)
        split = self._split_instruction_count(K, M, d_mid, eta)
        return split - original

    def _instruction_count(self, K: int, M: int, eta: int) -> int:
        """Estimate instruction count with the tail approximation."""
        return M * math.ceil(K / eta)

    def _split_instruction_count(self, K: int, M: int, d_mid: int, eta: int) -> int:
        """Estimate instruction count for the split path."""
        return (d_mid * math.ceil(K / eta)) + (M * math.ceil(d_mid / eta))

    def _estimate_split_speedup(self, K: int, M: int, d_mid: int, eta: int) -> float:
        """Estimate speedup for split strategy."""
        original = self._instruction_count(K, M, eta)
        split = self._split_instruction_count(K, M, d_mid, eta)
        if split <= 0:
            return 1.0
        return min(original / split, 5.0)
    
    def _calculate_valid_ranks(self, weight_shape: Tuple[int, ...], op_type: str) -> List[int]:
        """Compute valid rank values."""
        K, M = self._get_matrix_dimensions(weight_shape, op_type)
        
        valid_ranks = []
        
        # Candidate ranks based on K divisors
        for divisor in self.rank_divisors:
            candidate_rank = K // divisor
            
            # Round down to predefined rank candidates
            for rank in sorted(self.rank_candidates):
                if rank <= candidate_rank:
                    if rank not in valid_ranks:
                        valid_ranks.append(rank)
                    break
        
        # Filter to valid range
        valid_ranks = [r for r in valid_ranks 
                      if self.min_rank <= r <= min(self.max_rank, min(K, M) // 2)]
        
        return sorted(valid_ranks, reverse=True)  # Descending order
    
    def _estimate_quantization_speedup(self, bits: int, is_activation: bool = False) -> float:
        """Estimate speedup for quantization."""
        if is_activation:
            # Activation quantization speedups are conservative
            return {8: 1.2, 4: 1.8}.get(bits, 1.0)
        else:
            # Weight quantization speedups
            return {8: 1.5, 4: 2.5}.get(bits, 1.0)
    
    def _estimate_low_rank_speedup(self, weight_shape: Tuple[int, ...], rank: int) -> float:
        """Estimate speedup for low-rank decomposition."""
        if len(weight_shape) < 2:
            return 1.0
        
        # Compression ratio
        original_ops = np.prod(weight_shape)
        if len(weight_shape) == 4:  # Conv layer
            out_c, in_c, h, w = weight_shape
            compressed_ops = (in_c * h * w * rank) + (rank * out_c)
        else:  # FC layer
            M, K = weight_shape
            compressed_ops = (K * rank) + (rank * M)
        
        compression_ratio = original_ops / compressed_ops
        
        # RVV alignment bonus
        rvv_bonus = 1.2 if rank % self.w == 0 else 1.0
        
        return min(compression_ratio * rvv_bonus, 5.0)  # Cap max speedup
    
    def _estimate_mixed_speedup(self, weight_shape: Tuple[int, ...], rank: int, bits: int) -> float:
        """Estimate speedup for mixed strategies."""
        low_rank_speedup = self._estimate_low_rank_speedup(weight_shape, rank)
        quant_speedup = self._estimate_quantization_speedup(bits)
        
        # Mixed speedup includes an efficiency penalty
        efficiency_factor = 0.8
        return low_rank_speedup * quant_speedup * efficiency_factor
    
    def get_strategy_compatibility_matrix(self, layer_infos: List[LayerInfo]) -> Dict[str, List[str]]:
        """Build strategy compatibility matrix for pre-filtering."""
        compatibility = {}
        
        for layer_info in layer_infos:
            strategies = self.generate_strategies(layer_info)
            strategy_names = [f"{s.strategy_type.value}_{s.target}" for s in strategies]
            compatibility[layer_info.name] = strategy_names
        
        return compatibility
    
    def filter_strategies_by_budget(self, strategies: List[OptimizationStrategy], 
                                   layer_budget: float) -> List[OptimizationStrategy]:
        """Filter strategies by per-layer budget (heuristic)."""
        # Simple heuristic: more aggressive strategies have larger risk
        strategy_risk_scores = {
            StrategyType.ORIGINAL: 0.0,
            StrategyType.WEIGHT_QUANTIZATION: 0.3,
            StrategyType.ACTIVATION_QUANTIZATION: 0.4,
            StrategyType.SPLIT_CONSTRUCTION: 0.5,
            StrategyType.LOW_RANK: 0.6,
            StrategyType.MIXED: 0.8
        }
        
        filtered = []
        for strategy in strategies:
            base_risk = strategy_risk_scores[strategy.strategy_type]
            
            # Adjust risk by parameters
            if strategy.strategy_type == StrategyType.WEIGHT_QUANTIZATION:
                bits = strategy.parameters.get("bits", 8)
                risk = base_risk * (8 / bits)
            elif strategy.strategy_type in [StrategyType.LOW_RANK, StrategyType.SPLIT_CONSTRUCTION]:
                # Smaller ranks imply higher risk (simplified)
                risk = base_risk
            else:
                risk = base_risk
            
            # Keep strategy if risk is within budget
            if risk <= layer_budget * 10:  # Scale factor to avoid over-conservatism
                filtered.append(strategy)
        
        return filtered


def create_strategy_summary(strategies: List[OptimizationStrategy]) -> Dict[str, Any]:
    """Create a strategy summary."""
    summary = {
        "total_strategies": len(strategies),
        "by_type": {},
        "by_target": {},
        "expected_speedup_range": [1.0, 1.0]
    }
    
    speedups = []
    for strategy in strategies:
        # Count by type
        type_name = strategy.strategy_type.value
        summary["by_type"][type_name] = summary["by_type"].get(type_name, 0) + 1
        
        # Count by target
        target = strategy.target
        summary["by_target"][target] = summary["by_target"].get(target, 0) + 1
        
        # Collect speedups
        speedups.append(strategy.expected_speedup)
    
    if speedups:
        summary["expected_speedup_range"] = [min(speedups), max(speedups)]
    
    return summary


if __name__ == "__main__":
    # Test code
    from onnx_info_extractor import LayerInfo
    
    # Build a test layer
    test_layer = LayerInfo(
        name="test_conv",
        onnx_node_name="Conv_1",
        op_type="Conv",
        weight_shape=(512, 256, 3, 3),
        has_weights=True,
        mac_count=1000000,
        original_latency_ms=50.0
    )
    
    # Create strategy generator
    generator = RVVAwareStrategyGenerator(rvv_length=128)
    
    # Generate strategies
    strategies = generator.generate_strategies(test_layer)
    
    print(f"Generated {len(strategies)} strategies for {test_layer.name}")
    for i, strategy in enumerate(strategies):
        print(f"{i+1}. {strategy}")
    
    # Build strategy summary
    summary = create_strategy_summary(strategies)
    print(f"\nStrategy summary: {summary}")

# optimization_strategies.py (调整版本)
import numpy as np
import torch
import torch.nn as nn
from scipy.linalg import svd
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass

@dataclass
class OptimizationStrategy:
    """优化策略的数据结构"""
    layer_name: str
    op_type: str
    dtype: str = "float32"
    rank: Optional[int] = None
    rank_ratio: Optional[float] = None
    vectorization_efficiency: float = 1.0
    estimated_speedup: float = 1.0
    estimated_accuracy_loss: float = 0.0
    strategy_type: str = "none"  # "mixed_precision", "lowrank", "mixed_lowrank", "none"

class HessianSensitivityAnalyzer:
    """基于Hessian的精度损失估计器"""
    
    def __init__(self, model: torch.nn.Module, dataloader, loss_fn, device='cpu'):
        self.model = model
        self.dataloader = dataloader
        self.loss_fn = loss_fn
        self.device = device
        self.hessian_diag = {}
        self.computed = False
        
    def compute_hessian_diagonal(self, max_batches: int = 10):
        """计算损失函数对参数的Hessian对角元素"""
        print("Computing Hessian diagonal elements...")
        self.model.eval()
        self.model.to(self.device)
        
        # 初始化Hessian对角元素存储
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.hessian_diag[name] = torch.zeros_like(param)
        
        batch_count = 0
        for batch_idx, (data, target) in enumerate(self.dataloader):
            if batch_count >= max_batches:
                break
                
            data, target = data.to(self.device), target.to(self.device)
            
            # 清零梯度
            self.model.zero_grad()
            
            # 前向传播
            output = self.model(data)
            loss = self.loss_fn(output, target)
            
            # 计算一阶梯度
            first_grads = torch.autograd.grad(
                loss, self.model.parameters(), 
                create_graph=True, retain_graph=True
            )
            
            # 计算Hessian对角元素
            for (name, param), grad in zip(self.model.named_parameters(), first_grads):
                if param.requires_grad and grad is not None:
                    # 对于每个参数元素计算二阶导数
                    grad_flat = grad.flatten()
                    for i in range(min(100, grad_flat.numel())):  # 限制计算量
                        if grad_flat[i].requires_grad:
                            second_grad = torch.autograd.grad(
                                grad_flat[i], param, retain_graph=True
                            )[0]
                            # 累积对角元素
                            self.hessian_diag[name] += second_grad.abs()
            
            batch_count += 1
            if batch_count % 5 == 0:
                print(f"Processed {batch_count}/{max_batches} batches")
        
        # 平均化
        for name in self.hessian_diag:
            self.hessian_diag[name] /= batch_count
        
        self.computed = True
        print("Hessian diagonal computation completed!")
    
    def estimate_quantization_loss(self, layer_name: str, original_weights: torch.Tensor, 
                                 quantized_weights: torch.Tensor) -> float:
        """估计量化导致的精度损失"""
        if not self.computed or layer_name not in self.hessian_diag:
            # 如果没有Hessian信息，使用简单的L2范数估计
            delta_w = quantized_weights - original_weights
            return torch.norm(delta_w).item() * 0.001  # 经验缩放因子
        
        delta_w = quantized_weights - original_weights
        H_diag = self.hessian_diag[layer_name]
        
        # ΔL ≈ Σ(1/2 * H_ii * (Δw_i)^2)
        accuracy_loss = 0.5 * torch.sum(H_diag * delta_w**2).item()
        return accuracy_loss
    
    def estimate_lowrank_loss(self, layer_name: str, original_weights: torch.Tensor, 
                            rank: int) -> float:
        """估计低秩分解导致的精度损失"""
        # 执行SVD分解
        if original_weights.dim() == 2:
            U, s, Vt = torch.svd(original_weights)
        else:
            # 对于卷积权重，先reshape
            shape = original_weights.shape
            W_2d = original_weights.view(shape[0], -1)
            U, s, Vt = torch.svd(W_2d)
        
        # 重构低秩近似
        rank = min(rank, min(U.shape[1], Vt.shape[0]))
        lowrank_weights = U[:, :rank] @ torch.diag(s[:rank]) @ Vt[:rank, :]
        
        if original_weights.dim() > 2:
            lowrank_weights = lowrank_weights.view(original_weights.shape)
        
        return self.estimate_quantization_loss(layer_name, original_weights, lowrank_weights)

class RVVAwareRankSelector:
    """基于RISC-V Vector Extension的rank选择器"""
    
    def __init__(self, vector_length: int = 128):
        self.vector_length = vector_length
        self.dtype_widths = {"int8": 8, "int16": 16, "float32": 32, "float16": 16}
    
    def get_rvv_elements(self, dtype: str) -> int:
        """获取该数据类型下的RVV向量元素数量"""
        element_width = self.dtype_widths[dtype]
        return self.vector_length // element_width
    
    def get_small_rank(self, weight_shape: Tuple[int, ...], dtype: str) -> int:
        """为情况A计算小rank值"""
        rvv_elements = self.get_rvv_elements(dtype)
        max_dim = max(weight_shape)
        
        ratio = max_dim / rvv_elements
        
        if ratio >= 0.5:      # 原始是1/2以上
            small_rank = rvv_elements // 4  # 选择1/4
        elif ratio >= 0.25:   # 原始是1/4以上  
            small_rank = rvv_elements // 8  # 选择1/8
        else:                 # 原始很小
            small_rank = max(1, rvv_elements // 16)  # 选择1/16，至少为1
        
        return small_rank
    
    def get_lowrank_candidates(self, dtype: str) -> List[int]:
        """获取低秋分解的rank候选（用于情况B）"""
        rvv_elements = self.get_rvv_elements(dtype)
        return [
            rvv_elements,        # 1x
            rvv_elements // 2,   # 1/2x
            rvv_elements // 4    # 1/4x
        ]
    
    def estimate_vectorization_efficiency(self, rank: Optional[int], dtype: str, 
                                        weight_shape: Tuple[int, ...]) -> float:
        """估计向量化效率"""
        rvv_elements = self.get_rvv_elements(dtype)
        
        if rank is None:
            # 没有低秋分解，基于原始维度
            max_dim = max(weight_shape)
            utilization = min(1.0, max_dim / rvv_elements)
        else:
            # 有低秩分解，基于rank
            utilization = min(1.0, rank / rvv_elements)
        
        # 对齐效率
        if rank is None:
            alignment_dim = max(weight_shape)
        else:
            alignment_dim = rank
            
        if alignment_dim % rvv_elements == 0:
            alignment_efficiency = 1.0
        elif alignment_dim % (rvv_elements // 2) == 0:
            alignment_efficiency = 0.9
        elif alignment_dim % (rvv_elements // 4) == 0:
            alignment_efficiency = 0.8
        else:
            alignment_efficiency = 0.7
        
        # 数据类型效率
        dtype_efficiency = {"int8": 1.0, "int16": 0.95, "float32": 0.9, "float16": 0.85}[dtype]
        
        return utilization * alignment_efficiency * dtype_efficiency

class QuantizationSimulator:
    """量化模拟器"""
    
    @staticmethod
    def simulate_int8_quantization(weights: torch.Tensor) -> torch.Tensor:
        """模拟INT8量化"""
        w_min, w_max = weights.min(), weights.max()
        scale = (w_max - w_min) / 255.0
        zero_point = -w_min / scale
        
        quantized = torch.round(weights / scale + zero_point)
        quantized = torch.clamp(quantized, 0, 255)
        
        # 反量化
        dequantized = (quantized - zero_point) * scale
        return dequantized
    
    @staticmethod
    def simulate_int16_quantization(weights: torch.Tensor) -> torch.Tensor:
        """模拟INT16量化"""
        w_min, w_max = weights.min(), weights.max()
        scale = (w_max - w_min) / 65535.0
        zero_point = -w_min / scale
        
        quantized = torch.round(weights / scale + zero_point)
        quantized = torch.clamp(quantized, 0, 65535)
        
        dequantized = (quantized - zero_point) * scale
        return dequantized
    
    @staticmethod
    def simulate_float16_quantization(weights: torch.Tensor) -> torch.Tensor:
        """模拟FP16量化"""
        return weights.half().float()

class RVVAwareStrategyGenerator:
    """基于RVV的智能策略生成器"""
    
    def __init__(self, hessian_analyzer: HessianSensitivityAnalyzer,
                 rvv_selector: RVVAwareRankSelector,
                 supported_dtypes: List[str] = None):
        self.hessian_analyzer = hessian_analyzer
        self.rvv_selector = rvv_selector
        self.supported_dtypes = supported_dtypes or ["int8", "int16", "float16", "float32"]
        self.quantizer = QuantizationSimulator()
    
    def generate_strategies_for_layer(self, layer_info: Dict, 
                                    weights: torch.Tensor) -> List[OptimizationStrategy]:
        """
        为单个层生成智能策略候选
        根据权重维度与RVV长度的关系选择不同的策略组合
        """
        layer_name = layer_info["layer_name"]
        op_type = layer_info["op_type"]
        weight_shape = tuple(weights.shape)
        
        print(f"Generating strategies for {layer_name} with shape {weight_shape}")
        
        strategies = []
        
        # 判断是情况A还是情况B
        max_dim = max(weight_shape)
        
        # 以float32为基准判断
        rvv_elements_fp32 = self.rvv_selector.get_rvv_elements("float32")
        
        if max_dim <= rvv_elements_fp32:
            # 情况A: 维度适合向量化，主要用混合精度
            strategies = self._generate_case_a_strategies(layer_name, op_type, weights, weight_shape)
            print(f"  Case A (max_dim={max_dim} <= {rvv_elements_fp32}): Generated {len(strategies)} mixed-precision strategies")
        else:
            # 情况B: 维度过大，主要用低秩分解
            strategies = self._generate_case_b_strategies(layer_name, op_type, weights, weight_shape)
            print(f"  Case B (max_dim={max_dim} > {rvv_elements_fp32}): Generated {len(strategies)} low-rank strategies")
        
        return strategies
    
    def _generate_case_a_strategies(self, layer_name: str, op_type: str,
                                  weights: torch.Tensor, weight_shape: Tuple[int, ...]) -> List[OptimizationStrategy]:
        """
        情况A: max(weight_shape) <= rvv_elements
        策略: 混合精度、混合精度+小低秩、None
        """
        strategies = []
        
        # 1. None策略 (保持原始)
        strategies.append(OptimizationStrategy(
            layer_name=layer_name,
            op_type=op_type,
            dtype="float32",
            rank=None,
            vectorization_efficiency=1.0,
            estimated_speedup=1.0,
            estimated_accuracy_loss=0.0,
            strategy_type="none"
        ))
        
        # 2. 纯混合精度策略
        for dtype in ["int8", "int16", "float16"]:
            strategy = self._create_mixed_precision_strategy(
                layer_name, op_type, weights, dtype, None
            )
            strategies.append(strategy)
        
        # 3. 混合精度 + 小低秩策略
        for dtype in ["int8", "int16", "float16"]:
            small_rank = self.rvv_selector.get_small_rank(weight_shape, dtype)
            if small_rank < min(weight_shape):  # 只有当small_rank有意义时才添加
                strategy = self._create_mixed_precision_strategy(
                    layer_name, op_type, weights, dtype, small_rank
                )
                strategies.append(strategy)
        
        return strategies
    
    def _generate_case_b_strategies(self, layer_name: str, op_type: str,
                                  weights: torch.Tensor, weight_shape: Tuple[int, ...]) -> List[OptimizationStrategy]:
        """
        情况B: max(weight_shape) > rvv_elements  
        策略: 低秩、低秩+混合精度、None
        """
        strategies = []
        
        # 1. None策略 (保持原始)
        strategies.append(OptimizationStrategy(
            layer_name=layer_name,
            op_type=op_type,
            dtype="float32",
            rank=None,
            vectorization_efficiency=1.0,
            estimated_speedup=1.0,
            estimated_accuracy_loss=0.0,
            strategy_type="none"
        ))
        
        # 2. 纯低秋分解策略 (float32)
        rank_candidates = self.rvv_selector.get_lowrank_candidates("float32")
        for rank in rank_candidates:
            if rank < min(weight_shape):  # 确保rank有效
                strategy = self._create_lowrank_strategy(
                    layer_name, op_type, weights, "float32", rank
                )
                strategies.append(strategy)
        
        # 3. 低秩 + 混合精度策略
        for dtype in ["int8", "int16"]:  # 不包括float16，避免策略过多
            rank_candidates = self.rvv_selector.get_lowrank_candidates(dtype)
            for rank in rank_candidates:
                if rank < min(weight_shape):  # 确保rank有效
                    strategy = self._create_lowrank_strategy(
                        layer_name, op_type, weights, dtype, rank
                    )
                    strategies.append(strategy)
        
        return strategies
    
    def _create_mixed_precision_strategy(self, layer_name: str, op_type: str,
                                       weights: torch.Tensor, dtype: str, 
                                       rank: Optional[int]) -> OptimizationStrategy:
        """创建混合精度策略（可选择性包含小低秋）"""
        
        # 1. 估计量化的精度损失
        if dtype == "int8":
            quantized_weights = self.quantizer.simulate_int8_quantization(weights)
        elif dtype == "int16":
            quantized_weights = self.quantizer.simulate_int16_quantization(weights)
        elif dtype == "float16":
            quantized_weights = self.quantizer.simulate_float16_quantization(weights)
        else:
            quantized_weights = weights
        
        quant_loss = self.hessian_analyzer.estimate_quantization_loss(
            layer_name, weights, quantized_weights
        )
        
        # 2. 如果有低秩分解，估计额外的精度损失
        total_loss = quant_loss
        if rank is not None:
            lowrank_loss = self.hessian_analyzer.estimate_lowrank_loss(
                layer_name, quantized_weights, rank
            )
            total_loss += lowrank_loss
        
        # 3. 估计向量化效率
        vec_efficiency = self.rvv_selector.estimate_vectorization_efficiency(
            rank, dtype, weights.shape
        )
        
        # 4. 估计加速比
        # 量化带来的加速
        dtype_speedup = {"int8": 2.5, "int16": 1.8, "float16": 1.4, "float32": 1.0}[dtype]
        
        # 低秩分解带来的加速
        if rank is not None:
            original_ops = np.prod(weights.shape)
            lowrank_ops = weights.shape[0] * rank + rank * weights.shape[1]
            compute_speedup = original_ops / lowrank_ops
        else:
            compute_speedup = 1.0
        
        total_speedup = dtype_speedup * compute_speedup * vec_efficiency
        
        # 5. 确定策略类型
        if rank is not None:
            strategy_type = "mixed_lowrank"
        else:
            strategy_type = "mixed_precision"
        
        return OptimizationStrategy(
            layer_name=layer_name,
            op_type=op_type,
            dtype=dtype,
            rank=rank,
            rank_ratio=rank / min(weights.shape) if rank else None,
            vectorization_efficiency=vec_efficiency,
            estimated_speedup=total_speedup,
            estimated_accuracy_loss=total_loss,
            strategy_type=strategy_type
        )
    
    def _create_lowrank_strategy(self, layer_name: str, op_type: str,
                               weights: torch.Tensor, dtype: str, 
                               rank: int) -> OptimizationStrategy:
        """创建低秩分解策略（可选择性包含量化）"""
        
        # 1. 估计低秩分解的精度损失
        lowrank_loss = self.hessian_analyzer.estimate_lowrank_loss(
            layer_name, weights, rank
        )
        
        # 2. 如果有量化，估计额外的精度损失
        total_loss = lowrank_loss
        if dtype != "float32":
            # 简化：假设在低秩权重上量化的损失
            quant_loss_factor = {"int8": 0.002, "int16": 0.001}[dtype]
            total_loss += quant_loss_factor * torch.norm(weights).item()
        
        # 3. 估计向量化效率
        vec_efficiency = self.rvv_selector.estimate_vectorization_efficiency(
            rank, dtype, weights.shape
        )
        
        # 4. 估计加速比
        # 低秩分解的计算加速
        original_ops = np.prod(weights.shape)
        lowrank_ops = weights.shape[0] * rank + rank * weights.shape[1]
        compute_speedup = original_ops / lowrank_ops
        
        # 量化带来的额外加速
        dtype_speedup = {"int8": 2.5, "int16": 1.8, "float32": 1.0}[dtype]
        
        total_speedup = compute_speedup * dtype_speedup * vec_efficiency
        
        # 5. 确定策略类型
        if dtype != "float32":
            strategy_type = "mixed_lowrank"
        else:
            strategy_type = "lowrank"
        
        return OptimizationStrategy(
            layer_name=layer_name,
            op_type=op_type,
            dtype=dtype,
            rank=rank,
            rank_ratio=rank / min(weights.shape),
            vectorization_efficiency=vec_efficiency,
            estimated_speedup=total_speedup,
            estimated_accuracy_loss=total_loss,
            strategy_type=strategy_type
        )

# 主要接口函数（供第三个文件调用）
def create_strategy_generator(model: torch.nn.Module = None, 
                            dataloader = None, 
                            loss_fn = None,
                            rvv_vector_length: int = 128) -> RVVAwareStrategyGenerator:
    """
    创建策略生成器的工厂函数
    
    Args:
        model: PyTorch模型（可选，用于Hessian分析）
        dataloader: 验证数据（可选，用于Hessian分析）  
        loss_fn: 损失函数（可选，用于Hessian分析）
        rvv_vector_length: RVV向量长度
    
    Returns:
        RVVAwareStrategyGenerator实例
    """
    
    # 如果提供了模型和数据，创建Hessian分析器
    if model is not None and dataloader is not None and loss_fn is not None:
        hessian_analyzer = HessianSensitivityAnalyzer(model, dataloader, loss_fn)
        hessian_analyzer.compute_hessian_diagonal()
    else:
        print("Warning: No Hessian analysis will be performed (simplified accuracy estimation)")
        hessian_analyzer = HessianSensitivityAnalyzer(None, None, None)
    
    # 创建RVV选择器
    rvv_selector = RVVAwareRankSelector(vector_length=rvv_vector_length)
    
    # 创建策略生成器
    strategy_generator = RVVAwareStrategyGenerator(
        hessian_analyzer=hessian_analyzer,
        rvv_selector=rvv_selector,
        supported_dtypes=["int8", "int16", "float16", "float32"]
    )
    
    return strategy_generator

# 测试函数
def test_strategy_generation():
    """测试策略生成功能"""
    
    # 创建测试权重
    test_weights = {
        "small_layer": torch.randn(8, 12),      # Case A: 小矩阵
        "large_layer": torch.randn(512, 1024)   # Case B: 大矩阵
    }
    
    # 创建策略生成器（不使用Hessian）
    strategy_generator = create_strategy_generator()
    
    for layer_name, weights in test_weights.items():
        layer_info = {
            "layer_name": layer_name,
            "op_type": "nn.dense",
            "latency_ms": 10.0
        }
        
        strategies = strategy_generator.generate_strategies_for_layer(layer_info, weights)
        
        print(f"\n{layer_name} (shape: {weights.shape}):")
        for i, strategy in enumerate(strategies):
            print(f"  {i+1}. {strategy.strategy_type}: {strategy.dtype}, "
                  f"rank={strategy.rank}, speedup={strategy.estimated_speedup:.2f}x")

if __name__ == "__main__":
    test_strategy_generation()
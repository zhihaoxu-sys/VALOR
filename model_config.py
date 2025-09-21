"""
模型配置管理模块
提供配置验证、默认值设定、输入规范检查等功能
"""

import os
import json
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional
import numpy as np


@dataclass
class ModelConfig:
    """模型优化配置类"""
    onnx_path: str
    layers_json_path: str
    rvv_length: int
    accuracy_threshold: float
    input_shape: Tuple[int, ...]
    output_dir: str = "./optimized_models"
    calibration_samples: int = 32
    
    def __post_init__(self):
        """配置初始化后的验证"""
        self._validate_config()
        self._create_output_dir()
    
    def _validate_config(self):
        """验证配置参数的有效性"""
        # 验证文件路径
        if not os.path.exists(self.onnx_path):
            raise FileNotFoundError(f"ONNX model not found: {self.onnx_path}")
        
        if not os.path.exists(self.layers_json_path):
            raise FileNotFoundError(f"Layers JSON not found: {self.layers_json_path}")
        
        # 验证JSON格式
        try:
            with open(self.layers_json_path, 'r') as f:
                json_data = json.load(f)
                if "layer_mappings" not in json_data:
                    raise ValueError("JSON must contain 'layer_mappings' field")
                if not json_data["layer_mappings"]:
                    raise ValueError("layer_mappings cannot be empty")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}")
        
        # 验证RVV长度
        if self.rvv_length <= 0 or self.rvv_length % 32 != 0:
            raise ValueError(f"RVV length must be positive and multiple of 32, got {self.rvv_length}")
        
        # 验证精度阈值
        if not 0 < self.accuracy_threshold <= 1.0:
            raise ValueError(f"Accuracy threshold must be in (0, 1], got {self.accuracy_threshold}")
        
        # 验证输入shape
        if not self.input_shape or len(self.input_shape) < 2:
            raise ValueError(f"Input shape must have at least 2 dimensions, got {self.input_shape}")
        
        # 验证校准样本数
        if self.calibration_samples < 2 or self.calibration_samples > 64:
            raise ValueError(f"Calibration samples must be in [2, 64], got {self.calibration_samples}")
    
    def _create_output_dir(self):
        """创建输出目录"""
        os.makedirs(self.output_dir, exist_ok=True)
    
    @property
    def rvv_fp32_elements(self) -> int:
        """计算RVV在FP32情况下的向量元素数"""
        return self.rvv_length // 32
    
    def get_optimized_model_path(self) -> str:
        """获取优化后模型的保存路径"""
        base_name = os.path.splitext(os.path.basename(self.onnx_path))[0]
        return os.path.join(self.output_dir, f"{base_name}_optimized.onnx")
    
    def get_report_path(self) -> str:
        """获取优化报告的保存路径"""
        base_name = os.path.splitext(os.path.basename(self.onnx_path))[0]
        return os.path.join(self.output_dir, f"{base_name}_optimization_report.json")


class InputSpecification:
    """输入规范检查和生成类"""
    
    def __init__(self, input_shape: Tuple[int, ...], dtype: str = "float32"):
        self.input_shape = input_shape
        self.dtype = dtype
        self._validate_specification()
    
    def _validate_specification(self):
        """验证输入规范"""
        # 检查shape有效性
        if not all(dim > 0 for dim in self.input_shape):
            raise ValueError(f"All dimensions must be positive, got {self.input_shape}")
        
        # 检查dtype有效性
        valid_dtypes = ["float32", "float16", "int64", "int32"]
        if self.dtype not in valid_dtypes:
            raise ValueError(f"Unsupported dtype {self.dtype}, must be one of {valid_dtypes}")
    
    def generate_random_input(self, batch_size: int = 1) -> np.ndarray:
        """生成符合规范的随机输入数据"""
        full_shape = (batch_size,) + self.input_shape[1:]  # 假设第一维是batch
        
        if self.dtype == "float32":
            # 生成[-1, 1]范围的随机数，模拟标准化后的图像数据
            return np.random.uniform(-1.0, 1.0, full_shape).astype(np.float32)
        elif self.dtype == "float16":
            return np.random.uniform(-1.0, 1.0, full_shape).astype(np.float16)
        elif self.dtype in ["int64", "int32"]:
            # 对于整数类型，生成[0, 255]范围（常见于未标准化的图像）
            dtype_map = {"int64": np.int64, "int32": np.int32}
            return np.random.randint(0, 256, full_shape, dtype=dtype_map[self.dtype])
        
    def validate_input_data(self, data: np.ndarray) -> bool:
        """验证输入数据是否符合规范"""
        # 检查shape兼容性
        if len(data.shape) != len(self.input_shape):
            return False
        
        # 除了batch维度外，其他维度必须完全匹配
        if data.shape[1:] != self.input_shape[1:]:
            return False
        
        # 检查dtype
        expected_dtype = getattr(np, self.dtype)
        if data.dtype != expected_dtype:
            return False
        
        return True
    
    def get_input_info(self) -> Dict[str, Any]:
        """获取输入信息摘要"""
        return {
            "shape": self.input_shape,
            "dtype": self.dtype,
            "total_elements": np.prod(self.input_shape),
            "memory_size_mb": np.prod(self.input_shape) * self._get_dtype_size() / (1024 * 1024)
        }
    
    def _get_dtype_size(self) -> int:
        """获取数据类型的字节大小"""
        size_map = {
            "float32": 4,
            "float16": 2,
            "int64": 8,
            "int32": 4
        }
        return size_map[self.dtype]


def create_default_config(onnx_path: str, layers_json_path: str, 
                         input_shape: Tuple[int, ...], rvv_length: int = 128) -> ModelConfig:
    """创建默认配置的便捷函数"""
    return ModelConfig(
        onnx_path=onnx_path,
        layers_json_path=layers_json_path,
        rvv_length=rvv_length,
        accuracy_threshold=0.01,  # 默认1%精度下降阈值
        input_shape=input_shape,
        calibration_samples=32
    )


# 配置验证辅助函数
def validate_rvv_strategy_compatibility(rvv_length: int, layer_info: Dict[str, Any]) -> bool:
    """验证RVV配置与层策略的兼容性"""
    w = rvv_length // 32  # FP32情况下的向量元素数
    
    # 这里需要layer_info包含权重shape信息
    # 具体验证逻辑将在strategy_generator中实现
    # 此处仅做基础检查
    return w > 0 and w <= 64  # 合理的向量元素数范围


if __name__ == "__main__":
    # 测试代码
    try:
        # 测试配置创建
        config = create_default_config(
            onnx_path="test_model.onnx",
            layers_json_path="test_layers.json", 
            input_shape=(1, 3, 224, 224),
            rvv_length=128
        )
        print("Config validation passed")
        
        # 测试输入规范
        input_spec = InputSpecification((1, 3, 224, 224))
        random_input = input_spec.generate_random_input(batch_size=2)
        print(f"Generated random input shape: {random_input.shape}")
        print(f"Input info: {input_spec.get_input_info()}")
        
    except Exception as e:
        print(f"Test failed: {e}")
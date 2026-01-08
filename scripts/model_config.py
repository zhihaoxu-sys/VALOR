"""
Model configuration module.
Provides config validation, defaults, and input specification checks.
"""

import os
import json
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional
import numpy as np


@dataclass
class ModelConfig:
    """Model optimization configuration."""
    onnx_path: str
    layers_json_path: str
    rvv_length: int
    accuracy_threshold: float
    input_shape: Tuple[int, ...]
    output_dir: str = "./optimized_models"
    calibration_samples: int = 32
    
    def __post_init__(self):
        """Validate config after initialization."""
        self._validate_config()
        self._create_output_dir()
    
    def _validate_config(self):
        """Validate configuration parameters."""
        # Validate file paths
        if not os.path.exists(self.onnx_path):
            raise FileNotFoundError(f"ONNX model not found: {self.onnx_path}")
        
        if not os.path.exists(self.layers_json_path):
            raise FileNotFoundError(f"Layers JSON not found: {self.layers_json_path}")
        
        # Validate JSON format
        try:
            with open(self.layers_json_path, 'r') as f:
                json_data = json.load(f)
                if "layer_mappings" not in json_data:
                    raise ValueError("JSON must contain 'layer_mappings' field")
                if not json_data["layer_mappings"]:
                    raise ValueError("layer_mappings cannot be empty")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}")
        
        # Validate RVV length
        if self.rvv_length <= 0 or self.rvv_length % 32 != 0:
            raise ValueError(f"RVV length must be positive and multiple of 32, got {self.rvv_length}")
        
        # Validate accuracy threshold
        if not 0 < self.accuracy_threshold <= 1.0:
            raise ValueError(f"Accuracy threshold must be in (0, 1], got {self.accuracy_threshold}")
        
        # Validate input shape
        if not self.input_shape or len(self.input_shape) < 2:
            raise ValueError(f"Input shape must have at least 2 dimensions, got {self.input_shape}")
        
        # Validate calibration sample count
        if self.calibration_samples < 2 or self.calibration_samples > 64:
            raise ValueError(f"Calibration samples must be in [2, 64], got {self.calibration_samples}")
    
    def _create_output_dir(self):
        """Create the output directory."""
        os.makedirs(self.output_dir, exist_ok=True)
    
    @property
    def rvv_fp32_elements(self) -> int:
        """Compute RVV vector elements for FP32."""
        return self.rvv_length // 32
    
    def get_optimized_model_path(self) -> str:
        """Get the output path for the optimized model."""
        base_name = os.path.splitext(os.path.basename(self.onnx_path))[0]
        return os.path.join(self.output_dir, f"{base_name}_optimized.onnx")
    
    def get_report_path(self) -> str:
        """Get the output path for the optimization report."""
        base_name = os.path.splitext(os.path.basename(self.onnx_path))[0]
        return os.path.join(self.output_dir, f"{base_name}_optimization_report.json")


class InputSpecification:
    """Input specification validation and generation."""
    
    def __init__(self, input_shape: Tuple[int, ...], dtype: str = "float32"):
        self.input_shape = input_shape
        self.dtype = dtype
        self._validate_specification()
    
    def _validate_specification(self):
        """Validate the input specification."""
        # Validate shape
        if not all(dim > 0 for dim in self.input_shape):
            raise ValueError(f"All dimensions must be positive, got {self.input_shape}")
        
        # Validate dtype
        valid_dtypes = ["float32", "float16", "int64", "int32"]
        if self.dtype not in valid_dtypes:
            raise ValueError(f"Unsupported dtype {self.dtype}, must be one of {valid_dtypes}")
    
    def generate_random_input(self, batch_size: int = 1) -> np.ndarray:
        """Generate random input data that matches the specification."""
        full_shape = (batch_size,) + self.input_shape[1:]  # Assume first dim is batch
        
        if self.dtype == "float32":
            # Generate [-1, 1] data to mimic normalized images
            return np.random.uniform(-1.0, 1.0, full_shape).astype(np.float32)
        elif self.dtype == "float16":
            return np.random.uniform(-1.0, 1.0, full_shape).astype(np.float16)
        elif self.dtype in ["int64", "int32"]:
            # For integer types, generate [0, 255] values (common for unnormalized images)
            dtype_map = {"int64": np.int64, "int32": np.int32}
            return np.random.randint(0, 256, full_shape, dtype=dtype_map[self.dtype])
        
    def validate_input_data(self, data: np.ndarray) -> bool:
        """Validate input data against the specification."""
        # Check shape compatibility
        if len(data.shape) != len(self.input_shape):
            return False
        
        # All non-batch dimensions must match
        if data.shape[1:] != self.input_shape[1:]:
            return False
        
        # Check dtype
        expected_dtype = getattr(np, self.dtype)
        if data.dtype != expected_dtype:
            return False
        
        return True
    
    def get_input_info(self) -> Dict[str, Any]:
        """Return an input summary."""
        return {
            "shape": self.input_shape,
            "dtype": self.dtype,
            "total_elements": np.prod(self.input_shape),
            "memory_size_mb": np.prod(self.input_shape) * self._get_dtype_size() / (1024 * 1024)
        }
    
    def _get_dtype_size(self) -> int:
        """Get dtype size in bytes."""
        size_map = {
            "float32": 4,
            "float16": 2,
            "int64": 8,
            "int32": 4
        }
        return size_map[self.dtype]


def create_default_config(onnx_path: str, layers_json_path: str, 
                         input_shape: Tuple[int, ...], rvv_length: int = 128) -> ModelConfig:
    """Convenience helper to build a default config."""
    return ModelConfig(
        onnx_path=onnx_path,
        layers_json_path=layers_json_path,
        rvv_length=rvv_length,
        accuracy_threshold=0.01,  # Default 1% accuracy drop threshold
        input_shape=input_shape,
        calibration_samples=32
    )


# Configuration validation helpers
def validate_rvv_strategy_compatibility(rvv_length: int, layer_info: Dict[str, Any]) -> bool:
    """Validate RVV configuration compatibility with a layer."""
    w = rvv_length // 32  # Vector elements for FP32
    
    # layer_info should include weight shape information.
    # Detailed checks live in strategy_generator.
    # This is a basic sanity check.
    return w > 0 and w <= 64  # Reasonable vector element range


if __name__ == "__main__":
    # Test code
    try:
        # Test config creation
        config = create_default_config(
            onnx_path="test_model.onnx",
            layers_json_path="test_layers.json", 
            input_shape=(1, 3, 224, 224),
            rvv_length=128
        )
        print("Config validation passed")
        
        # Test input specification
        input_spec = InputSpecification((1, 3, 224, 224))
        random_input = input_spec.generate_random_input(batch_size=2)
        print(f"Generated random input shape: {random_input.shape}")
        print(f"Input info: {input_spec.get_input_info()}")
        
    except Exception as e:
        print(f"Test failed: {e}")

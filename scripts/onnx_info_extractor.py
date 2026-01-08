"""
ONNX model info extraction module.
Parses JSON mappings, extracts layer info, computes MACs, and builds test data.
"""

import os
import json
import onnx
import numpy as np
import onnxruntime as ort
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass

from model_config import ModelConfig, InputSpecification


@dataclass
class LayerInfo:
    """Layer info data structure."""
    name: str
    onnx_node_name: str
    op_type: str
    weight_shape: Optional[Tuple[int, ...]]
    has_weights: bool
    mac_count: int
    original_latency_ms: float
    input_shape: Optional[Tuple[int, ...]] = None
    output_shape: Optional[Tuple[int, ...]] = None


class ONNXNodeInfoExtractor:
    """ONNX node info extractor."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = self._load_onnx_model()
        self.graph = self.model.graph
        self.json_data = self._load_json_data()
        self.input_spec = InputSpecification(config.input_shape)
        
        # Precomputed metadata
        self.initializer_names = {init.name for init in self.graph.initializer}
        self.node_name_to_node = {node.name: node for node in self.graph.node}
        self.shape_info = self._get_shape_info()
    
    def _load_onnx_model(self) -> onnx.ModelProto:
        """Load the ONNX model."""
        try:
            model = onnx.load(self.config.onnx_path)
            onnx.checker.check_model(model)
            return model
        except Exception as e:
            raise ValueError(f"Failed to load ONNX model: {e}")
    
    def _load_json_data(self) -> Dict[str, Any]:
        """Load JSON data."""
        try:
            with open(self.config.layers_json_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load JSON data: {e}")
    
    def _get_shape_info(self) -> Dict[str, Tuple[int, ...]]:
        """Get tensor shape info."""
        shape_info = {}
        
        try:
            # Try shape inference first
            inferred_model = onnx.shape_inference.infer_shapes(self.model)
            for value_info in inferred_model.graph.value_info:
                if value_info.type.tensor_type.shape.dim:
                    shape = tuple(
                        dim.dim_value if dim.dim_value > 0 else -1 
                        for dim in value_info.type.tensor_type.shape.dim
                    )
                    shape_info[value_info.name] = shape
        except Exception:
            # Fall back to forward pass if inference fails
            try:
                shape_info = self._get_shape_by_forward()
            except Exception:
                print("Warning: Could not infer shapes, MAC calculation may be inaccurate")
        
        return shape_info
    
    def _get_shape_by_forward(self) -> Dict[str, Tuple[int, ...]]:
        """Infer shapes via a forward pass."""
        shape_info = {}
        
        try:
            # Create session with profiling enabled
            sess_options = ort.SessionOptions()
            sess_options.enable_profiling = True
            
            session = ort.InferenceSession(self.config.onnx_path, sess_options)
            
            # Generate dummy input
            dummy_input = self.input_spec.generate_random_input(batch_size=1)
            input_name = session.get_inputs()[0].name
            
            # Run inference to collect intermediate outputs
            outputs = session.run(None, {input_name: dummy_input})
            
            # Extract shapes from profiling output (simplified).
            # A real implementation would be more involved.
            
        except Exception as e:
            print(f"Warning: Forward pass failed: {e}")
        
        return shape_info
    
    def extract_layer_info(self) -> List[LayerInfo]:
        """Extract layer info in the JSON order (latency-sorted)."""
        layer_infos = []
        
        for layer_key, layer_data in self.json_data["layer_mappings"].items():
            # Extract ONNX node names
            onnx_node_names = layer_data["onnx_nodes"]
            if not onnx_node_names:
                continue
            
            # Use the first node as the representative (usually only one)
            onnx_node_name = onnx_node_names[0]
            
            # Locate the node in the ONNX graph
            if onnx_node_name not in self.node_name_to_node:
                print(f"Warning: Node {onnx_node_name} not found in ONNX graph")
                continue
            
            onnx_node = self.node_name_to_node[onnx_node_name]
            
            # Check for trainable weights
            has_weights = self._has_trainable_weights(onnx_node)
            
            # Get weight shape
            weight_shape = None
            if has_weights:
                weight_shape = self._get_weight_shape(onnx_node)
            
            # Compute MACs
            mac_count = self._calculate_mac(onnx_node, weight_shape)
            
            # Get input/output shapes
            input_shape, output_shape = self._get_io_shapes(onnx_node)
            
            # Build LayerInfo
            layer_info = LayerInfo(
                name=layer_key,
                onnx_node_name=onnx_node_name,
                op_type=onnx_node.op_type,
                weight_shape=weight_shape,
                has_weights=has_weights,
                mac_count=mac_count,
                original_latency_ms=layer_data["original_latency_ms"],
                input_shape=input_shape,
                output_shape=output_shape
            )
            
            layer_infos.append(layer_info)
        
        return layer_infos
    
    def _has_trainable_weights(self, node: onnx.NodeProto) -> bool:
        """Check whether the node has trainable weights."""
        for input_name in node.input:
            if input_name in self.initializer_names:
                return True
        return False
    
    def _get_weight_shape(self, node: onnx.NodeProto) -> Optional[Tuple[int, ...]]:
        """Get the node weight shape."""
        for input_name in node.input:
            if input_name in self.initializer_names:
                # Find the weight initializer
                for init in self.graph.initializer:
                    if init.name == input_name:
                        return tuple(init.dims)
        return None
    
    def _get_io_shapes(self, node: onnx.NodeProto) -> Tuple[Optional[Tuple[int, ...]], Optional[Tuple[int, ...]]]:
        """Get input/output shapes for the node."""
        input_shape = None
        output_shape = None
        
        # Input shape
        if node.input:
            input_name = node.input[0]  # Use first input
            if input_name in self.shape_info:
                input_shape = self.shape_info[input_name]
        
        # Output shape
        if node.output:
            output_name = node.output[0]  # Use first output
            if output_name in self.shape_info:
                output_shape = self.shape_info[output_name]
        
        return input_shape, output_shape
    
    def _calculate_mac(self, node: onnx.NodeProto, weight_shape: Optional[Tuple[int, ...]]) -> int:
        """Compute MACs for a node."""
        if not weight_shape:
            return 0
        
        op_type = node.op_type
        
        if op_type == "Conv":
            return self._calculate_conv_mac(node, weight_shape)
        elif op_type in ["MatMul", "Gemm"]:
            return self._calculate_gemm_mac(weight_shape)
        elif op_type == "DepthwiseConv":
            return self._calculate_depthwise_conv_mac(node, weight_shape)
        else:
            # Approximate other ops using weight element count
            return np.prod(weight_shape)
    
    def _calculate_conv_mac(self, node: onnx.NodeProto, weight_shape: Tuple[int, ...]) -> int:
        """Compute MACs for Conv."""
        # weight_shape: (out_c, in_c, k_h, k_w)
        out_c, in_c, k_h, k_w = weight_shape
        
        # Output spatial size
        output_shape = self._get_io_shapes(node)[1]
        if output_shape and len(output_shape) >= 4:
            # Assume NCHW layout
            out_h, out_w = output_shape[2], output_shape[3]
        else:
            # If output shape is unknown, estimate from input
            input_shape = self._get_io_shapes(node)[0]
            if input_shape and len(input_shape) >= 4:
                # Simplified assumption: stride=1, padding preserves size
                out_h, out_w = input_shape[2], input_shape[3]
            else:
                # Final fallback: use 224x224
                out_h, out_w = 224, 224
        
        mac = out_c * in_c * k_h * k_w * out_h * out_w
        return mac
    
    def _calculate_gemm_mac(self, weight_shape: Tuple[int, ...]) -> int:
        """Compute MACs for GEMM/MatMul."""
        # weight_shape is typically (out_dim, in_dim) or (in_dim, out_dim)
        if len(weight_shape) == 2:
            return weight_shape[0] * weight_shape[1]
        else:
            # Fallback to total element count
            return np.prod(weight_shape)
    
    def _calculate_depthwise_conv_mac(self, node: onnx.NodeProto, weight_shape: Tuple[int, ...]) -> int:
        """Compute MACs for DepthwiseConv."""
        # DepthwiseConv weights: (out_c, 1, k_h, k_w) or (out_c, k_h, k_w)
        if len(weight_shape) >= 3:
            out_c = weight_shape[0]
            k_h = weight_shape[-2]
            k_w = weight_shape[-1]
        else:
            return np.prod(weight_shape)
        
        # Output spatial size
        output_shape = self._get_io_shapes(node)[1]
        if output_shape and len(output_shape) >= 4:
            out_h, out_w = output_shape[2], output_shape[3]
        else:
            # Fallback
            out_h, out_w = 224, 224
        
        mac = out_c * k_h * k_w * out_h * out_w
        return mac
    
    def generate_test_data(self, num_samples: int = None) -> List[np.ndarray]:
        """Generate test data for calibration."""
        if num_samples is None:
            num_samples = self.config.calibration_samples
        
        test_data = []
        for _ in range(num_samples):
            sample = self.input_spec.generate_random_input(batch_size=1)
            test_data.append(sample)
        
        return test_data
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get model summary info."""
        layer_infos = self.extract_layer_info()
        
        total_mac = sum(layer.mac_count for layer in layer_infos)
        total_latency = sum(layer.original_latency_ms for layer in layer_infos)
        
        return {
            "model_path": self.config.onnx_path,
            "total_target_layers": len(layer_infos),
            "total_mac": total_mac,
            "total_original_latency_ms": total_latency,
            "layers_with_weights": sum(1 for layer in layer_infos if layer.has_weights),
            "layers_without_weights": sum(1 for layer in layer_infos if not layer.has_weights),
            "input_shape": self.config.input_shape,
            "rvv_length": self.config.rvv_length
        }


if __name__ == "__main__":
    # Test code
    from model_config import create_default_config
    
    try:
        # Create test config
        config = create_default_config(
            onnx_path="test_model.onnx",
            layers_json_path="test_layers.json",
            input_shape=(1, 3, 224, 224)
        )
        
        # Create extractor
        extractor = ONNXNodeInfoExtractor(config)
        
        # Extract layer info
        layer_infos = extractor.extract_layer_info()
        print(f"Extracted {len(layer_infos)} layers")
        
        for layer in layer_infos:
            print(f"Layer: {layer.name}, Op: {layer.op_type}, "
                  f"MAC: {layer.mac_count}, Has weights: {layer.has_weights}")
        
        # Generate test data
        test_data = extractor.generate_test_data(num_samples=5)
        print(f"Generated {len(test_data)} test samples")
        
        # Get model summary
        summary = extractor.get_model_summary()
        print(f"Model summary: {summary}")
        
    except Exception as e:
        print(f"Test failed: {e}")

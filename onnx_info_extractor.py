"""
ONNX模型信息提取模块
负责解析JSON文件、提取目标层信息、计算MAC、生成测试数据等
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
    """层信息数据结构"""
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
    """ONNX节点信息提取器"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = self._load_onnx_model()
        self.graph = self.model.graph
        self.json_data = self._load_json_data()
        self.input_spec = InputSpecification(config.input_shape)
        
        # 预计算信息
        self.initializer_names = {init.name for init in self.graph.initializer}
        self.node_name_to_node = {node.name: node for node in self.graph.node}
        self.shape_info = self._get_shape_info()
    
    def _load_onnx_model(self) -> onnx.ModelProto:
        """加载ONNX模型"""
        try:
            model = onnx.load(self.config.onnx_path)
            onnx.checker.check_model(model)
            return model
        except Exception as e:
            raise ValueError(f"Failed to load ONNX model: {e}")
    
    def _load_json_data(self) -> Dict[str, Any]:
        """加载JSON数据"""
        try:
            with open(self.config.layers_json_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load JSON data: {e}")
    
    def _get_shape_info(self) -> Dict[str, Tuple[int, ...]]:
        """获取tensor的shape信息"""
        shape_info = {}
        
        try:
            # 尝试使用shape inference
            inferred_model = onnx.shape_inference.infer_shapes(self.model)
            for value_info in inferred_model.graph.value_info:
                if value_info.type.tensor_type.shape.dim:
                    shape = tuple(
                        dim.dim_value if dim.dim_value > 0 else -1 
                        for dim in value_info.type.tensor_type.shape.dim
                    )
                    shape_info[value_info.name] = shape
        except Exception:
            # 如果shape inference失败，使用实际forward获取
            try:
                shape_info = self._get_shape_by_forward()
            except Exception:
                print("Warning: Could not infer shapes, MAC calculation may be inaccurate")
        
        return shape_info
    
    def _get_shape_by_forward(self) -> Dict[str, Tuple[int, ...]]:
        """通过实际forward获取shape信息"""
        shape_info = {}
        
        try:
            # 创建会话，启用详细输出
            sess_options = ort.SessionOptions()
            sess_options.enable_profiling = True
            
            session = ort.InferenceSession(self.config.onnx_path, sess_options)
            
            # 生成dummy输入
            dummy_input = self.input_spec.generate_random_input(batch_size=1)
            input_name = session.get_inputs()[0].name
            
            # 运行推理获取中间结果
            outputs = session.run(None, {input_name: dummy_input})
            
            # 从profiling信息中提取shape（这里简化处理）
            # 实际实现中可能需要更复杂的shape提取逻辑
            
        except Exception as e:
            print(f"Warning: Forward pass failed: {e}")
        
        return shape_info
    
    def extract_layer_info(self) -> List[LayerInfo]:
        """提取层信息，按JSON中的顺序返回（已按latency排序）"""
        layer_infos = []
        
        for layer_key, layer_data in self.json_data["layer_mappings"].items():
            # 提取ONNX节点名称
            onnx_node_names = layer_data["onnx_nodes"]
            if not onnx_node_names:
                continue
            
            # 取第一个节点作为代表（大多数情况下只有一个）
            onnx_node_name = onnx_node_names[0]
            
            # 在ONNX图中查找对应节点
            if onnx_node_name not in self.node_name_to_node:
                print(f"Warning: Node {onnx_node_name} not found in ONNX graph")
                continue
            
            onnx_node = self.node_name_to_node[onnx_node_name]
            
            # 检查是否有可训练权重
            has_weights = self._has_trainable_weights(onnx_node)
            
            # 获取权重shape
            weight_shape = None
            if has_weights:
                weight_shape = self._get_weight_shape(onnx_node)
            
            # 计算MAC
            mac_count = self._calculate_mac(onnx_node, weight_shape)
            
            # 获取输入输出shape
            input_shape, output_shape = self._get_io_shapes(onnx_node)
            
            # 创建LayerInfo对象
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
        """检查节点是否有可训练权重"""
        for input_name in node.input:
            if input_name in self.initializer_names:
                return True
        return False
    
    def _get_weight_shape(self, node: onnx.NodeProto) -> Optional[Tuple[int, ...]]:
        """获取节点的权重shape"""
        for input_name in node.input:
            if input_name in self.initializer_names:
                # 找到权重initializer
                for init in self.graph.initializer:
                    if init.name == input_name:
                        return tuple(init.dims)
        return None
    
    def _get_io_shapes(self, node: onnx.NodeProto) -> Tuple[Optional[Tuple[int, ...]], Optional[Tuple[int, ...]]]:
        """获取节点的输入输出shape"""
        input_shape = None
        output_shape = None
        
        # 获取输入shape
        if node.input:
            input_name = node.input[0]  # 取第一个输入
            if input_name in self.shape_info:
                input_shape = self.shape_info[input_name]
        
        # 获取输出shape
        if node.output:
            output_name = node.output[0]  # 取第一个输出
            if output_name in self.shape_info:
                output_shape = self.shape_info[output_name]
        
        return input_shape, output_shape
    
    def _calculate_mac(self, node: onnx.NodeProto, weight_shape: Optional[Tuple[int, ...]]) -> int:
        """计算节点的MAC操作数"""
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
            # 其他类型暂时返回权重元素数作为近似
            return np.prod(weight_shape)
    
    def _calculate_conv_mac(self, node: onnx.NodeProto, weight_shape: Tuple[int, ...]) -> int:
        """计算Conv层的MAC"""
        # weight_shape: (out_c, in_c, k_h, k_w)
        out_c, in_c, k_h, k_w = weight_shape
        
        # 获取输出空间尺寸
        output_shape = self._get_io_shapes(node)[1]
        if output_shape and len(output_shape) >= 4:
            # 假设NCHW格式
            out_h, out_w = output_shape[2], output_shape[3]
        else:
            # 如果无法获取输出尺寸，使用输入尺寸估算
            input_shape = self._get_io_shapes(node)[0]
            if input_shape and len(input_shape) >= 4:
                # 简化假设：stride=1, padding保持尺寸
                out_h, out_w = input_shape[2], input_shape[3]
            else:
                # 最后的fallback：使用常见的224x224
                out_h, out_w = 224, 224
        
        mac = out_c * in_c * k_h * k_w * out_h * out_w
        return mac
    
    def _calculate_gemm_mac(self, weight_shape: Tuple[int, ...]) -> int:
        """计算GEMM/MatMul层的MAC"""
        # weight_shape通常是 (out_dim, in_dim) 或 (in_dim, out_dim)
        if len(weight_shape) == 2:
            return weight_shape[0] * weight_shape[1]
        else:
            # 其他情况返回总元素数
            return np.prod(weight_shape)
    
    def _calculate_depthwise_conv_mac(self, node: onnx.NodeProto, weight_shape: Tuple[int, ...]) -> int:
        """计算DepthwiseConv层的MAC"""
        # DepthwiseConv: weight_shape通常是 (out_c, 1, k_h, k_w) 或 (out_c, k_h, k_w)
        if len(weight_shape) >= 3:
            out_c = weight_shape[0]
            k_h = weight_shape[-2]
            k_w = weight_shape[-1]
        else:
            return np.prod(weight_shape)
        
        # 获取输出空间尺寸
        output_shape = self._get_io_shapes(node)[1]
        if output_shape and len(output_shape) >= 4:
            out_h, out_w = output_shape[2], output_shape[3]
        else:
            # fallback
            out_h, out_w = 224, 224
        
        mac = out_c * k_h * k_w * out_h * out_w
        return mac
    
    def generate_test_data(self, num_samples: int = None) -> List[np.ndarray]:
        """生成用于校准的测试数据"""
        if num_samples is None:
            num_samples = self.config.calibration_samples
        
        test_data = []
        for _ in range(num_samples):
            sample = self.input_spec.generate_random_input(batch_size=1)
            test_data.append(sample)
        
        return test_data
    
    def get_model_summary(self) -> Dict[str, Any]:
        """获取模型摘要信息"""
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
    # 测试代码
    from model_config import create_default_config
    
    try:
        # 创建测试配置
        config = create_default_config(
            onnx_path="test_model.onnx",
            layers_json_path="test_layers.json",
            input_shape=(1, 3, 224, 224)
        )
        
        # 创建提取器
        extractor = ONNXNodeInfoExtractor(config)
        
        # 提取层信息
        layer_infos = extractor.extract_layer_info()
        print(f"Extracted {len(layer_infos)} layers")
        
        for layer in layer_infos:
            print(f"Layer: {layer.name}, Op: {layer.op_type}, "
                  f"MAC: {layer.mac_count}, Has weights: {layer.has_weights}")
        
        # 生成测试数据
        test_data = extractor.generate_test_data(num_samples=5)
        print(f"Generated {len(test_data)} test samples")
        
        # 获取模型摘要
        summary = extractor.get_model_summary()
        print(f"Model summary: {summary}")
        
    except Exception as e:
        print(f"Test failed: {e}")
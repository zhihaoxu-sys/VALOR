# ilp_graph_optimizer.py (ONNX版本 - 基于映射关系)
import onnx
import onnx.helper as helper
import onnx.numpy_helper as numpy_helper
import numpy as np
import json
import torch
from typing import Dict, List, Tuple, Optional, Set
from scipy.optimize import milp, LinearConstraint, Bounds
import logging

# 导入原有的优化策略（不需要修改）
from optimization_strategies import (
    create_strategy_generator,
    RVVAwareStrategyGenerator,
    OptimizationStrategy
)

class ONNXGraphOptimizer:
    """ONNX图优化器，基于ONNX节点名称直接优化"""
    
    def __init__(self, onnx_model_path: str):
        """
        初始化ONNX图优化器
        
        Args:
            onnx_model_path: ONNX模型文件路径
        """
        self.model = onnx.load(onnx_model_path)
        self.graph = self.model.graph
        
        # 创建节点和权重的映射
        self.node_map = {node.name: node for node in self.graph.node}
        self.initializer_map = {init.name: init for init in self.graph.initializer}
        
        logging.info(f"Loaded ONNX model with {len(self.graph.node)} nodes, {len(self.graph.initializer)} initializers")
        
    def apply_optimization_strategies(self, onnx_node_optimizations: Dict[str, OptimizationStrategy]) -> onnx.ModelProto:
        """
        应用所有优化策略到ONNX图
        
        Args:
            onnx_node_optimizations: ONNX节点名称到优化策略的映射
            
        Returns:
            优化后的ONNX模型
        """
        logging.info(f"Applying optimizations to {len(onnx_node_optimizations)} ONNX nodes...")
        
        # 按优化类型分组处理
        for onnx_node_name, strategy in onnx_node_optimizations.items():
            logging.info(f"Applying {strategy.strategy_type} to ONNX node: {onnx_node_name}")
            
            if strategy.strategy_type == "low_rank_decomposition":
                self._apply_low_rank_decomposition(onnx_node_name, strategy)
            elif strategy.strategy_type == "channel_pruning":
                self._apply_channel_pruning(onnx_node_name, strategy)
            elif strategy.strategy_type == "quantization":
                self._apply_quantization(onnx_node_name, strategy)
            elif strategy.strategy_type == "none":
                logging.info(f"Keeping {onnx_node_name} unchanged")
            else:
                logging.warning(f"Unknown optimization type: {strategy.strategy_type}")
        
        # 清理图
        self._cleanup_graph()
        
        logging.info("ONNX graph optimization completed")
        return self.model
    
    def _apply_low_rank_decomposition(self, onnx_node_name: str, strategy: OptimizationStrategy):
        """应用低秩分解优化"""
        # 直接根据ONNX节点名称找到节点
        target_node = self._find_onnx_node_by_name(onnx_node_name)
        if not target_node:
            logging.warning(f"Could not find ONNX node: {onnx_node_name}")
            return
        
        # 获取权重
        if len(target_node.input) < 2:
            logging.warning(f"ONNX node {onnx_node_name} doesn't have weight input")
            return
            
        weight_name = target_node.input[1]
        if weight_name not in self.initializer_map:
            logging.warning(f"Could not find weight initializer for ONNX node: {onnx_node_name}")
            return
        
        weight_tensor = self.initializer_map[weight_name]
        weight_array = numpy_helper.to_array(weight_tensor)
        
        # 应用低秩分解
        rank = strategy.rank if strategy.rank else min(weight_array.shape) // 2
        
        if target_node.op_type == "Conv":
            self._decompose_conv_node(target_node, weight_array, rank)
        elif target_node.op_type in ["MatMul", "Gemm"]:
            self._decompose_matmul_node(target_node, weight_array, rank)
        else:
            logging.warning(f"Unsupported node type for low-rank decomposition: {target_node.op_type}")
        
        logging.info(f"Applied low-rank decomposition to {onnx_node_name} with rank {rank}")
    
    def _apply_channel_pruning(self, onnx_node_name: str, strategy: OptimizationStrategy):
        """应用通道剪枝优化"""
        target_node = self._find_onnx_node_by_name(onnx_node_name)
        if not target_node or len(target_node.input) < 2:
            return
        
        weight_name = target_node.input[1]
        if weight_name not in self.initializer_map:
            return
        
        weight_tensor = self.initializer_map[weight_name]
        weight_array = numpy_helper.to_array(weight_tensor)
        
        # 计算要保留的通道（基于权重重要性）
        pruning_ratio = getattr(strategy, 'pruning_ratio', 0.5)
        channels_to_keep = self._select_important_channels(weight_array, pruning_ratio)
        
        # 创建剪枝后的权重
        if target_node.op_type == "Conv":
            # 对于卷积，剪枝输出通道
            pruned_weight = weight_array[channels_to_keep, :, :, :]
        elif target_node.op_type in ["MatMul", "Gemm"]:
            # 对于全连接，剪枝输出维度
            pruned_weight = weight_array[channels_to_keep, :]
        else:
            logging.warning(f"Unsupported node type for channel pruning: {target_node.op_type}")
            return
        
        # 更新权重
        new_weight_name = f"{weight_name}_pruned"
        new_weight_tensor = numpy_helper.from_array(pruned_weight, new_weight_name)
        self.graph.initializer.append(new_weight_tensor)
        self.initializer_map[new_weight_name] = new_weight_tensor
        
        # 更新节点输入
        target_node.input[1] = new_weight_name
        
        # 处理bias（如果存在）
        if len(target_node.input) > 2:
            bias_name = target_node.input[2]
            if bias_name in self.initializer_map:
                bias_array = numpy_helper.to_array(self.initializer_map[bias_name])
                pruned_bias = bias_array[channels_to_keep]
                new_bias_name = f"{bias_name}_pruned"
                new_bias_tensor = numpy_helper.from_array(pruned_bias, new_bias_name)
                self.graph.initializer.append(new_bias_tensor)
                self.initializer_map[new_bias_name] = new_bias_tensor
                target_node.input[2] = new_bias_name
        
        logging.info(f"Applied channel pruning to {onnx_node_name}, kept {len(channels_to_keep)} channels")
    
    def _apply_quantization(self, onnx_node_name: str, strategy: OptimizationStrategy):
        """应用量化优化"""
        target_node = self._find_onnx_node_by_name(onnx_node_name)
        if not target_node:
            return
        
        # 简化的量化实现：将权重转换为指定数据类型
        if len(target_node.input) >= 2:
            weight_name = target_node.input[1]
            if weight_name in self.initializer_map:
                weight_tensor = self.initializer_map[weight_name]
                weight_array = numpy_helper.to_array(weight_tensor)
                
                # 根据目标数据类型进行量化
                if strategy.dtype == "int8":
                    # 简单的线性量化到int8
                    scale = np.max(np.abs(weight_array)) / 127.0
                    quantized_weight = np.round(weight_array / scale).astype(np.int8)
                elif strategy.dtype == "float16":
                    quantized_weight = weight_array.astype(np.float16)
                else:
                    quantized_weight = weight_array
                
                # 更新权重
                new_weight_name = f"{weight_name}_quantized"
                new_weight_tensor = numpy_helper.from_array(quantized_weight, new_weight_name)
                self.graph.initializer.append(new_weight_tensor)
                self.initializer_map[new_weight_name] = new_weight_tensor
                target_node.input[1] = new_weight_name
        
        logging.info(f"Applied {strategy.dtype} quantization to {onnx_node_name}")
    
    def _find_onnx_node_by_name(self, onnx_node_name: str) -> Optional[onnx.NodeProto]:
        """根据ONNX节点名称精确找到节点"""
        # 直接查找
        if onnx_node_name in self.node_map:
            return self.node_map[onnx_node_name]
        
        # 如果没找到，尝试模糊匹配（可能节点名称有细微差异）
        for node_name, node in self.node_map.items():
            if onnx_node_name in node_name or node_name in onnx_node_name:
                logging.info(f"Found approximate match: {node_name} for target: {onnx_node_name}")
                return node
        
        logging.warning(f"ONNX node not found: {onnx_node_name}")
        return None
    
    def _decompose_conv_node(self, conv_node: onnx.NodeProto, weight_array: np.ndarray, rank: int):
        """分解卷积节点"""
        out_channels, in_channels, kh, kw = weight_array.shape
        
        # 将卷积权重重塑为矩阵进行SVD
        weight_matrix = weight_array.reshape(out_channels, -1)
        U, s, Vt = np.linalg.svd(weight_matrix, full_matrices=False)
        
        # 截断到指定rank
        rank = min(rank, len(s))
        U_truncated = U[:, :rank]
        s_truncated = s[:rank]
        Vt_truncated = Vt[:rank, :]
        
        # 创建两个卷积层
        # 第一层：1x1卷积 in_channels -> rank
        W1 = (Vt_truncated * np.sqrt(s_truncated)[:, None]).reshape(rank, in_channels, 1, 1)
        # 第二层：kh x kw卷积 rank -> out_channels  
        W2 = (U_truncated * np.sqrt(s_truncated)).T.reshape(out_channels, rank, kh, kw)
        
        # 创建新的权重
        w1_name = f"{conv_node.name}_lowrank_1_weight"
        w2_name = f"{conv_node.name}_lowrank_2_weight"
        
        w1_tensor = numpy_helper.from_array(W1, w1_name)
        w2_tensor = numpy_helper.from_array(W2, w2_name)
        
        self.graph.initializer.extend([w1_tensor, w2_tensor])
        self.initializer_map[w1_name] = w1_tensor
        self.initializer_map[w2_name] = w2_tensor
        
        # 创建中间张量
        intermediate_name = f"{conv_node.name}_intermediate"
        
        # 创建第一个卷积节点
        conv1_node = helper.make_node(
            'Conv',
            inputs=[conv_node.input[0], w1_name],
            outputs=[intermediate_name],
            name=f"{conv_node.name}_lowrank_1",
            kernel_shape=[1, 1],
            pads=[0, 0, 0, 0]
        )
        
        # 创建第二个卷积节点
        conv2_attrs = {}
        for attr in conv_node.attribute:
            if attr.name in ['kernel_shape', 'pads', 'strides', 'dilations', 'group']:
                conv2_attrs[attr.name] = helper.get_attribute_value(attr)
        
        conv2_inputs = [intermediate_name, w2_name]
        if len(conv_node.input) > 2:  # 有bias
            conv2_inputs.append(conv_node.input[2])
        
        conv2_node = helper.make_node(
            'Conv',
            inputs=conv2_inputs,
            outputs=conv_node.output,
            name=f"{conv_node.name}_lowrank_2",
            **conv2_attrs
        )
        
        # 替换原节点
        node_index = list(self.graph.node).index(conv_node)
        self.graph.node.remove(conv_node)
        self.graph.node.insert(node_index, conv1_node)
        self.graph.node.insert(node_index + 1, conv2_node)
        
        # 添加中间值信息
        intermediate_vi = helper.make_tensor_value_info(
            intermediate_name, onnx.TensorProto.FLOAT, [1, rank, None, None]
        )
        self.graph.value_info.append(intermediate_vi)
    
    def _decompose_matmul_node(self, matmul_node: onnx.NodeProto, weight_array: np.ndarray, rank: int):
        """分解MatMul/Gemm节点"""
        # 对权重矩阵进行SVD分解
        U, s, Vt = np.linalg.svd(weight_array, full_matrices=False)
        
        rank = min(rank, len(s))
        U_truncated = U[:, :rank]
        s_truncated = s[:rank]
        Vt_truncated = Vt[:rank, :]
        
        # 创建两个矩阵：W = U_truncated @ diag(s_truncated) @ Vt_truncated
        W1 = Vt_truncated * np.sqrt(s_truncated)[:, None]  # [rank, in_features]
        W2 = U_truncated * np.sqrt(s_truncated)  # [out_features, rank]
        
        # 创建新的权重
        w1_name = f"{matmul_node.name}_lowrank_1_weight"
        w2_name = f"{matmul_node.name}_lowrank_2_weight"
        
        w1_tensor = numpy_helper.from_array(W1.T, w1_name)  # 转置用于MatMul
        w2_tensor = numpy_helper.from_array(W2.T, w2_name)
        
        self.graph.initializer.extend([w1_tensor, w2_tensor])
        self.initializer_map[w1_name] = w1_tensor
        self.initializer_map[w2_name] = w2_tensor
        
        # 创建中间张量
        intermediate_name = f"{matmul_node.name}_intermediate"
        
        # 创建两个MatMul节点
        matmul1_node = helper.make_node(
            'MatMul',
            inputs=[matmul_node.input[0], w1_name],
            outputs=[intermediate_name],
            name=f"{matmul_node.name}_lowrank_1"
        )
        
        matmul2_inputs = [intermediate_name, w2_name]
        matmul2_node = helper.make_node(
            'MatMul',
            inputs=matmul2_inputs,
            outputs=matmul_node.output,
            name=f"{matmul_node.name}_lowrank_2"
        )
        
        # 如果原来是Gemm且有bias，需要添加Add节点
        if matmul_node.op_type == "Gemm" and len(matmul_node.input) > 2:
            bias_name = matmul_node.input[2]
            final_output_name = f"{matmul_node.name}_final_output"
            matmul2_node.output[0] = final_output_name
            
            add_node = helper.make_node(
                'Add',
                inputs=[final_output_name, bias_name],
                outputs=matmul_node.output,
                name=f"{matmul_node.name}_add_bias"
            )
            
            # 替换原节点
            node_index = list(self.graph.node).index(matmul_node)
            self.graph.node.remove(matmul_node)
            self.graph.node.insert(node_index, matmul1_node)
            self.graph.node.insert(node_index + 1, matmul2_node)
            self.graph.node.insert(node_index + 2, add_node)
        else:
            # 替换原节点
            node_index = list(self.graph.node).index(matmul_node)
            self.graph.node.remove(matmul_node)
            self.graph.node.insert(node_index, matmul1_node)
            self.graph.node.insert(node_index + 1, matmul2_node)
    
    def _select_important_channels(self, weight_array: np.ndarray, pruning_ratio: float) -> List[int]:
        """基于权重重要性选择要保留的通道"""
        if len(weight_array.shape) == 4:  # Conv weight [out, in, h, w]
            channel_importance = np.sum(np.abs(weight_array), axis=(1, 2, 3))
        elif len(weight_array.shape) == 2:  # FC weight [out, in]
            channel_importance = np.sum(np.abs(weight_array), axis=1)
        else:
            # 默认保留前半部分
            num_channels = weight_array.shape[0]
            channels_to_keep = int(num_channels * (1 - pruning_ratio))
            return list(range(channels_to_keep))
        
        num_channels = len(channel_importance)
        channels_to_keep = int(num_channels * (1 - pruning_ratio))
        
        # 选择重要性最高的通道
        important_indices = np.argsort(channel_importance)[-channels_to_keep:]
        return sorted(important_indices.tolist())
    
    def _cleanup_graph(self):
        """清理图中未使用的initializer"""
        # 收集所有被使用的初始化器名称
        used_initializers = set()
        
        for node in self.graph.node:
            for input_name in node.input:
                if input_name in self.initializer_map:
                    used_initializers.add(input_name)
        
        # 移除未使用的initializer
        self.graph.initializer[:] = [
            init for init in self.graph.initializer 
            if init.name in used_initializers
        ]
        
        # 更新映射
        self.initializer_map = {init.name: init for init in self.graph.initializer}


class MappingBasedILPSolver:
    """基于映射关系的ILP求解器"""
    
    def __init__(self, accuracy_threshold: float = 0.02, rvv_vector_length: int = 128):
        self.accuracy_threshold = accuracy_threshold
        self.rvv_vector_length = rvv_vector_length
    
    def solve_optimization_problem(self, 
                                  layer_mapping_results: Dict,
                                  onnx_model_path: str) -> Dict[str, OptimizationStrategy]:
        """
        基于层映射结果求解ILP优化问题
        
        Args:
            layer_mapping_results: 层映射结果（来自matching工具）
            onnx_model_path: ONNX模型路径
            
        Returns:
            ONNX节点名称到最优策略的映射
        """
        print("=== Solving ILP optimization problem based on layer mapping ===")
        
        # 1. 从映射结果中提取有效的层信息
        valid_layers = self._extract_valid_layers(layer_mapping_results)
        
        if not valid_layers:
            print("No valid mapped layers found!")
            return {}
        
        # 2. 从ONNX模型提取权重
        onnx_weights = self._extract_onnx_weights(onnx_model_path, valid_layers)
        
        # 3. 创建策略生成器
        strategy_generator = create_strategy_generator(
            model=None,
            dataloader=None,
            loss_fn=None,
            rvv_vector_length=self.rvv_vector_length
        )
        
        # 4. 为每个ONNX节点生成策略候选
        onnx_node_strategies = {}
        all_variables = []
        variable_map = {}
        
        var_idx = 0
        for layer_info in valid_layers:
            onnx_nodes = layer_info['onnx_nodes']
            
            for onnx_node_name in onnx_nodes:
                if onnx_node_name in onnx_weights:
                    weights = onnx_weights[onnx_node_name]
                    
                    # 使用原始layer信息生成策略
                    strategies = strategy_generator.generate_strategies_for_layer(
                        layer_info['original_layer_info'], weights
                    )
                    
                    if strategies:
                        onnx_node_strategies[onnx_node_name] = strategies
                        
                        for i, strategy in enumerate(strategies):
                            variable_map[var_idx] = (onnx_node_name, i, strategy)
                            all_variables.append(var_idx)
                            var_idx += 1
        
        if not onnx_node_strategies:
            print("No valid strategies generated for ONNX nodes!")
            return {}
        
        print(f"Generated {len(all_variables)} decision variables for {len(onnx_node_strategies)} ONNX nodes")
        
        # 5. 构建和求解ILP问题
        c, A_eq, b_eq, A_ub, b_ub, bounds = self._build_ilp_problem(
            onnx_node_strategies, variable_map, valid_layers
        )
        result = self._solve_milp(c, A_eq, b_eq, A_ub, b_ub, bounds)
        
        # 6. 解析结果
        optimal_strategies = self._parse_solution(result, variable_map)
        
        return optimal_strategies
    
    def _extract_valid_layers(self, layer_mapping_results: Dict) -> List[Dict]:
        """从映射结果中提取有效的层信息"""
        valid_layers = []
        
        # layer_mapping_results的格式：
        # {
        #   "layer_mappings": {
        #     "tvm_layer_name": {
        #       "onnx_nodes": ["node1", "node2"],
        #       "similarity_score": 0.8,
        #       "original_info": {...}
        #     }
        #   }
        # }
        
        layer_mappings = layer_mapping_results.get("layer_mappings", {})
        
        for tvm_layer_name, mapping_info in layer_mappings.items():
            onnx_nodes = mapping_info.get("onnx_nodes", [])
            
            if onnx_nodes:  # 有成功映射的ONNX节点
                layer_info = {
                    'tvm_layer_name': tvm_layer_name,
                    'onnx_nodes': onnx_nodes,
                    'similarity_score': mapping_info.get("similarity_score", 0.0),
                    'original_layer_info': mapping_info.get("original_info", {})
                }
                
                # 补充缺失的信息
                if 'layer_name' not in layer_info['original_layer_info']:
                    layer_info['original_layer_info']['layer_name'] = tvm_layer_name
                
                valid_layers.append(layer_info)
        
        print(f"Extracted {len(valid_layers)} valid mapped layers")
        return valid_layers
    
    def _extract_onnx_weights(self, onnx_model_path: str, valid_layers: List[Dict]) -> Dict[str, torch.Tensor]:
        """从ONNX模型提取权重"""
        model = onnx.load(onnx_model_path)
        initializer_map = {init.name: init for init in model.graph.initializer}
        node_map = {node.name: node for node in model.graph.node}
        
        weights = {}
        
        for layer_info in valid_layers:
            onnx_nodes = layer_info['onnx_nodes']
            
            for onnx_node_name in onnx_nodes:
                if onnx_node_name in node_map:
                    node = node_map[onnx_node_name]
                    
                    # 查找该节点的权重
                    if len(node.input) >= 2:
                        weight_name = node.input[1]
                        
                        if weight_name in initializer_map:
                            weight_array = numpy_helper.to_array(initializer_map[weight_name])
                            weights[onnx_node_name] = torch.from_numpy(weight_array)
                            continue
                
                # 如果没找到权重，使用占位符
                original_info = layer_info['original_layer_info']
                if "weight_shape" in original_info and original_info["weight_shape"]:
                    shape = original_info["weight_shape"]
                    weights[onnx_node_name] = torch.randn(*shape)
                else:
                    # 默认权重
                    weights[onnx_node_name] = torch.randn(64, 64)
        
        print(f"Extracted weights for {len(weights)} ONNX nodes")
        return weights
    
    def _build_ilp_problem(self, onnx_node_strategies: Dict, variable_map: Dict, valid_layers: List[Dict]):
        """构建ILP问题"""
        n_vars = len(variable_map)
        onnx_node_names = list(onnx_node_strategies.keys())
        
        # 构建延迟映射
        latency_map = {}
        for layer_info in valid_layers:
            original_latency = layer_info['original_layer_info'].get('latency_ms', 1.0)
            onnx_nodes = layer_info['onnx_nodes']
            
            # 平均分配延迟到每个ONNX节点
            latency_per_node = original_latency / len(onnx_nodes) if onnx_nodes else original_latency
            
            for onnx_node_name in onnx_nodes:
                latency_map[onnx_node_name] = latency_per_node
        
        # 目标函数：最小化总延迟
        c = np.zeros(n_vars)
        
        for var_idx, (onnx_node_name, strategy_idx, strategy) in variable_map.items():
            baseline_latency = latency_map.get(onnx_node_name, 1.0)
            if strategy.estimated_speedup > 0:
                optimized_latency = baseline_latency / strategy.estimated_speedup
            else:
                optimized_latency = baseline_latency
            c[var_idx] = optimized_latency
        
        # 等式约束：每个ONNX节点只能选择一个策略
        n_nodes = len(onnx_node_names)
        A_eq = np.zeros((n_nodes, n_vars))
        b_eq = np.ones(n_nodes)
        
        for node_idx, onnx_node_name in enumerate(onnx_node_names):
            for var_idx, (var_node_name, strategy_idx, strategy) in variable_map.items():
                if var_node_name == onnx_node_name:
                    A_eq[node_idx, var_idx] = 1
        
        # 不等式约束：总精度损失不超过阈值
        A_ub = np.zeros((1, n_vars))
        b_ub = np.array([self.accuracy_threshold])
        
        for var_idx, (onnx_node_name, strategy_idx, strategy) in variable_map.items():
            A_ub[0, var_idx] = strategy.estimated_accuracy_loss
        
        # 变量边界
        bounds = Bounds(lb=0, ub=1)
        
        return c, A_eq, b_eq, A_ub, b_ub, bounds
    
    def _solve_milp(self, c, A_eq, b_eq, A_ub, b_ub, bounds):
        """求解MILP"""
        constraints = []
        if A_eq is not None and A_eq.size > 0:
            constraints.append(LinearConstraint(A_eq, b_eq, b_eq))
        if A_ub is not None and A_ub.size > 0:
            constraints.append(LinearConstraint(A_ub, -np.inf, b_ub))
        
        integrality = np.ones(len(c), dtype=int)
        
        try:
            result = milp(
                c=c,
                constraints=constraints,
                bounds=bounds,
                integrality=integrality,
                options={'disp': False, 'time_limit': 300}
            )
            
            if result.success:
                print(f"✓ ILP solved successfully! Optimal value: {result.fun:.4f}")
            else:
                print(f"✗ ILP solving failed: {result.message}")
                
        except Exception as e:
            print(f"✗ MILP solver error: {e}")
            class FailedResult:
                def __init__(self):
                    self.success = False
                    self.x = None
            result = FailedResult()
        
        return result
    
    def _parse_solution(self, result, variable_map: Dict) -> Dict[str, OptimizationStrategy]:
        """解析ILP结果"""
        if not result.success or result.x is None:
            # 返回所有ONNX节点的"不变"策略
            strategies = {}
            processed_nodes = set()
            for var_idx, (onnx_node_name, strategy_idx, strategy) in variable_map.items():
                if onnx_node_name not in processed_nodes and strategy.strategy_type == "none":
                    strategies[onnx_node_name] = strategy
                    processed_nodes.add(onnx_node_name)
            return strategies
        
        optimal_strategies = {}
        for var_idx, value in enumerate(result.x):
            if value > 0.5:  # 选中的策略
                onnx_node_name, strategy_idx, strategy = variable_map[var_idx]
                optimal_strategies[onnx_node_name] = strategy
                print(f"✓ {onnx_node_name}: {strategy.strategy_type} (speedup: {strategy.estimated_speedup:.2f}x)")
        
        return optimal_strategies


def optimize_onnx_with_mapping_based_ilp(onnx_model_path: str,
                                        layer_mapping_results_path: str,
                                        output_path: str,
                                        accuracy_threshold: float = 0.02,
                                        rvv_vector_length: int = 128) -> str:
    """
    基于层映射结果使用ILP优化ONNX模型的主函数
    
    Args:
        onnx_model_path: 原始ONNX模型路径
        layer_mapping_results_path: 层映射结果JSON文件路径
        output_path: 优化后模型保存路径
        accuracy_threshold: 精度损失阈值
        rvv_vector_length: RVV向量长度
        
    Returns:
        优化后的模型路径
    """
    logging.basicConfig(level=logging.INFO)
    
    print("=== ONNX Model Optimization with Mapping-based ILP ===")
    
    # 1. 读取层映射结果
    print(f"1. Loading layer mapping results from {layer_mapping_results_path}...")
    try:
        with open(layer_mapping_results_path, 'r') as f:
            layer_mapping_results = json.load(f)
        
        total_mappings = len(layer_mapping_results.get("layer_mappings", {}))
        print(f"✓ Loaded {total_mappings} layer mappings")
    except FileNotFoundError:
        print(f"✗ Error: {layer_mapping_results_path} not found!")
        return None
    
    # 2. 求解ILP优化问题
    print("\n2. Solving mapping-based ILP optimization problem...")
    solver = MappingBasedILPSolver(accuracy_threshold, rvv_vector_length)
    onnx_optimization_strategies = solver.solve_optimization_problem(
        layer_mapping_results, onnx_model_path
    )
    
    if not onnx_optimization_strategies:
        print("No optimization strategies found, copying original model")
        import shutil
        shutil.copy2(onnx_model_path, output_path)
        return output_path
    
    # 3. 应用优化到ONNX图
    print(f"\n3. Applying optimizations to ONNX model...")
    optimizer = ONNXGraphOptimizer(onnx_model_path)
    optimized_model = optimizer.apply_optimization_strategies(onnx_optimization_strategies)
    
    # 4. 保存优化后的模型
    print(f"\n4. Saving optimized model to {output_path}...")
    onnx.save(optimized_model, output_path)
    
    # 5. 验证模型
    try:
        onnx.checker.check_model(output_path)
        print("✓ Optimized model validation passed")
    except Exception as e:
        print(f"⚠ Model validation warning: {e}")
    
    print(f"\n✓ Optimization completed! Optimized model saved to: {output_path}")
    return output_path


# 使用示例
if __name__ == "__main__":
    # 示例使用
    optimized_path = optimize_onnx_with_mapping_based_ilp(
        onnx_model_path="/doc2/zhzh/models_tvm/yolov4.onnx",
        layer_mapping_results_path="simplified_matching_results.json",  # 来自mapping工具的结果
        output_path="/doc2/zhzh/models_tvm/yolov4_optimized_model.onnx",
        accuracy_threshold=0.02,
        rvv_vector_length=128
    )
    
    if optimized_path:
        print(f"Optimization successful! Output: {optimized_path}")
    else:
        print("Optimization failed!")
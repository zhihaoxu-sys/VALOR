# simplified_tvm_onnx_matcher.py
import json
import re
from typing import Dict, List, Tuple, Optional
import logging

class SimplifiedTVMONNXMatcher:
    """简化的TVM到ONNX匹配器，基于核心模式匹配"""
    
    def __init__(self, top_k_layers_path: str, tvm_onnx_mapping_path: str):
        """
        初始化匹配器
        
        Args:
            top_k_layers_path: top_k_layers.json文件路径
            tvm_onnx_mapping_path: tvm_onnx_mapping.json文件路径
        """
        self.top_k_layers = self._load_json(top_k_layers_path)
        self.tvm_onnx_mapping = self._load_json(tvm_onnx_mapping_path)
        
        # 预处理映射数据
        self.processed_mapping = self._preprocess_mapping()
        
        print(f"Loaded {len(self.top_k_layers)} top-k layers")
        print(f"Loaded {len(self.tvm_onnx_mapping)} TVM-ONNX mappings")
        
    def _load_json(self, file_path: str) -> Dict:
        """加载JSON文件"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                # 如果是top_k_layers，可能是列表；如果是mapping，是字典
                if isinstance(data, dict) and "tvm_to_onnx_mapping" in data:
                    return data["tvm_to_onnx_mapping"]
                return data
        except Exception as e:
            logging.error(f"Failed to load {file_path}: {e}")
            return {}
    
    def _preprocess_mapping(self) -> Dict:
        """预处理映射数据，提取核心模式"""
        processed = {}
        
        for tvm_name, onnx_nodes in self.tvm_onnx_mapping.items():
            core_pattern = self._extract_core_pattern(tvm_name)
            processed[tvm_name] = {
                'onnx_nodes': onnx_nodes,
                'core_pattern': core_pattern
            }
        
        return processed
    
    def _extract_core_pattern(self, tvm_name: str) -> str:
        """
        提取TVM名称的核心模式
        
        例如：
        tvmgen_default_fused_layout_transform_layout_transform_nn_contrib_conv2d_NCHWc_expand_dims_expa_261c3e1a08fa8056__14_1
        -> nn_contrib_conv2d_NCHWc_14_1
        
        tvmgen_default_fused_nn_contrib_conv2d_NCHWc_add_14_1
        -> nn_contrib_conv2d_NCHWc_add_14_1
        """
        
        # 1. 移除前缀 tvmgen_default_fused_
        pattern = tvm_name.replace("tvmgen_default_fused_", "")
        
        # 2. 移除多余的layout_transform前缀
        pattern = re.sub(r'^(?:layout_transform_)*', '', pattern)
        
        # 3. 找到主要算子开始的位置（nn_contrib_conv2d, nn_dense, add等）
        main_op_match = re.search(r'(nn_contrib_conv2d_NCHWc|nn_conv2d|nn_dense|add|multiply|exp|log|tanh)', pattern)
        
        if main_op_match:
            # 从主要算子开始截取
            start_pos = main_op_match.start()
            core_part = pattern[start_pos:]
            
            # 4. 移除哈希值部分 (如 _261c3e1a08fa8056__)
            core_part = re.sub(r'_[a-f0-9]{16}__', '_', core_part)
            
            # 5. 移除中间的expand_dims_expa等
            core_part = re.sub(r'_expand_dims_expa_', '_', core_part)
            core_part = re.sub(r'_layout_transform_', '_', core_part)
            
            # 6. 清理多余的下划线
            core_part = re.sub(r'_+', '_', core_part)
            core_part = core_part.strip('_')
            
            return core_part
        
        # 如果没找到主要算子，返回清理后的整个字符串
        pattern = re.sub(r'_[a-f0-9]{16}__', '_', pattern)
        pattern = re.sub(r'_+', '_', pattern)
        return pattern.strip('_')
    
    def find_best_match(self, target_layer_name: str) -> Optional[Tuple[str, List[str], float]]:
        """
        为目标层找到最佳匹配
        
        Args:
            target_layer_name: 目标TVM层名称
            
        Returns:
            (匹配的TVM名称, ONNX节点列表, 相似度分数) 或 None
        """
        target_pattern = self._extract_core_pattern(target_layer_name)
        
        print(f"  Target pattern: {target_pattern}")
        
        best_match = None
        best_score = 0.0
        
        for tvm_name, data in self.processed_mapping.items():
            candidate_pattern = data['core_pattern']
            
            # 计算匹配分数
            score = self._calculate_pattern_similarity(target_pattern, candidate_pattern)
            
            if score > best_score:
                best_score = score
                best_match = (tvm_name, data['onnx_nodes'], score)
        
        # 如果最佳分数大于阈值，返回匹配结果
        if best_score > 0.6:  # 降低阈值
            return best_match
        
        return None
    
    def _calculate_pattern_similarity(self, target: str, candidate: str) -> float:
        """计算两个核心模式的相似度"""
        
        # 1. 完全匹配
        if target == candidate:
            return 1.0
        
        # 2. 提取关键组件进行比较
        target_components = self._extract_key_components(target)
        candidate_components = self._extract_key_components(candidate)
        
        # 3. 计算组件匹配度
        scores = []
        
        # 主算子匹配（权重最高）
        if target_components['main_op'] == candidate_components['main_op']:
            scores.append(0.5)  # 50%权重
        else:
            scores.append(0.0)
        
        # 后缀数字匹配
        if target_components['suffix'] == candidate_components['suffix']:
            scores.append(0.3)  # 30%权重
        else:
            # 部分数字匹配
            target_nums = set(target_components['suffix'].split('_'))
            candidate_nums = set(candidate_components['suffix'].split('_'))
            common_nums = len(target_nums & candidate_nums)
            total_nums = len(target_nums | candidate_nums)
            if total_nums > 0:
                scores.append(0.3 * common_nums / total_nums)
            else:
                scores.append(0.0)
        
        # 其他算子匹配
        target_other = set(target_components['other_ops'])
        candidate_other = set(candidate_components['other_ops'])
        if target_other or candidate_other:
            common_ops = len(target_other & candidate_other)
            total_ops = len(target_other | candidate_other)
            scores.append(0.2 * common_ops / total_ops if total_ops > 0 else 0.1)
        else:
            scores.append(0.2)  # 都没有其他算子，给满分
        
        return sum(scores)
    
    def _extract_key_components(self, pattern: str) -> Dict:
        """提取模式的关键组件"""
        components = {
            'main_op': '',
            'other_ops': [],
            'suffix': ''
        }
        
        # 提取主算子
        main_ops = ['nn_contrib_conv2d_NCHWc', 'nn_conv2d', 'nn_dense']
        for op in main_ops:
            if op in pattern:
                components['main_op'] = op
                break
        
        # 提取其他算子
        other_ops = ['add', 'multiply', 'exp', 'log', 'tanh', 'relu', 'sigmoid']
        for op in other_ops:
            if op in pattern:
                components['other_ops'].append(op)
        
        # 提取后缀数字
        suffix_match = re.search(r'(\d+(?:_\d+)*)$', pattern)
        if suffix_match:
            components['suffix'] = suffix_match.group(1)
        
        return components
    
    def match_all_layers(self) -> Dict[str, Dict]:
        """为所有top-k层找到匹配"""
        results = {}
        
        print("\n=== Matching TVM layers to ONNX nodes ===")
        
        for layer_info in self.top_k_layers:
            layer_name = layer_info['layer_name']
            
            print(f"\nProcessing: {layer_name}")
            
            # 查找最佳匹配
            best_match = self.find_best_match(layer_name)
            
            if best_match:
                matched_tvm_name, onnx_nodes, score = best_match
                
                results[layer_name] = {
                    'original_layer_info': layer_info,
                    'matched_tvm_name': matched_tvm_name,
                    'similarity_score': score,
                    'onnx_nodes': onnx_nodes
                }
                
                print(f"  ✓ Best match: {matched_tvm_name}")
                print(f"  ✓ Similarity: {score:.3f}")
                print(f"  ✓ ONNX nodes: {onnx_nodes}")
            else:
                print(f"  ✗ No good match found")
                results[layer_name] = {
                    'original_layer_info': layer_info,
                    'matched_tvm_name': None,
                    'similarity_score': 0.0,
                    'onnx_nodes': []
                }
        
        return results
    
    def debug_show_all_patterns(self):
        """调试：显示所有模式"""
        print("\n=== Debug: All Core Patterns ===")
        
        print("\nTop-k layers patterns:")
        for layer_info in self.top_k_layers:
            layer_name = layer_info['layer_name']
            pattern = self._extract_core_pattern(layer_name)
            print(f"  {layer_name} -> {pattern}")
        
        print(f"\nMapping patterns (showing first 10):")
        count = 0
        for tvm_name, data in self.processed_mapping.items():
            pattern = data['core_pattern']
            print(f"  {tvm_name} -> {pattern}")
            count += 1
            if count >= 10:
                print(f"  ... and {len(self.processed_mapping) - 10} more")
                break
    
    def save_results(self, results: Dict, output_path: str):
        """保存匹配结果"""
        # 准备保存数据
        save_data = {
            'matching_summary': {
                'total_layers': len(results),
                'successfully_matched': len([r for r in results.values() if r['matched_tvm_name']]),
                'match_rate': len([r for r in results.values() if r['matched_tvm_name']]) / len(results) if results else 0
            },
            'layer_mappings': {}
        }
        
        # 简化格式保存
        for layer_name, result in results.items():
            save_data['layer_mappings'][layer_name] = {
                'onnx_nodes': result['onnx_nodes'],
                'similarity_score': result['similarity_score'],
                'matched_tvm_name': result['matched_tvm_name'],
                'original_latency_ms': result['original_layer_info']['latency_ms'],
                'original_op_type': result['original_layer_info']['op_type']
            }
        
        with open(output_path, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"\n✓ Results saved to {output_path}")

def match_tvm_to_onnx_simplified(top_k_layers_path: str,
                                tvm_onnx_mapping_path: str,
                                output_path: str = "simplified_matching_results.json",
                                debug: bool = False) -> Dict:
    """
    简化的TVM到ONNX匹配主函数
    
    Args:
        top_k_layers_path: top_k_layers.json路径
        tvm_onnx_mapping_path: tvm_onnx_mapping.json路径
        output_path: 结果保存路径
        debug: 是否显示调试信息
        
    Returns:
        匹配结果字典
    """
    print("=== Simplified TVM Layer to ONNX Node Matching ===")
    
    # 创建匹配器
    matcher = SimplifiedTVMONNXMatcher(top_k_layers_path, tvm_onnx_mapping_path)
    
    # 调试模式：显示所有模式
    if debug:
        matcher.debug_show_all_patterns()
    
    # 执行匹配
    results = matcher.match_all_layers()
    
    # 保存结果
    matcher.save_results(results, output_path)
    
    # 统计信息
    total_layers = len(results)
    matched_layers = len([r for r in results.values() if r['matched_tvm_name']])
    
    print(f"\n=== Final Summary ===")
    print(f"Total layers: {total_layers}")
    print(f"Successfully matched: {matched_layers}")
    print(f"Match rate: {matched_layers/total_layers*100:.1f}%")
    
    return results

# 使用示例
if __name__ == "__main__":
    # 使用简化匹配器
    results = match_tvm_to_onnx_simplified(
        top_k_layers_path="top_k_layers_yolov4.json",
        tvm_onnx_mapping_path="tvm_onnx_mapping.json",
        output_path="simplified_matching_results.json",
        debug=True  # 开启调试，看看模式提取是否正确
    )
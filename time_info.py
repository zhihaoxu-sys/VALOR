import json
import tvm

class AccurateTVMLayerInfoUpdater:
    def __init__(self, lib_path, json_path):
        self.lib = tvm.runtime.load_module(lib_path)
        with open(json_path, 'r') as f:
            self.graph_json = json.load(f)
        
        # 解析所有信息
        self.shapes = self._parse_shape_info()
        self.storage_info = self._parse_storage_info()
        self.dtypes = self._parse_dtype_info()
        
        print(f"Parsed {len(self.shapes)} shapes from graph")
        
    def _parse_shape_info(self):
        """解析图中的shape信息"""
        shapes = []
        if "attrs" in self.graph_json and "shape" in self.graph_json["attrs"]:
            shape_data = self.graph_json["attrs"]["shape"]
            if len(shape_data) >= 2 and shape_data[0] == "list_shape":
                shapes = shape_data[1]
        return shapes
    
    def _parse_storage_info(self):
        """解析存储信息"""
        storage_ids = []
        if "attrs" in self.graph_json and "storage_id" in self.graph_json["attrs"]:
            storage_data = self.graph_json["attrs"]["storage_id"]
            if len(storage_data) >= 2 and storage_data[0] == "list_int":
                storage_ids = storage_data[1]
        return storage_ids
    
    def _parse_dtype_info(self):
        """解析数据类型信息"""
        dtypes = []
        if "attrs" in self.graph_json and "dltype" in self.graph_json["attrs"]:
            dtype_data = self.graph_json["attrs"]["dltype"]
            if len(dtype_data) >= 2 and dtype_data[0] == "list_str":
                dtypes = dtype_data[1]
        return dtypes

    def update_top_k_layers_info(self, top_k_json_path="top_k_layers.json"):
        """
        读取top_k_layers.json，根据其中的layer_name获取准确信息并更新文件
        """
        # 读取现有的top_k_layers.json
        with open(top_k_json_path, 'r') as f:
            top_k_layers = json.load(f)
        
        print(f"Updating {len(top_k_layers)} layers from {top_k_json_path}")
        
        # 为每个层获取准确信息
        for layer_info in top_k_layers:
            layer_name = layer_info["layer_name"]
            print(f"\n=== Processing {layer_name} ===")
            
            # 获取准确的层信息
            accurate_info = self._get_accurate_layer_info(layer_name)
            
            if accurate_info:
                # 更新层信息
                layer_info["op_type"] = accurate_info["op_type"]
                layer_info["input_shape"] = accurate_info["input_shape"]
                layer_info["weight_shape"] = accurate_info["weight_shape"]
                layer_info["output_shape"] = accurate_info["output_shape"]
                
                print(f"✓ Updated {layer_name}:")
                print(f"  Op Type: {layer_info['op_type']}")
                print(f"  Input Shape: {layer_info['input_shape']}")
                print(f"  Weight Shape: {layer_info['weight_shape']}")
                print(f"  Output Shape: {layer_info['output_shape']}")
            else:
                print(f"✗ Failed to get accurate info for {layer_name}")
        
        # 保存更新后的文件
        with open(top_k_json_path, 'w') as f:
            json.dump(top_k_layers, f, indent=2)
        
        print(f"\n✓ Updated {top_k_json_path} with accurate layer information")
        return top_k_layers

    def _get_accurate_layer_info(self, layer_name):
        """获取单个层的准确信息"""
        # 找到对应节点
        target_node = None
        target_idx = -1
        
        for i, node in enumerate(self.graph_json["nodes"]):
            if node.get("name") == layer_name:
                target_node = node
                target_idx = i
                break
        
        if not target_node:
            print(f"  ✗ Node {layer_name} not found in graph")
            return None
        
        print(f"  Found node at index {target_idx}")
        
        # 提取操作类型
        op_type = self._extract_op_type(target_node)
        
        # 获取输出shape
        output_shape_raw = self.shapes[target_idx] if target_idx < len(self.shapes) else None
        
        # 分析输入
        input_shape = []
        weight_shape = []
        
        if "inputs" in target_node:
            inputs_info = []
            for j, input_ref in enumerate(target_node["inputs"]):
                if isinstance(input_ref, list) and len(input_ref) >= 1:
                    input_node_idx = input_ref[0]
                    input_shape_raw = self.shapes[input_node_idx] if input_node_idx < len(self.shapes) else None
                    
                    # 获取输入节点信息
                    input_node = self.graph_json["nodes"][input_node_idx] if input_node_idx < len(self.graph_json["nodes"]) else {}
                    
                    inputs_info.append({
                        "index": j,
                        "node_idx": input_node_idx,
                        "node_op": input_node.get("op", "unknown"),
                        "raw_shape": input_shape_raw,
                        "is_param": input_node.get("op") == "null"
                    })
            
            # 第一个输入通常是数据
            if inputs_info and inputs_info[0]["raw_shape"]:
                input_shape = self._convert_nchwc_to_standard(inputs_info[0]["raw_shape"], "data")
                print(f"  Input shape: {inputs_info[0]['raw_shape']} -> {input_shape}")
            
            # 找权重 (null节点且不是第一个输入)
            for inp in inputs_info[1:]:
                if inp["is_param"] and inp["raw_shape"]:
                    weight_shape = self._convert_nchwc_to_standard(inp["raw_shape"], "weight")
                    print(f"  Weight shape: {inp['raw_shape']} -> {weight_shape}")
                    break
        
        # 输出shape
        output_shape = []
        if output_shape_raw:
            output_shape = self._convert_nchwc_to_standard(output_shape_raw, "data")
            print(f"  Output shape: {output_shape_raw} -> {output_shape}")
        
        return {
            "op_type": op_type,
            "input_shape": input_shape,
            "weight_shape": weight_shape,
            "output_shape": output_shape
        }

    def _extract_op_type(self, node):
        """提取操作类型"""
        if node.get("op") == "tvm_op":
            # 从func_name中提取更详细的信息
            if "attrs" in node and "func_name" in node["attrs"]:
                func_name = node["attrs"]["func_name"]
                
                # 分析融合操作
                op_parts = []
                if "conv2d" in func_name.lower():
                    if "nchwc" in func_name.lower():
                        op_parts.append("nn.contrib_conv2d_NCHWc")
                    else:
                        op_parts.append("nn.conv2d")
                
                if "add" in func_name.lower():
                    # 计算add的数量
                    add_count = func_name.lower().count("add")
                    if add_count == 1:
                        op_parts.append("add")
                    else:
                        op_parts.append(f"add(x{add_count})")
                
                if "relu" in func_name.lower():
                    op_parts.append("relu")
                    
                if "layout_transform" in func_name.lower():
                    op_parts.append("layout_transform")
                
                return " + ".join(op_parts) if op_parts else func_name
            else:
                return "tvm_op"
        else:
            return node.get("op", "unknown")

    def _convert_nchwc_to_standard(self, nchwc_shape, shape_type="data"):
        """
        将NCHWc格式转换为标准格式
        shape_type: "data" (NCHW) 或 "weight" (OIHW)
        """
        if not isinstance(nchwc_shape, list) or len(nchwc_shape) == 0:
            return nchwc_shape
        
        # 如果已经是标准格式，直接返回
        if len(nchwc_shape) <= 4:
            return nchwc_shape
        
        if len(nchwc_shape) == 5:
            # NCHW1c: [N, C_outer, H, W, C_inner] -> [N, C_outer*C_inner, H, W]
            n, c_outer, h, w, c_inner = nchwc_shape
            return [n, c_outer * c_inner, h, w]
            
        elif len(nchwc_shape) == 6:
            # 权重: OIHW1i4o: [O_outer, I_outer, KH, KW, I_inner, O_inner] -> [O_outer*O_inner, I_outer*I_inner, KH, KW]
            o_outer, i_outer, kh, kw, i_inner, o_inner = nchwc_shape
            return [o_outer * o_inner, i_outer * i_inner, kh, kw]
            
        else:
            # 其他格式，保守处理
            print(f"    Warning: Unusual shape format {nchwc_shape} for {shape_type}")
            return nchwc_shape

def update_top_k_layers_accurately(lib_path="./compiler/bin/YoloV4_x86_64.tar", 
                                  json_path="./compiler/bin/YoloV4_x86_64.json",
                                  top_k_json_path="top_k_layers_yolov4"):
    """
    根据top_k_layers.json中的layer_name，准确更新层信息
    """
    updater = AccurateTVMLayerInfoUpdater(lib_path, json_path)
    return updater.update_top_k_layers_info(top_k_json_path)

# 使用示例
if __name__ == "__main__":
    print("=== Accurately Updating top_k_layers.json ===")
    updated_layers = update_top_k_layers_accurately(
        lib_path="./compiler/bin/YoloV4_x86_64.tar",
        json_path="./compiler/bin/YoloV4_x86_64.json",
        top_k_json_path="top_k_layers_yolov4.json"
    )
    
    print(f"\n✓ Successfully updated {len(updated_layers)} layers with accurate shapes")
    print("Ready for low-rank decomposition analysis!")
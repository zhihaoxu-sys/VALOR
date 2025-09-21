import tvm
from tvm.contrib import graph_executor
from tvm.runtime.vm import VirtualMachine
from tvm.contrib.debugger import debug_executor
import numpy as np
import json
import time
import os

# 1. 基本设置
lib = tvm.runtime.load_module("./compiler/bin/YoloV4_x86_64.tar")
dev = tvm.device("llvm", 0)
debug_output_dir = "./debug_output"
os.makedirs(debug_output_dir, exist_ok=True)


# 2. 使用 graph_executor 进行基本性能分析
def analyze_performance():
    print("=== Performance Analysis ===")
    module = graph_executor.GraphModule(lib["default"](dev))

    # 设置随机输入
    input_data = np.random.rand(1, 416, 416, 3).astype("float32")
    module.set_input("input_1:0", input_data)

    # 多次运行取平均
    times = []
    num_runs = 10

    for i in range(num_runs):
        start = time.time()
        module.run()
        end = time.time()
        times.append((end - start) * 1000)  # 转换为毫秒

    print(f"Average inference time: {np.mean(times):.2f} ms")
    print(f"Median inference time: {np.median(times):.2f} ms")
    print(f"Std dev: {np.std(times):.2f} ms")


# 3. 使用 debug_executor 进行详细分析
def analyze_with_debugger():
    print("\n=== Detailed Layer Analysis ===")
    try:
        # 如果有json文件
        with open("./compiler/bin/YoloV4_x86_64.json", "r") as f:
            graph_json = f.read()

        debug_mod = debug_executor.create(graph_json, lib, dev, dump_root=debug_output_dir)

        # 设置输入
        input_data = np.random.rand(1, 416, 416, 3).astype("float32")
        debug_mod.set_input("input_1:0", input_data)

        # 运行并收集每层信息
        debug_mod.run()

        # 获取中间层输出
        print("\nAnalyzing intermediate layers:")
        node_info = debug_mod.get_graph_nodes()
        for i, node in enumerate(node_info):
            try:
                output = debug_mod.get_node_output(i)
                print(f"\nNode {i}: {node['name']}")
                print(f"Shape: {output.shape}")
                print(f"Data type: {output.dtype}")
                print(f"Mean value: {np.mean(output.numpy()):.6f}")
                print(f"Std dev: {np.std(output.numpy()):.6f}")
            except Exception as e:
                print(f"Error analyzing node {i}: {str(e)}")

    except Exception as e:
        print(f"Error in debugger analysis: {str(e)}")
        print("Falling back to basic profiling...")


# 4. 分析计算图结构
def analyze_graph_structure():
    print("\n=== Graph Structure Analysis ===")
    try:
        with open("./compiler/bin/ArcFace_x86_64.json", "r") as f:
            graph = json.load(f)

        # 分析节点类型统计
        node_types = {}
        for node in graph["nodes"]:
            op_type = node["op"]
            node_types[op_type] = node_types.get(op_type, 0) + 1

        print("\nNode type statistics:")
        for op_type, count in node_types.items():
            print(f"{op_type}: {count}")

    except Exception as e:
        print(f"Error analyzing graph structure: {str(e)}")


# 5. 主分析流程
def main():
    print("Starting TVM model analysis...\n")

    # 运行性能分析
    # analyze_performance()

    # 运行调试器分析
    analyze_with_debugger()

    # 分析图结构
    analyze_graph_structure()


if __name__ == "__main__":
    main()
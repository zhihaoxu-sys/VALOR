import re
import io
import sys
import os
import time
import numpy as np
import contextlib
import tvm
from tvm.contrib import graph_executor
from tvm.contrib.debugger import debug_executor
import json

# Basic setup
lib = tvm.runtime.load_module("./compiler/bin/YoloV4_riscv64.tar")
dev = tvm.device("llvm", 0)
debug_output_dir = "./debug_output"
os.makedirs(debug_output_dir, exist_ok=True)


def capture_logs(func, *args, **kwargs):
    """
    Capture all stdout/stderr output (including C++ logs) from func.
    Returns: (result, log_str) where result is func's return value or None if suppressed.
    """
    # create pipe
    rfd, wfd = os.pipe()
    # backup fds
    old_stdout = os.dup(1)
    old_stderr = os.dup(2)
    try:
        # redirect to pipe
        os.dup2(wfd, 1)
        os.dup2(wfd, 2)
        os.close(wfd)
        # run func
        try:
            result = func(*args, **kwargs)
        except ValueError as e:
            # suppress known RISC-V TVM debugger error
            if "tvm.relay._save_param_dict" in str(e):
                result = None
            else:
                raise
    finally:
        # restore
        os.dup2(old_stdout, 1)
        os.dup2(old_stderr, 2)
        os.close(old_stdout)
        os.close(old_stderr)
    # read logs
    logs = []
    while True:
        chunk = os.read(rfd, 4096)
        if not chunk:
            break
        logs.append(chunk)
    os.close(rfd)
    return result, b"".join(logs).decode('utf-8', errors='ignore')


def analyze_performance():
    print("=== Performance Analysis ===")
    module = graph_executor.GraphModule(lib["default"](dev))
    input_data = np.random.rand(1, 1, 128, 32).astype("float32")
    module.set_input("data", input_data)

    times = []
    for _ in range(10):
        start = time.time()
        module.run()
        end = time.time()
        times.append((end - start) * 1000)

    print(f"Average inference time: {np.mean(times):.2f} ms")
    print(f"Median inference time: {np.median(times):.2f} ms")
    print(f"Std dev: {np.std(times):.2f} ms")


def analyze_with_debugger(top_k=5):
    print("\n=== Detailed Layer Analysis + Top-K Slowest Ops ===")
    graph_json = open("./compiler/bin/YoloV4_riscv64.json").read()
    debug_mod = debug_executor.create(graph_json, lib, dev, dump_root=debug_output_dir)
    input_data = np.random.rand(1, 416, 416, 3).astype("float32")
    debug_mod.set_input("input_1:0", input_data)

    # define a safe run that ignores the missing global function error
    def safe_run():
        try:
            debug_mod.run()
        except ValueError as e:
            if "tvm.relay._save_param_dict" in str(e):
                return
            raise

    # capture logs from safe_run
    _, log_str = capture_logs(safe_run)
    log_lines = log_str.splitlines()

    # parse op timings
    times = []
    current = None
    for line in log_lines:
        m_op = re.search(r'Op #(\d+)\s+(\S+):', line)
        if m_op:
            idx = int(m_op.group(1))
            name = m_op.group(2)
            current = (idx, name)
            continue
        m_time = re.search(r'Iteration:\s*\d+:\s*([\d\.]+)\s*us/iter', line)
        if m_time and current:
            us = float(m_time.group(1))
            times.append((current[0], current[1], us))
            current = None

    # show top-k
    times.sort(key=lambda x: x[2], reverse=True)
    top_k_layers = []
    print(f"\nTop {top_k} slowest operators (μs):")
    for idx, name, us in times[:top_k]:
        print(f"  Op #{idx:2d} ({name}): {us:.2f} μs")
            # 创建层信息字典
        latency_ms = us / 1000.0
        layer_info = {
            "layer_name": name,
            "op_type": "",
            "latency_ms": round(latency_ms, 3),
            "input_shape": [],
            "weight_shape": []
        }
        top_k_layers.append(layer_info)
    with open("top_k_layers_yolov4.json", 'w') as f:
        json.dump(top_k_layers, f, indent=2)

    print(f"\nTop {top_k} layers info saved to top_k_layers_yolov4.json")
    
    # # optional: per-layer stats
    # print("\nAnalyzing intermediate layers:")
    # for i, node in enumerate(debug_mod.get_graph_nodes()):
    #     try:
    #         out = debug_mod.get_node_output(i).numpy()
    #         print(f"\nNode {i}: {node['name']}")
    #         print(f"  shape: {out.shape}, dtype: {out.dtype}")
    #         print(f"  mean: {out.mean():.6f}, std: {out.std():.6f}")
    #     except Exception as e:
    #         print(f"  [!] Skipping node {i}: {e}")


if __name__ == "__main__":
    analyze_with_debugger(top_k=5)

from optimization_pipeline import optimize_onnx_model

# 运行优化
optimized_model_path, report = optimize_onnx_model(
    onnx_path="your_model.onnx",           # 原始ONNX模型路径
    layers_json_path="layers_info.json",   # 你提供的JSON文件
    input_shape=(1, 3, 224, 224),         # 模型输入shape
    rvv_length=128,                        # RVV向量长度
    accuracy_threshold=0.01                # 精度阈值(1%)
)

print(f"优化完成！模型保存在：{optimized_model_path}")
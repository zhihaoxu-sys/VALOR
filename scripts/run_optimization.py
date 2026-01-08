from optimization_pipeline import optimize_onnx_model

# Run optimization
optimized_model_path, report = optimize_onnx_model(
    onnx_path="your_model.onnx",           # Path to the original ONNX model
    layers_json_path="layers_info.json",   # Path to the layer mapping JSON
    input_shape=(1, 3, 224, 224),          # Model input shape
    rvv_length=128,                        # RVV vector length
    accuracy_threshold=0.01                # Accuracy threshold (1%)
)

print(f"Optimization complete! Model saved to: {optimized_model_path}")

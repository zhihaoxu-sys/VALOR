import torch
from transformers import T5ForConditionalGeneration
import numpy as np

from torch_mlir import fx
from torch.export import Dim

import os
os.environ['TORCH_LOGS'] = '+dynamic'
#from torch_mlir.dynamo import run_backend

Dim_auto = Dim.AUTO

class T5Wrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained('t5-small')

    def forward(self, input_ids,decoder_input_ids):  # 移除 attention_mask 参数

        input_ids = input_ids.to(torch.long)
        if decoder_input_ids is not None:
            decoder_input_ids = decoder_input_ids.to(torch.long)
        else:
            # 如果没有提供 decoder_input_ids，用默认的起始 token（通常是 input_ids 的第一个 token）
            decoder_input_ids = input_ids[:, :1]
        outputs = self.model(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            return_dict=True,
            # use_cache = False,  # 禁用缓存可能有助于避免一些动态shape问题
            # output_attentions = False,
            # output_hidden_states = False
        )
        return outputs.logits


# 加载预训练的 T5 模型
#base_model = T5ForConditionalGeneration.from_pretrained('t5-small')
model = T5Wrapper()
model.eval()
# import onnx
# import torch
# from onnx2pytorch import ConvertModel
# # 1. 加载 ONNX 模型
# onnx_model = onnx.load("/doc2/zhzh/models_tvm/t5-encoder-12.onnx")
#
#
# # 2. 创建一个包装类来处理转换后的模型
# class ONNXModelWrapper(torch.nn.Module):
#     def __init__(self, onnx_model):
#         super().__init__()
#         self.model = ConvertModel(onnx_model, experimental=True)  # 添加 experimental 参数
#
#     def forward(self, input_ids, attention_mask=None):
#         return self.model(input_ids, attention_mask)
#
#
# try:
#     # 3. 创建包装模型
#     wrapped_model = ONNXModelWrapper(onnx_model)
#     wrapped_model.eval()
# except Exception as e:
#     print(f"Error during conversion: {e}")
# 打印模型结构，确保转换成功
# 定义输入张量
batch_size = 1
seq_length = 16
input_ids = torch.ones((batch_size, seq_length), dtype=torch.long)
decoder_input_ids = torch.ones((batch_size, seq_length), dtype=torch.long)

model.eval()
with torch.no_grad():
    output = model(input_ids,decoder_input_ids)

print("PyTorch Model output:", output)

# dynamic_shapes = {
#     "input_ids": {0: Dim_auto, 1: Dim_auto},
#     "decoder_input_ids": {0: Dim_auto, 1: Dim_auto}
# }

d2 = Dim("seq_length", min=1,max=512)

dynamic_shapes = {
    "input_ids": {  # 第一个输入 (input_ids)
        1: d2
    },
    "decoder_input_ids": {  # 第二个输入 (decoder_input_ids)
        1: d2
    }
}


# dynamic_shapes = {
#     "input_dict": {
#         "input_ids": {0: Dim_auto, 1: Dim_auto},  # 嵌套结构
#     }
# }
# example_inputs = {
#     "input_ids": example_inputs,
#     # "attention_mask": attention_mask,
#     # "decoder_input_ids": decoder_input_ids,
#     # "decoder_attention_mask": decoder_attention_mask
# }

# 尝试使用 torch-mlir 导出模型
try:
    traced_model = torch.jit.trace(model, (input_ids, decoder_input_ids))
    module = fx.export_and_import(
        model,
        input_ids = input_ids,
        decoder_input_ids = decoder_input_ids,
        # attention_mask = attention_mask,
        # decoder_input_ids = decoder_input_ids,
        # decoder_attention_mask = decoder_attention_mask,
        dynamic_shapes = dynamic_shapes,
        output_type="linalg-on-tensors",
        func_name="forward"
    )
    print("MLIR model exported successfully.")
    # with open("/doc2/zhzh/models_tvm/t5_model.mlir", "w") as f:
    #     f.write(str(module))
    # print("MLIR model saved to 't5_model.mlir'.")
    # # 验证不同输入大小
    # from torch_mlir import runtime
    # from torch_mlir.dialects import torch
    #
    # # 加载 MLIR 模型
    # with open("/doc2/zhzh/models_tvm/t5_model.mlir", "r") as f:
    #     mlir_module = f.read()
    backend = run_backend.RefBackendLinalgOnTensorsBackend()
    compiled = backend.compile(module)
    fx_module = backend.load(compiled)

    # 使用 torch_mlir 的运行时加载 MLIR 模型


    print("\nValidating dynamic shapes...")
    test_sizes = [
        (2, 20, 12),  # 不同的 batch_size 和序列长度
        (4, 32, 15),  # 更大的尺寸
        (1, 8, 5)  # 更小的尺寸
    ]

    for batch, seq_len, tgt_len in test_sizes:
        print(f"\nTesting with batch={batch}, seq_len={seq_len}, target_len={tgt_len}")
        test_inputs = (
            torch.ones((batch, seq_len), dtype=torch.long),
            torch.ones((batch, seq_len), dtype=torch.long),
            torch.ones((batch, tgt_len), dtype=torch.long),
            torch.ones((batch, tgt_len), dtype=torch.long)
        )

        # PyTorch 模型输出
        torch_output = model(*test_inputs)
        print(f"PyTorch output shape: {torch_output.shape}")

        # MLIR 模型输出
        mlir_output = torch.from_numpy(getattr(fx_module, "forward")(*test_inputs))
        print(f"MLIR output shape: {mlir_output.shape}")

        # 验证输出形状和值
        torch.testing.assert_close(torch_output, mlir_output, rtol=1e-3, atol=1e-3)
        print("✓ Shapes match and outputs are close")

    print("\nAll validations passed successfully!")
except Exception as e:
    print(f"Error during export: {e}")

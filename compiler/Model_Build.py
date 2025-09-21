import os
import torch
import onnx
import tvm
from tvm import relay
from typing import Dict, Tuple


class ModelBuilder:
    devices: Dict[str, tvm.target.Target] = {
        "riscv64": tvm.target.Target(
            "llvm -mtriple=riscv64-unknown-linux-gnu -mcpu=generic-rv64 -mabi=lp64d -mattr=+64bit,+m,+f,+d"),
        "x86_64": tvm.target.Target("llvm")
    }

    MODEL_CONFIGS = {
        "imagenet": {
            "path": "../models/resnet18-v2-7.onnx",
            "input_shape": {"data": (1, 3, 224, 224)},
            "loader": "onnx"
        },
        "kws": {
            "path": "../models/resnet18-kws-best-acc.pt",
            "input_shape": [("data", (1, 1, 128, 32))],
            "loader": "pytorch"
        },
        "resnet50": {
            "path": "../models/resnet50-v2-7.onnx",
            "input_shape": {"data": (1, 3, 224, 224)},
            "loader": "onnx"
        },
        "retinanet": {
            "path": "./models/retinanet-9.onnx",
            "input_shape": {"input_0": (1, 3, 480, 640)},
            "loader": "onnx"
        },
        "YoloV4": {
            "path": "/doc2/zhzh/models_tvm/yolov4.onnx",
            "input_shape": {"input_1:0": (1, 416, 416, 3)},
            "loader": "onnx"
        },
        "ArcFace": {
            "path": "/doc2/zhzh/models_tvm/modified_model.onnx",
            "input_shape": {"data": (1, 3, 112, 112)},
            "loader": "onnx"
        },
        "T5": {
            "path": "/doc2/zhzh/models_tvm/t5-encoder-12.onnx",
            "input_shape": {"input_ids": (1, 512)},
            "loader": "onnx"
        },
        "T5-decoder": {
            "path": "/doc2/zhzh/models_tvm/t5-decoder-with-lm-head-12.onnx",
            "input_shape": {
                "input_ids": (relay.Any(), relay.Any()),
                "encoder_hidden_states": (relay.Any(), relay.Any(), 768)
            },
            "loader": "onnx"
        },
        "Bert": {
            "path": "/doc2/zhzh/models_tvm/bertsquad-12.onnx",
            "input_shape": {
                "unique_ids_raw_output___9:0": (1,),
                "segment_ids:0": (1, 256),
                "input_mask:0": (1, 256),
                "input_ids:0": (1, 256)
            },
            "loader": "onnx"
        }
    }

    def __init__(self, platform: str):
        if platform not in self.devices:
            raise ValueError(f"Platform {platform} is not supported. Available platforms: {list(self.devices.keys())}")
        self.platform = platform
        self.device = self.devices[platform]

    def _load_and_compile_model(self, model_name: str) -> Tuple[tvm.runtime.Module, str]:
        if model_name not in self.MODEL_CONFIGS:
            raise ValueError(f"Model {model_name} not supported. Available models: {list(self.MODEL_CONFIGS.keys())}")

        config = self.MODEL_CONFIGS[model_name]

        try:
            # Load model
            if config["loader"] == "onnx":
                model = onnx.load(config["path"])
                mod, params = relay.frontend.from_onnx(model, config["input_shape"])
            elif config["loader"] == "pytorch":
                model = torch.jit.load(config["path"])
                mod, params = relay.frontend.from_pytorch(model, config["input_shape"])
            else:
                raise ValueError(f"Unsupported loader type: {config['loader']}")

            # Compile model
            pass_context = {"relay.backend.use_auto_scheduler": True} if model_name == "T5-decoder" else {}
            with tvm.transform.PassContext(opt_level=3, config=pass_context):
                lib = relay.build(mod, target=self.device, params=params)
                graph_json = lib.get_graph_json()

            return lib, graph_json

        except Exception as e:
            raise RuntimeError(f"Error processing model {model_name}: {str(e)}")

    def build(self, model_name: str) -> None:
        """Build and export the specified model."""
        binary_path = f"./bin/{model_name}_{self.platform}.tar"
        json_path = f"./bin/{model_name}_{self.platform}.json"

        # Skip if files already exist
        if os.path.exists(binary_path) and os.path.exists(json_path):
            print(f"Model {model_name} already built. Skipping...")
            return

        # Create bin directory if it doesn't exist
        os.makedirs(os.path.dirname(binary_path), exist_ok=True)

        # Build model
        print(f"Building {model_name} for {self.platform}...")
        try:
            lib, graph_json = self._load_and_compile_model(model_name)

            # Export library and JSON
            lib.export_library(binary_path)
            with open(json_path, "w") as f:
                f.write(graph_json)

            print(f"Successfully built {model_name} for {self.platform}")

        except Exception as e:
            print(f"Failed to build {model_name}: {str(e)}")
            if os.path.exists(binary_path):
                os.remove(binary_path)
            if os.path.exists(json_path):
                os.remove(json_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Model Builder for TVM")
    parser.add_argument("--model", type=str, required=True,
                        help=f"Model to build: {', '.join(ModelBuilder.MODEL_CONFIGS.keys())}")
    parser.add_argument("--platform", type=str, required=True,
                        help="Target platform (riscv64, x86_64)")

    args = parser.parse_args()

    builder = ModelBuilder(args.platform)
    builder.build(args.model)
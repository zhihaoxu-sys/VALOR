import os

import torch
import onnx

import tvm
from tvm import relay

from typing import Dict


class ModelBuilder:
  devices: Dict[str, tvm.target.Target] = {
    "riscv64" : tvm.target.Target("llvm -mtriple=riscv64-unknown-linux-gnu -mcpu=generic-rv64 -mabi=lp64d -mattr=+64bit,+m,+f,+d"),
    "x86_64"  : tvm.target.Target("llvm")
  }

  def __init__(self, platform: str):
    try:
      self.platform = platform
      self.device = ModelBuilder.devices[platform]
    except Exception as e:
      print(f" platform {platform} is not supported: {e} \n")
    
  
  @staticmethod
  def _imagenet_resnet18(target: tvm.target.Target) -> tvm.runtime.Module:
    try:
      model = onnx.load("./models/resnet18-v2-7.onnx")
    except Exception as e:
      print(f" error loading the model: {e} \n")

    input_shape = (1, 3, 224, 224)
    shape_dict = {"data": input_shape}
    mod, params = relay.frontend.from_onnx(model, shape_dict)
    with tvm.transform.PassContext(opt_level=3, config={}):
      lib = relay.build(mod, target=target, params=params)
    return lib

  @staticmethod
  def _kws_resnet18(target: tvm.target.Target) -> tvm.runtime.Module:
    try:
      model = torch.jit.load("./models/resnet18-kws-best-acc.pt")
    except Exception as e:
      print(f" error loading the model: {e} \n")

    input_shape = (1, 1, 128, 32)
    shape_list = [("data", input_shape)]
    mod, params = relay.frontend.from_pytorch(model, shape_list)
    with tvm.transform.PassContext(opt_level=3, config={}):
      lib = relay.build(mod, target=target, params=params)
    return lib

  def build(self,  model: str) -> None:
    binary_path = f"./bin/{model}_{self.platform}.tar"
    if model == "imagenet":
      if not os.path.exists(binary_path):
        imagenet_module = ModelBuilder._imagenet_resnet18(self.device)
        imagenet_module.export_library(binary_path)
    if model == "kws":
      if not os.path.exists(binary_path):
        kws_module = ModelBuilder._kws_resnet18(self.device)
        kws_module.export_library(binary_path)


  @staticmethod
  def _resnet50(target: tvm.target.Target) -> tvm.runtime.Module:
      try:
          model = onnx.load("./models/resnet50-v2-7.onnx")
      except Exception as e:
          print(f" error loading the model: {e} \n")
    
      # ResNet50 ?????? ResNet18 ??
      input_shape = (1, 3, 224, 224)
      shape_dict = {"data": input_shape}
    
      # ??? Relay
      mod, params = relay.frontend.from_onnx(model, shape_dict)
    
      # ??
      with tvm.transform.PassContext(opt_level=3, config={}):
          lib = relay.build(mod, target=target, params=params)
      return lib
  
  def build(self, model: str) -> None:
      binary_path = f"./bin/{model}_{self.platform}.tar"
    
      # ?? ResNet50 ?????
      if model == "resnet50":
          if not os.path.exists(binary_path):
              resnet50_module = ModelBuilder._resnet50(self.device)
              resnet50_module.export_library(binary_path)
    
      # ???????
      elif model == "imagenet":
          if not os.path.exists(binary_path):
              imagenet_module = ModelBuilder._imagenet_resnet18(self.device)
              imagenet_module.export_library(binary_path)
      elif model == "kws":
          if not os.path.exists(binary_path):
              kws_module = ModelBuilder._kws_resnet18(self.device)
              kws_module.export_library(binary_path)
              
  @staticmethod
  def _retinanet(target: tvm.target.Target) -> tvm.runtime.Module:
    try:
        model = onnx.load("./models/retinanet-9.onnx")  # ?????????
    except Exception as e:
        print(f" error loading the model: {e} \n")
    
    # RetinaNet ?????
    input_shape = (1, 3, 480, 640)  # ??????????
    shape_dict = {"input_0": input_shape}  
    mod, params = relay.frontend.from_onnx(model, shape_dict)
    with tvm.transform.PassContext(opt_level=3, config={}):
        lib = relay.build(mod, target=target, params=params)
    return lib

  def build(self, model: str) -> None:
    binary_path = f"./bin/{model}_{self.platform}.tar"
    if model == "retinanet":
        if not os.path.exists(binary_path):
            retinanet_module = ModelBuilder._retinanet(self.device)
            retinanet_module.export_library(binary_path)

  @staticmethod
  def _YoloV4(target: tvm.target.Target) -> tvm.runtime.Module:
    try:
        model = onnx.load("/doc2/zhzh/models_tvm/yolov4.onnx")
    except Exception as e:
        print(f" error loading the model: {e} \n")
    
    # RetinaNet ?????
    input_shape = (1, 416, 416, 3)  # ??????????
    shape_dict = {"input_1:0": input_shape}
    mod, params = relay.frontend.from_onnx(model, shape_dict)
    with tvm.transform.PassContext(opt_level=3, config={}):
        lib = relay.build(mod, target=target, params=params)
        graph_json = lib.get_graph_json()
    return lib, graph_json

  def build(self, model: str) -> None:
    binary_path = f"./bin/{model}_{self.platform}.tar"
    json_path = f"./bin/{model}_{self.platform}.json"
    # ?? RetinaNet ?????
    if model == "YoloV4":
        if not os.path.exists(binary_path):
            YoloV4_module, graph_json = ModelBuilder._YoloV4(self.device)
            # 导出模型
            YoloV4_module.export_library(binary_path)
            # 保存 JSON
            with open(json_path, "w") as f:
                f.write(graph_json)

  @staticmethod
  def _ArcFace(target: tvm.target.Target) -> tvm.runtime.Module:
      try:
          model = onnx.load("/doc2/zhzh/models_tvm/modified_model.onnx")
      except Exception as e:
          print(f" error loading the model: {e} \n")

      # RetinaNet ?????
      input_shape = (1, 3,112, 112)  # ??????????
      shape_dict = {"data": input_shape}
      mod, params = relay.frontend.from_onnx(model, shape_dict)
      with tvm.transform.PassContext(opt_level=3, config={}):
          lib = relay.build(mod, target=target, params=params)
      return lib

  def build(self, model: str) -> None:
      binary_path = f"../bin/{model}_{self.platform}.tar"
      # ?? RetinaNet ?????
      if model == "ArcFace":
          if not os.path.exists(binary_path):
              ArcFace_module = ModelBuilder._ArcFace(self.device)
              ArcFace_module.export_library(binary_path)

  @staticmethod
  def _T5_encoder(target: tvm.target.Target) -> tvm.runtime.Module:
      try:
          model = onnx.load("/doc2/zhzh/models_tvm/t5-encoder-12.onnx")
          print(f"Model: {model}")
          # model_decoder = onnx.load("/doc2/zhzh/models_tvm/modified_model.onnx")
      except Exception as e:
          print(f" error loading the model: {e} \n")

      # RetinaNet ?????
      input_shape = (1, 512)
      shape_dict = {"input_ids": input_shape}
      mod, params = relay.frontend.from_onnx(model, shape_dict)
      with tvm.transform.PassContext(opt_level=3, config={}):
          lib = relay.build(mod, target=target, params=params)
      return lib

  def build(self, model: str) -> None:
      binary_path = f"../bin/{model}_{self.platform}.tar"
      # ?? RetinaNet ?????
      if model == "T5":
          if not os.path.exists(binary_path):
              T5_encoder_module = ModelBuilder._T5_encoder(self.device)
              T5_encoder_module.export_library(binary_path)
              print(f"Model: {T5_encoder_module}")


  @staticmethod
  def _T5_decoder(target: tvm.target.Target) -> tvm.runtime.Module:
      try:
          #model_encoder = onnx.load("/doc2/zhzh/models_tvm/modified_model.onnx")
          model_decoder = onnx.load("/doc2/zhzh/models_tvm/t5-decoder-with-lm-head-12.onnx")
      except Exception as e:
          print(f" error loading the model: {e} \n")

      input_shape = (relay.Any(), relay.Any())
      encoder_hidden_states_shape = (relay.Any(), relay.Any(), 768)      # ??????????
      shape_dict = {
          "input_ids": input_shape,
          "encoder_hidden_states": encoder_hidden_states_shape,
      }
      mod, params = relay.frontend.from_onnx(model_decoder, shape_dict)
      config = {
          "relay.backend.use_auto_scheduler": True,
          "relay.FuseOps.max_depth": 1
      }
      with tvm.transform.PassContext(opt_level=3, config=config):
          lib = relay.build(mod, target=target, params=params)
      return lib

  def build(self, model: str) -> None:
      binary_path = f"../bin/{model}_{self.platform}.tar"
      # ?? RetinaNet ?????
      if model == "T5-decoder":
          if not os.path.exists(binary_path):
              T5_decoder_module = ModelBuilder._T5_decoder(self.device)
              T5_decoder_module.export_library(binary_path)

  @staticmethod
  def _Bert(target: tvm.target.Target) -> tvm.runtime.Module:
      try:
          model = onnx.load("/doc2/zhzh/models_tvm/bertsquad-12.onnx")
      except Exception as e:
          print(f" error loading the model: {e} \n")

      # 定义输入名称和形状（确保与 ONNX 模型匹配）

      input_shape = {
          "unique_ids_raw_output___9:0":(1,),
          "segment_ids:0": (1, 256),  # 假设批大小为1，序列长度为256
          "input_mask:0": (1, 256),
          "input_ids:0": (1, 256)
      }

      try:
          # 从 ONNX 模型转换为 Relay 模型
          mod, params = relay.frontend.from_onnx(model, input_shape)
          # 编译 Relay 模型
          with tvm.transform.PassContext(opt_level=3):
              lib = relay.build(mod, target=target, params=params)
          return lib
      except Exception as e:
          print(f"Error during TVM model compilation: {e}")
          return None

  def build(self, model: str) -> None:
      binary_path = f"../bin/{model}_{self.platform}.tar"
      # ?? RetinaNet ?????
      if model == "Bert":
          if not os.path.exists(binary_path):
              Bert_module = ModelBuilder._Bert(self.device)
              Bert_module.export_library(binary_path)


if __name__ == "__main__":
    import argparse

    # 定义命令行参数
    parser = argparse.ArgumentParser(description="Model Builder for TVM")
    parser.add_argument("--model", type=str, required=True, help="Specify the model to build (e.g., resnet50, retinanet, YoloV4)")
    parser.add_argument("--platform", type=str, required=True, help="Target platform (e.g., riscv64, x86_64)")
    args = parser.parse_args()

    # 初始化 ModelBuilder
    builder = ModelBuilder(args.platform)
    print(args.model)
    builder.build(args.model)
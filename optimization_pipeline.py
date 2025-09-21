"""
优化管道模块
整合所有组件，提供完整的模型优化流程
"""

import os
import json
import time
import onnx
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import asdict

from model_config import ModelConfig, create_default_config
from onnx_info_extractor import ONNXNodeInfoExtractor
from strategy_generator import RVVAwareStrategyGenerator, create_strategy_summary
from mse_evaluator import MSEAccuracyEstimator
from strategy_searcher import GreedyStrategySearcher, SearchResult


class OptimizationReport:
    """优化报告类"""
    
    def __init__(self):
        self.model_info = {}
        self.optimization_config = {}
        self.layer_analysis = {}
        self.search_results = {}
        self.final_performance = {}
        self.timing_breakdown = {}
        self.warnings = []
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "model_info": self.model_info,
            "optimization_config": self.optimization_config,
            "layer_analysis": self.layer_analysis,
            "search_results": self.search_results,
            "final_performance": self.final_performance,
            "timing_breakdown": self.timing_breakdown,
            "warnings": self.warnings,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def save_to_file(self, file_path: str):
        """保存报告到文件"""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


class OptimizationPipeline:
    """完整的优化管道"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.report = OptimizationReport()
        
        # 初始化各个组件
        self.extractor = None
        self.generator = None
        self.evaluator = None
        self.searcher = None
        
        # 记录时间
        self.timing = {}
    
    def optimize_onnx_model(self) -> Tuple[str, Dict[str, Any]]:
        """执行完整的模型优化流程"""
        print("="*60)
        print("Starting ONNX Model Optimization Pipeline")
        print("="*60)
        
        total_start_time = time.time()
        
        try:
            # 1. 初始化和模型分析
            self._initialize_components()
            layer_infos = self._analyze_model()
            
            # 2. 策略生成
            strategies_per_layer = self._generate_strategies(layer_infos)
            
            # 3. 策略搜索
            search_result = self._search_optimal_strategies(layer_infos, strategies_per_layer)
            
            # 4. 最终验证和模型生成
            optimized_model_path = self._finalize_optimization(search_result)
            
            # 5. 生成报告
            total_time = time.time() - total_start_time
            self._generate_final_report(search_result, total_time)
            
            print("="*60)
            print("Optimization Pipeline Completed Successfully!")
            print(f"Optimized model saved to: {optimized_model_path}")
            print(f"Optimization report saved to: {self.config.get_report_path()}")
            print("="*60)
            
            return optimized_model_path, self.report.to_dict()
        
        except Exception as e:
            print(f"Error in optimization pipeline: {e}")
            self.report.warnings.append(f"Pipeline failed: {str(e)}")
            raise
    
    def _initialize_components(self):
        """初始化所有组件"""
        print("1. Initializing components...")
        
        start_time = time.time()
        
        # 初始化各个组件
        self.extractor = ONNXNodeInfoExtractor(self.config)
        self.generator = RVVAwareStrategyGenerator(self.config.rvv_length)
        self.evaluator = MSEAccuracyEstimator(self.config)
        self.searcher = GreedyStrategySearcher(self.config, self.evaluator)
        
        self.timing["initialization"] = time.time() - start_time
        
        # 记录配置信息
        self.report.optimization_config = {
            "onnx_path": self.config.onnx_path,
            "layers_json_path": self.config.layers_json_path,
            "rvv_length": self.config.rvv_length,
            "accuracy_threshold": self.config.accuracy_threshold,
            "input_shape": self.config.input_shape,
            "calibration_samples": self.config.calibration_samples
        }
        
        print(f"   ✓ Components initialized in {self.timing['initialization']:.2f}s")
    
    def _analyze_model(self) -> List:
        """分析模型结构"""
        print("2. Analyzing model structure...")
        
        start_time = time.time()
        
        # 提取层信息
        layer_infos = self.extractor.extract_layer_info()
        
        # 获取模型摘要
        model_summary = self.extractor.get_model_summary()
        
        self.timing["model_analysis"] = time.time() - start_time
        
        # 记录模型信息
        self.report.model_info = model_summary
        self.report.layer_analysis = {
            "total_target_layers": len(layer_infos),
            "layers_with_weights": sum(1 for layer in layer_infos if layer.has_weights),
            "layers_without_weights": sum(1 for layer in layer_infos if not layer.has_weights),
            "total_mac_count": sum(layer.mac_count for layer in layer_infos),
            "total_original_latency": sum(layer.original_latency_ms for layer in layer_infos),
            "layer_details": [
                {
                    "name": layer.name,
                    "onnx_node_name": layer.onnx_node_name,
                    "op_type": layer.op_type,
                    "weight_shape": layer.weight_shape,
                    "has_weights": layer.has_weights,
                    "mac_count": layer.mac_count,
                    "original_latency_ms": layer.original_latency_ms
                }
                for layer in layer_infos
            ]
        }
        
        print(f"   ✓ Found {len(layer_infos)} target layers")
        print(f"   ✓ {sum(1 for layer in layer_infos if layer.has_weights)} layers with trainable weights")
        print(f"   ✓ Total MAC operations: {sum(layer.mac_count for layer in layer_infos):,}")
        print(f"   ✓ Model analysis completed in {self.timing['model_analysis']:.2f}s")
        
        return layer_infos
    
    def _generate_strategies(self, layer_infos: List) -> Dict[str, List]:
        """生成优化策略"""
        print("3. Generating optimization strategies...")
        
        start_time = time.time()
        
        strategies_per_layer = {}
        total_strategies = 0
        
        for layer in layer_infos:
            if layer.has_weights:
                strategies = self.generator.generate_strategies(layer)
                strategies_per_layer[layer.name] = strategies
                total_strategies += len(strategies)
                
                print(f"   ✓ Layer {layer.name} ({layer.op_type}): {len(strategies)} strategies")
            else:
                strategies_per_layer[layer.name] = []
                print(f"   - Layer {layer.name} ({layer.op_type}): skipped (no weights)")
        
        self.timing["strategy_generation"] = time.time() - start_time
        
        # 创建策略摘要
        all_strategies = []
        for strategies in strategies_per_layer.values():
            all_strategies.extend(strategies)
        
        strategy_summary = create_strategy_summary(all_strategies)
        
        self.report.layer_analysis["strategy_summary"] = {
            "total_strategies_generated": total_strategies,
            "strategies_per_layer": {name: len(strategies) 
                                   for name, strategies in strategies_per_layer.items()},
            "strategy_type_distribution": strategy_summary["by_type"],
            "strategy_target_distribution": strategy_summary["by_target"]
        }
        
        print(f"   ✓ Generated {total_strategies} total strategies")
        print(f"   ✓ Strategy generation completed in {self.timing['strategy_generation']:.2f}s")
        
        return strategies_per_layer
    
    def _search_optimal_strategies(self, layer_infos: List, strategies_per_layer: Dict) -> SearchResult:
        """搜索最优策略组合"""
        print("4. Searching for optimal strategy combination...")
        
        start_time = time.time()
        
        # 执行搜索
        search_result = self.searcher.search_optimal_strategies(layer_infos, strategies_per_layer)
        
        self.timing["strategy_search"] = time.time() - start_time
        
        # 记录搜索结果
        self.report.search_results = {
            "search_time_seconds": search_result.search_time_seconds,
            "total_evaluations": search_result.total_evaluations,
            "final_accuracy_loss": search_result.accuracy_loss,
            "estimated_latency_improvement": search_result.estimated_latency_improvement,
            "selected_strategies": [
                {
                    "layer_name": strategy.layer_name,
                    "strategy_type": strategy.strategy_type.value,
                    "parameters": strategy.parameters,
                    "target": strategy.target,
                    "expected_speedup": getattr(strategy, 'expected_speedup', 1.0)
                }
                for strategy in search_result.strategies
            ]
        }
        
        print(f"   ✓ Search completed in {search_result.search_time_seconds:.2f}s")
        print(f"   ✓ Total evaluations: {search_result.total_evaluations}")
        print(f"   ✓ Final accuracy loss: {search_result.accuracy_loss:.6f}")
        print(f"   ✓ Estimated latency improvement: {search_result.estimated_latency_improvement:.2f}x")
        
        return search_result
    
    def _finalize_optimization(self, search_result: SearchResult) -> str:
        """最终验证和模型生成"""
        print("5. Generating optimized model...")
        
        start_time = time.time()
        
        # 应用最终策略到模型
        optimized_model = self.evaluator.apply_strategies_to_onnx(search_result.strategies)
        
        # 保存优化后的模型
        optimized_model_path = self.config.get_optimized_model_path()
        onnx.save(optimized_model, optimized_model_path)
        
        # 最终验证
        validation_test_data = self.extractor.generate_test_data(num_samples=16)
        
        # 计算真实的性能指标
        final_mse = self.evaluator.evaluate_mse(
            self.evaluator.original_model, optimized_model, validation_test_data
        )
        
        # 测量实际延迟
        try:
            original_latency = self.evaluator.measure_latency(
                self.config.onnx_path, validation_test_data
            )
            optimized_latency = self.evaluator.measure_latency(
                optimized_model_path, validation_test_data
            )
            actual_latency_improvement = original_latency / optimized_latency
        except Exception as e:
            print(f"   Warning: Could not measure actual latency: {e}")
            actual_latency_improvement = search_result.estimated_latency_improvement
            self.report.warnings.append(f"Latency measurement failed: {str(e)}")
        
        # 计算模型大小变化
        original_size = os.path.getsize(self.config.onnx_path)
        optimized_size = os.path.getsize(optimized_model_path)
        size_reduction = (original_size - optimized_size) / original_size
        
        self.timing["finalization"] = time.time() - start_time
        
        # 记录最终性能
        self.report.final_performance = {
            "final_accuracy_loss": final_mse,
            "estimated_latency_improvement": search_result.estimated_latency_improvement,
            "actual_latency_improvement": actual_latency_improvement,
            "original_model_size_mb": original_size / (1024 * 1024),
            "optimized_model_size_mb": optimized_size / (1024 * 1024),
            "model_size_reduction_ratio": size_reduction,
            "accuracy_threshold_met": final_mse <= self.config.accuracy_threshold
        }
        
        print(f"   ✓ Final accuracy loss: {final_mse:.6f} (threshold: {self.config.accuracy_threshold:.6f})")
        print(f"   ✓ Actual latency improvement: {actual_latency_improvement:.2f}x")
        print(f"   ✓ Model size reduction: {size_reduction*100:.1f}%")
        print(f"   ✓ Model finalization completed in {self.timing['finalization']:.2f}s")
        
        # 检查是否满足精度要求
        if final_mse > self.config.accuracy_threshold:
            warning_msg = f"Final accuracy loss ({final_mse:.6f}) exceeds threshold ({self.config.accuracy_threshold:.6f})"
            print(f"   ⚠ Warning: {warning_msg}")
            self.report.warnings.append(warning_msg)
        
        return optimized_model_path
    
    def _generate_final_report(self, search_result: SearchResult, total_time: float):
        """生成最终报告"""
        print("6. Generating optimization report...")
        
        # 记录时间分解
        self.report.timing_breakdown = {
            "total_time_seconds": total_time,
            "initialization_seconds": self.timing.get("initialization", 0),
            "model_analysis_seconds": self.timing.get("model_analysis", 0),
            "strategy_generation_seconds": self.timing.get("strategy_generation", 0),
            "strategy_search_seconds": self.timing.get("strategy_search", 0),
            "finalization_seconds": self.timing.get("finalization", 0)
        }
        
        # 保存报告
        report_path = self.config.get_report_path()
        self.report.save_to_file(report_path)
        
        print(f"   ✓ Optimization report saved to: {report_path}")
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """获取优化摘要"""
        return {
            "success": len(self.report.warnings) == 0,
            "accuracy_loss": self.report.final_performance.get("final_accuracy_loss", 0),
            "latency_improvement": self.report.final_performance.get("actual_latency_improvement", 1.0),
            "model_size_reduction": self.report.final_performance.get("model_size_reduction_ratio", 0),
            "total_time": self.report.timing_breakdown.get("total_time_seconds", 0),
            "warnings": self.report.warnings
        }


def optimize_onnx_model(onnx_path: str, layers_json_path: str, input_shape: Tuple[int, ...],
                       rvv_length: int = 128, accuracy_threshold: float = 0.01,
                       output_dir: str = "./optimized_models") -> Tuple[str, Dict[str, Any]]:
    """便捷的模型优化函数"""
    
    # 创建配置
    config = ModelConfig(
        onnx_path=onnx_path,
        layers_json_path=layers_json_path,
        rvv_length=rvv_length,
        accuracy_threshold=accuracy_threshold,
        input_shape=input_shape,
        output_dir=output_dir
    )
    
    # 创建优化管道
    pipeline = OptimizationPipeline(config)
    
    # 执行优化
    return pipeline.optimize_onnx_model()


if __name__ == "__main__":
    # 测试代码
    try:
        # 示例用法
        optimized_model_path, report = optimize_onnx_model(
            onnx_path="/doc2/zhzh/models_tvm/yolov4.onnx",
            layers_json_path="simplified_matching_results.json",
            input_shape=(1, 416, 416, 3),
            rvv_length=128,
            accuracy_threshold=0.01
        )
        
        print(f"Optimization completed!")
        print(f"Optimized model: {optimized_model_path}")
        print(f"Summary: {report}")
        
    except Exception as e:
        print(f"Optimization failed: {e}")
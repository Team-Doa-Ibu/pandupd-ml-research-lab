"""
Universal Model Export Utility

A comprehensive script for exporting machine learning models to multiple formats.
Supports PyTorch, PyTorch Lightning, and various model architectures.
"""

import os
import json
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime

import torch
import torch.nn as nn
import torchvision.models as models
from torch.jit import ScriptModule

try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    warnings.warn("ONNX not available. Install with: pip install onnx onnxruntime")

try:
    import lightning as L
    LIGHTNING_AVAILABLE = True
except ImportError:
    LIGHTNING_AVAILABLE = False


class ModelExporter:
    """Universal model exporter supporting multiple formats and architectures."""
    
    def __init__(self, output_dir: str = "exported_models"):
        """
        Initialize the model exporter.
        
        Args:
            output_dir: Directory to save exported models
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.supported_formats = ['pt', 'weights', 'onnx', 'torchscript', 'info']
        
    def detect_model_type(self, model: Union[nn.Module, str, Path]) -> Dict[str, Any]:
        """
        Detect model type and extract basic information.
        
        Args:
            model: Model instance or path to model file
            
        Returns:
            Dictionary containing model metadata
        """
        info = {
            'model_type': 'unknown',
            'framework': 'pytorch',
            'is_lightning': False,
            'architecture': 'custom',
            'input_shape': None,
            'num_classes': None
        }
        
        if isinstance(model, (str, Path)):
            # Load model from file
            model_path = Path(model)
            if model_path.suffix == '.ckpt' and LIGHTNING_AVAILABLE:
                info['is_lightning'] = True
                info['model_type'] = 'lightning_checkpoint'
            elif model_path.suffix in ['.pt', '.pth']:
                info['model_type'] = 'pytorch_model'
            return info
            
        if LIGHTNING_AVAILABLE and isinstance(model, L.LightningModule):
            info['is_lightning'] = True
            info['model_type'] = 'lightning_module'
            
        if isinstance(model, ScriptModule):
            info['model_type'] = 'torchscript'
            
        # Detect architecture
        class_name = model.__class__.__name__
        info['architecture'] = class_name
        
        # Try to detect common architectures
        if 'resnet' in class_name.lower():
            info['architecture'] = 'ResNet'
        elif 'inception' in class_name.lower():
            info['architecture'] = 'InceptionV3'
        elif 'efficientnet' in class_name.lower():
            info['architecture'] = 'EfficientNet'
        elif 'vit' in class_name.lower() or 'vision_transformer' in class_name.lower():
            info['architecture'] = 'VisionTransformer'
            
        return info
        
    def get_model_info(self, model: nn.Module) -> Dict[str, Any]:
        """
        Extract comprehensive model information.
        
        Args:
            model: PyTorch model
            
        Returns:
            Dictionary containing detailed model information
        """
        info = self.detect_model_type(model)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        # Get model size in MB
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        model_size_mb = (param_size + buffer_size) / 1024 / 1024
        
        # Try to infer input shape
        input_shape = self._infer_input_shape(model)
        
        info.update({
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'frozen_parameters': frozen_params,
            'model_size_mb': round(model_size_mb, 2),
            'input_shape': input_shape,
            'export_timestamp': datetime.now().isoformat(),
            'pytorch_version': torch.__version__,
        })
        
        return info
        
    def _infer_input_shape(self, model: nn.Module) -> Optional[List[int]]:
        """
        Try to infer the expected input shape of the model.
        
        Args:
            model: PyTorch model
            
        Returns:
            Inferred input shape or None if cannot determine
        """
        # Common input shapes for different architectures
        common_shapes = {
            'InceptionV3': [3, 299, 299],
            'ResNet': [3, 224, 224],
            'EfficientNet': [3, 224, 224],
            'VisionTransformer': [3, 224, 224],
        }
        
        model_info = self.detect_model_type(model)
        arch = model_info.get('architecture', '')
        
        if arch in common_shapes:
            return common_shapes[arch]
            
        # Try to find first Conv2d layer to infer input channels
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                in_channels = module.in_channels
                # Assume square images, common sizes
                for size in [224, 299, 256, 512]:
                    return [in_channels, size, size]
                    
        return None
        
    def _generate_dummy_input(self, model: nn.Module, batch_size: int = 1) -> torch.Tensor:
        """
        Generate dummy input for the model.
        
        Args:
            model: PyTorch model
            batch_size: Batch size for dummy input
            
        Returns:
            Dummy input tensor
        """
        input_shape = self._infer_input_shape(model)
        
        if input_shape is None:
            # Default fallback
            input_shape = [3, 224, 224]
            
        full_shape = [batch_size] + input_shape
        return torch.randn(*full_shape)
        
    def export_complete_model(self, 
                            model: nn.Module, 
                            filename: str,
                            optimize: bool = False) -> str:
        """
        Export complete PyTorch model.
        
        Args:
            model: PyTorch model to export
            filename: Output filename (without extension)
            optimize: Whether to optimize the model
            
        Returns:
            Path to exported model
        """
        model.eval()
        
        if optimize:
            # Apply basic optimizations
            model = torch.jit.optimize_for_inference(torch.jit.script(model))
            
        output_path = self.output_dir / f"{filename}_complete.pt"
        torch.save(model, output_path)
        
        print(f"Complete model saved to: {output_path}")
        return str(output_path)
        
    def export_weights_only(self, 
                          model: nn.Module, 
                          filename: str) -> str:
        """
        Export model weights (state dict) only.
        
        Args:
            model: PyTorch model
            filename: Output filename (without extension)
            
        Returns:
            Path to exported weights
        """
        output_path = self.output_dir / f"{filename}_weights.pt"
        torch.save(model.state_dict(), output_path)
        
        print(f"Model weights saved to: {output_path}")
        return str(output_path)
        
    def export_to_onnx(self, 
                      model: nn.Module, 
                      filename: str,
                      input_shape: Optional[List[int]] = None,
                      dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
                      opset_version: int = 11) -> Optional[str]:
        """
        Export model to ONNX format.
        
        Args:
            model: PyTorch model
            filename: Output filename (without extension)
            input_shape: Custom input shape [C, H, W]
            dynamic_axes: Dynamic axes for variable input sizes
            opset_version: ONNX opset version
            
        Returns:
            Path to exported ONNX model or None if failed
        """
        if not ONNX_AVAILABLE:
            print("ONNX not available. Skipping ONNX export.")
            return None
            
        model.eval()
        
        # Generate dummy input
        if input_shape:
            dummy_input = torch.randn(1, *input_shape)
        else:
            dummy_input = self._generate_dummy_input(model)
            
        output_path = self.output_dir / f"{filename}.onnx"
        
        try:
            # Default dynamic axes for batch size
            if dynamic_axes is None:
                dynamic_axes = {
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
                
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes=dynamic_axes,
                verbose=False
            )
            
            # Verify ONNX model
            onnx_model = onnx.load(str(output_path))
            onnx.checker.check_model(onnx_model)
            
            print(f"ONNX model saved to: {output_path}")
            return str(output_path)
            
        except Exception as e:
            print(f"Failed to export ONNX model: {e}")
            return None
            
    def export_to_torchscript(self, 
                            model: nn.Module, 
                            filename: str,
                            method: str = 'trace') -> Optional[str]:
        """
        Export model to TorchScript format.
        
        Args:
            model: PyTorch model
            filename: Output filename (without extension)
            method: 'trace' or 'script'
            
        Returns:
            Path to exported TorchScript model or None if failed
        """
        model.eval()
        output_path = self.output_dir / f"{filename}_torchscript.pt"
        
        try:
            if method == 'trace':
                dummy_input = self._generate_dummy_input(model)
                traced_model = torch.jit.trace(model, dummy_input)
            else:  # script
                traced_model = torch.jit.script(model)
                
            traced_model.save(str(output_path))
            
            print(f"TorchScript model saved to: {output_path}")
            return str(output_path)
            
        except Exception as e:
            print(f"Failed to export TorchScript model: {e}")
            return None
            
    def export_model_info(self, 
                         model: nn.Module, 
                         filename: str,
                         additional_info: Optional[Dict] = None) -> str:
        """
        Export model information as JSON.
        
        Args:
            model: PyTorch model
            filename: Output filename (without extension)
            additional_info: Additional metadata to include
            
        Returns:
            Path to exported info file
        """
        info = self.get_model_info(model)
        
        if additional_info:
            info.update(additional_info)
            
        output_path = self.output_dir / f"{filename}_info.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2, default=str)
            
        print(f"Model info saved to: {output_path}")
        return str(output_path)
        
    def export_all_formats(self, 
                          model: nn.Module,
                          filename: str,
                          formats: Optional[List[str]] = None,
                          **kwargs) -> Dict[str, Optional[str]]:
        """
        Export model to all specified formats.
        
        Args:
            model: PyTorch model
            filename: Base filename (without extension)
            formats: List of formats to export ('pt', 'weights', 'onnx', 'torchscript', 'info')
            **kwargs: Additional arguments for specific export functions
            
        Returns:
            Dictionary mapping format to export path
        """
        if formats is None:
            formats = ['pt', 'weights', 'onnx', 'info']
            
        results = {}
        
        print(f"Exporting model '{filename}' to formats: {formats}")
        
        if 'pt' in formats:
            results['complete_model'] = self.export_complete_model(model, filename, **kwargs.get('pt_args', {}))
            
        if 'weights' in formats:
            results['weights'] = self.export_weights_only(model, filename)
            
        if 'onnx' in formats:
            results['onnx'] = self.export_to_onnx(model, filename, **kwargs.get('onnx_args', {}))
            
        if 'torchscript' in formats:
            results['torchscript'] = self.export_to_torchscript(model, filename, **kwargs.get('torchscript_args', {}))
            
        if 'info' in formats:
            results['info'] = self.export_model_info(model, filename, kwargs.get('additional_info', {}))
            
        print(f"Export completed. Files saved to: {self.output_dir}")
        return results


def load_model_from_checkpoint(checkpoint_path: str, 
                             model_class: Optional[type] = None) -> nn.Module:
    """
    Load model from Lightning checkpoint or PyTorch state dict.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model_class: Model class for loading state dict (optional for .ckpt files)
        
    Returns:
        Loaded PyTorch model
    """
    checkpoint_path = Path(checkpoint_path)
    
    if checkpoint_path.suffix == '.ckpt' and LIGHTNING_AVAILABLE:
        # Load Lightning checkpoint
        if model_class:
            model = model_class.load_from_checkpoint(checkpoint_path)
        else:
            raise ValueError("model_class required for loading Lightning checkpoints")
    else:
        # Load PyTorch model or state dict
        if checkpoint_path.suffix in ['.pt', '.pth']:
            try:
                # Try loading complete model first with weights_only=False for older models
                model = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            except Exception as e:
                print(f"Failed to load complete model: {e}")
                # Fallback to state dict
                if model_class is None:
                    raise ValueError("model_class required for loading state dict")
                model = model_class()
                state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
                model.load_state_dict(state_dict)
        else:
            raise ValueError(f"Unsupported checkpoint format: {checkpoint_path.suffix}")
            
    return model


# Convenience functions
def export_model(model: Union[nn.Module, str, Path],
                filename: str,
                output_dir: str = "exported_models",
                formats: List[str] = None,
                **kwargs) -> Dict[str, Optional[str]]:
    """
    Convenience function to export a model to multiple formats.
    
    Args:
        model: PyTorch model or path to model file
        filename: Base filename for exports
        output_dir: Output directory
        formats: List of export formats
        **kwargs: Additional arguments
        
    Returns:
        Dictionary mapping format to export path
    """
    exporter = ModelExporter(output_dir)
    
    if isinstance(model, (str, Path)):
        # Load model from file
        model_path = Path(model)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        model = torch.load(model_path, map_location='cpu')
        
    return exporter.export_all_formats(model, filename, formats, **kwargs)


if __name__ == "__main__":
    # Example usage
    print("Universal Model Export Utility")
    print("Available formats:", ['pt', 'weights', 'onnx', 'torchscript', 'info'])
    
    # Example with a simple model
    model = models.resnet18(pretrained=True)
    exporter = ModelExporter("test_exports")
    
    results = exporter.export_all_formats(
        model, 
        "resnet18_example",
        formats=['pt', 'weights', 'info']
    )
    
    print("Export results:", results)

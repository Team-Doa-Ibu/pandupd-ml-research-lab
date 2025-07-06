"""
Convert existing PyTorch models to ONNX format
Specifically designed for the spiral drawing classification models
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys
import warnings

# Add utils to path to import the export utility
sys.path.append(str(Path(__file__).parent / "utils"))
from export import ModelExporter, load_model_from_checkpoint

class InceptionV3SpiralClassifier(nn.Module):
    """Inception V3 model for spiral classification"""
    
    def __init__(self, num_classes=2):
        super().__init__()
        # Load pretrained Inception V3
        self.inception = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=False)
        
        # Modify the final classifier for spiral classification
        num_features = self.inception.fc.in_features
        self.inception.fc = nn.Linear(num_features, num_classes)
        
        # Set aux_logits to None for inference
        self.inception.aux_logits = False
        
    def forward(self, x):
        return self.inception(x)

def convert_spiral_models():
    """Convert all spiral drawing models to ONNX format"""
    
    models_dir = Path("spiral-drawing-digital-engine/models")
    output_dir = Path("spiral-drawing-digital-engine/models/onnx")
    
    if not models_dir.exists():
        print(f"Models directory not found: {models_dir}")
        return
    
    # Create ONNX output directory
    output_dir.mkdir(exist_ok=True)
    
    # Initialize exporter
    exporter = ModelExporter(str(output_dir))
    
    # Model files to convert
    model_files = {
        "inception_v3_spiral_production.pt": "inception_v3_spiral_production",
    }
    
    for model_file, model_name in model_files.items():
        model_path = models_dir / model_file
        
        if not model_path.exists():
            print(f"Model file not found: {model_path}")
            continue
            
        print(f"\nConverting {model_file}...")
        
        try:
            # Load the model
            if "weights" in model_file:
                # This is a state dict file
                print("Loading weights into Inception V3 architecture...")
                model = InceptionV3SpiralClassifier(num_classes=2)
                state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
                model.load_state_dict(state_dict)
            else:
                # This is a complete model file
                print("Loading complete model...")
                try:
                    model = torch.load(model_path, map_location='cpu', weights_only=False)
                except Exception as e:
                    print(f"Failed to load as complete model: {e}")
                    print("Trying to load as state dict...")
                    model = InceptionV3SpiralClassifier(num_classes=2)
                    state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
                    model.load_state_dict(state_dict)
            
            model.eval()
            
            # Convert to ONNX with Inception V3 specific settings
            onnx_path = exporter.export_to_onnx(
                model, 
                model_name,
                input_shape=[3, 299, 299],  # Inception V3 input size
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                },
                opset_version=11
            )
            
            if onnx_path:
                print(f"✓ Successfully converted {model_file} to ONNX")
                
                # Also export model info
                info_path = exporter.export_model_info(
                    model, 
                    model_name,
                    additional_info={
                        'original_file': str(model_file),
                        'task': 'spiral_classification',
                        'classes': ['healthy', 'parkinson'],
                        'input_size': [299, 299],
                        'preprocessing': 'normalize with ImageNet stats'
                    }
                )
                print(f"✓ Model info saved: {info_path}")
            else:
                print(f"✗ Failed to convert {model_file} to ONNX")
                
        except Exception as e:
            print(f"✗ Error converting {model_file}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\nConversion completed. ONNX models saved to: {output_dir}")

def test_onnx_models():
    """Test the converted ONNX models"""
    try:
        import onnxruntime as ort
        import numpy as np
    except ImportError:
        print("ONNXRuntime not available. Install with: pip install onnxruntime")
        return
    
    onnx_dir = Path("spiral-drawing-digital-engine/models/onnx")
    
    for onnx_file in onnx_dir.glob("*.onnx"):
        print(f"\nTesting {onnx_file.name}...")
        
        try:
            # Create ONNX Runtime session
            session = ort.InferenceSession(str(onnx_file))
            
            # Get model input info
            input_info = session.get_inputs()[0]
            input_shape = input_info.shape
            print(f"Input shape: {input_shape}")
            
            # Create dummy input
            dummy_input = np.random.randn(1, 3, 299, 299).astype(np.float32)
            
            # Run inference
            outputs = session.run(None, {input_info.name: dummy_input})
            print(f"Output shape: {outputs[0].shape}")
            print(f"✓ {onnx_file.name} is working correctly")
            
        except Exception as e:
            print(f"✗ Error testing {onnx_file.name}: {e}")

if __name__ == "__main__":
    print("Spiral Drawing Model Converter")
    print("=" * 40)
    
    # Convert models to ONNX
    convert_spiral_models()
    
    # Test the converted models
    print("\n" + "=" * 40)
    print("Testing converted ONNX models...")
    test_onnx_models()
    
    print("\nDone! Your models are now available in ONNX format.")
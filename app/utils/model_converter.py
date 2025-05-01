"""
Model conversion utilities.
This module provides utilities for converting models to different formats.
"""

import os
import logging
import json
import shutil
import tempfile
from typing import Dict, Any, Optional, List

import torch
import torch.nn as nn
import torch.jit as jit

# Configure logging
logger = logging.getLogger(__name__)

class ModelConverter:
    """
    Class for converting models to different formats.
    """
    
    def __init__(self, storage_dir: str = '/tmp/model-trainer'):
        """
        Initialize model converter.
        
        Args:
            storage_dir: Directory for storing conversion artifacts
        """
        self.storage_dir = storage_dir
        self.exports_dir = os.path.join(storage_dir, 'exports')
        
        # Create exports directory if it doesn't exist
        os.makedirs(self.exports_dir, exist_ok=True)
    
    def convert_to_flask(
        self,
        model: nn.Module,
        tokenizer: Any,
        export_id: str,
        quantize: bool = False,
        example_inputs: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, Any]:
        """
        Convert model for use in Flask applications.
        
        Args:
            model: Model to convert
            tokenizer: Tokenizer for the model
            export_id: Export ID
            quantize: Whether to quantize the model
            example_inputs: Example inputs for tracing
            
        Returns:
            Dictionary with conversion result
        """
        logger.info(f"Converting model to Flask format (export_id={export_id}, quantize={quantize})")
        
        # Create export directory
        export_dir = os.path.join(self.exports_dir, export_id, 'flask')
        os.makedirs(export_dir, exist_ok=True)
        
        try:
            # Ensure model is in evaluation mode
            model.eval()
            
            # Quantize model if requested
            if quantize:
                logger.info("Quantizing model")
                try:
                    from torch.quantization import quantize_dynamic
                    model = quantize_dynamic(
                        model,
                        {torch.nn.Linear},
                        dtype=torch.qint8
                    )
                    logger.info("Model quantized to int8")
                except Exception as e:
                    logger.warning(f"Error quantizing model: {e}, continuing with full precision")
            
            # Create example inputs if not provided
            if example_inputs is None:
                example_inputs = {
                    'input_ids': torch.zeros((1, 10), dtype=torch.long),
                    'attention_mask': torch.ones((1, 10), dtype=torch.long)
                }
            
            # Export model to TorchScript
            with torch.no_grad():
                try:
                    # Try to trace the model
                    traced_model = jit.trace(
                        model,
                        (example_inputs['input_ids'], example_inputs['attention_mask'])
                    )
                    logger.info("Model traced successfully")
                except Exception as e:
                    logger.warning(f"Error tracing model: {e}, trying to script instead")
                    try:
                        # Try to script the model instead
                        scripted_model = jit.script(model)
                        traced_model = scripted_model
                        logger.info("Model scripted successfully")
                    except Exception as e2:
                        logger.error(f"Error scripting model: {e2}")
                        raise
            
            # Save model
            model_path = os.path.join(export_dir, 'model.pt')
            traced_model.save(model_path)
            logger.info(f"Model saved to {model_path}")
            
            # Save tokenizer
            try:
                tokenizer.save_pretrained(export_dir)
                logger.info(f"Tokenizer saved to {export_dir}")
            except Exception as e:
                logger.warning(f"Error saving tokenizer: {e}, trying alternate method")
                try:
                    # Save tokenizer configuration
                    if hasattr(tokenizer, 'to_dict'):
                        with open(os.path.join(export_dir, 'tokenizer_config.json'), 'w') as f:
                            json.dump(tokenizer.to_dict(), f, indent=2)
                    
                    # Save vocabulary
                    if hasattr(tokenizer, 'get_vocab'):
                        with open(os.path.join(export_dir, 'vocab.json'), 'w') as f:
                            json.dump(tokenizer.get_vocab(), f, indent=2)
                    
                    logger.info(f"Tokenizer configuration saved to {export_dir}")
                except Exception as e2:
                    logger.error(f"Error saving tokenizer configuration: {e2}")
            
            # Create model metadata
            metadata = {
                'export_id': export_id,
                'format': 'flask',
                'quantized': quantize,
                'model_type': type(model).__name__,
                'input_shapes': {
                    'input_ids': [-1, -1],
                    'attention_mask': [-1, -1]
                },
                'output_shapes': {
                    'logits': [-1, -1, -1]
                }
            }
            
            # Save metadata
            with open(os.path.join(export_dir, 'metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Create example Flask application
            self._create_flask_example(export_dir)
            
            # Create zip file
            zip_path = os.path.join(self.exports_dir, export_id, 'flask_model.zip')
            shutil.make_archive(
                os.path.splitext(zip_path)[0],
                'zip',
                export_dir
            )
            
            return {
                'status': 'success',
                'format': 'flask',
                'export_id': export_id,
                'export_path': zip_path,
                'export_size': os.path.getsize(zip_path),
                'metadata': metadata
            }
            
        except Exception as e:
            logger.exception(f"Error converting model to Flask format: {e}")
            return {
                'status': 'error',
                'format': 'flask',
                'export_id': export_id,
                'error': str(e)
            }
    
    def convert_to_coreml(
        self,
        model: nn.Module,
        tokenizer: Any,
        export_id: str,
        optimization_level: int = 1,
        example_inputs: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, Any]:
        """
        Convert model to Apple's CoreML format.
        
        Args:
            model: Model to convert
            tokenizer: Tokenizer for the model
            export_id: Export ID
            optimization_level: Optimization level (0-3)
            example_inputs: Example inputs for conversion
            
        Returns:
            Dictionary with conversion result
        """
        logger.info(f"Converting model to CoreML format (export_id={export_id}, optimization_level={optimization_level})")
        
        # Create export directory
        export_dir = os.path.join(self.exports_dir, export_id, 'coreml')
        os.makedirs(export_dir, exist_ok=True)
        
        try:
            # Check if coremltools is available
            try:
                import coremltools as ct
            except ImportError:
                logger.error("coremltools package is not available for CoreML export")
                return {
                    'status': 'error',
                    'format': 'coreml',
                    'export_id': export_id,
                    'error': "coremltools package is not available"
                }
            
            # Ensure model is in evaluation mode
            model.eval()
            
            # Create example inputs if not provided
            if example_inputs is None:
                example_inputs = {
                    'input_ids': torch.zeros((1, 10), dtype=torch.long),
                    'attention_mask': torch.ones((1, 10), dtype=torch.long)
                }
            
            # Convert to ONNX first
            onnx_path = os.path.join(export_dir, 'model.onnx')
            
            # Export to ONNX
            with torch.no_grad():
                torch.onnx.export(
                    model,
                    (example_inputs['input_ids'], example_inputs['attention_mask']),
                    onnx_path,
                    input_names=['input_ids', 'attention_mask'],
                    output_names=['logits'],
                    dynamic_axes={
                        'input_ids': {0: 'batch', 1: 'sequence'},
                        'attention_mask': {0: 'batch', 1: 'sequence'},
                        'logits': {0: 'batch', 1: 'sequence', 2: 'vocab'}
                    },
                    opset_version=12
                )
            
            logger.info(f"Model exported to ONNX at {onnx_path}")
            
            # Load ONNX model
            onnx_model = ct.converters.onnx.load(onnx_path)
            
            # Set optimization parameters based on level
            if optimization_level >= 3:
                compute_precision = ct.precision.FLOAT16
                compute_units = ct.ComputeUnit.ALL
            elif optimization_level == 2:
                compute_precision = ct.precision.FLOAT16
                compute_units = ct.ComputeUnit.CPU_AND_GPU
            elif optimization_level == 1:
                compute_precision = ct.precision.FLOAT32
                compute_units = ct.ComputeUnit.CPU_ONLY
            else:
                compute_precision = ct.precision.FLOAT32
                compute_units = ct.ComputeUnit.CPU_ONLY
            
            # Convert to CoreML
            mlmodel = ct.convert(
                onnx_model,
                convert_to="mlprogram",
                compute_precision=compute_precision,
                compute_units=compute_units
            )
            
            # Add metadata
            mlmodel.author = "Model Trainer"
            mlmodel.license = "MIT"
            mlmodel.version = "1.0"
            mlmodel.short_description = f"Language model exported from Model Trainer"
            
            # Save CoreML model
            mlmodel_path = os.path.join(export_dir, 'model.mlmodel')
            mlmodel.save(mlmodel_path)
            logger.info(f"Model saved to {mlmodel_path}")
            
            # Save tokenizer configuration
            try:
                if hasattr(tokenizer, 'to_dict'):
                    with open(os.path.join(export_dir, 'tokenizer_config.json'), 'w') as f:
                        json.dump(tokenizer.to_dict(), f, indent=2)
                
                # Save vocabulary
                if hasattr(tokenizer, 'get_vocab'):
                    with open(os.path.join(export_dir, 'vocab.json'), 'w') as f:
                        json.dump(tokenizer.get_vocab(), f, indent=2)
                
                logger.info(f"Tokenizer configuration saved to {export_dir}")
            except Exception as e:
                logger.warning(f"Error saving tokenizer configuration: {e}")
            
            # Create model metadata
            metadata = {
                'export_id': export_id,
                'format': 'coreml',
                'optimization_level': optimization_level,
                'compute_precision': str(compute_precision),
                'compute_units': str(compute_units),
                'model_type': type(model).__name__,
                'input_shapes': {
                    'input_ids': [-1, -1],
                    'attention_mask': [-1, -1]
                },
                'output_shapes': {
                    'logits': [-1, -1, -1]
                }
            }
            
            # Save metadata
            with open(os.path.join(export_dir, 'metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Create example Swift code
            self._create_swift_example(export_dir)
            
            # Create zip file
            zip_path = os.path.join(self.exports_dir, export_id, 'coreml_model.zip')
            shutil.make_archive(
                os.path.splitext(zip_path)[0],
                'zip',
                export_dir
            )
            
            return {
                'status': 'success',
                'format': 'coreml',
                'export_id': export_id,
                'export_path': zip_path,
                'export_size': os.path.getsize(zip_path),
                'metadata': metadata
            }
            
        except Exception as e:
            logger.exception(f"Error converting model to CoreML format: {e}")
            return {
                'status': 'error',
                'format': 'coreml',
                'export_id': export_id,
                'error': str(e)
            }
    
    def _create_flask_example(self, export_dir: str) -> None:
        """
        Create example Flask application for the exported model.
        
        Args:
            export_dir: Directory where model is exported
        """
        app_code = """
from flask import Flask, request, jsonify
import torch
import os
import json
from transformers import AutoTokenizer

app = Flask(__name__)

# Load model and tokenizer
@app.before_first_request
def load_model():
    global model, tokenizer
    
    # Load model
    model_path = os.path.join(os.path.dirname(__file__), 'model.pt')
    model = torch.jit.load(model_path)
    model.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(os.path.dirname(__file__))

@app.route('/generate', methods=['POST'])
def generate():
    # Get request data
    data = request.json
    prompt = data.get('prompt', '')
    max_length = data.get('max_length', 50)
    temperature = data.get('temperature', 1.0)
    
    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors='pt')
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_length,
            temperature=temperature,
            do_sample=temperature > 0.0
        )
    
    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return jsonify({
        'prompt': prompt,
        'response': response,
        'tokens': len(outputs[0])
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'model_loaded': 'model' in globals()
    })

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
"""
        
        requirements = """
flask>=2.0.1
torch>=1.13.0
transformers>=4.30.0
"""
        
        readme = """
# Flask Model Server

This is an example Flask application for serving the exported model.

## Installation

1. Install the required packages:

```bash
pip install -r requirements.txt
```

2. Run the server:

```bash
python app.py
```

## API Endpoints

### Generate Text

```
POST /generate
```

Request body:
```json
{
    "prompt": "Once upon a time",
    "max_length": 50,
    "temperature": 0.7
}
```

Response:
```json
{
    "prompt": "Once upon a time",
    "response": "Once upon a time there was a great king who ruled...",
    "tokens": 62
}
```

### Health Check

```
GET /health
```

Response:
```json
{
    "status": "ok",
    "model_loaded": true
}
```
"""
        
        # Write files
        with open(os.path.join(export_dir, 'app.py'), 'w') as f:
            f.write(app_code.strip())
        
        with open(os.path.join(export_dir, 'requirements.txt'), 'w') as f:
            f.write(requirements.strip())
        
        with open(os.path.join(export_dir, 'README.md'), 'w') as f:
            f.write(readme.strip())
    
    def _create_swift_example(self, export_dir: str) -> None:
        """
        Create example Swift code for the exported CoreML model.
        
        Args:
            export_dir: Directory where model is exported
        """
        swift_code = """
import CoreML
import Foundation

class LanguageModel {
    private let model: MLModel
    private let tokenizer: Tokenizer
    
    init() throws {
        // Load the model
        let modelURL = Bundle.main.url(forResource: "model", withExtension: "mlmodel")!
        model = try MLModel(contentsOf: modelURL)
        
        // Initialize tokenizer
        tokenizer = try Tokenizer()
    }
    
    func generate(prompt: String, maxLength: Int = 50, temperature: Double = 1.0) throws -> String {
        // Tokenize the prompt
        let inputTokens = tokenizer.encode(text: prompt)
        
        // Create input array
        var inputIds = inputTokens
        var attentionMask = [Int](repeating: 1, count: inputTokens.count)
        
        // Generate tokens
        for _ in 0..<maxLength {
            // Create input dictionary
            let inputDict: [String: Any] = [
                "input_ids": MLMultiArray(inputIds),
                "attention_mask": MLMultiArray(attentionMask)
            ]
            
            // Run inference
            let output = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: inputDict))
            
            // Get logits
            let logits = output.featureValue(for: "logits")!.multiArrayValue!
            
            // Get next token (simplified - sampling logic would be more complex in real app)
            let nextToken = getNextToken(logits: logits, temperature: temperature)
            
            // Append to input
            inputIds.append(nextToken)
            attentionMask.append(1)
            
            // Check for end of sequence token
            if nextToken == tokenizer.eosToken {
                break
            }
        }
        
        // Decode the generated tokens
        return tokenizer.decode(tokens: inputIds)
    }
    
    private func getNextToken(logits: MLMultiArray, temperature: Double) -> Int {
        // Simplified token sampling logic
        // In a real app, you would need to implement proper temperature sampling
        
        // Get the last token logits
        let lastTokenLogits = logits[logits.shape[0] - 1, logits.shape[1] - 1]
        
        // Just return the argmax for this example
        return 0 // Placeholder - actual implementation needed
    }
}

// Placeholder for a tokenizer class
class Tokenizer {
    let eosToken: Int
    
    init() throws {
        // Load vocabulary from vocab.json
        eosToken = 50256 // Typical GPT-2 EOS token, adjust as needed
    }
    
    func encode(text: String) -> [Int] {
        // Placeholder for encoding logic
        return [0] // Placeholder - actual implementation needed
    }
    
    func decode(tokens: [Int]) -> String {
        // Placeholder for decoding logic
        return "" // Placeholder - actual implementation needed
    }
}
"""
        
        readme = """
# CoreML Model Usage

This is an example of how to use the exported CoreML model in a Swift application.

## Integration

1. Add the `model.mlmodel` file to your Xcode project
2. Implement the tokenizer based on the provided `vocab.json` and `tokenizer_config.json`
3. Use the sample code as a starting point for integrating the model

## Sample Code

The provided `ModelUsage.swift` file contains example code for:

1. Loading the CoreML model
2. Implementing a basic tokenizer
3. Text generation with the model

## Notes

- The actual tokenizer implementation is not included as it depends on your specific requirements
- You'll need to implement the token sampling logic based on the logits
- For more advanced usage, consider using Apple's CreateML framework as well
"""
        
        # Write files
        with open(os.path.join(export_dir, 'ModelUsage.swift'), 'w') as f:
            f.write(swift_code.strip())
        
        with open(os.path.join(export_dir, 'README.md'), 'w') as f:
            f.write(readme.strip())


# Example usage
if __name__ == "__main__":
    converter = ModelConverter()
    print("Model converter initialized")

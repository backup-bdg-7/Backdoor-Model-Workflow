# Advanced AI Model Training Platform

A complete solution for training, monitoring, and deploying AI models using datasets from various sources. This platform allows you to train models on a curated set of datasets and export them in formats compatible with Flask applications and Apple's CoreML for iOS/macOS applications.

## Features

- **Multi-service Architecture**: Distributed system designed for Render.com deployment with memory-efficient processing
- **Dataset Management**: Support for streaming and batched processing of multiple datasets
- **Model Training**: Configurable transformer-based models with various sizes and hyperparameters
- **Real-time Monitoring**: WebSocket-based monitoring of training progress with visualizations
- **Export Capabilities**: Export trained models for Flask applications or convert to Apple's CoreML format
- **User-friendly Interface**: Simple web interface for managing training jobs and exports

## Architecture

The system is divided into four main components:

1. **API Service**: RESTful API for handling user requests and coordinating other services
2. **Training Service**: Background worker responsible for training models and exporting them
3. **Monitoring Service**: Collects and provides training metrics with real-time updates
4. **Frontend**: Web interface for interacting with the system

## Deployment on Render.com

This project is designed to be deployed on [Render.com](https://render.com) using multiple services to efficiently distribute the workload and manage memory usage.

### Prerequisites

- A Render.com account
- Git repository with this codebase

### Deployment Steps

1. **Fork or clone this repository** to your own GitHub account
2. **Connect your GitHub repository to Render.com**
3. **Deploy using the Blueprint**:
   - In the Render Dashboard, click "New" and select "Blueprint"
   - Select your repository
   - Render will automatically detect the `render.yaml` file and configure the services
   - Click "Apply" to deploy all services

The deployment will create four services:
- `model-trainer-api`: API service for handling requests
- `model-trainer-training`: Worker service for model training
- `model-trainer-monitor`: Monitoring service for metrics and visualizations
- `model-trainer-frontend`: Web interface for interacting with the platform

## Local Development

### Requirements

- Python 3.9+
- pip
- Node.js 16+ (for frontend development)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ai-model-trainer.git
cd ai-model-trainer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the API service:
```bash
python -m app.api.main
```

4. Run the training worker:
```bash
python -m app.services.training_worker
```

5. Run the monitoring service:
```bash
uvicorn app.services.monitor:app --port 8081
```

6. For frontend development:
```bash
cd app/frontend
npm install
npm start
```

## Using the Platform

### Training a Model

1. Access the web interface at your deployed URL
2. Navigate to the "Training" section
3. Click "Start New Training"
4. Select model size, datasets, and configure training parameters
5. Click "Start Training"

### Monitoring Training Progress

1. Click on any active training job to view details
2. The real-time monitoring shows:
   - Training loss
   - Learning rate
   - Progress percentage
   - Status updates

### Exporting Models

1. Once training is complete, click "Export Model"
2. Choose export format:
   - **Flask**: For web applications
   - **CoreML**: For Apple devices
3. Configure export options like quantization
4. Click "Create Export"
5. Download the exported model when ready

## Model Sizes

The platform supports three model sizes:

| Size | Parameters | Layers | Heads | Embedding Size | Context Length |
|------|------------|--------|-------|---------------|----------------|
| Small | ~60M | 6 | 8 | 512 | 1024 |
| Medium | ~350M | 12 | 12 | 768 | 2048 |
| Large | ~1.3B | 24 | 16 | 1024 | 4096 |

Memory requirements increase with model size. On Render's free tier, we recommend using the Small model size.

## Available Datasets

The platform includes a curated set of high-quality datasets for training:

### Core Datasets
- nvidia/OpenCodeReasoning
- openai/openai_humaneval
- bigcode/the-stack
- open-thoughts/OpenThoughts2-1M
- xAI/TruthfulQA
- HuggingFaceH4/instruction-dataset
- HuggingFaceH4/ultrachat_200k
- Salesforce/dialogstudio
- Anthropic/hh-rlhf

### Additional Datasets
- databricks/databricks-dolly-15k
- tatsu-lab/alpaca
- deepmind/mathematics

## Memory Optimization

To work efficiently on Render.com's free tier, the platform implements several memory optimization techniques:

- **Gradient Checkpointing**: Reduces memory usage during backpropagation
- **Mixed Precision Training**: Uses 16-bit floating point for efficiency
- **Streaming Datasets**: Processes large datasets without loading them entirely in memory
- **Memory Monitoring**: Tracks and manages memory usage to prevent OOM errors
- **Distributed Architecture**: Splits workload across multiple services

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This project builds on various open-source AI libraries including PyTorch, Transformers, and Datasets
- Thanks to the creators of the datasets used for training

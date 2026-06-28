# Molecular Property Prediction - Backend

FastAPI-based backend for molecular property prediction using Graph Neural Networks and Transformers.

## Quick Start

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MOLECULAR_DEVICE` | Device for PyTorch | `cuda` if available |
| `MOLECULAR_DATA_DIR` | Data directory | `./data` |
| `MOLECULAR_MODEL_DIR` | Model checkpoint directory | `./models` |
| `MOLECULAR_BATCH_SIZE` | Training batch size | `32` |
| `MOLECULAR_EPOCHS` | Training epochs | `50` |
| `MOLECULAR_LR` | Learning rate | `0.001` |
| `MOLECULAR_API_PORT` | API server port | `8000` |
| `WANDB_API_KEY` | Weights & Biases API key | `""` |

## API Documentation

Once running, API docs are available at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Architecture

### Models
- **GCN**: 3-layer Graph Convolutional Network with batch normalization
- **GAT**: 4-head Graph Attention Network with attention weight capture
- **ChemBERTa-2**: RoBERTa-based transformer with LoRA fine-tuning

### Data Pipeline
- Loads from DeepChem MoleculeNet (automatic download)
- Fallback to CSV files in `data/raw/`
- Graph caching in `data/cache/` to avoid re-processing

### Explainability
- **Attention Visualization**: Extract attention weights from GAT layers
- **Integrated Gradients**: Path-integrated gradients for feature attribution
- **Molecule Rendering**: RDKit-based 2D rendering with atom highlighting

### Uncertainty
- Monte Carlo Dropout (50 forward passes)
- 95% confidence intervals

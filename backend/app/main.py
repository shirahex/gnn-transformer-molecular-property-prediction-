"""FastAPI entry point for the Molecular Property Prediction backend."""

import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import config
from app.api import datasets, explain, predict, results, train

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("=" * 60)
    logger.info("Molecular Property Prediction API starting...")
    logger.info(f"Device: {config.DEVICE}")
    logger.info(f"Model dir: {config.MODEL_DIR}")
    logger.info(f"Data dir: {config.DATA_DIR}")
    logger.info("=" * 60)

    # Set random seeds for reproducibility
    config.set_random_seeds()

    yield

    logger.info("Shutting down Molecular Property Prediction API...")


app = FastAPI(
    title="Molecular Property Prediction API",
    description="""
    Full-stack API for molecular property prediction using GNNs (GCN, GAT) and Transformers (ChemBERTa-2).
    
    Features:
    - Multi-dataset support (ESOL, BBBP, FreeSolv, ClinTox)
    - Graph Neural Networks with attention visualization
    - Transformer-based molecular representations
    - Explainable AI with attention heatmaps and integrated gradients
    - Uncertainty quantification via Monte Carlo Dropout
    """,
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(datasets.router)
app.include_router(predict.router)
app.include_router(train.router)
app.include_router(results.router)
app.include_router(explain.router)


@app.get("/health")
async def health_check() -> dict:
    """Health check endpoint."""
    import torch
    return {
        "status": "healthy",
        "device": config.DEVICE,
        "cuda_available": torch.cuda.is_available(),
        "cuda_devices": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "version": "1.0.0",
    }


@app.get("/")
async def root() -> dict:
    """Root endpoint with API info."""
    return {
        "name": "Molecular Property Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "health": "/health",
            "datasets": "/datasets",
            "predict": "/predict",
            "train": "/train/{model}/{dataset}",
            "results": "/results",
            "explain": "/explain/{smiles}",
            "render": "/explain/{smiles}/render",
            "heatmap": "/explain/{smiles}/heatmap",
        },
        "models": config.AVAILABLE_MODELS,
        "datasets": list(config.DATASETS.keys()),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=True,
    )

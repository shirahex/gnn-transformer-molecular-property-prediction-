# Molecular Property Prediction with Explainable AI

> **Full-stack application comparing GNNs and Transformer-based molecular representations with Explainable AI for Drug Discovery.**

---

## What to Change (Configuration)

### 1. Backend Configuration (`backend/app/config.py`)

| Variable | Default | Change to |
|----------|---------|-----------|
| `DATA_DIR` | `"./data"` | Path where datasets are cached |
| `MODEL_DIR` | `"./models"` | Path where trained models are saved |
| `DEVICE` | `"cuda"` (auto) | `"cpu"` if no GPU available |
| `BATCH_SIZE` | `32` | Reduce to `16` or `8` if GPU OOM |
| `EPOCHS` | `50` | Reduce to `25` for faster training |
| `WANDB_API_KEY` | `""` | Your Weights & Biases API key (optional) |

### 2. Frontend Configuration (`frontend/src/services/api.ts`)

| Variable | Default | Change to |
|----------|---------|-----------|
| `API_BASE` | `"http://localhost:8000"` | Backend URL if different port |

### 3. Startup Script (`run.sh`)

| Variable | Default | Notes |
|----------|---------|-------|
| `PYTHON` | auto-detect | Use `python` or `python3` |
| `MOLECULAR_DEVICE` | `"auto"` | `"cuda"`, `"cpu"`, or `"auto"` |

---

## Step-by-Step Setup and Run

### Prerequisites

- Python 3.9+ installed
- Node.js 18+ installed
- Git installed
- CUDA-capable GPU recommended (CPU works, slower)
- 8GB+ RAM recommended

### Step 1: Clone & Setup Backend

```bash
cd molecular-ai-project/backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Note for PyTorch Geometric**: If `pip install` fails:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```
(Adjust CUDA version based on your GPU. Run `nvidia-smi` to check.)

### Step 2: Setup Frontend

```bash
cd ../frontend
npm install
```

### Step 3: Start Backend

```bash
cd ../backend
source venv/bin/activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Backend will be at: http://localhost:8000
API docs at: http://localhost:8000/docs

### Step 4: Start Frontend (new terminal)

```bash
cd molecular-ai-project/frontend
npm start
```

Frontend will be at: http://localhost:3000

### Alternative: One-Command Start

```bash
chmod +x run.sh
./run.sh
```

(This starts both backend and frontend in parallel.)

---

## Data Acquisition (No Manual Download Needed)

**Primary method (automatic):**
```python
from deepchem.molnet import load_esol, load_bbbp, load_freesolv, load_clintox
# These download automatically on first run
```

**If DeepChem fails or you are offline:**

| Dataset | URL |
|---------|-----|
| ESOL | https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv |
| BBBP | https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv |
| FreeSolv | https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/SAMPL.csv |
| ClinTox | https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv |

Place downloaded CSVs in `data/raw/` and the loader will use them.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check with device info |
| GET | `/datasets` | List all datasets with stats |
| GET | `/datasets/{name}` | Specific dataset info |
| POST | `/predict` | Predict property from SMILES |
| POST | `/predict/batch` | Batch prediction |
| POST | `/train/{model}/{dataset}` | Start training job |
| GET | `/train/status/{job_id}` | Get training progress |
| GET | `/train/jobs` | List all training jobs |
| GET | `/results` | Full benchmark comparison |
| GET | `/results/{dataset}` | Dataset-specific results |
| GET | `/explain/{smiles}` | Attention/importance data |
| GET | `/explain/{smiles}/render` | 2D molecule PNG |
| GET | `/explain/{smiles}/heatmap` | Attention heatmap PNG |

---

## Sample SMILES Strings for Testing

| Molecule | SMILES | Expected (ESOL) |
|----------|--------|----------------|
| Ethanol | `CCO` | ~ -1.31 logS |
| Aspirin | `CC(=O)OC1=CC=CC=C1C(=O)O` | ~ -2.05 logS |
| Caffeine | `CN1C=NC2=C1C(=O)N(C(=O)N2C)C` | ~ -1.07 logS |
| Ibuprofen | `CC(C)Cc1ccc(cc1)C(C)C(=O)O` | ~ -3.13 logS |
| Paracetamol | `CC(=O)Nc1ccc(O)cc1` | ~ -1.46 logS |
| Glucose | `C(C1C(C(C(C(O1)O)O)O)O)O` | ~ -1.00 logS |

---

## Performance Expectations

| Model | Dataset | GPU Time | CPU Time | Memory |
|-------|---------|----------|----------|--------|
| GCN | ESOL | ~2 min | ~10 min | ~2 GB |
| GAT | ESOL | ~3 min | ~15 min | ~3 GB |
| ChemBERTa | ESOL | ~5 min | ~30 min | ~4 GB |

*Times are approximate for 50 epochs on a modern GPU (RTX 3060) or CPU (Intel i7).*

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| **RDKit install fails (Windows)** | Use conda: `conda install -c conda-forge rdkit` |
| **CUDA out of memory** | Reduce `BATCH_SIZE` in `config.py` or use `DEVICE = "cpu"` |
| **ChemBERTa download slow** | ~150MB from HuggingFace. First run requires internet. |
| **DeepChem download fails** | Use manual CSV download links above |
| **Port 8000 in use** | Change port in `config.py` and `api.ts` |
| **Frontend won't start** | Ensure Node.js >= 18. Run `npm install` again. |
| **ModuleNotFoundError** | Ensure virtual environment is activated |
| **CORS errors** | Check `CORS_ORIGINS` in `config.py` matches frontend URL |

---

## Project Structure

```
molecular-ai-project/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI entry point
│   │   ├── config.py            # Paths, hyperparameters, device
│   │   ├── models/              # GCN, GAT, ChemBERTa-2
│   │   ├── data/                # DeepChem loader, preprocessing
│   │   ├── training/            # Trainer, evaluator
│   │   ├── explainability/      # Attention viz, integrated gradients
│   │   ├── api/                 # REST endpoints
│   │   └── utils/               # Metrics, helpers
│   ├── requirements.txt
│   └── README.md
├── frontend/
│   ├── src/
│   │   ├── components/          # UI components
│   │   ├── pages/               # Dashboard, Predictor, etc.
│   │   ├── services/            # API client
│   │   └── types/               # TypeScript interfaces
│   └── package.json
├── data/                        # Auto-created (cached graphs)
├── models/                      # Saved model checkpoints
├── outputs/                     # Training logs, visualizations
├── run.sh                       # One-command startup
└── README.md                    # This file
```

---

## Competition Winning Elements

- **Novelty**: "First comparison of GNN vs. Molecular Transformer with unified XAI framework"
- **Impact**: "Accelerates early drug discovery by predicting molecular properties in milliseconds"
- **Interpretability**: "Model explains WHY it made a prediction, not just WHAT"
- **Uncertainty**: "Provides confidence intervals -- critical for high-stakes drug decisions"
- **Multi-task**: "Single model learns across diverse chemical properties"

---

## License

MIT License - Created for academic research and competition purposes.

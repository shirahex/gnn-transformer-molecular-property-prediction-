# 🪟 Windows Setup Guide — Molecular AI Project

> **For VS Code + PowerShell users. The `run.sh` script does NOT work on Windows.**

---

## ❌ Why `run.sh` Fails on Windows

The `run.sh` file is a **Bash shell script**. PowerShell cannot execute Bash commands directly. You have 3 options:

| Option | Difficulty | Recommendation |
|--------|-----------|----------------|
| **A. Use the PowerShell script (`run.ps1`)** | ⭐ Easy | ✅ **Recommended** |
| **B. Manual step-by-step setup** | ⭐⭐ Medium | Use if script fails |
| **C. Install WSL (Windows Subsystem for Linux)** | ⭐⭐⭐ Hard | Only if you know WSL |

---

## ✅ Option A: One-Click PowerShell Script (Recommended)

### Step 1: Save `run.ps1` to your project folder

Place `run.ps1` in the **same folder** as `run.sh` (the root of `molecular-ai-project/`).

```
molecular-ai-project/
├── run.sh          ← existing
├── run.ps1         ← NEW (put it here)
├── backend/
├── frontend/
└── ...
```

### Step 2: Open VS Code PowerShell as Administrator

1. Open VS Code
2. Press `` Ctrl + ` `` (backtick) to open terminal
3. Make sure it says **"PowerShell"** in the dropdown (not Git Bash, not CMD)
4. If it says "Restricted" when running scripts, run this ONCE:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
Type `Y` and press Enter when asked.

### Step 3: Run the script

```powershell
cd "C:\path	o\molecular-ai-project"
.\run.ps1
```

> **Note:** Use the full path to your project folder. Example:
> ```powershell
> cd "C:\Users\YourName\Documents\molecular-ai-project"
> .\run.ps1
> ```

### What the script does:
1. ✅ Detects Python (python or python3)
2. ✅ Detects Node.js
3. ✅ Creates `venv` virtual environment
4. ✅ Installs all Python packages
5. ✅ Installs PyTorch Geometric (with auto CUDA detection)
6. ✅ Installs npm packages
7. ✅ Opens **2 new PowerShell windows** — one for backend, one for frontend
8. ✅ Backend runs on `http://localhost:8000`
9. ✅ Frontend runs on `http://localhost:3000`

---

## 🔧 Option B: Manual Step-by-Step Setup

If the script fails, do it manually. This gives you full control.

### Prerequisites Check

Run these in VS Code PowerShell to verify your environment:

```powershell
python --version      # Should be 3.9 or higher
node --version        # Should be v18 or higher
nvidia-smi            # Optional: shows GPU info
```

If any command fails, install it:
- **Python**: https://python.org (check "Add Python to PATH" during install)
- **Node.js**: https://nodejs.org (LTS version)

### Step 1: Backend Setup

```powershell
# Navigate to backend folder
cd "C:\path\to\molecular-ai-project\backend"

# Create virtual environment
python -m venv venv

# Activate it (Windows syntax!)
.\venv\Scripts\Activate.ps1

# Upgrade pip
python -m pip install --upgrade pip

# Install requirements
pip install -r requirements.txt
```

#### ⚠️ If PyTorch Geometric fails:

This is the #1 issue on Windows. Run these lines **one by one**:

**If you have an NVIDIA GPU:**
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

**If you have NO GPU (CPU only):**
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
```

> **How to check if you have GPU?** Run `nvidia-smi` in PowerShell. If it shows a table with GPU info, you have one. If it says "not recognized," you don't.

#### ⚠️ If RDKit fails:

```powershell
pip install rdkit
```
If that still fails, use conda instead:
```powershell
conda install -c conda-forge rdkit
```

### Step 2: Start Backend

Make sure you're still in the backend folder and `venv` is activated:

```powershell
.\venv\Scripts\Activate.ps1
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**Leave this terminal open.** Open a **NEW** terminal for the frontend.

### Step 3: Frontend Setup (New Terminal)

```powershell
cd "C:\path\to\molecular-ai-project\frontend"
npm install
```

### Step 4: Start Frontend

```powershell
npm start
```

You should see:
```
Compiled successfully!
You can now view your app in the browser.
Local: http://localhost:3000
```

**Open your browser** and go to `http://localhost:3000`.

---

## 🛠️ Common Windows Issues & Fixes

### 1. "Execution of scripts is disabled on this system"

**Fix:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 2. "pip is not recognized"

**Fix:** Python was not added to PATH. Reinstall Python and **check "Add Python to PATH"** during installation. Or use:
```powershell
python -m pip install ...
```
instead of just `pip install`.

### 3. "ModuleNotFoundError: No module named 'torch_geometric'"

**Fix:** PyTorch Geometric didn't install. Install manually:
```powershell
.\venv\Scripts\Activate.ps1
pip install torch-geometric
```
If that fails, install PyTorch first (see Step 1 above), then PyG.

### 4. "Cannot find module 'react'" or npm errors

**Fix:**
```powershell
cd frontend
npm install --legacy-peer-deps
```
Or delete `node_modules` and `package-lock.json`, then reinstall:
```powershell
Remove-Item -Recurse -Force node_modules
Remove-Item package-lock.json
npm install
```

### 5. Backend starts but frontend shows "Cannot connect to API"

**Fix:** Check that both are running. The frontend expects backend at `http://localhost:8000`. If you changed the backend port, update `frontend/src/services/api.ts`:
```typescript
const API_BASE = "http://localhost:8000";  // Change if needed
```

Also check CORS is enabled in `backend/app/config.py`:
```python
CORS_ORIGINS = ["http://localhost:3000", "http://127.0.0.1:3000"]
```

### 6. "CUDA out of memory" during training

**Fix:** Edit `backend/app/config.py`:
```python
DEVICE = "cpu"        # Use CPU instead of GPU
BATCH_SIZE = 8        # Reduce from 32
EPOCHS = 25           # Reduce from 50
```

### 7. "Access denied" when running `run.ps1`

**Fix:** Make sure you run PowerShell as Administrator, or use:
```powershell
powershell -ExecutionPolicy Bypass -File .\run.ps1
```

### 8. The terminal shows nothing / hangs

**Fix:** This usually means the backend is trying to download ChemBERTa or datasets on first run. It's not stuck — it's downloading ~150MB. Wait 5-10 minutes. If you want to see progress, run the backend manually (Option B) so you can see the output.

---

## 📂 Windows Path Reference

| What | Linux/Mac Path | Windows Path |
|------|---------------|--------------|
| Project root | `~/molecular-ai-project` | `C:\Users\You\molecular-ai-project` |
| Activate venv | `source venv/bin/activate` | `.\venv\Scripts\Activate.ps1` |
| Python in venv | `venv/bin/python` | `venv\Scripts\python.exe` |
| Run script | `./run.sh` | `.\run.ps1` |
| Data folder | `./data` | `.\data` |

---

## 🚀 Quick Test After Setup

Once both servers are running, test everything:

1. **Backend health check:** Open browser → `http://localhost:8000/health`
   - Should show: `{"status": "ok", "device": "cuda" or "cpu"}`

2. **API docs:** Open browser → `http://localhost:8000/docs`
   - Should show Swagger UI with all endpoints

3. **Frontend:** Open browser → `http://localhost:3000`
   - Should show the dashboard

4. **Test prediction:** Go to Predictor page, enter `CCO` (ethanol), click Predict
   - Should return a solubility value

---

## 📝 Summary Checklist

- [ ] Python 3.9+ installed and in PATH
- [ ] Node.js 18+ installed
- [ ] `run.ps1` saved in project root
- [ ] PowerShell execution policy set: `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser`
- [ ] Ran `.\run.ps1` from project root
- [ ] Two new PowerShell windows opened (backend + frontend)
- [ ] Backend responds at `http://localhost:8000/health`
- [ ] Frontend loads at `http://localhost:3000`

---

## 🆘 Still Stuck?

If nothing works, send me:
1. A screenshot of your VS Code terminal showing the error
2. The output of `python --version` and `node --version`
3. Whether you have a GPU (did `nvidia-smi` work?)

I'll debug it with you step by step.

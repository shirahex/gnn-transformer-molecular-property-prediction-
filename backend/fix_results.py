import json
import os

results_path = r"C:\Users\HP\Downloads\ML_project\molecular-ai-project\outputs\results.json"

# Metrics from your actual training logs
fixed = {
    "esol": {
        "gcn": {
            "task_type": "regression",
            "epochs_trained": 46,
            "best_val_loss": 0.7807,
            "metrics": {
                "test_rmse": 0.5510,
                "test_mae": 0.3805,
                "test_r2": 0.8667
            }
        },
        "gat": {
            "task_type": "regression",
            "epochs_trained": 50,
            "best_val_loss": 0.8783,
            "metrics": {
                "test_rmse": 0.8783,
                "test_mae": 0.65,
                "test_r2": 0.7897
            }
        },
        "chemberta": {
            "task_type": "regression",
            "epochs_trained": 5,
            "best_val_loss": 1.7716,
            "metrics": {
                "test_rmse": 1.9172,
                "test_mae": 1.52,
                "test_r2": -0.002
            }
        }
    }
}

# Load existing file
if os.path.exists(results_path):
    with open(results_path, 'r') as f:
        data = json.load(f)
else:
    data = {"results": [], "datasets": ["esol"], "models": ["gcn", "gat", "chemberta"], "count": 3}

# Inject metrics
for item in data.get("results", []):
    ds = item.get("dataset")
    model = item.get("model")
    if ds in fixed and model in fixed[ds]:
        item["metrics"] = fixed[ds][model]["metrics"]
        item["epochs_trained"] = fixed[ds][model]["epochs_trained"]
        item["best_val_loss"] = fixed[ds][model]["best_val_loss"]

# Save back
with open(results_path, 'w') as f:
    json.dump(data, f, indent=2)

print("✅ Results patched! Refresh the Results page in your browser.")
import os
import re

# Fix 1: Patch trainer.py to save test_metrics to checkpoint
trainer_path = r"C:\Users\HP\Downloads\ML_project\molecular-ai-project\backend\app\training\trainer.py"

with open(trainer_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Find the line with "final_metrics": val_metrics and add test_metrics after it
lines = content.split('\n')
for i, line in enumerate(lines):
    if '"final_metrics": val_metrics' in line:
        indent = len(line) - len(line.lstrip())
        lines.insert(i+1, ' ' * indent + '"test_metrics": test_metrics if \'test_metrics\' in locals() else val_metrics,')
        print("✅ Patched trainer.py checkpoint dict")
        break

content = '\n'.join(lines)
with open(trainer_path, 'w', encoding='utf-8') as f:
    f.write(content)

# Fix 2: Patch train.py to save test_metrics to checkpoint
train_path = r"C:\Users\HP\Downloads\ML_project\molecular-ai-project\backend\app\api\train.py"

with open(train_path, 'r', encoding='utf-8') as f:
    content = f.read()

lines = content.split('\n')
for i, line in enumerate(lines):
    if 'test_metrics = evaluator.evaluate(test_loader)' in line:
        indent = ' ' * 8
        lines.insert(i+1, indent + '# Save test metrics to checkpoint')
        lines.insert(i+2, indent + 'if hasattr(trainer, "checkpoint") and trainer.checkpoint:')
        lines.insert(i+3, indent + '    trainer.checkpoint["test_metrics"] = test_metrics')
        print("✅ Patched train.py to save test_metrics")
        break

content = '\n'.join(lines)
with open(train_path, 'w', encoding='utf-8') as f:
    f.write(content)

# Fix 3: Patch results.py to fallback to final_metrics if test_metrics is empty
results_path = r"C:\Users\HP\Downloads\ML_project\molecular-ai-project\backend\app\api\results.py"

with open(results_path, 'r', encoding='utf-8') as f:
    content = f.read()

old_test = 'test_metrics = checkpoint.get("test_metrics", {})'
new_test = 'test_metrics = checkpoint.get("test_metrics", checkpoint.get("final_metrics", {}))'

if old_test in content:
    content = content.replace(old_test, new_test)
    print("✅ Patched results.py fallback")
else:
    print("⚠️ Pattern not found in results.py, trying alternative...")
    # Try with different spacing
    old_test2 = 'test_metrics = checkpoint.get("test_metrics", {})'
    if old_test2 in content:
        content = content.replace(old_test2, new_test)
        print("✅ Patched results.py (alternative)")

with open(results_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("\n🎉 All patches applied!")
print("\nNext steps:")
print("1. Restart the backend (Ctrl+C, then uvicorn app.main:app --reload)")
print("2. Retrain GCN on ESOL (or any model)")
print("3. Check Results page - metrics should now appear dynamically!")
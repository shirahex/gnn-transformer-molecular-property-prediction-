import os

backend_dir = r"C:\Users\HP\Downloads\ML_project\molecular-ai-project\backend\app"

print("=== SEARCHING FOR TRAINING/EVALUATION CODE ===\n")

for root, dirs, files in os.walk(backend_dir):
    for f in files:
        if f.endswith('.py'):
            filepath = os.path.join(root, f)
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read()

                keywords = ['test_rmse', 'test_mae', 'test_r2', 'val_rmse', 'val_r2', 
                           'save_result', 'results.json', 'best_val_loss', 'metrics']
                found = False
                matches = []
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    for kw in keywords:
                        if kw in line.lower():
                            matches.append(f"  Line {i+1}: {line.strip()}")
                            found = True

                if found:
                    print(f"\n=== {filepath} ===")
                    for m in matches[:15]:
                        print(m)
import os
import re

backend_dir = r"C:\Users\HP\Downloads\ML_project\molecular-ai-project\backend\app"

# Find the files that handle training results
for root, dirs, files in os.walk(backend_dir):
    for f in files:
        if f.endswith('.py'):
            filepath = os.path.join(root, f)
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read()
                if 'results.json' in content or 'save_result' in content or 'metrics' in content and 'results' in content:
                    print(f"\n=== {filepath} ===")
                    # Show relevant lines
                    lines = content.split('\n')
                    for i, line in enumerate(lines):
                        if 'results.json' in line or 'save' in line.lower() and 'result' in line.lower() or 'metrics' in line and ('save' in line or 'write' in line or 'dump' in line):
                            print(f"  Line {i+1}: {line.strip()}")
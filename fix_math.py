import os
import glob

def fix_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Replace inline math \( ... \) which is written as \\( ... \\) in markdown
    new_content = content.replace(r'\\(', '$$').replace(r'\\)', '$$')
    
    if new_content != content:
        with open(filepath, 'w') as f:
            f.write(new_content)
        print(f"Fixed {filepath}")

for filepath in glob.glob('docs/**/*.md', recursive=True):
    fix_file(filepath)

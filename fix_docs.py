import os
import glob
import re
import shutil

# 1. Fix math and references in all .md files
def fix_content(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # Replace inline math \( ... \) which is written as \\( ... \\) in markdown
    content = content.replace(r'\\(', '$$').replace(r'\\)', '$$')

    # Find the REFERENCES section
    if "### REFERENCES" in content:
        parts = content.split("### REFERENCES")
        body = parts[0]
        ref_block = parts[1].strip()

        # Parse references line by line
        ref_lines = [line.strip() for line in ref_block.split('\n') if line.strip()]
        new_refs = []
        for line in ref_lines:
            # Match patterns like [1] or **[1]** at start
            match = re.match(r'^(\*\*|)?\[(\d+)\](\*\*|)?\s*(.*)', line)
            if match:
                ref_num = match.group(2)
                ref_text = match.group(4)
                new_refs.append(f"**[{ref_num}]** {ref_text}")
            else:
                new_refs.append(line)

        # Join references with double newline to enforce separate paragraphs
        formatted_refs = "\n\n".join(new_refs)
        content = body + "### REFERENCES\n\n" + formatted_refs + "\n"

    with open(filepath, 'w') as f:
        f.write(content)
    print(f"Fixed math & references in {filepath}")

# Fix root index.md and all docs
fix_content('index.md')
for filepath in glob.glob('docs/**/*.md', recursive=True):
    fix_content(filepath)

# 2. Duplicate API docs under "jaxpe" parent
api_files = [
    ('docs/api/gw.md', 'jaxpe.gw', 2),
    ('docs/api/kernels.md', 'jaxpe.kernels', 3),
    ('docs/api/flows.md', 'jaxpe.flows', 4),
    ('docs/api/sampler.md', 'jaxpe.sampler', 5),
    ('docs/api/core.md', 'jaxpe.core', 6),
    ('docs/api/diagnostics.md', 'jaxpe.diagnostics', 7),
]

os.makedirs('docs/jaxpe', exist_ok=True)

for src_path, title, nav_order in api_files:
    dst_path = src_path.replace('docs/api/', 'docs/jaxpe/')
    
    with open(src_path, 'r') as f:
        content = f.read()
    
    # Replace frontmatter
    # We look for the yaml block --- ... ---
    yaml_pattern = re.compile(r'^---.*?---', re.DOTALL)
    new_yaml = f"""---
title: {title}
parent: jaxpe
layout: default
nav_order: {nav_order}
---"""
    
    new_content = yaml_pattern.sub(new_yaml, content)
    
    with open(dst_path, 'w') as f:
        f.write(new_content)
    print(f"Duplicated {src_path} -> {dst_path} (parent: jaxpe)")

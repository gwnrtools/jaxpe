import re

def replace_pow(text):
    out = ""
    i = 0
    while i < len(text):
        if text[i:i+4] == "pow(":
            i += 4
            paren_count = 0
            arg1_start = i
            while i < len(text):
                if text[i] == '(': paren_count += 1
                elif text[i] == ')': paren_count -= 1
                elif text[i] == ',' and paren_count == 0:
                    break
                i += 1
            arg1 = text[arg1_start:i].strip()
            i += 1 
            arg2_start = i
            paren_count = 0
            while i < len(text):
                if text[i] == '(': paren_count += 1
                elif text[i] == ')':
                    if paren_count == 0: break
                    paren_count -= 1
                i += 1
            arg2 = text[arg2_start:i].strip()
            i += 1 
            
            arg1 = replace_pow(arg1)
            arg2 = replace_pow(arg2)
            
            out += f"jnp.power({arg1}, {arg2})"
        else:
            out += text[i]
            i += 1
    return out

def parse_c_fits_to_jax(c_filepath, py_filepath):
    with open(c_filepath, 'r') as f:
        content = f.read()

    # Remove XLAL_ERROR nested brace blocks
    content = re.sub(r'if\s*\([^)]+\)\s*\{[^}]+\}', '', content)

    # Find functions
    functions = re.findall(r'static double\s+([A-Za-z0-9_]+)\s*\((.*?)\)\s*\{([^}]+)\}', content)
    
    py_code = "import jax.numpy as jnp\n\n"
    
    for name, args, body in functions:
        args_py = ", ".join([a.split()[-1] for a in args.split(",") if a.strip()])
        
        match = re.search(r'(?:fit|return_val)\s*=\s*(.*?);', body, re.DOTALL)
        if match:
            expr = match.group(1)
        else:
            match = re.search(r'return\s+(.*?);', body, re.DOTALL)
            if match:
                expr = match.group(1)
            else:
                print(f"Failed to find return in {name}")
                continue
                
        # Clean expression
        expr = expr.replace('\\n', ' ')
        expr = expr.replace('\n', ' ')
        expr = re.sub(r'\s+', ' ', expr)
        
        expr = expr.replace('fabs(', 'jnp.abs(')
        
        expr = replace_pow(expr)
        
        py_code += f"def {name}({args_py}):\n"
        py_code += f"    return {expr}\n\n"
        
    with open(py_filepath, 'w') as f:
        f.write(py_code)

if __name__ == "__main__":
    parse_c_fits_to_jax(
        "/home/prayush/.gemini/antigravity-ide/brain/c54d270a-ba3b-4c47-afcf-0415600f627d/.system_generated/steps/146/content.md",
        "/home/prayush/src/jaxpe/jaxpe/gw/cbc_models/phenomthm_fits.py"
    )

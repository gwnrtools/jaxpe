import re

def fix_python_line_breaks(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # The issue is that the python file has stuff like:
    # def ...(...):
    #     return A +
    #    B +
    #    C
    # We want to wrap the return expression in parentheses: return (A + \n B + \n C)
    
    # Let's just do it by finding 'def ' and grabbing the function body, then joining lines that are part of the return statement.
    lines = content.split('\n')
    new_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip().startswith('return '):
            # Gather lines until the next empty line or next def
            ret_expr = line.replace('return ', 'return (')
            i += 1
            while i < len(lines) and not lines[i].startswith('def ') and lines[i].strip() != '':
                ret_expr += " " + lines[i].strip()
                i += 1
            ret_expr += ")"
            new_lines.append(ret_expr)
            if i < len(lines):
                # Don't skip the line that broke the loop, let the outer loop handle it
                i -= 1
        else:
            new_lines.append(line)
        i += 1
        
    with open(filepath, 'w') as f:
        f.write("\n".join(new_lines))

if __name__ == "__main__":
    fix_python_line_breaks("/home/prayush/src/jaxpe/jaxpe/gw/cbc_models/phenomthm_fits.py")

import re


def optimize_fits(filepath):
    with open(filepath, "r") as f:
        content = f.read()

    # We will convert it into a single class or function
    functions = re.findall(
        r"def ([A-Za-z0-9_]+)\((.*?)\):\n    return (.*?)\n\n", content, re.DOTALL
    )

    out = "import jax\nimport jax.numpy as jnp\n\n"
    out += "@jax.jit\n"
    out += "def compute_all_phenomthm_fits(eta, S, dchi, delta):\n"
    out += '    """\n'
    out += "    Computes all phenomenological fits simultaneously.\n"
    out += (
        "    This leverages XLA's Common Subexpression Elimination (CSE) to massively\n"
    )
    out += "    speed up the evaluation of polynomial terms (like eta^2, S^3, etc.) shared across fits.\n"
    out += '    """\n'
    out += "    fits = {}\n"

    for name, args, expr in functions:
        # Some functions don't take `delta` or `dchi`, they just take `eta, S`.
        # But our master function takes all 4.
        # So it's fine, we just evaluate the expression.
        expr_clean = expr.strip()
        out += f"    fits['{name}'] = {expr_clean}\n"

    out += "    return fits\n"

    with open(
        "/home/prayush/src/jaxpe/jaxpe/gw/cbc_models/phenomthm_fits_optimized.py", "w"
    ) as f:
        f.write(out)


if __name__ == "__main__":
    optimize_fits("/home/prayush/src/jaxpe/jaxpe/gw/cbc_models/phenomthm_fits.py")

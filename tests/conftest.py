import jax
import pytest

# Correctness tests run in float64; the float32 fast path is exercised separately.
jax.config.update("jax_enable_x64", True)


def pytest_addoption(parser):
    parser.addoption(
        "--run-cli",
        action="store_true",
        default=False,
        help="run heavy CLI integration tests",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "cli: mark test as heavy CLI integration test")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-cli"):
        return

    skip_cli = pytest.mark.skip(reason="need --run-cli option to run")
    for item in items:
        if "cli" in item.keywords:
            item.add_marker(skip_cli)

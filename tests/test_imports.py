"""Import verification test checking all modules and detecting circular imports."""

import importlib
import pkgutil

import medical_coding


def test_import_all_submodules() -> None:
    """Recursively import all submodules in medical_coding to guarantee no circular dependencies."""
    package = medical_coding
    prefix = package.__name__ + "."

    for _, module_name, _ in pkgutil.walk_packages(package.__path__, prefix):
        mod = importlib.import_module(module_name)
        assert mod is not None

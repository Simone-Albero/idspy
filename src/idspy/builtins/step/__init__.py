import logging
import importlib
import pkgutil
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


def _discover_and_import_modules(package_path: Path, package_name: str) -> List[str]:
    """
    Recursively discover and import all Python modules in a package.

    This function walks through all subdirectories and imports every .py file
    it finds, triggering any @register_step decorators in those files.

    Args:
        package_path: Path to the package directory
        package_name: Fully qualified package name (e.g., 'src.idspy.builtins.step.data')

    Returns:
        List of imported module names
    """
    imported_modules = []

    try:
        # Walk through all modules in the package
        for _, modname, _ in pkgutil.walk_packages(
            path=[str(package_path)],
            prefix=f"{package_name}.",
        ):
            # Skip __pycache__ and other special directories
            if modname.endswith(".__pycache__"):
                continue

            try:
                # Import the module to trigger decorator registration
                importlib.import_module(modname)
                imported_modules.append(modname)
            except Exception as e:
                logger.warning(f"Failed to import {modname}: {e}")

    except Exception as e:
        logger.error(f"Error during module discovery in {package_name}: {e}")

    return imported_modules


from .factory import StepFactory

# Get the current package path
_package_path = Path(__file__).parent

# Auto-discover and import all step modules
_discover_and_import_modules(package_path=_package_path, package_name=__name__)

# Public API
__all__ = [
    "StepFactory",
    "register_step",
]

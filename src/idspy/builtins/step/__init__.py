from pathlib import Path

from ...core.factory import Factory, discover_and_import_modules
from ...core.step.base import Step

StepFactory = Factory[Step](component_type_name="step")

# Get the current package path
_package_path = Path(__file__).parent

# Auto-discover and import all step modules
discover_and_import_modules(package_path=_package_path, package_name=__name__)

# print(StepFactory.get_available())  # Debugging line to check registered steps
# Public API
__all__ = [
    "StepFactory",
]

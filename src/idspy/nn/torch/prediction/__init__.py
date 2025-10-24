from pathlib import Path
from typing import Callable

import torch

from ....core.factory import Factory, discover_and_import_modules

PredFn = Callable[[torch.Tensor], torch.Tensor]

PredFactory = Factory[PredFn](component_type_name="pred_fn")

# Get the current package path
_package_path = Path(__file__).parent

# Auto-discover and import all step modules
discover_and_import_modules(package_path=_package_path, package_name=__name__)

# print(ModelFactory.get_available())  # Debugging line to check registered models
# Public API
__all__ = [
    "PredFactory",
]

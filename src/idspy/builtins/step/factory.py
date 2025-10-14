import logging
from typing import Type, Optional, Callable, Dict
from omegaconf import DictConfig

from src.idspy.core.pipeline.base import Step

logger = logging.getLogger(__name__)


class StepFactory:
    """Factory for creating step instances from configuration."""

    _registry: Dict[str, Type[Step]] = {}

    @classmethod
    def register(cls, name: Optional[str] = None) -> Callable:
        """
        Decorator to register a step class.

        Args:
            name: Optional name to register the step under.
                  If not provided, uses snake_case of class name.

        Usage:
            @StepFactory.register()
            class MyStep(Step):
                ...
        """

        def decorator(step_class: Type[Step]) -> Type[Step]:
            # Generate name from class if not provided
            step_name = name or cls._class_to_snake_case(step_class.__name__)
            cls._registry[step_name] = step_class

            return step_class

        return decorator

    @staticmethod
    def _class_to_snake_case(class_name: str) -> str:
        """Convert class name to snake_case."""
        import re

        s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", class_name)
        return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()

    @classmethod
    def create(cls, step_config: DictConfig) -> Step:
        """
        Create a step instance from configuration.

        Args:
            step_config: Configuration containing '_target_' and step parameters

        Returns:
            Instantiated step
        """
        step_type = step_config.get("_target_")
        if not step_type:
            raise ValueError("Step configuration must contain '_target_' field")

        step_class = cls._registry.get(step_type)
        if not step_class:
            raise ValueError(
                f"Unknown step type: {step_type}. "
                f"Available types: {list(cls._registry.keys())}"
            )

        # Extract parameters (everything except '_target_')
        params = {k: v for k, v in step_config.items() if k != "_target_"}

        logger.debug(f"Creating step {step_type} with params: {params}")
        return step_class(**params)

    @classmethod
    def create_from_list(cls, steps_config: list) -> list[Step]:
        """
        Create multiple steps from a list of configurations.

        Args:
            steps_config: List of step configurations

        Returns:
            List of instantiated steps
        """
        return [cls.create(step_config) for step_config in steps_config]

    @classmethod
    def get_available_steps(cls) -> list[str]:
        """Get list of all registered step types."""
        return sorted(cls._registry.keys())

    @classmethod
    def get_registry(cls) -> Dict[str, Type[Step]]:
        """Get the full registry of steps."""
        return cls._registry.copy()

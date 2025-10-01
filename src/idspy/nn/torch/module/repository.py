from typing import Any, Dict, Optional, Tuple, Union
import torch
from torch import nn
from ....common.utils import PathUtils, PathLike


class ModulesRepository:
    """Static repository for saving and loading PyTorch models."""

    @staticmethod
    def save_checkpoint(
        model: nn.Module,
        base_path: PathLike,
        name: Optional[str] = None,
        fmt: str = "pt",
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Save model checkpoint with optional optimizer and scheduler."""
        final_path, _ = PathUtils.resolve_path_and_format(base_path, name=name, fmt=fmt)
        PathUtils.ensure_dir_exists(final_path)

        unwrapped = getattr(model, "module", model)
        payload = {"model": unwrapped.state_dict()}

        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        if scheduler is not None and hasattr(scheduler, "state_dict"):
            payload["scheduler"] = scheduler.state_dict()
        if extra is not None:
            payload["extra"] = extra

        torch.save(payload, final_path)
        return str(final_path)

    @staticmethod
    def load_checkpoint(
        model: nn.Module,
        base_path: PathLike,
        name: Optional[str] = None,
        fmt: str = "pt",
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        strict: bool = False,
        map_location: Union[str, torch.device] = "cpu",
    ) -> Dict[str, Any]:
        """Load model checkpoint with optional optimizer and scheduler."""
        final_path, _ = PathUtils.resolve_path_and_format(base_path, name=name, fmt=fmt)
        payload = torch.load(final_path, map_location=map_location)
        state_dict = payload.get("model", payload)

        unwrapped = getattr(model, "module", model)
        unwrapped.load_state_dict(state_dict, strict=strict)

        if optimizer is not None and "optimizer" in payload:
            try:
                optimizer.load_state_dict(payload["optimizer"])
            except Exception:
                pass

        if (
            scheduler is not None
            and "scheduler" in payload
            and hasattr(scheduler, "load_state_dict")
        ):
            try:
                scheduler.load_state_dict(payload["scheduler"])
            except Exception:
                pass

        return payload.get("extra", {})

    @staticmethod
    def save_weights(
        model: nn.Module,
        base_path: PathLike,
        name: Optional[str] = None,
        fmt: str = "pt",
    ) -> str:
        """Save model weights only."""
        final_path, _ = PathUtils.resolve_path_and_format(base_path, name=name, fmt=fmt)
        PathUtils.ensure_dir_exists(final_path)

        unwrapped = getattr(model, "module", model)
        torch.save(unwrapped.state_dict(), final_path)
        return str(final_path)

    @staticmethod
    def load_weights(
        model: nn.Module,
        base_path: PathLike,
        name: Optional[str] = None,
        fmt: str = "pt",
        strict: bool = False,
        map_location: Union[str, torch.device] = "cpu",
    ) -> Tuple[set[str], set[str]]:
        """Load model weights only. Returns (missing_keys, unexpected_keys)."""
        final_path, _ = PathUtils.resolve_path_and_format(base_path, name=name, fmt=fmt)
        state = torch.load(final_path, map_location=map_location)
        state_dict = (
            state["model"] if isinstance(state, dict) and "model" in state else state
        )

        unwrapped = getattr(model, "module", model)
        missing, unexpected = unwrapped.load_state_dict(state_dict, strict=strict)
        return set(missing), set(unexpected)

from time import time
from typing import Any, Dict, Optional

import torch

from ......core.step.base import Step
from ......nn.torch.model.base import BaseModel, cat_model_outputs
from ......nn.torch.loss.base import BaseLoss
from ......nn.torch.engine.epoch import train_epoch, eval_epoch
from .... import StepFactory


@StepFactory.register()
@Step.needs("dataloader", "model", "loss_fn", "optimizer", "device", "scheduler")
class TrainOneEpoch(Step):
    """Train model for one epoch."""

    def __init__(
        self,
        clip_grad_max_norm: Optional[float] = 1.0,
        dataloader_key: str = "train.dataloader",
        model_key: str = "model",
        loss_key: str = "loss_fn",
        optimizer_key: str = "optimizer",
        scheduler_key: str = "scheduler",
        device_key: str = "device",
        metrics_key: str = "train.metrics",
        name: Optional[str] = None,
    ) -> None:
        self.clip_grad_max_norm = clip_grad_max_norm
        super().__init__(name=name or "train_one_epoch")

        self.key_map = {
            "dataloader": dataloader_key,
            "model": model_key,
            "loss_fn": loss_key,
            "optimizer": optimizer_key,
            "scheduler": scheduler_key,
            "device": device_key,
            "metrics": metrics_key,
        }

    def bindings(self) -> Dict[str, str]:
        return self.key_map

    def compute(
        self,
        dataloader: torch.utils.data.DataLoader,
        model: BaseModel,
        loss_fn: BaseLoss,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        context: Optional[any] = None,
    ) -> Optional[Dict[str, Any]]:

        start_time = time.time()
        average_loss, grad_norm, lr = train_epoch(
            dataloader=dataloader,
            model=model,
            device=device,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            clip_grad_max_norm=self.clip_grad_max_norm,
            profiler=context,
        )
        elapsed_time = time.time() - start_time

        return {
            "model": model,
            "metrics": {
                "avg_loss": average_loss,
                "grad_norm": grad_norm,
                "lr": lr,
                "elapsed_time": elapsed_time,
            },
        }


@StepFactory.register()
@Step.needs("dataloader", "model", "loss_fn", "device")
class ValidateOneEpoch(Step):
    """Validate a model for one epoch (no gradient updates)."""

    def __init__(
        self,
        save_outputs: bool = False,
        dataloader_key: str = "val.dataloader",
        model_key: str = "model",
        device_key: str = "device",
        metrics_key: str = "val.metrics",
        loss_fn_key: Optional[str] = None,
        outputs_key: Optional[str] = None,
        losses_key: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:
        self.save_outputs = save_outputs
        super().__init__(name=name or "validate_one_epoch")

        self.key_map = {
            "dataloader": dataloader_key,
            "model": model_key,
            "device": device_key,
            "metrics": metrics_key,
        }

        for key, key_name in [
            ("loss_fn", loss_fn_key),
            ("outputs", outputs_key),
            ("losses", losses_key),
        ]:
            if key_name is not None:
                self.key_map[key] = key_name

    def bindings(self) -> Dict[str, str]:
        return self.key_map

    def compute(
        self,
        dataloader: torch.utils.data.DataLoader,
        model: BaseModel,
        device: torch.device,
        loss_fn: Optional[BaseLoss] = None,
        context: Optional[any] = None,
    ) -> Optional[Dict[str, Any]]:
        start_time = time.time()
        average_loss, losses, model_outputs = eval_epoch(
            dataloader=dataloader,
            model=model,
            device=device,
            loss_fn=loss_fn,
            save_outputs=self.save_outputs,
            profiler=context,
        )
        elapsed_time = time.time() - start_time

        out = {
            "model": model,
            "metrics": {"avg_loss": average_loss, "elapsed_time": elapsed_time},
        }

        if self.key_map.get("losses") is not None:
            out["losses"] = losses

        if self.key_map.get("outputs") is not None:
            output_key = self.key_map["outputs"]
            model_outputs = cat_model_outputs(model_outputs, dim=0)
            for key in model_outputs.keys():
                out[f"{output_key}.{key}"] = (
                    model_outputs[key].to(torch.device("cpu")).detach()
                )

        return out

from typing import Optional, List, Tuple

from tqdm import tqdm
import torch
from torch.nn.utils import clip_grad_norm_

from ..model.base import BaseModel, ModelOutput
from ..loss.base import BaseLoss
from ....data.torch.batch import Batch, ensure_batch


def _forward_and_loss(
    model: BaseModel,
    batch: Batch,
    loss_fn: Optional[BaseLoss] = None,
) -> Tuple[ModelOutput, Optional[torch.Tensor]]:
    """Forward pass with optional loss computation."""
    outputs = model(batch.features)

    if loss_fn is None or batch.targets is None:
        return outputs, None

    loss = loss_fn(*model.for_loss(outputs, batch.targets))
    return outputs, loss


@torch.no_grad()
def eval_epoch(
    dataloader: torch.utils.data.DataLoader,
    model: BaseModel,
    device: torch.device,
    loss_fn: Optional[BaseLoss] = None,
    save_outputs: bool = False,
    profiler: Optional[torch.profiler.profile] = None,
) -> Tuple[float, List[ModelOutput]]:
    """Evaluation epoch: forward pass and optional loss computation."""
    model.eval()

    total_loss = 0.0
    outputs_list = []
    pbar = tqdm(dataloader, desc="Evaluating", unit="batch")

    for batch in pbar:
        batch = ensure_batch(batch).to(device, non_blocking=True)
        outputs, loss = _forward_and_loss(model, batch, loss_fn)

        if save_outputs:
            outputs_list.append(outputs.detach().to(torch.device("cpu")))

        if loss is not None:
            total_loss += loss.item()

        if profiler is not None:
            profiler.step()

        if loss is not None:
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / len(dataloader) if dataloader else 0.0
    return avg_loss, outputs_list


def train_epoch(
    dataloader: torch.utils.data.DataLoader,
    model: BaseModel,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    loss_fn: BaseLoss,
    clip_grad_max_norm: Optional[float] = 1.0,
    profiler: Optional[torch.profiler.profile] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
) -> Tuple[float, Optional[float], Optional[float | List[float]]]:
    """Training epoch: forward, backward, optimizer step with optional gradient clipping."""
    model.train()

    total_loss = 0.0
    grad_norm = None
    pbar = tqdm(dataloader, desc="Training", unit="batch")

    lrs = []

    for batch in pbar:
        batch = ensure_batch(batch).to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        _, loss = _forward_and_loss(model, batch, loss_fn)
        loss.backward()

        if clip_grad_max_norm is not None:
            grad_norm = float(
                clip_grad_norm_(model.parameters(), max_norm=clip_grad_max_norm)
            )

        optimizer.step()
        total_loss += loss.item()

        if profiler is not None:
            profiler.step()

        if scheduler is not None:
            scheduler.step()
            lrs.append(optimizer.param_groups[0].get("lr"))

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / len(dataloader) if dataloader else 0.0

    if lrs == []:
        lrs = optimizer.param_groups[0].get("lr")

    return avg_loss, grad_norm, lrs

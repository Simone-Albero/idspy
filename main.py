import logging

import torch
from omegaconf import DictConfig

from src.idspy.common.logging import setup_logging
from src.idspy.common.utils import set_seeds
from src.idspy.common.config import load_config

from src.idspy.core.storage.dict import DictStorage
from src.idspy.core.pipeline.base import PipelineEvent
from src.idspy.core.pipeline.observable import (
    ObservablePipeline,
    ObservableFittablePipeline,
    ObservableRepeatablePipeline,
)
from src.idspy.core.events.bus import EventBus

from src.idspy.nn.torch.helper import get_device
from src.idspy.nn.torch.model.classifier import TabularClassifier
from src.idspy.nn.torch.loss.classification import ClassificationLoss

from src.idspy.builtins.handler.logging import Logger

from src.idspy.builtins.step import StepFactory

setup_logging()
logger = logging.getLogger(__name__)


def main(cfg: DictConfig):
    set_seeds(cfg.seed)

    # Log available steps
    logger.info(f"Registered {len(StepFactory.get_available())} steps")

    # Setup device
    if cfg.device == "auto":
        device = get_device()
    else:
        device = torch.device(cfg.device)

    # Setup model
    model = TabularClassifier(
        num_numeric=len(cfg.data.numerical_columns),
        cat_cardinalities=[cfg.max_frequency_levels]
        * len(cfg.data.categorical_columns),
        out_features=cfg.model.out_features,
        hidden_dims=cfg.model.hidden_dims,
        dropout=cfg.model.dropout,
    ).to(device)

    loss = ClassificationLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.optimizer.lr)

    # Setup storage
    storage = DictStorage(
        {
            "device": device,
            "model": model,
            "loss_fn": loss,
            "optimizer": optimizer,
            "seed": cfg.seed,
            "stop_pipeline": False,
        }
    )

    # Setup event bus
    bus = EventBus()
    bus.subscribe(callback=Logger(), event_type=PipelineEvent.STEP_START)

    # Build preprocessing pipeline from config
    logger.info("Building preprocessing pipeline from config...")

    # Create fitted pipeline
    fitted_steps = StepFactory.create_from_list(cfg.pipeline.preprocessing.fitted_steps)
    fit_aware_pipeline = ObservableFittablePipeline(
        steps=fitted_steps,
        name="fit_aware_pipeline",
        bus=bus,
        storage=storage,
    )

    # Create base preprocessing steps
    base_steps = StepFactory.create_from_list(cfg.pipeline.preprocessing.base_steps)
    save_steps = StepFactory.create_from_list(cfg.pipeline.preprocessing.save_steps)

    # Combine all preprocessing steps
    preprocessing_pipeline = ObservablePipeline(
        steps=base_steps + [fit_aware_pipeline] + save_steps,
        storage=storage,
        bus=bus,
        name="preprocessing_pipeline",
    )

    # Build training pipeline from config
    logger.info("Building training pipeline from config...")
    training_steps = StepFactory.create_from_list(cfg.pipeline.training.base_steps)

    training_pipeline = ObservableRepeatablePipeline(
        steps=training_steps,
        bus=bus,
        count=cfg.loop.train.epochs,
        clear_storage=False,
        predicate=lambda storage: storage.get("stop_pipeline"),
        name="training_pipeline",
        storage=storage,
    )

    # Run pipelines
    logger.info("Running preprocessing pipeline...")
    preprocessing_pipeline.run()

    logger.info("Running training pipeline...")
    training_pipeline.run()

    logger.info("Pipelines completed.\n(-_-) Exiting.")


if __name__ == "__main__":
    cfg = load_config(config_path="configs", config_name="config")
    main(cfg)

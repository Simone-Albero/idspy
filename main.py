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

from src.idspy.builtins.handler.logging import Logger

from src.idspy.builtins.step import StepFactory

setup_logging()
logger = logging.getLogger(__name__)


def preprocessing_pipeline(cfg: DictConfig, storage: DictStorage):
    bus = EventBus()
    bus.subscribe(callback=Logger(), event_type=PipelineEvent.STEP_START)

    logger.info("Building preprocessing pipeline from config...")

    # Collecting steps
    base_steps = StepFactory.create_from_list(cfg.pipeline.preprocessing.base_steps)
    fitted_steps = StepFactory.create_from_list(cfg.pipeline.preprocessing.fitted_steps)
    final_steps = StepFactory.create_from_list(cfg.pipeline.preprocessing.final_steps)

    fit_aware_pipeline = ObservableFittablePipeline(
        steps=fitted_steps,
        name="fit_aware_pipeline",
        bus=bus,
        storage=storage,
    )

    full_steps = base_steps + [fit_aware_pipeline] + final_steps

    preprocessing_pipeline = ObservablePipeline(
        steps=full_steps,
        storage=storage,
        bus=bus,
        name="preprocessing_pipeline",
    )

    logger.info("Running preprocessing pipeline...")
    preprocessing_pipeline.run()


def training_pipeline(cfg: DictConfig, storage: DictStorage):
    bus = EventBus()
    bus.subscribe(callback=Logger(), event_type=PipelineEvent.STEP_START)

    logger.info("Building training pipeline from config...")

    setup_steps = StepFactory.create_from_list(cfg.pipeline.training.setup_steps)
    training_steps = StepFactory.create_from_list(cfg.pipeline.training.base_steps)
    final_steps = StepFactory.create_from_list(cfg.pipeline.training.final_steps)

    training_pipeline = ObservableRepeatablePipeline(
        steps=training_steps,
        bus=bus,
        count=cfg.loop.train.epochs,
        clear_storage=False,
        predicate=lambda storage: storage.get("stop_pipeline"),
        name="training_pipeline",
        storage=storage,
    )

    full_steps = setup_steps + [training_pipeline] + final_steps
    full_pipeline = ObservablePipeline(
        steps=full_steps,
        storage=storage,
        bus=bus,
        name="full_pipeline",
    )

    logger.info("Running training pipeline...")
    full_pipeline.run()


def testing_pipeline(cfg: DictConfig, storage: DictStorage):
    bus = EventBus()
    bus.subscribe(callback=Logger(), event_type=PipelineEvent.STEP_START)

    logger.info("Building testing pipeline from config...")

    setup_steps = StepFactory.create_from_list(cfg.pipeline.testing.setup_steps)
    testing_steps = StepFactory.create_from_list(cfg.pipeline.testing.base_steps)

    full_steps = setup_steps + testing_steps
    full_pipeline = ObservablePipeline(
        steps=full_steps,
        storage=storage,
        bus=bus,
        name="full_pipeline",
    )

    logger.info("Running testing pipeline...")
    full_pipeline.run()


def main(cfg: DictConfig):
    set_seeds(cfg.seed)

    # Log available steps
    logger.info(f"Registered {len(StepFactory.get_available())} steps")

    # Setup device
    if cfg.device == "auto":
        device = get_device()
    else:
        device = torch.device(cfg.device)

    # Setup storage
    storage = DictStorage(
        {
            "device": device,
            "seed": cfg.seed,
            "stop_pipeline": False,
        }
    )

    if cfg.stage == "preprocessing":
        preprocessing_pipeline(cfg, storage)
    elif cfg.stage == "training":
        training_pipeline(cfg, storage)
    elif cfg.stage == "testing":
        testing_pipeline(cfg, storage)


if __name__ == "__main__":
    cfg = load_config(config_path="configs", config_name="config")
    main(cfg)

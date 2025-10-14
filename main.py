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

from src.idspy.data.schema import Schema, ColumnRole
from src.idspy.data.tab_accessor import TabAccessor
from src.idspy.data.torch.batch import default_collate

from src.idspy.nn.torch.helper import get_device
from src.idspy.nn.torch.model.classifier import TabularClassifier
from src.idspy.nn.torch.loss.classification import ClassificationLoss

from src.idspy.builtins.handler.logging import Logger

from src.idspy.builtins.step.data.io import SaveData, LoadData
from src.idspy.builtins.step.data.adjust import DropNulls
from src.idspy.builtins.step.data.map import FrequencyMap, LabelMap
from src.idspy.builtins.step.data.scale import StandardScale
from src.idspy.builtins.step.data.split import (
    AllocateSplitPartitions,
    AllocateTargets,
    StratifiedSplit,
)

from src.idspy.builtins.step.nn.torch.builder.dataloader import BuildDataLoader
from src.idspy.builtins.step.nn.torch.builder.dataset import BuildDataset
from src.idspy.builtins.step.nn.torch.engine.train import TrainOneEpoch
from src.idspy.builtins.step.nn.torch.engine.validate import (
    ValidateOneEpoch,
    MakePredictions,
)
from src.idspy.builtins.step.nn.torch.engine.tensor import CatTensors
from src.idspy.builtins.step.nn.torch.engine.early_stopping import EarlyStopping
from src.idspy.builtins.step.nn.torch.model.io import LoadModelWeights, SaveModelWeights
from src.idspy.builtins.step.nn.torch.metric.classification import ClassificationMetrics
from src.idspy.builtins.step.nn.torch.log.tensorboard import (
    MetricsLogger,
    WeightsLogger,
)

from src.idspy.builtins.step.ml.cluster.score import ClusteringScores

setup_logging()
logger = logging.getLogger(__name__)


def main(cfg: DictConfig):
    set_seeds(cfg.seed)

    schema = Schema()
    schema.add(cfg.data.target_columns, ColumnRole.TARGET)
    schema.add(cfg.data.numerical_columns, ColumnRole.NUMERICAL)
    schema.add(cfg.data.categorical_columns, ColumnRole.CATEGORICAL)

    if cfg.device == "auto":
        device = get_device()
    else:
        device = torch.device(cfg.device)

    model = TabularClassifier(
        num_numeric=len(schema.numerical),
        cat_cardinalities=[cfg.max_frequency_levels] * len(schema.categorical),
        out_features=cfg.model.out_features,
        hidden_dims=cfg.model.hidden_dims,
        dropout=cfg.model.dropout,
    ).to(device)

    loss = ClassificationLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.optimizer.lr)

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

    bus = EventBus()
    bus.subscribe(callback=Logger(), event_type=PipelineEvent.STEP_START)

    fit_aware_pipeline = ObservableFittablePipeline(
        steps=[
            StandardScale(),
            FrequencyMap(max_levels=cfg.max_frequency_levels),
            LabelMap(),
        ],
        name="fit_aware_pipeline",
        bus=bus,
        storage=storage,
    )

    preprocessing_pipeline = ObservablePipeline(
        steps=[
            LoadData(
                file_path=cfg.paths.data_raw,
                file_name=cfg.data.file_name,
                fmt=cfg.data.format,
            ),
            DropNulls(),
            # DownsampleToMinority(class_column=schema.columns(ColumnRole.TARGET)[0]),
            StratifiedSplit(class_column=schema.target),
            fit_aware_pipeline,
            SaveData(
                file_path=cfg.paths.data_processed,
                file_name=cfg.data.file_name,
                fmt=cfg.data.format,
            ),
        ],
        storage=storage,
        bus=bus,
        name="preprocessing_pipeline",
    )

    training_pipeline = ObservableRepeatablePipeline(
        steps=[
            LoadData(
                file_path=cfg.paths.data_processed,
                file_name=cfg.data.file_name,
                fmt=cfg.data.format,
            ),
            AllocateSplitPartitions(),
            AllocateTargets(df_key="test.data", targets_key="test.targets"),
            BuildDataset(df_key="train.data", dataset_key="train.dataset"),
            BuildDataLoader(
                dataset_key="train.dataset",
                dataloader_key="train.dataloader",
                batch_size=cfg.loops.train.dataloader.batch_size,
                num_workers=cfg.loops.train.dataloader.num_workers,
                shuffle=cfg.loops.train.dataloader.shuffle,
                pin_memory=cfg.loops.train.dataloader.pin_memory,
                collate_fn=default_collate,
            ),
            BuildDataset(df_key="test.data", dataset_key="test.dataset"),
            BuildDataLoader(
                dataset_key="test.dataset",
                dataloader_key="test.dataloader",
                batch_size=cfg.loops.test.dataloader.batch_size,
                num_workers=cfg.loops.test.dataloader.num_workers,
                shuffle=cfg.loops.test.dataloader.shuffle,
                pin_memory=cfg.loops.test.dataloader.pin_memory,
                collate_fn=default_collate,
            ),
            TrainOneEpoch(),
            MetricsLogger(log_dir=cfg.paths.logs, metrics_key="train.metrics"),
            SaveModelWeights(
                file_path=cfg.paths.models,
                file_name=cfg.model.name,
            ),
            ValidateOneEpoch(
                dataloader_key="test.dataloader",
                metrics_key="test.metrics",
                outputs_key="test.outputs",
                save_outputs=True,
            ),
            EarlyStopping(
                min_delta=0.001,
                metrics_key="test.metrics",
                model_key="model",
                stop_key="stop_pipeline",
            ),
            CatTensors(
                inputs_key="test.outputs",
                input_section="logits",
                outputs_key="test.logits",
            ),
            CatTensors(
                inputs_key="test.outputs",
                input_section="latents",
                outputs_key="test.latents",
            ),
            MakePredictions(
                pred_fn=lambda x: torch.argmax(x, dim=1),
                inputs_key="test.logits",
                outputs_key="test.predictions",
            ),
            ClassificationMetrics(
                predictions_key="test.predictions",
                targets_key="test.targets",
                metrics_key="test.metrics",
            ),
            MetricsLogger(log_dir="resources/logs", metrics_key="test.metrics"),
            WeightsLogger(log_dir="resources/logs", model_key="model"),
            # ClusteringScores(
            #     vectors_key="test.latents",
            #     targets_key="test.targets",
            #     outputs_key="test.clustering_scores",
            #     scale_inputs=True,
            # ),
        ],
        bus=bus,
        count=1,
        clear_storage=False,
        predicate=lambda storage: storage.get("stop_pipeline"),
        name="training_pipeline",
        storage=storage,
    )

    # preprocessing_pipeline.run()
    training_pipeline.run()
    print(storage.get(["test.metrics"]))
    print(storage.get(["test.clustering_scores"]))


if __name__ == "__main__":
    cfg = load_config(config_path="configs", config_name="config")
    main(cfg)

import logging

import torch

from src.idspy.common.logging import setup_logging
from src.idspy.common.utils import set_seeds

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
set_seeds(42)


def main():
    schema = Schema()
    schema.add(["Attack"], ColumnRole.TARGET)
    schema.add(
        [
            "IN_BYTES",
            "IN_PKTS",
            "OUT_BYTES",
            "OUT_PKTS",
            "FLOW_DURATION_MILLISECONDS",
            "DURATION_IN",
            "DURATION_OUT",
            "MIN_TTL",
            "MAX_TTL",
            "LONGEST_FLOW_PKT",
            "SHORTEST_FLOW_PKT",
            "MIN_IP_PKT_LEN",
            "MAX_IP_PKT_LEN",
            "SRC_TO_DST_SECOND_BYTES",
            "DST_TO_SRC_SECOND_BYTES",
            "RETRANSMITTED_IN_BYTES",
            "RETRANSMITTED_IN_PKTS",
            "RETRANSMITTED_OUT_BYTES",
            "RETRANSMITTED_OUT_PKTS",
            "SRC_TO_DST_AVG_THROUGHPUT",
            "DST_TO_SRC_AVG_THROUGHPUT",
            "NUM_PKTS_UP_TO_128_BYTES",
            "NUM_PKTS_128_TO_256_BYTES",
            "NUM_PKTS_256_TO_512_BYTES",
            "NUM_PKTS_512_TO_1024_BYTES",
            "NUM_PKTS_1024_TO_1514_BYTES",
            "TCP_WIN_MAX_IN",
            "TCP_WIN_MAX_OUT",
            "DNS_TTL_ANSWER",
        ],
        ColumnRole.NUMERICAL,
    )
    schema.add(
        [
            "L4_SRC_PORT",
            "L4_DST_PORT",
            "PROTOCOL",
            "L7_PROTO",
            "TCP_FLAGS",
            "CLIENT_TCP_FLAGS",
            "SERVER_TCP_FLAGS",
            "ICMP_TYPE",
            "ICMP_IPV4_TYPE",
            "DNS_QUERY_ID",
            "DNS_QUERY_TYPE",
        ],
        ColumnRole.CATEGORICAL,
    )

    device = get_device()
    model = TabularClassifier(
        num_features=len(schema.numerical),
        cat_cardinalities=[20] * len(schema.categorical),
        num_classes=15,
        hidden_dims=[128, 64],
        dropout=0.1,
    ).to(device)
    loss = ClassificationLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    storage = DictStorage(
        {
            "device": device,
            "model": model,
            "loss_fn": loss,
            "optimizer": optimizer,
            "seed": 42,
            "stop_pipeline": False,
        }
    )

    bus = EventBus()
    bus.subscribe(callback=Logger(), event_type=PipelineEvent.STEP_START)

    fit_aware_pipeline = ObservableFittablePipeline(
        steps=[
            StandardScale(),
            FrequencyMap(max_levels=20),
            LabelMap(),
        ],
        name="fit_aware_pipeline",
        bus=bus,
        storage=storage,
    )

    preprocessing_pipeline = ObservablePipeline(
        steps=[
            LoadData(
                file_path="resources/data/dataset_v2/cic_2018_v2.csv",
                schema=schema,
            ),
            DropNulls(),
            # DownsampleToMinority(class_column=schema.columns(ColumnRole.TARGET)[0]),
            StratifiedSplit(class_column=schema.target),
            fit_aware_pipeline,
            SaveData(
                file_path="resources/data/processed",
                file_name="cic_2018_v2",
                fmt="parquet",
            ),
        ],
        storage=storage,
        bus=bus,
        name="preprocessing_pipeline",
    )

    training_pipeline = ObservableRepeatablePipeline(
        steps=[
            LoadData(file_path="resources/data/processed/cic_2018_v2.parquet"),
            AllocateSplitPartitions(),
            AllocateTargets(df_key="test.data", targets_key="test.targets"),
            BuildDataset(df_key="train.data", dataset_key="train.dataset"),
            BuildDataLoader(
                dataset_key="train.dataset",
                dataloader_key="train.dataloader",
                batch_size=512,
                num_workers=6,
                shuffle=True,
                collate_fn=default_collate,
            ),
            BuildDataset(df_key="test.data", dataset_key="test.dataset"),
            BuildDataLoader(
                dataset_key="test.dataset",
                dataloader_key="test.dataloader",
                batch_size=1024,
                num_workers=6,
                shuffle=False,
                collate_fn=default_collate,
            ),
            TrainOneEpoch(),
            MetricsLogger(log_dir="resources/logs", metrics_key="train.metrics"),
            SaveModelWeights(file_path="resources/models/cic_2018_v2/model.pt"),
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
    main()

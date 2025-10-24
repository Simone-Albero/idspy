from typing import Optional, Dict, Any

import numpy as np
from sklearn.manifold import TSNE

from ....core.step.base import Step
from ....plot.vectors import vectors_plot
from ..helpers import sample_vectors_and_labels
from .. import StepFactory


@StepFactory.register()
@Step.needs("vectors", "labels")
class VectorsProjectionPlot(Step):

    def __init__(
        self,
        name: Optional[str] = None,
        vectors_key: str = "vectors",
        labels_key: str = "labels",
        output_key: str = "projection_plot",
        n_components: int = 2,
        sample_size: int = 30000,
        class_names: Optional[list] = None,
    ) -> None:

        super().__init__(name=name or "vectors_projection")
        self.class_names = class_names
        self.n_components = n_components
        self.sample_size = sample_size
        self.key_map = {
            "vectors": vectors_key,
            "labels": labels_key,
            "output": output_key,
        }

    def bindings(self) -> Dict[str, str]:
        return self.key_map

    def compute(
        self, vectors: np.ndarray, labels: np.ndarray
    ) -> Optional[Dict[str, Any]]:

        X_sampled, labels_sampled = sample_vectors_and_labels(
            vectors,
            labels,
            sample_size=self.sample_size,
        )

        X_compressed = TSNE(
            n_components=self.n_components,
            perplexity=30,
            max_iter=1000,
            random_state=42,
        ).fit_transform(X_sampled)

        return {
            "output": {
                f"{self.n_components}d_projection": vectors_plot(
                    X_compressed, labels_sampled
                )
            }
        }

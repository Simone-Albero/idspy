from typing import Optional, Dict, Any

import numpy as np
from sklearn.manifold import TSNE

from ....core.step.base import Step
from ....plot.vectors import vectors_plot
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
        class_names: Optional[list] = None,
    ) -> None:

        super().__init__(name=name or "vectors_projection")
        self.class_names = class_names
        self.n_components = n_components
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

        X_compressed = TSNE(
            n_components=self.n_components,
            perplexity=30,
            max_iter=1000,
            random_state=42,
        ).fit_transform(vectors)

        return {
            "output": {
                f"{self.n_components}d_projection": vectors_plot(X_compressed, labels)
            }
        }

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    """
    Strong re-ranker. Default: BAAI/bge-reranker-v2-m3 (multilingual).
    """

    def __init__(
        self, model_name: str = "BAAI/bge-reranker-v2-m3", device: Optional[str] = None
    ):
        self.model = CrossEncoder(model_name, device=device)

    def rerank(
        self, query: str, texts: List[str], top_k: int
    ) -> List[Tuple[int, float]]:
        """
        Returns list of (index_in_texts, score) sorted by score desc.
        """

        pairs = [[query, t] for t in texts]
        scores = self.model.predict(pairs, show_progress_bar=False)
        order = np.argsort(-scores)[:top_k]
        return [(int(i), float(scores[int(i)])) for i in order]

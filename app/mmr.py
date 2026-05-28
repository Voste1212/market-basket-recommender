from __future__ import annotations

from typing import Iterable

import numpy as np


def mmr_rerank(
    candidates: list[dict],
    similarity_matrix: dict[tuple[str, str], float],
    top_k: int = 5,
    lambda_param: float = 0.75,
) -> list[dict]:
    """Rerank candidates using Maximal Marginal Relevance.

    Args:
        candidates: list of recommendation dictionaries. Each must contain
            `product_name` and `final_score`.
        similarity_matrix: dictionary keyed by (product_a, product_b).
        top_k: number of recommendations to return.
        lambda_param: relevance/diversity trade-off. Higher value means more
            emphasis on relevance.
    """
    if not candidates:
        return []

    selected: list[dict] = []
    remaining = candidates.copy()

    while remaining and len(selected) < top_k:
        if not selected:
            best = max(remaining, key=lambda item: item.get("final_score", 0.0))
        else:
            scored = []
            for candidate in remaining:
                relevance = candidate.get("final_score", 0.0)
                max_similarity = max(
                    similarity_matrix.get(
                        (candidate["product_name"], chosen["product_name"]),
                        similarity_matrix.get((chosen["product_name"], candidate["product_name"]), 0.0),
                    )
                    for chosen in selected
                )
                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_similarity
                scored.append((mmr_score, candidate))
            best = max(scored, key=lambda pair: pair[0])[1]

        selected.append(best)
        remaining = [item for item in remaining if item["product_name"] != best["product_name"]]

    return selected

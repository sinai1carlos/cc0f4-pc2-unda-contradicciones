from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

import config


def recall_at_k(ranking: list[str], relevant: list[str], k: int) -> float:
    return len(set(relevant) & set(ranking[:k])) / len(set(relevant))


def reciprocal_rank(ranking: list[str], relevant: list[str]) -> float:
    rel = set(relevant)
    for rank, pid in enumerate(ranking, start=1):
        if pid in rel:
            return 1.0 / rank
    return 0.0


def dcg_at_k(ranking: list[str], relevant: list[str], k: int) -> float:
    rel = set(relevant)
    return sum(
        (1.0 if pid in rel else 0.0) / math.log2(i + 1)
        for i, pid in enumerate(ranking[:k], start=1)
    )


def ndcg_at_k(ranking: list[str], relevant: list[str], k: int) -> float:
    ideal = dcg_at_k(list(relevant), relevant, k)
    return dcg_at_k(ranking, relevant, k) / ideal if ideal > 0 else 0.0


def first_relevant_rank(ranking: list[str], relevant: list[str]) -> int | None:
    rel = set(relevant)
    for rank, pid in enumerate(ranking, start=1):
        if pid in rel:
            return rank
    return None


def classify_failure(ranking: list[str], relevant: list[str], top_k: int = 3) -> str:
    """Taxonomía del Cuaderno 7 sobre los candidatos devueltos."""
    if recall_at_k(ranking, relevant, len(ranking)) == 0:
        return "RETRIEVAL_MISS"
    top = recall_at_k(ranking, relevant, top_k)
    if top == 0:
        return "RANKING_ERROR"
    if top < 1:
        return "PARTIAL_RECALL"
    return "RETRIEVAL_OK"


def paired_bootstrap(frame: pd.DataFrame, metric: str, cond_a: str, cond_b: str) -> dict[str, Any]:
    """Diferencia media cond_b - cond_a remuestreando consultas completas."""
    a = frame[frame.condition == cond_a].set_index("query_id")[metric].sort_index()
    b = frame[frame.condition == cond_b].set_index("query_id")[metric].sort_index()
    if list(a.index) != list(b.index):
        raise ValueError("comparación pareada con consultas distintas")
    deltas = b.to_numpy(float) - a.to_numpy(float)
    rng = np.random.default_rng(config.SEED)
    idx = rng.integers(0, len(deltas), size=(config.BOOTSTRAP_RESAMPLES, len(deltas)))
    means = deltas[idx].mean(axis=1)
    low, high = np.quantile(means, [config.BOOTSTRAP_ALPHA / 2, 1 - config.BOOTSTRAP_ALPHA / 2])
    return {
        "metrica": metric,
        "comparacion": f"{cond_b} - {cond_a}",
        "delta_media": round(float(deltas.mean()), 4),
        "ic95_inf": round(float(low), 4),
        "ic95_sup": round(float(high), 4),
        "ic_cruza_cero": bool(low <= 0 <= high),
        "consultas_mejoran": int((deltas > 0).sum()),
        "consultas_empeoran": int((deltas < 0).sum()),
    }

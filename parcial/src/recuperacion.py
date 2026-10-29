"""Recuperadores del experimento 1.

A = denso (E5 + FAISS IndexFlatIP)
B = A + BM25 fusionados con RRF
C = B + reranking con Cross-Encoder

La unidad recuperada es el artículo: cada chunk contiene un solo passage, así que
el ranking de chunks coincide con el ranking de passage_id que usan los qrels.
"""

from __future__ import annotations

import math
import re
import time
from collections import Counter, defaultdict
from typing import Any

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

import config

TOKEN_RE = re.compile(r"[a-záéíóúñü0-9]+", re.IGNORECASE)

# Lista de stopwords del Cuaderno 7. Se conservan términos que cambian elsentido de una norma (no, sin, puede, debe).
BM25_PRESERVED_TERMS = frozenset({"no", "sin", "si", "puede", "pueden", "debe", "más"})
SPANISH_STOPWORDS = frozenset({
    "a", "al", "algo", "como", "con", "cuando", "de", "del", "donde",
    "e", "el", "ella", "ellas", "ellos", "en", "entre", "era", "es",
    "esa", "ese", "eso", "esta", "este", "esto", "ha", "hay", "la",
    "las", "le", "les", "lo", "los", "o", "para", "pero", "por",
    "que", "qué", "se", "su", "sus", "también", "un", "una", "uno",
    "unos", "unas", "y", "ya",
})


def tokenize_bm25(text: str) -> list[str]:
    tokens = [m.group(0).lower() for m in TOKEN_RE.finditer(text)]
    return [t for t in tokens if t in BM25_PRESERVED_TERMS or t not in SPANISH_STOPWORDS]


def _sorted_ranking(chunks, scores, depth: int) -> list[dict[str, Any]]:
    order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), chunks[i]["passage_id"]))[:depth]
    return [{**chunks[i], "score": float(scores[i])} for i in order]


class BM25:
    def __init__(self, chunks: list[dict[str, Any]], k1: float = config.BM25_K1, b: float = config.BM25_B):
        self.chunks = chunks
        self.k1, self.b = k1, b
        docs = [tokenize_bm25(c["text"]) for c in chunks]
        self.tf = [Counter(d) for d in docs]
        self.lengths = np.array([len(d) for d in docs], dtype=float)
        self.avgdl = float(self.lengths.mean())
        df = Counter(t for d in docs for t in set(d))
        n = len(docs)
        self.idf = {t: math.log(1.0 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.postings: dict[str, list[int]] = defaultdict(list)
        for i, tf in enumerate(self.tf):
            for t in tf:
                self.postings[t].append(i)

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(len(self.chunks))
        for t in tokenize_bm25(query):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i in self.postings[t]:
                f = self.tf[i][t]
                norm = self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avgdl)
                out[i] += idf * f * (self.k1 + 1) / (f + norm)
        return out

    def rank(self, query: str, depth: int) -> list[dict[str, Any]]:
        return _sorted_ranking(self.chunks, self.scores(query), depth)


class DenseRetriever:
    """E5 + FAISS IndexFlatIP con vectores normalizados (producto interno = coseno)."""

    def __init__(self, chunks: list[dict[str, Any]], real: bool = config.RUN_REAL_RETRIEVAL):
        self.chunks = chunks
        self.real = real
        texts = [c["text"] for c in chunks]
        if real:
            import faiss
            from sentence_transformers import SentenceTransformer

            # FAISS y PyTorch usan OpenMP; con un índice exacto de 2115 vectores un hilo basta
            # y se evita que ambos compitan por los hilos durante la generación.
            faiss.omp_set_num_threads(1)
            self.model = SentenceTransformer(config.DENSE_MODEL_ID, device="cpu")
            self.max_seq_length = self.model.max_seq_length
            self.token_lengths = [
                len(self.model.tokenizer("passage: " + t, add_special_tokens=True, truncation=False)["input_ids"])
                for t in texts
            ]
            self.vectors = self.model.encode(
                ["passage: " + t for t in texts], batch_size=32, normalize_embeddings=True,
                convert_to_numpy=True, show_progress_bar=False,
            ).astype("float32")
            self.index = faiss.IndexFlatIP(self.vectors.shape[1])
            self.index.add(self.vectors)
            self.implementation = f"{config.DENSE_MODEL_ID} + FAISS IndexFlatIP"
        else:
            self.vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2))
            x = self.vectorizer.fit_transform(texts)
            self.svd = TruncatedSVD(n_components=128, random_state=config.SEED)
            self.vectors = normalize(self.svd.fit_transform(x)).astype("float32")
            self.index = None
            self.max_seq_length = None
            self.token_lengths = None
            self.implementation = "TF-IDF + SVD (solo validación del software)"

    def encode_query(self, query: str) -> np.ndarray:
        if self.real:
            return self.model.encode(["query: " + query], normalize_embeddings=True,
                                     convert_to_numpy=True, show_progress_bar=False).astype("float32")
        return normalize(self.svd.transform(self.vectorizer.transform([query]))).astype("float32")

    def rank(self, query: str, depth: int) -> list[dict[str, Any]]:
        q = self.encode_query(query)
        if self.index is not None:
            scores, idx = self.index.search(q, depth)
            full = np.full(len(self.chunks), -np.inf)
            full[idx[0]] = scores[0]
            return _sorted_ranking(self.chunks, full, depth)
        return _sorted_ranking(self.chunks, self.vectors @ q[0], depth)


def reciprocal_rank_fusion(rankings: list[list[dict[str, Any]]], depth: int = config.CANDIDATE_DEPTH,
                           constant: int = config.RRF_CONSTANT) -> list[dict[str, Any]]:
    """RRF(d) = suma de 1 / (k + rango). Fusiona posiciones, no scores."""
    fused: dict[str, float] = defaultdict(float)
    by_id: dict[str, dict[str, Any]] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            fused[item["passage_id"]] += 1.0 / (constant + rank)
            by_id[item["passage_id"]] = item
    ordered = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))[:depth]
    return [{**by_id[pid], "score": score} for pid, score in ordered]


class Reranker:
    def __init__(self, real: bool = config.RUN_REAL_RERANKER):
        self.real = real
        if real:
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(config.RERANKER_MODEL_ID, max_length=512, device="cpu")
            self.implementation = config.RERANKER_MODEL_ID
        else:
            self.implementation = "solapamiento léxico (solo validación del software)"

    def rerank(self, query: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self.real:
            scores = self.model.predict([[query, c["text"]] for c in candidates], show_progress_bar=False)
        else:
            q = set(tokenize_bm25(query))
            scores = [len(q & set(tokenize_bm25(c["text"]))) / max(len(q), 1) for c in candidates]
        return _sorted_ranking(candidates, scores, len(candidates))


CONDITIONS = {"A": "denso", "B": "denso + BM25 + RRF", "C": "denso + BM25 + RRF + Cross-Encoder"}


def _ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def run_condition(condition: str, query: str, dense: DenseRetriever, bm25: BM25,
                  reranker: Reranker, depth: int = config.CANDIDATE_DEPTH):
    """Ejecuta una condición y mide la latencia de cada etapa por separado."""
    t = {"dense_ms": 0.0, "bm25_ms": 0.0, "rrf_ms": 0.0, "reranker_ms": 0.0}
    start_total = time.perf_counter()
    s = time.perf_counter()
    dense_ranking = dense.rank(query, depth)
    t["dense_ms"] = _ms(s)
    if condition == "A":
        ranking = dense_ranking
    else:
        s = time.perf_counter()
        lexical = bm25.rank(query, depth)
        t["bm25_ms"] = _ms(s)
        s = time.perf_counter()
        ranking = reciprocal_rank_fusion([dense_ranking, lexical], depth)
        t["rrf_ms"] = _ms(s)
        if condition == "C":
            s = time.perf_counter()
            ranking = reranker.rerank(query, ranking)
            t["reranker_ms"] = _ms(s)
    t["total_ms"] = _ms(start_total)
    return ranking, t

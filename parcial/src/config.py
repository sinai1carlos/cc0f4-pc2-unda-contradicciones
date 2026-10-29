"""Configuración fija de los tres experimentos .

Todos los valores se fijan antes de ejecutar y no se ajustan después de ver
resultados. Los modelos se pueden cambiar con variables de entorno, pero los
resultados reportados usan los valores por defecto.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
PROMPTS_DIR = ROOT / "prompts"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

CORPUS_PATH = DATA_DIR / "corpus_cc_peru_v2.json"
QUERIES_PATH = DATA_DIR / "consultas.jsonl"
QRELS_PATH = DATA_DIR / "qrels.json"
REFERENCES_PATH = DATA_DIR / "respuestas_referencia.jsonl"
HOLIDAYS_PATH = DATA_DIR / "feriados_peru_2026.json"

SEED = 42

# Recuperación (experimento 1)
DENSE_MODEL_ID = os.getenv("CC0F4_DENSE_MODEL_ID", "intfloat/multilingual-e5-small")
RERANKER_MODEL_ID = os.getenv("CC0F4_RERANKER_MODEL_ID", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
CANDIDATE_DEPTH = 20
RRF_CONSTANT = 60
TOP_K_VALUES = [1, 3, 5]
NDCG_K = 5
BM25_K1 = 1.5
BM25_B = 0.75
BOOTSTRAP_RESAMPLES = 5000
BOOTSTRAP_ALPHA = 0.05

# RAG (experimento 2)
GENERATOR_MODEL_ID = os.getenv("CC0F4_GENERATOR_MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
RAG_CONDITION = "C"
RAG_TOP_K = 3
MAX_NEW_TOKENS = 220

# Modos de ejecución. Con 0 se valida el software sin descargar modelos;
# esos números no son resultados del examen.
RUN_REAL_RETRIEVAL = os.getenv("CC0F4_RUN_REAL_RETRIEVAL", "1") == "1"
RUN_REAL_RERANKER = os.getenv("CC0F4_RUN_REAL_RERANKER", "1") == "1"
RUN_REAL_LLM = os.getenv("CC0F4_RUN_REAL_LLM", "1") == "1"

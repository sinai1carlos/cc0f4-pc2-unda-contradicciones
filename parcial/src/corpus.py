"""Carga del corpus del Código Civil, las consultas, los qrels y las referencias."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import config


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 16), b""):
            digest.update(block)
    return digest.hexdigest()


def passage_text(article: dict[str, Any]) -> str:
    """Texto que se indexa: número de artículo, sumilla si existe y contenido."""
    head = f"Artículo {article['articulo']}."
    if article.get("sumilla"):
        head += f" {article['sumilla']}."
    return f"{head} {article['texto']}"


def load_passages() -> list[dict[str, Any]]:
    """Un passage por artículo vigente. El identificador es el `id` del corpus."""
    articles = json.loads(config.CORPUS_PATH.read_text(encoding="utf-8"))
    passages = []
    for art in articles:
        if not art["vigente"]:
            continue
        passages.append({
            "passage_id": art["id"],
            "articulo": art["articulo"],
            "libro": art["libro_nombre"],
            "revisar_spij": art["revisar_spij"],
            "num_palabras": art["num_palabras"],
            "text": passage_text(art),
        })
    return passages


def load_benchmark() -> tuple[list[dict], dict[str, list[str]], list[dict]]:
    queries = read_jsonl(config.QUERIES_PATH)
    qrels = json.loads(config.QRELS_PATH.read_text(encoding="utf-8"))
    references = read_jsonl(config.REFERENCES_PATH)
    return queries, qrels, references


def validate_benchmark(passages, queries, qrels, references) -> dict[str, int]:
    ids = [p["passage_id"] for p in passages]
    if len(ids) != len(set(ids)):
        raise ValueError("passage_id duplicado en el corpus")
    known = set(ids)
    qids = [q["query_id"] for q in queries]
    if len(qids) != len(set(qids)):
        raise ValueError("query_id duplicado")
    if set(qids) != set(qrels):
        raise ValueError("consultas y qrels no tienen los mismos query_id")
    for qid, gold in qrels.items():
        if not gold:
            raise ValueError(f"qrels vacíos para {qid}")
        missing = [pid for pid in gold if pid not in known]
        if missing:
            raise ValueError(f"{qid}: qrels fuera del corpus vigente {missing}")
    for ref in references:
        if not ref["control_negativo"] and ref["query_id"] not in qrels:
            raise ValueError(f"referencia sin consulta: {ref['query_id']}")
        for pid in ref["evidencia_esperada"]:
            if pid not in known:
                raise ValueError(f"evidencia esperada fuera del corpus: {pid}")
    return {
        "passages": len(passages),
        "consultas": len(queries),
        "pares_relevantes": sum(len(v) for v in qrels.values()),
        "consultas_rag": sum(1 for r in references if not r["control_negativo"]),
        "controles_negativos": sum(1 for r in references if r["control_negativo"]),
    }

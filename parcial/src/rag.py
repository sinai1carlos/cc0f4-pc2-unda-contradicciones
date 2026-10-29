"""Generación closed-book y RAG con salida JSON validada (experimento 2)."""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from typing import Any

from jsonschema import Draft7Validator

import config

ANSWER_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "properties": {
        "respuesta": {"type": "string", "minLength": 1, "maxLength": 1200},
        "citas": {"type": "array", "items": {"type": "string", "pattern": "^cc_art_[0-9]+(_[A-Z])?$"}, "maxItems": 5},
        "abstencion": {"type": "boolean"},
    },
    "required": ["respuesta", "citas", "abstencion"],
    "additionalProperties": False,
}
VALIDATOR = Draft7Validator(ANSWER_SCHEMA)

SYSTEM_PROMPT = (config.PROMPTS_DIR / "rag_sistema.txt").read_text(encoding="utf-8").strip()
RAG_TEMPLATE = (config.PROMPTS_DIR / "rag_usuario.txt").read_text(encoding="utf-8")
CLOSED_TEMPLATE = (config.PROMPTS_DIR / "closed_book_usuario.txt").read_text(encoding="utf-8")


def build_evidence(ranking: list[dict[str, Any]], top_k: int = config.RAG_TOP_K) -> str:
    return "\n\n".join(f"[{c['passage_id']}] {c['text']}" for c in ranking[:top_k])


def build_messages(question: str, ranking: list[dict[str, Any]] | None) -> list[dict[str, str]]:
    if ranking is None:
        user = CLOSED_TEMPLATE.format(pregunta=question)
    else:
        user = RAG_TEMPLATE.format(pregunta=question, evidencia=build_evidence(ranking))
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def extract_json(text: str) -> dict[str, Any] | None:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return None
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def validate_output(raw: str, context_ids: list[str]) -> dict[str, Any]:
    """Tres niveles: JSON legible, cumple el schema, citas dentro del contexto entregado."""
    obj = extract_json(raw)
    out = {"json_valido": obj is not None, "schema_valido": False, "errores_schema": [],
           "respuesta": None, "citas": [], "abstencion": None,
           "citas_fuera_de_contexto": [], "citas_en_contexto": False}
    if obj is None:
        return out
    errors = sorted(VALIDATOR.iter_errors(obj), key=lambda e: list(e.path))
    out["errores_schema"] = [f"{'/'.join(map(str, e.path)) or 'raiz'}: {e.message}" for e in errors]
    out["schema_valido"] = not errors
    if isinstance(obj.get("respuesta"), str):
        out["respuesta"] = obj["respuesta"]
    if isinstance(obj.get("citas"), list):
        out["citas"] = [str(c) for c in obj["citas"]]
    if isinstance(obj.get("abstencion"), bool):
        out["abstencion"] = obj["abstencion"]
    allowed = set(context_ids)
    out["citas_fuera_de_contexto"] = [c for c in out["citas"] if c not in allowed]
    out["citas_en_contexto"] = out["schema_valido"] and not out["citas_fuera_de_contexto"]
    return out


class Generator:
    """Qwen con decoding greedy. Sin modelo real devuelve un texto fijo que no es JSON."""

    def __init__(self, real: bool = config.RUN_REAL_LLM):
        self.real = real
        self.model_id = config.GENERATOR_MODEL_ID if real else "sin modelo (validación del software)"
        if real:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.torch = torch
            self.tokenizer = AutoTokenizer.from_pretrained(config.GENERATOR_MODEL_ID)
            self.model = AutoModelForCausalLM.from_pretrained(config.GENERATOR_MODEL_ID, torch_dtype="auto")
            self.model.eval()

    def prompt_tokens(self, messages: list[dict[str, str]]) -> int | None:
        if not self.real:
            return None
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return len(self.tokenizer(text)["input_ids"])

    def generate(self, messages: list[dict[str, str]]) -> tuple[str, float]:
        if not self.real:
            return "Modo de validación: no se cargó el generador.", 0.0
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt")
        start = time.perf_counter()
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=config.MAX_NEW_TOKENS, do_sample=False,
                                      temperature=None, top_p=None, top_k=None)
        ms = (time.perf_counter() - start) * 1000.0
        return self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True), ms

    def next_token_distribution(self, messages: list[dict[str, str]], top: int = 5) -> list[tuple[str, float]]:
        """Softmax de los logits del primer token generado (evidencia de la Semana 2)."""
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt")
        with self.torch.no_grad():
            logits = self.model(**inputs).logits[0, -1]
        probs = self.torch.softmax(logits.float(), dim=-1)
        values, idx = probs.topk(top)
        return [(self.tokenizer.decode([int(i)]), round(float(v), 4)) for v, i in zip(values, idx)]


def token_f1_proxy(prediction: str, reference: str) -> float:
    """Solapamiento de tokens: solo para depurar, no mide corrección (Cuaderno 7)."""
    pred = Counter(t.lower() for t in re.findall(r"\w+", prediction or ""))
    gold = Counter(t.lower() for t in re.findall(r"\w+", reference))
    common = sum((pred & gold).values())
    if not common:
        return 0.0
    p, r = common / sum(pred.values()), common / sum(gold.values())
    return round(2 * p * r / (p + r), 3)


from __future__ import annotations

import asyncio
import calendar
import json
import time
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

import config

TOOL_NAME = "plazos.computar_vencimiento"
DIAS = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]


class PlazoInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    fecha_inicio: date = Field(description="Día en que empieza el plazo (por ejemplo, la notificación).")
    cantidad: int = Field(ge=1, le=3650, description="Número de unidades del plazo.")
    unidad: Literal["dias", "meses", "anios"]
    computo_dias: Literal["naturales", "habiles"] = Field(
        default="naturales", description="Art. 183 inc. 1: por defecto días naturales.")
    naturaleza: Literal["general", "caducidad"] = Field(
        default="general", description="La caducidad no se traslada al día hábil siguiente (art. 2007).")


class PlazoOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    fecha_vencimiento: date
    fecha_vencimiento_nominal: date
    dia_semana_vencimiento: str
    dias_inhabiles_saltados: int = Field(ge=0)
    reglas_aplicadas: list[str] = Field(min_length=1)
    fundamento: list[str] = Field(min_length=1)


ErrorKind = Literal["protocol", "validation", "domain", "transient", "timeout", "execution", "output_contract"]


class ToolError(BaseModel):
    code: str
    kind: ErrorKind
    retryable: bool
    message: str


class ToolResult(BaseModel):
    status: Literal["ok", "error"]
    data: dict[str, Any] | None = None
    error: ToolError | None = None


class DomainError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class TransientToolError(Exception):
    pass


@dataclass(frozen=True)
class Policy:
    max_attempts: int = 3
    backoff_s: tuple[float, ...] = (0.05, 0.10)
    per_attempt_timeout_s: float = 0.20
    overall_deadline_s: float = 0.80


POLICY = Policy()

# Especificación tipo function calling (Semana 6): nombre, descripción y JSON Schema de entrada.
TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": "Calcula la fecha de vencimiento de un plazo según el art. 183 del Código Civil "
                       "(y el art. 2007 para caducidad), con el calendario de feriados nacionales 2026.",
        "parameters": PlazoInput.model_json_schema(),
    },
}


def _load_holidays() -> tuple[int, set[date]]:
    data = json.loads(config.HOLIDAYS_PATH.read_text(encoding="utf-8"))
    return data["anio"], {date.fromisoformat(d) for d, _ in data["feriados"]}


async def cargar_calendario(fault: str, attempt: int) -> tuple[int, set[date]]:
    """Fuente externa simulada del calendario. `fault` es un parámetro del experimento."""
    if fault == "transitorio_una_vez" and attempt == 1:
        raise TransientToolError("calendario no disponible temporalmente")
    if fault == "transitorio_siempre":
        raise TransientToolError("calendario no disponible temporalmente")
    if fault == "lento":
        await asyncio.sleep(0.5)
    return _load_holidays()


def es_inhabil(d: date, feriados: set[date]) -> bool:
    return d.weekday() >= 5 or d in feriados


def sumar_meses(d: date, meses: int) -> date:
    """Art. 183 inc. 2: mismo día del mes de vencimiento o, si falta, el último día de ese mes."""
    year, month0 = divmod(d.month - 1 + meses, 12)
    year, month = d.year + year, month0 + 1
    return date(year, month, min(d.day, calendar.monthrange(year, month)[1]))


async def computar_vencimiento(args: PlazoInput, fault: str, attempt: int) -> dict[str, Any]:
    if args.computo_dias == "habiles" and args.unidad != "dias":
        raise DomainError("habiles_requiere_dias", "El cómputo por días hábiles solo aplica a plazos en días.")
    anio, feriados = await cargar_calendario(fault, attempt)
    if args.fecha_inicio.year != anio:
        raise DomainError("fuera_de_calendario", f"El calendario cargado solo cubre {anio}.")

    reglas: list[str] = []
    saltados = 0
    if args.unidad == "dias" and args.computo_dias == "naturales":
        nominal = args.fecha_inicio + timedelta(days=args.cantidad)
        reglas.append("art. 183 inc. 1: plazo por días, computado en días naturales")
    elif args.unidad == "dias":
        nominal, contados = args.fecha_inicio, 0
        while contados < args.cantidad:
            nominal += timedelta(days=1)
            if nominal.year != anio:
                raise DomainError("fuera_de_calendario", f"El vencimiento sale del calendario {anio}.")
            if es_inhabil(nominal, feriados):
                saltados += 1
            else:
                contados += 1
        reglas.append("art. 183 inc. 1: plazo por días hábiles, establecido por la ley o el acto")
    else:
        meses = args.cantidad * (12 if args.unidad == "anios" else 1)
        nominal = sumar_meses(args.fecha_inicio, meses)
        inciso = "inc. 3 (años, con la regla del inc. 2)" if args.unidad == "anios" else "inc. 2"
        reglas.append(f"art. 183 {inciso}: vence el día correspondiente del mes final")
    reglas.append("art. 183 inc. 4: excluye el día inicial e incluye el día del vencimiento")
    if nominal.year != anio:
        raise DomainError("fuera_de_calendario", f"El vencimiento sale del calendario {anio}.")

    final = nominal
    fundamento = ["cc_art_183"]
    if es_inhabil(final, feriados):
        if args.naturaleza == "caducidad":
            reglas.append("art. 2007: la caducidad se produce el último día aunque sea inhábil")
            fundamento.append("cc_art_2007")
        else:
            while es_inhabil(final, feriados):
                final += timedelta(days=1)
                saltados += 1
                if final.year != anio:
                    raise DomainError("fuera_de_calendario", f"El vencimiento sale del calendario {anio}.")
            reglas.append("art. 183 inc. 5: el último día era inhábil; vence el primer día hábil siguiente")

    if fault == "salida_corrupta":
        return {"fecha_vencimiento": "pronto", "reglas_aplicadas": []}
    return {
        "fecha_vencimiento": final.isoformat(),
        "fecha_vencimiento_nominal": nominal.isoformat(),
        "dia_semana_vencimiento": DIAS[final.weekday()],
        "dias_inhabiles_saltados": saltados,
        "reglas_aplicadas": reglas,
        "fundamento": fundamento,
    }


TOOL_IMPLS = {TOOL_NAME: computar_vencimiento}


def _error(kind: ErrorKind, code: str, message: str, retryable: bool) -> ToolResult:
    return ToolResult(status="error", error=ToolError(code=code, kind=kind, retryable=retryable, message=message))


def should_retry(error: ToolError, attempt: int, policy: Policy = POLICY) -> bool:
    """Solo se reintentan errores transitorios o de timeout y mientras queden intentos."""
    return error.retryable and attempt < policy.max_attempts


async def ejecutar(tool_name: str, raw_args: dict[str, Any], fault: str = "ninguna",
                   policy: Policy = POLICY) -> dict[str, Any]:
    started = time.perf_counter()
    traces: list[dict[str, Any]] = []

    def finish(result: ToolResult, attempts: int) -> dict[str, Any]:
        return {"tool": tool_name, "args": raw_args, "fault": fault, "attempts": attempts,
                "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                "result": result.model_dump(mode="json"), "traces": traces}

    if tool_name not in TOOL_IMPLS:
        return finish(_error("protocol", "herramienta_no_permitida", f"Herramienta no permitida: {tool_name}", False), 0)
    try:
        args = PlazoInput.model_validate(raw_args)
    except ValidationError as exc:
        detail = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
        return finish(_error("validation", "entrada_invalida", detail, False), 0)

    attempt = 0
    while True:
        attempt += 1
        remaining = policy.overall_deadline_s - (time.perf_counter() - started)
        if remaining <= 0:
            return finish(_error("timeout", "limite_total_agotado", "Se agotó el límite temporal total.", False), attempt - 1)
        t0 = time.perf_counter()
        try:
            raw = await asyncio.wait_for(TOOL_IMPLS[tool_name](args, fault, attempt),
                                         timeout=min(policy.per_attempt_timeout_s, remaining))
            traces.append({"attempt": attempt, "outcome": "ok", "ms": round((time.perf_counter() - t0) * 1000, 1)})
            break
        except DomainError as exc:
            result = _error("domain", exc.code, str(exc), False)
        except TransientToolError as exc:
            result = _error("transient", "calendario_no_disponible", str(exc), True)
        except asyncio.TimeoutError:
            result = _error("timeout", "tiempo_agotado", "El intento superó el tiempo máximo.", True)
        except Exception as exc:  # error no previsto: no se reintenta
            result = _error("execution", "error_interno", type(exc).__name__, False)
        traces.append({"attempt": attempt, "outcome": result.error.kind,
                       "ms": round((time.perf_counter() - t0) * 1000, 1)})
        if not should_retry(result.error, attempt, policy):
            return finish(result, attempt)
        wait = policy.backoff_s[min(attempt - 1, len(policy.backoff_s) - 1)]
        if policy.overall_deadline_s - (time.perf_counter() - started) <= wait:
            return finish(result, attempt)
        await asyncio.sleep(wait)

    try:
        data = PlazoOutput.model_validate(raw).model_dump(mode="json")
    except ValidationError as exc:
        return finish(_error("output_contract", "salida_invalida", f"{len(exc.errors())} campos no cumplen el contrato", False), attempt)
    return finish(ToolResult(status="ok", data=data), attempt)

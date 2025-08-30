# ai/observaciones.py
from __future__ import annotations

import os
import time
import json
from typing import List, Dict, Tuple
from dataclasses import dataclass

import polars as pl

# SDK oficial Google GenAI (no confundir con librerías legacy)
# pip install google-genai polars pydantic
from pydantic import BaseModel
from google import genai
from google.genai import types as genai_types


# ======== Config =========
MODEL_NAME = "gemini-2.5-flash"  # modelo pedido
MAX_ATTEMPTS = 3  # intentos por batch
RETRY_SECONDS = 10  # pausa entre intentos
BATCH_SIZE = 100  # tamaño de lote (ajustable)

INPUT_CSV = "observaciones.csv"  # generado previamente (id, observaciones)
OUTPUT_CSV = (
    "observaciones_final.csv"  # id, observaciones_original, observaciones_final
)


# ======== Esquema de salida estructurada ========
class Correction(BaseModel):
    id: int
    observaciones_final: str


# Para evitar problemas con modelos anidados, usamos directamente list[Correction] como schema
# (la SDK soporta response_schema con Pydantic / enums / JSON schema).
# https://ai.google.dev/gemini-api/docs/structured-output
# https://github.com/googleapis/python-genai
ResponseSchema = list[Correction]


def load_input() -> pl.DataFrame:
    df = pl.read_csv(INPUT_CSV)
    # Normalizamos nombres esperados
    if "observaciones" not in df.columns or "id" not in df.columns:
        raise ValueError("Se espera un CSV con columnas: id, observaciones")
    return df.select(["id", "observaciones"])


def chunked(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def build_prompt() -> str:
    # Instrucciones en español, explícitas y concisas:
    return (
        "Tarea: Corregí ortografía, mayúsculas/minúsculas, espacios y signos de puntuación "
        "de cada observación SIN perder información ni resumir. No agregues contenido. "
        "No cambies el significado. Conservá números, nombres propios y detalles.\n\n"
        "Devolvé SOLO JSON válido ajustado al esquema (lista de objetos), "
        "sin texto adicional."
    )


def call_gemini_batch(
    client: genai.Client,
    batch_items: list[dict],
) -> list[Correction]:
    """
    Llama al modelo para un batch de entradas.
    batch_items: [{'id': int, 'observaciones': '...'}, ...]
    Retorna lista de Correction (id, observaciones_final).
    """
    prompt = build_prompt()

    # Contenido: instrucciones + datos de entrada
    contents = [
        prompt,
        "Entradas (lista de objetos con 'id' y 'observaciones'):",
        json.dumps(batch_items, ensure_ascii=False),
        "Salida esperada: lista JSON de objetos {id: int, observaciones_final: string}.",
    ]

    # Configuramos salida JSON + schema (lista de Correction)
    cfg = genai_types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=ResponseSchema,  # list[Correction]
        # Opcional: temperature baja para consistencia
        temperature=0.2,
    )

    resp = client.models.generate_content(
        model=MODEL_NAME,
        contents=contents,
        config=cfg,
    )

    # La SDK expone .parsed cuando se usa response_schema
    # https://googleapis.github.io/python-genai/  (ver GenerateContentResponse)
    try:
        parsed = resp.parsed  # -> list[Correction] si el schema se respetó
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass

    # Fallback: intentar parsear JSON manualmente desde .text
    text = (resp.text or "").strip()
    if not text:
        return []
    try:
        data = json.loads(text)
        # data esperado: lista de dicts con id y observaciones_final
        out: list[Correction] = []
        if isinstance(data, list):
            for item in data:
                if (
                    isinstance(item, dict)
                    and "id" in item
                    and "observaciones_final" in item
                ):
                    out.append(Correction(**item))
        return out
    except Exception:
        return []


def process_all(df_in: pl.DataFrame) -> pl.DataFrame:
    """
    Procesa todas las observaciones con reintentos parciales.
    Devuelve DataFrame: id, observaciones_original, observaciones_final
    SOLO con filas efectivamente procesadas por la IA.
    """
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    records = df_in.to_dicts()  # [{'id':..., 'observaciones':...}, ...]
    pending = records[:]  # observaciones aún no corregidas
    results: Dict[int, str] = {}

    attempt = 1
    while pending and attempt <= MAX_ATTEMPTS:
        new_pending: list[dict] = []

        for batch in chunked(pending, BATCH_SIZE):
            # Mapeo rápido de ids en el batch
            batch_ids = {item["id"] for item in batch}

            # Llamada al modelo
            try:
                corr_list = call_gemini_batch(client, batch)
            except Exception as e:
                # Si hubo error de red o rate-limit, reintentar todo el batch
                # (ver límites y mejores prácticas de reintentos)
                # https://ai.google.dev/gemini-api/docs/rate-limits
                corr_list = []

            # Registrar resultados correctos
            got_ids = set()
            for c in corr_list:
                if c.id in batch_ids and isinstance(c.observaciones_final, str):
                    results[c.id] = c.observaciones_final
                    got_ids.add(c.id)

            # Re-encolar los que faltan
            for item in batch:
                if item["id"] not in got_ids:
                    new_pending.append(item)

        if new_pending and attempt < MAX_ATTEMPTS:
            time.sleep(RETRY_SECONDS)

        pending = new_pending
        attempt += 1

    # Construir salida solo para ids procesados
    if not results:
        return pl.DataFrame(
            schema={
                "id": pl.Int64,
                "observaciones_original": pl.Utf8,
                "observaciones_final": pl.Utf8,
            }
        )

    df_res = pl.DataFrame(
        {"id": list(results.keys()), "observaciones_final": list(results.values())}
    ).with_columns(pl.col("id").cast(pl.Int64))

    out = (
        df_in.rename({"observaciones": "observaciones_original"})
        .join(df_res, on="id", how="inner")
        .select(["id", "observaciones_original", "observaciones_final"])
        .sort("id")
    )
    return out


def main():
    df_in = load_input()
    df_out = process_all(df_in)
    df_out.write_csv(OUTPUT_CSV)
    print(f"✅ Generado {OUTPUT_CSV} con {df_out.height} filas procesadas")


if __name__ == "__main__":
    # Requiere GEMINI_API_KEY en el entorno
    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("Falta variable de entorno GEMINI_API_KEY")
    main()

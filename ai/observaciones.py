# ai/observaciones.py
import os
import json
import time
from typing import List, Dict, Tuple

import polars as pl
from pydantic import BaseModel
from google import genai


# ======== Config =========
MODEL_NAME = "gemini-2.5-flash"
INPUT_CSV = "observaciones.csv"
OUTPUT_CSV = "observaciones_final.csv"
BATCH_SIZE = 100  # tamaño de lote; ajusta si lo necesitás
MAX_ATTEMPTS = 3
RETRY_SLEEP_SEC = 10


class ObservacionFix(BaseModel):
    id: int
    observaciones_final: str


def _mk_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Falta la variable de entorno GEMINI_API_KEY.")
    return genai.Client(api_key=api_key)


def _build_contents(batch_items: List[Tuple[int, str]]) -> str:
    """Construye el prompt con instrucciones y payload JSON."""
    entrada = [{"id": i, "observaciones_original": t} for i, t in batch_items]
    prompt = (
        "Corrige ortografía, gramática, tildes, mayúsculas/minúsculas, espacios y signos de puntuación "
        "en español para cada observación, sin perder información ni cambiar el significado. "
        "No agregues ni elimines datos. Mantén números, fechas y nombres propios cuando existan.\n\n"
        "Devuelve una lista JSON que cumpla exactamente con el esquema, un objeto por cada item de entrada. "
        "Respeta los ids.\n\n"
        "ENTRADA:\n"
        f"{json.dumps(entrada, ensure_ascii=False)}\n\n"
        "ESQUEMA DE SALIDA (JSON): lista de objetos con campos:\n"
        "- id: entero (copiar desde la entrada)\n"
        "- observaciones_final: string (texto corregido)\n"
    )
    return prompt


def _call_gemini(client: genai.Client, items: List[Tuple[int, str]]) -> Dict[int, str]:
    """
    Llama al modelo con response_schema tipado (pydantic) y devuelve {id: observaciones_final}.
    Puede devolver subconjunto si el modelo responde parcialmente.
    """
    if not items:
        return {}

    prompt = _build_contents(items)
    # Respuesta estructurada: application/json + response_schema
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": List[ObservacionFix],
        },
    )

    out_map: Dict[int, str] = {}
    try:
        parsed: List[ObservacionFix] = response.parsed  # objetos pydantic
        for obj in parsed:
            out_map[obj.id] = obj.observaciones_final
    except Exception:
        # Si no pudo parsear, intentar parsear como JSON raw (por si vino texto)
        try:
            data = json.loads(response.text)
            if isinstance(data, list):
                for obj in data:
                    if (
                        isinstance(obj, dict)
                        and "id" in obj
                        and "observaciones_final" in obj
                    ):
                        out_map[int(obj["id"])] = str(obj["observaciones_final"])
        except Exception:
            # Si tampoco, devolver vacío (se reintenta en el caller)
            return {}

    return out_map


def _retry_batch(
    client: genai.Client, batch_items: List[Tuple[int, str]]
) -> Dict[int, str]:
    """
    Reintenta hasta MAX_ATTEMPTS: si la respuesta es parcial, vuelve a consultar
    únicamente por los ids faltantes, con 10s entre intentos.
    """
    remaining = list(batch_items)
    results: Dict[int, str] = {}

    for attempt in range(1, MAX_ATTEMPTS + 1):
        got = _call_gemini(client, remaining)
        # Incorporar lo que llegó
        if got:
            results.update(got)
        # Calcular faltantes
        done_ids = set(results.keys())
        remaining = [(i, t) for i, t in remaining if i not in done_ids]

        print(
            f"[Batch] Intento {attempt}: recibidos {len(got)} / acumulados {len(results)} / faltan {len(remaining)}"
        )

        if not remaining:
            break

        if attempt < MAX_ATTEMPTS:
            time.sleep(RETRY_SLEEP_SEC)

    return results


def main():
    # 1) Leer observaciones.csv (id, observaciones)
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"No se encontró {INPUT_CSV} en el repo público.")

    df = pl.read_csv(INPUT_CSV)
    # columnas esperadas: id, observaciones
    if not {"id", "observaciones"}.issubset(set(df.columns)):
        raise KeyError(
            "Se esperan columnas 'id' y 'observaciones' en observaciones.csv."
        )

    # Asegurar tipos
    df = df.select(
        pl.col("id").cast(pl.Int64),
        pl.col("observaciones").cast(pl.Utf8).str.strip_chars(),
    )

    rows = list(zip(df["id"].to_list(), df["observaciones"].to_list()))
    print(f"Total de observaciones a procesar: {len(rows)}")

    client = _mk_client()

    # 2) Procesar por lotes con reintentos y completitud
    results: Dict[int, str] = {}
    for start in range(0, len(rows), BATCH_SIZE):
        batch = rows[start : start + BATCH_SIZE]
        print(f"\nProcesando batch {start//BATCH_SIZE + 1} (items: {len(batch)})...")
        fixed_map = _retry_batch(client, batch)
        results.update(fixed_map)

    print(f"\nProcesadas por IA: {len(results)} / {len(rows)}")

    # 3) Construir DataFrame SOLO con ids procesados por IA
    if results:
        out_df = (
            df.filter(pl.col("id").is_in(list(results.keys())))
            .with_columns(
                pl.col("observaciones").alias("observaciones_original"),
            )
            .with_columns(
                pl.col("id"),
                pl.col("observaciones_original"),
                pl.Series(
                    name="observaciones_final",
                    values=[
                        results[i]
                        for i in df.filter(pl.col("id").is_in(list(results.keys())))[
                            "id"
                        ].to_list()
                    ],
                ),
            )
            .select(["id", "observaciones_original", "observaciones_final"])
            .sort("id")
        )
    else:
        out_df = pl.DataFrame(
            {"id": [], "observaciones_original": [], "observaciones_final": []}
        )

    # 4) Guardar resultado
    out_df.write_csv(OUTPUT_CSV)
    print(f"✅ Generado {OUTPUT_CSV} con {out_df.height} filas")


if __name__ == "__main__":
    main()

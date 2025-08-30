import os, time, json
from pathlib import Path
from typing import Dict, List
import polars as pl
from pydantic import BaseModel
from google import genai
from google.genai import types as genai_types

MODEL_NAME = "gemini-2.5-flash"
MAX_ATTEMPTS = 3
RETRY_SECONDS = 10
BATCH_SIZE = 100

INPUT_CSV = Path("data/observaciones.csv")  # id, observaciones
OUTPUT_CSV = Path("data/observaciones_final.csv")  # id, observaciones_final


class Correction(BaseModel):
    id: int
    observaciones_final: str


ResponseSchema = list[Correction]


def load_input() -> pl.DataFrame:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"No existe {INPUT_CSV}")
    df = pl.read_csv(INPUT_CSV)
    if not {"id", "observaciones"}.issubset(df.columns):
        raise ValueError("Se espera CSV con columnas: id, observaciones")
    return df.select(["id", "observaciones"])


def chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def prompt() -> str:
    return (
        "Tarea: Corregí ortografía, mayúsculas/minúsculas, espacios y signos de puntuación "
        "de cada observación SIN perder información ni resumir. No agregues contenido ni cambies el significado. "
        "Devolvé SOLO JSON válido con [{id:int, observaciones_final:string}]."
    )


def call_batch(client: genai.Client, batch: List[dict]) -> List[Correction]:
    contents = [
        prompt(),
        "Entradas (lista de {id, observaciones}):",
        json.dumps(batch, ensure_ascii=False),
        "Salida: lista JSON de {id, observaciones_final}.",
    ]
    cfg = genai_types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=ResponseSchema,
        temperature=0.2,
    )
    resp = client.models.generate_content(
        model=MODEL_NAME, contents=contents, config=cfg
    )
    try:
        parsed = resp.parsed
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass
    txt = (resp.text or "").strip()
    if not txt:
        return []
    try:
        data = json.loads(txt)
        out = []
        if isinstance(data, list):
            for it in data:
                if isinstance(it, dict) and "id" in it and "observaciones_final" in it:
                    out.append(Correction(**it))
        return out
    except Exception:
        return []


def main():
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    df = load_input()
    records = df.to_dicts()
    pending = records[:]
    results: Dict[int, str] = {}
    attempt = 1
    while pending and attempt <= MAX_ATTEMPTS:
        new_pending = []
        for batch in chunked(pending, BATCH_SIZE):
            ids = {x["id"] for x in batch}
            try:
                corr = call_batch(client, batch)
            except Exception:
                corr = []
            got = set()
            for c in corr:
                if c.id in ids and isinstance(c.observaciones_final, str):
                    results[c.id] = c.observaciones_final
                    got.add(c.id)
            for item in batch:
                if item["id"] not in got:
                    new_pending.append(item)
        if new_pending and attempt < MAX_ATTEMPTS:
            time.sleep(RETRY_SECONDS)
        pending = new_pending
        attempt += 1

    if not results:
        raise RuntimeError("No se generaron correcciones.")
    out = (
        pl.DataFrame(
            {"id": list(results.keys()), "observaciones_final": list(results.values())}
        )
        .with_columns(pl.col("id").cast(pl.Int64))
        .sort("id")
    )
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.write_csv(OUTPUT_CSV)
    print(f"✅ {OUTPUT_CSV} con {out.height} filas")


if __name__ == "__main__":
    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("Falta GEMINI_API_KEY")
    main()

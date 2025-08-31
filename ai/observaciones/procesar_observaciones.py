import os, time, json
from pathlib import Path
from typing import Dict, List
import polars as pl
from pydantic import BaseModel
from google import genai
from google.genai import types as genai_types
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = "gemini-2.5-flash"
MAX_ATTEMPTS = 3
RETRY_SECONDS = 10
BATCH_SIZE = 100

INPUT_CSV = Path("data/observaciones.csv")  # columnas: id, observaciones
OUTPUT_CSV = Path(
    "data/observaciones_final.csv"
)  # columnas: id, observaciones_original, observaciones_final


class Correction(BaseModel):
    id: int
    observaciones_final: str


ResponseSchema = list[Correction]


def _read_input() -> pl.DataFrame:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"No existe {INPUT_CSV}")
    df = pl.read_csv(INPUT_CSV)
    if not {"id", "observaciones"}.issubset(df.columns):
        raise ValueError("Se espera CSV con columnas: id, observaciones")
    return df.select(
        pl.col("id").cast(pl.Int64),
        pl.col("observaciones").cast(pl.Utf8),
    ).unique(subset="id", keep="last")


def _read_or_empty_output() -> pl.DataFrame:
    if OUTPUT_CSV.exists():
        df = pl.read_csv(OUTPUT_CSV)
        expected = {"id", "observaciones_original", "observaciones_final"}
        if not expected.issubset(df.columns):
            raise ValueError(f"{OUTPUT_CSV} debe tener columnas {expected}")
        return df.select(
            pl.col("id").cast(pl.Int64),
            pl.col("observaciones_original").cast(pl.Utf8),
            pl.col("observaciones_final").cast(pl.Utf8),
        ).unique(subset="id", keep="last")
    return pl.DataFrame(
        schema={
            "id": pl.Int64,
            "observaciones_original": pl.Utf8,
            "observaciones_final": pl.Utf8,
        }
    )


def _prompt() -> str:
    return (
        "Tarea: Corregí ortografía, mayúsculas/minúsculas, espacios y signos de puntuación "
        "de cada observación SIN perder información ni resumir. No agregues contenido ni cambies el significado. "
        "Devolvé SOLO JSON válido con [{id:int, observaciones_final:string}]."
    )


def _call_batch(client: genai.Client, batch: List[dict]) -> List[Correction]:
    contents = [
        _prompt(),
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


def procesar() -> pl.DataFrame:
    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("Falta GEMINI_API_KEY")

    df_in = _read_input()
    df_out = _read_or_empty_output()

    # Determinar pendientes:
    # - si no existe output => procesar TODO
    # - si existe => procesar ids con observaciones_final nula (y que sigan vigentes en observaciones.csv)
    if df_out.is_empty():
        df_todo = df_in
    else:
        pendientes_ids = df_out.filter(pl.col("observaciones_final").is_null()).select(
            "id"
        )
        df_todo = df_in.join(pendientes_ids, on="id", how="inner")

    if df_todo.height == 0:
        print("✅ procesar_observaciones: no hay pendientes. Nada que hacer.")
        # reordenar/reescribir por las dudas
        df_out.sort("id").write_csv(OUTPUT_CSV, include_header=True)
        return df_out.sort("id")

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    records = df_todo.to_dicts()
    pending = records[:]
    results: Dict[int, str] = {}
    attempt = 1

    while pending and attempt <= MAX_ATTEMPTS:
        new_pending = []
        for i in range(0, len(pending), BATCH_SIZE):
            batch = pending[i : i + BATCH_SIZE]
            ids = {x["id"] for x in batch}
            try:
                corr = _call_batch(client, batch)
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

    df_results = pl.DataFrame(
        {"id": list(results.keys()), "observaciones_final": list(results.values())}
    ).with_columns(pl.col("id").cast(pl.Int64))

    if df_out.is_empty():
        # Creamos desde cero
        final_df = (
            df_in.join(df_results, on="id", how="left")
            .select(
                "id",
                pl.col("observaciones").alias("observaciones_original"),
                "observaciones_final",
            )
            .sort("id")
        )
    else:
        # Actualizamos observaciones_final para los ids procesados, y sincronizamos original a lo vigente
        final_df = (
            df_out.join(df_results, on="id", how="left", suffix="_new")
            .join(df_in, on="id", how="left")  # trae observaciones vigentes
            .with_columns(
                # si hay nuevo resultado, usarlo; sino mantener el actual
                pl.coalesce(
                    [pl.col("observaciones_final_new"), pl.col("observaciones_final")]
                ).alias("observaciones_final"),
                # asegurar que el original coincida con lo vigente
                pl.when(pl.col("observaciones").is_not_null())
                .then(pl.col("observaciones"))
                .otherwise(pl.col("observaciones_original"))
                .alias("observaciones_original"),
            )
            .select("id", "observaciones_original", "observaciones_final")
            .unique(subset="id", keep="last")
            .sort("id")
        )

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    final_df.write_csv(OUTPUT_CSV, include_header=True)
    print(
        f"✅ procesar_observaciones: +{df_results.height} filas procesadas. Total={final_df.height}"
    )

    return final_df


if __name__ == "__main__":
    procesar()

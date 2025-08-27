import argparse
import polars as pl
from pathlib import Path
from google import genai

INPUT_CSV = None
OUTPUT_CSV = Path("ai/output.csv")
PROCESADAS_CSV = Path("ai/observaciones_procesadas.csv")


def procesar_texto(text: str) -> str:
    client = genai.Client()
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=text,
        config={
            "response_mime_type": "text/plain",
        },
    )
    return response.text.strip()


def main(input_path: str):
    df = pl.read_csv(input_path)
    df = df.with_columns(
        pl.col("observacion_original").apply(procesar_texto).alias("observaciones")
    )
    df_out = df.select(["id", "observaciones"])
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_out.write_csv(OUTPUT_CSV)

    df_proc = (
        pl.read_csv(PROCESADAS_CSV)
        if PROCESADAS_CSV.exists()
        else pl.DataFrame(df_out.schema)
    )
    df_all = pl.concat([df_proc, df_out]).unique(subset=["observaciones"])
    df_all.write_csv(PROCESADAS_CSV)

    print(f"Salida guardada en: {OUTPUT_CSV}")
    print(f"Procesadas consolidadas en: {PROCESADAS_CSV}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=str, required=True, help="Ruta al CSV de observaciones"
    )
    args = parser.parse_args()
    INPUT_CSV = args.input
    main(INPUT_CSV)

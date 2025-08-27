from pathlib import Path
import polars as pl
from google import genai

INPUT_PATH = Path("ai/observaciones.csv")
OUTPUT_PATH = Path("ai/output.csv")
PROCESADAS_PATH = Path("ai/observaciones_procesadas.csv")
PROMPT = (
    "Corrige la ortografía, puntuación, mayúsculas/minúsculas, espacios y estructura de la observación "
    "sin alterar su contenido. Devuelve solo el texto corregido."
)


def procesar_con_gemini(text: str) -> str:
    client = genai.Client()
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=text,
        config={
            "response_mime_type": "text/plain",
        },
    )
    return response.text.strip()


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"No se encontró: {INPUT_PATH}")

    df = pl.read_csv(INPUT_PATH)

    # Aplica corrección a cada observación
    df = df.with_columns(
        [
            pl.col("observacion_original")
            .apply(lambda x: procesar_con_gemini(x))
            .alias("observaciones")
        ]
    )

    # Mantener la estructura: id, observaciones
    df.select(["id", "observaciones"]).write_csv(OUTPUT_PATH)

    print(f"Output generado: {OUTPUT_PATH}")

    # Actualizar observaciones_procesadas.csv (concatenar o reemplazar del día)
    df_processed = df.select(["id", "observaciones"])
    df_proc_old = (
        pl.read_csv(PROCESADAS_PATH)
        if PROCESADAS_PATH.exists()
        else pl.DataFrame(columns=df_processed.columns)
    )
    df_final = pl.concat([df_proc_old, df_processed]).unique(subset=["observaciones"])
    df_final.write_csv(PROCESADAS_PATH)

    print(f"Observaciones procesadas consolidadas en: {PROCESADAS_PATH}")


if __name__ == "__main__":
    main()

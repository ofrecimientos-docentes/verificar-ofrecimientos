# core/observaciones/preparar_observaciones.py
from __future__ import annotations

from pathlib import Path
import polars as pl

INPUT_CSV = Path("data/observaciones.csv")  # columnas: id, observaciones
OUTPUT_CSV = Path(
    "data/observaciones_final.csv"
)  # columnas: id, observaciones_original, observaciones_final


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


def preparar() -> pl.DataFrame:
    """
    Sincroniza observaciones_final con observaciones, y deja pendientes las que deban re-procesarse.
    Reglas:
      - ids que no estén en observaciones.csv => se eliminan de observaciones_final.
      - ids nuevos => se agregan con observaciones_original y observaciones_final = null.
      - si cambió el texto de observaciones => se actualiza observaciones_original y se vacía observaciones_final (requiere reproceso).
    Devuelve el DF final ya listo (pendientes = filas con observaciones_final nula).
    """
    df_in = _read_input()
    df_out = _read_or_empty_output()

    # 1) eliminar antiguos (ids no vigentes)
    df_out = df_out.join(df_in.select("id"), on="id", how="inner")

    # 2) actualizar originales y vaciar finales si cambió el texto
    df_out = (
        df_out.join(
            df_in, on="id", how="left"
        )  # agrega columna "observaciones" (vigente)
        .with_columns(
            pl.when(
                pl.col("observaciones").is_not_null()
                & (pl.col("observaciones") != pl.col("observaciones_original"))
            )
            .then(pl.col("observaciones"))
            .otherwise(pl.col("observaciones_original"))
            .alias("observaciones_original"),
            pl.when(
                pl.col("observaciones").is_not_null()
                & (pl.col("observaciones") != pl.col("observaciones_original"))
            )
            .then(pl.lit(None, dtype=pl.Utf8))  # invalidar final si cambió el original
            .otherwise(pl.col("observaciones_final"))
            .alias("observaciones_final"),
        )
        .drop("observaciones")
    )

    # 3) agregar ids nuevos (final nulo para que queden pendientes)
    faltantes = df_in.join(df_out.select("id"), on="id", how="anti")
    nuevos = faltantes.select(
        pl.col("id"),
        pl.col("observaciones").alias("observaciones_original"),
        pl.lit(None, dtype=pl.Utf8).alias("observaciones_final"),
    )

    final_df = (
        pl.concat([df_out, nuevos], how="vertical")
        .unique(subset="id", keep="last")
        .sort("id")
    )

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    final_df.write_csv(OUTPUT_CSV, include_header=True)

    print(
        f"🧭 preparar_observaciones: {final_df.height} filas vigentes en {OUTPUT_CSV} "
        f"(nuevas={nuevos.height}, revalidadas={(final_df['observaciones_final'].is_null()).sum()})"
    )
    return final_df


if __name__ == "__main__":
    preparar()

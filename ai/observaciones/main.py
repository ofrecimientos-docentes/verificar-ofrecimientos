from pathlib import Path
import sys

from ai.observaciones.preparar_observaciones import preparar
from ai.observaciones.procesar_observaciones import procesar


INPUT_CSV = Path("data/observaciones.csv")


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"No se puede continuar: falta {INPUT_CSV}. Este archivo es obligatorio."
        )
    # 1) sincronizar estado (vigentes, antiguas, cambios de texto)
    preparar()
    # 2) procesar únicamente las pendientes
    procesar()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Error en observaciones: {e}", file=sys.stderr)
        raise

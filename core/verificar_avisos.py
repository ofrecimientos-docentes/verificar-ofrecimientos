from pathlib import Path
from urllib.parse import urljoin
from html import unescape

import requests
from bs4 import BeautifulSoup
import polars as pl


URL = "https://educacion.sanjuan.edu.ar/mesj/LlamadosdePrensa/OfrecimientosdeHsC%C3%A1tedraCargosaCubrir.aspx"
BASE = "https://educacion.sanjuan.edu.ar"
OUT_PATH = Path("data/avisos.csv")


def fetch_html(url: str) -> str:
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0 Safari/537.36",
        "Accept-Language": "es-AR,es;q=0.9,en;q=0.8",
    }
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    # La página viene en UTF-8; BeautifulSoup manejará bien los acentos.
    return resp.text


def parse_avisos(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "lxml")

    # Tabla principal con class="Normal"
    table = soup.select_one("table.Normal")
    if not table:
        raise RuntimeError("No se encontró la tabla con class 'Normal'.")

    tbody = table.find("tbody") or table

    avisos: list[dict] = []

    for tr in tbody.find_all("tr"):
        classes = tr.get("class", [])
        if isinstance(classes, str):
            classes = [classes]

        # Saltar filas de subtítulo
        if "SubHead" in classes:
            continue

        # Enlace del título: preferimos el <td class="TÍTULOCell">, con fallback a
        # cualquier <a> cuya etiqueta no sea 'DESCARGAR'.
        a_tag = None
        titulo_cell = tr.find("td", class_="TÍTULOCell")
        if titulo_cell:
            a_tag = titulo_cell.find("a", href=True)

        if not a_tag:
            for cand in tr.find_all("a", href=True):
                if cand.get_text(strip=True).upper() != "DESCARGAR":
                    a_tag = cand
                    break

        if not a_tag:
            # Nada útil en esta fila
            continue

        titulo = " ".join(a_tag.get_text(strip=True).split())
        href = unescape(a_tag["href"])
        url_abs = urljoin(BASE, href)

        avisos.append({"nombre": titulo, "url": url_abs})

    if not avisos:
        raise RuntimeError("No se extrajeron avisos. ¿Cambió el HTML de la página?")

    return avisos


def save_as_csv(rows: list[dict], out_path: Path) -> None:
    df = pl.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(out_path)


def main():
    html = fetch_html(URL)
    avisos = parse_avisos(html)
    save_as_csv(avisos, OUT_PATH)
    print(f"Guardado {len(avisos)} avisos en {OUT_PATH}")


if __name__ == "__main__":
    main()

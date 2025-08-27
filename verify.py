import asyncio
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import aiohttp
import polars as pl
import requests
from bs4 import BeautifulSoup
import json5

DAYS = 7
URL = (
    "https://educacion.sanjuan.edu.ar/mesj/"
    "LlamadosdePrensa/OfrecimientosdeHsCátedraCargosaCubrir.aspx"
)
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/138.0.0.0 Safari/537.36"
    ),
    "X-Requested-With": "XMLHttpRequest",
    "Referer": URL,
}

OUT_PATH = Path("ofrecimientos.parquet")


def _get_hidden(soup: BeautifulSoup, name: str) -> str:
    tag = soup.find("input", {"name": name})
    return tag["value"] if tag and tag.has_attr("value") else ""


def _bootstrap_viewstate() -> tuple[str, str]:
    resp = requests.get(URL, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    viewstate = _get_hidden(soup, "__VIEWSTATE")
    viewstate_gen = _get_hidden(soup, "__VIEWSTATEGENERATOR")
    return viewstate, viewstate_gen


async def _fetch_date(
    session: aiohttp.ClientSession, fecha_iso: str, viewstate: str, viewstate_gen: str
) -> list[dict]:
    cfg = {
        "config": {
            "extraParams": {
                "idEstablecimiento": 0,
                "idNivel": 0,
                "idTurno": 0,
                "idSituacionDeRevista": 0,
                "fechaEQ": fecha_iso,
            }
        }
    }
    payload = {
        "__EVENTTARGET": "dnn$ctr1569$OfrecimientosDePlazas$ResourceManager1",
        "__EVENTARGUMENT": "dnn_ctr1569_OfrecimientosDePlazas_Store_Plazas|postback|refresh",
        "__VIEWSTATE": viewstate,
        "__VIEWSTATEGENERATOR": viewstate_gen,
        "submitDirectEventConfig": json.dumps(cfg),
        "__ExtNetDirectEventMarker": "delta=true",
    }
    params = {"_dc": str(int(time.time() * 1000))}
    async with session.post(
        URL, params=params, data=payload, headers=HEADERS, timeout=30
    ) as resp:
        text = await resp.text()

    soup = BeautifulSoup(text, "html.parser")
    ta = soup.find("textarea")
    if not ta:
        return []
    parsed = json5.loads(ta.get_text(strip=True))
    return parsed.get("serviceResponse", {}).get("data", {}).get("data", [])


async def _gather_all(viewstate: str, viewstate_gen: str) -> list[dict]:
    fechas = [
        (datetime.now() + timedelta(days=i)).strftime("%Y-%m-%dT00:00:00")
        for i in range(DAYS)
    ]
    async with aiohttp.ClientSession() as session:
        tasks = [_fetch_date(session, f, viewstate, viewstate_gen) for f in fechas]
        results = await asyncio.gather(*tasks)
    return [row for sub in results for row in sub]


def main() -> None:
    viewstate, viewstate_gen = _bootstrap_viewstate()
    rows = asyncio.run(_gather_all(viewstate, viewstate_gen))
    df = pl.DataFrame(rows) if rows else pl.DataFrame()

    print(f"➡️ Ofrecimientos obtenidos: {df.height}")
    if df.height:
        df.write_parquet(OUT_PATH)
        print(f"✅ Guardado en {OUT_PATH}")
    else:
        print("⚠️ No se obtuvieron datos.")


if __name__ == "__main__":
    main()

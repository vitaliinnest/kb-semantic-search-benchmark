"""
Take cropped screenshots of the Benchmark Explorer pages for the thesis.
Each shot uses a per-page viewport height — full_page=False crops to viewport.
Saves PNG files in docs/screenshots/.
"""
import sys
import pathlib
import asyncio
from urllib.parse import quote

sys.stdout.reconfigure(encoding='utf-8')

ROOT = pathlib.Path("D:/repos/kb-semantic-search-benchmark")
OUT = ROOT / "docs" / "screenshots"
OUT.mkdir(parents=True, exist_ok=True)

BASE = "http://127.0.0.1:5000"
BGE_MODEL = quote("SBERT (BAAI/bge-m3)", safe="")

SHOTS = [
    # (name, url, viewport_h)
    ("explorer_index_tech",     f"{BASE}/benchmark/explorer?domain=tech",                1300),
    ("explorer_index_legal",    f"{BASE}/benchmark/explorer?domain=legal&filter=spread", 1300),
    ("explorer_query_q1",       f"{BASE}/benchmark/explorer/q1?domain=tech",             1500),
    ("explorer_query_L001",     f"{BASE}/benchmark/explorer/L001?domain=legal",          1500),
    ("explorer_query_M001",     f"{BASE}/benchmark/explorer/M001?domain=medical",        1500),
    # Drill-down: queries where BGE-M3 is leader
    ("explorer_drill_bge_m3",   f"{BASE}/benchmark/explorer?domain=tech&model={BGE_MODEL}", 1300),
]


async def main():
    from playwright.async_api import async_playwright
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        for name, url, vh in SHOTS:
            context = await browser.new_context(viewport={"width": 1400, "height": vh})
            page = await context.new_page()
            print(f"  capturing: {name}  vh={vh}")
            await page.goto(url, wait_until="networkidle")
            await page.wait_for_timeout(500)
            out_path = OUT / f"{name}.png"
            await page.screenshot(path=str(out_path), full_page=False)
            print(f"    -> {out_path} ({out_path.stat().st_size:,} bytes)")
            await context.close()
        await browser.close()
    print(f"\nAll screenshots saved to: {OUT}")


asyncio.run(main())

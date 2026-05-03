"""
Take screenshots of the Benchmark Explorer pages for use in the thesis.
Saves PNG files in docs/screenshots/.
"""
import sys
import pathlib
import asyncio

sys.stdout.reconfigure(encoding='utf-8')

ROOT = pathlib.Path("D:/repos/kb-semantic-search-benchmark")
OUT = ROOT / "docs" / "screenshots"
OUT.mkdir(parents=True, exist_ok=True)

BASE = "http://127.0.0.1:5000"
SHOTS = [
    # (name, url, viewport_h, full_page)
    ("explorer_index_tech",   f"{BASE}/benchmark/explorer?domain=tech",                None, True),
    ("explorer_index_legal",  f"{BASE}/benchmark/explorer?domain=legal&filter=spread", None, True),
    ("explorer_query_q1",     f"{BASE}/benchmark/explorer/q1?domain=tech",             None, True),
    ("explorer_query_L001",   f"{BASE}/benchmark/explorer/L001?domain=legal",          None, True),
    ("explorer_query_M001",   f"{BASE}/benchmark/explorer/M001?domain=medical",        None, True),
]


async def main():
    from playwright.async_api import async_playwright
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(viewport={"width": 1400, "height": 900})
        page = await context.new_page()
        for name, url, _, full in SHOTS:
            print(f"  capturing: {name}  {url}")
            await page.goto(url, wait_until="networkidle")
            await page.wait_for_timeout(500)
            out_path = OUT / f"{name}.png"
            await page.screenshot(path=str(out_path), full_page=full)
            print(f"    -> {out_path} ({out_path.stat().st_size:,} bytes)")
        await browser.close()
    print(f"\nAll screenshots saved to: {OUT}")


asyncio.run(main())

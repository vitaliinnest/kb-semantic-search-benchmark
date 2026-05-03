"""
Take screenshots of the updated KB Search UI and replace the corresponding
imageN.png files inside the master's thesis .docx, then repack the .docx.

Usage (Flask must be running on :5000):
    .venv/Scripts/python.exe scripts/update_thesis_screenshots.py

Mapping is figure-aware: each shot is captured at a sensible viewport, then
written directly to unpacked_docx/word/media/imageN.png. After all captures
finish, the .docx is repacked from unpacked_docx/.
"""
import sys
import shutil
import zipfile
import asyncio
from pathlib import Path
from urllib.parse import quote, urlencode

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
DOCX_PATH = ROOT / "2026_M_PI_Nesterenko_VV.docx"
UNPACKED  = ROOT / "unpacked_docx"
MEDIA_DIR = UNPACKED / "word" / "media"
BASE = "http://127.0.0.1:5000"

# (target_image_name, url, viewport_height, label_for_log)
SHOTS = [
    # ── 5.4: documents (tech) ─────────────────────────────────────────────
    ("image4.png",  f"{BASE}/documents?domain=tech",                       1500, "5.4  documents tech"),
    # ── 5.6: build (tech) ─────────────────────────────────────────────────
    ("image6.png",  f"{BASE}/build?domain=tech",                           1100, "5.6  build tech"),
    # ── 5.7: search hero, no query (tech) ─────────────────────────────────
    ("image7.png",  f"{BASE}/?domain=tech",                                900,  "5.7  search hero tech"),
    # ── 5.8: search results tech, BGE-M3 ──────────────────────────────────
    ("image8.png",  f"{BASE}/?" + urlencode({
        "domain": "tech", "model": "bge-m3", "top_k": 5,
        "q": "виявлення аномалій у фінансових транзакціях"
    }), 1500, "5.8  search tech bge"),
    # ── 5.9: search results legal, BGE-M3 ─────────────────────────────────
    ("image9.png",  f"{BASE}/?" + urlencode({
        "domain": "legal", "model": "bge-m3", "top_k": 5,
        "q": "підстави обмеження цивільної дієздатності судом"
    }), 1500, "5.9  search legal bge"),
    # ── 5.10: search results medical, BGE-M3 ──────────────────────────────
    ("image10.png", f"{BASE}/?" + urlencode({
        "domain": "medical", "model": "bge-m3", "top_k": 5,
        "q": "механізм газообміну в легенях та транспорт кисню кров'ю"
    }), 1500, "5.10 search medical bge"),
    # ── 5.12 / 5.21 (reused): benchmark tech (model overview) ─────────────
    ("image12.png", f"{BASE}/benchmark?domain=tech",                       1500, "5.12/5.21 benchmark tech"),
    # ── 5.14 / 5.24 (reused): selection tech ──────────────────────────────
    ("image14.png", f"{BASE}/benchmark/selection?domain=tech",             1500, "5.14/5.24 selection tech"),
    # ── 5.15: explorer index (tech) ───────────────────────────────────────
    ("image30.png", f"{BASE}/benchmark/explorer?domain=tech",              1400, "5.15 explorer index tech"),
    # ── 5.16: explorer detail page (q1, tech) ─────────────────────────────
    ("image31.png", f"{BASE}/benchmark/explorer/q1?domain=tech",           1700, "5.16 explorer detail q1"),
    # ── 5.17: explorer drill-down by leader BGE-M3 (tech) ─────────────────
    ("image32.png", f"{BASE}/benchmark/explorer?domain=tech&model=" + quote("BGE-M3", safe=""), 1300, "5.17 explorer drill bge"),
    # ── 5.18: search results tech, BGE-M3 (different query) ───────────────
    ("image15.png", f"{BASE}/?" + urlencode({
        "domain": "tech", "model": "bge-m3", "top_k": 5,
        "q": "machine learning algorithms overview"
    }), 1500, "5.18 search tech bge en"),
    # ── 5.19: search results tech, nomic (contrast) ───────────────────────
    ("image16.png", f"{BASE}/?" + urlencode({
        "domain": "tech", "model": "nomic", "top_k": 5,
        "q": "алгоритми сортування та їх складність"
    }), 1500, "5.19 search tech nomic"),
    # ── 5.20: search results medical, E5-base ─────────────────────────────
    ("image17.png", f"{BASE}/?" + urlencode({
        "domain": "medical", "model": "e5-base", "top_k": 5,
        "q": "першочергова допомога при серцевому нападі"
    }), 1500, "5.20 search medical e5"),
    # ── 5.22: benchmark legal ─────────────────────────────────────────────
    ("image18.png", f"{BASE}/benchmark?domain=legal",                      1500, "5.22 benchmark legal"),
    # ── 5.23: benchmark medical ───────────────────────────────────────────
    ("image19.png", f"{BASE}/benchmark?domain=medical",                    1500, "5.23 benchmark medical"),
    # ── 5.25: selection legal ─────────────────────────────────────────────
    ("image20.png", f"{BASE}/benchmark/selection?domain=legal",            1500, "5.25 selection legal"),
    # ── 5.26: selection medical ───────────────────────────────────────────
    ("image21.png", f"{BASE}/benchmark/selection?domain=medical",          1500, "5.26 selection medical"),
]

VIEWPORT_W = 1400


def ensure_unpacked():
    """Re-extract the current .docx into unpacked_docx/ for fresh state."""
    if UNPACKED.exists():
        shutil.rmtree(UNPACKED)
    with zipfile.ZipFile(DOCX_PATH) as zf:
        zf.extractall(UNPACKED)
    print(f"[1/3] unpacked {DOCX_PATH.name} -> {UNPACKED}/")


async def take_shots():
    print(f"[2/3] capturing {len(SHOTS)} screenshots…")
    from playwright.async_api import async_playwright
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        for img_name, url, vh, label in SHOTS:
            ctx = await browser.new_context(
                viewport={"width": VIEWPORT_W, "height": vh},
                locale="uk-UA",
            )
            page = await ctx.new_page()
            try:
                await page.goto(url, wait_until="networkidle", timeout=120_000)
                await page.wait_for_timeout(900)
                target = MEDIA_DIR / img_name
                await page.screenshot(path=str(target), full_page=False)
                size_kb = target.stat().st_size // 1024
                print(f"   OK  {img_name:<13} ({size_kb:>4} KB)  {label}")
            except Exception as exc:
                print(f"   FAIL {img_name:<13}  {label}  -> {exc}")
            finally:
                await ctx.close()
        await browser.close()


def repack_docx():
    """Repack unpacked_docx/ into the .docx (same path)."""
    tmp = DOCX_PATH.with_suffix(".docx.tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(UNPACKED.rglob("*")):
            if path.is_file():
                arcname = path.relative_to(UNPACKED).as_posix()
                zf.write(path, arcname)
    DOCX_PATH.unlink()
    tmp.rename(DOCX_PATH)
    size_kb = DOCX_PATH.stat().st_size // 1024
    print(f"[3/3] repacked -> {DOCX_PATH.name} ({size_kb:,} KB)")


def main():
    if not DOCX_PATH.exists():
        sys.exit(f"thesis not found: {DOCX_PATH}")
    ensure_unpacked()
    if not MEDIA_DIR.exists():
        sys.exit(f"media dir missing: {MEDIA_DIR}")
    asyncio.run(take_shots())
    repack_docx()
    print("\nDONE. Open the docx in Word to verify.")


if __name__ == "__main__":
    main()

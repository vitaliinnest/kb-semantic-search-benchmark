"""
Автоматичні скріншоти ключових сторінок KB Search для звіту.

Оновлено під актуальний застосунок (BM25, E5-base, BGE-M3, nomic, Qwen3).

Використання:
    .venv\\Scripts\\python.exe scripts\\make_screenshots.py --start
    # або, якщо сервер вже запущений на :5000:
    .venv\\Scripts\\python.exe scripts\\make_screenshots.py
"""

import argparse
import subprocess
import sys
import time
import shutil
from pathlib import Path
import os

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

ROOT_DIR = Path(__file__).resolve().parent.parent
BASE_URL = "http://127.0.0.1:5000"
OUT_DIR = ROOT_DIR / "screenshots"

DOMAINS = ["tech", "legal", "medical"]

# Сторінки без пошуку (по одному скріну на домен)
STATIC_PAGES = [
    ("home",      "/"),
    ("documents", "/documents"),
    ("raw",       "/raw"),
    ("build",     "/build"),
    ("benchmark", "/benchmark"),
    ("selection", "/benchmark/selection"),
]

# Цільові скріншоти, що йдуть у текст звіту як рисунки.
# Ключ — назва файла, значення — параметри запиту.
THESIS_SHOTS: list[dict] = [
    # Рисунок 5.4 — перегляд документів і чанків (tech)
    {"name": "fig_5_04_documents_tech", "route": "/documents", "params": {"domain": "tech"}, "wait_ms": 1000},
    # Рисунок 5.6 — побудовані індекси (tech)
    {"name": "fig_5_06_build_tech", "route": "/build", "params": {"domain": "tech"}, "wait_ms": 1000},
    # Рисунок 5.7 — головна сторінка пошуку (tech, без запиту)
    {"name": "fig_5_07_search_home_tech", "route": "/", "params": {"domain": "tech"}, "wait_ms": 1000},
    # Рисунок 5.8 — пошук, технічний домен, BGE-M3
    {"name": "fig_5_08_search_tech_bge", "route": "/", "params": {
        "domain": "tech", "model": "bge-m3", "top_k": 5,
        "q": "виявлення аномалій у фінансових транзакціях"
    }, "wait_ms": 2500},
    # Рисунок 5.9 — пошук, юридичний домен, BGE-M3
    {"name": "fig_5_09_search_legal_bge", "route": "/", "params": {
        "domain": "legal", "model": "bge-m3", "top_k": 5,
        "q": "підстави обмеження цивільної дієздатності судом"
    }, "wait_ms": 2500},
    # Рисунок 5.10 — пошук, медичний домен, BGE-M3
    {"name": "fig_5_10_search_medical_bge", "route": "/", "params": {
        "domain": "medical", "model": "bge-m3", "top_k": 5,
        "q": "механізм газообміну в легенях та транспорт кисню кров'ю"
    }, "wait_ms": 2500},
    # Рисунок 5.12 — benchmark, перегляд (tech)
    {"name": "fig_5_12_benchmark_tech", "route": "/benchmark", "params": {"domain": "tech"}, "wait_ms": 1500},
    # Рисунок 5.14 — багатокритеріальний вибір (tech)
    {"name": "fig_5_14_selection_tech", "route": "/benchmark/selection", "params": {"domain": "tech"}, "wait_ms": 1500},
    # Рисунок 5.15 — пошук BGE-M3 для якісного аналізу (tech, англомовний запит)
    {"name": "fig_5_15_search_tech_bge_en", "route": "/", "params": {
        "domain": "tech", "model": "bge-m3", "top_k": 5,
        "q": "machine learning algorithms overview"
    }, "wait_ms": 2500},
    # Рисунок 5.16 — пошук nomic для контрасту (tech)
    {"name": "fig_5_16_search_tech_nomic", "route": "/", "params": {
        "domain": "tech", "model": "nomic", "top_k": 5,
        "q": "алгоритми сортування та їх складність"
    }, "wait_ms": 2500},
    # Рисунок 5.17 — пошук E5-base для якісного аналізу (medical)
    {"name": "fig_5_17_search_medical_e5", "route": "/", "params": {
        "domain": "medical", "model": "e5-base", "top_k": 5,
        "q": "першочергова допомога при серцевому нападі"
    }, "wait_ms": 2500},
    # Рисунок 5.18 — benchmark, технічний домен
    {"name": "fig_5_18_benchmark_tech", "route": "/benchmark", "params": {"domain": "tech"}, "wait_ms": 1500},
    # Рисунок 5.19 — benchmark, юридичний домен
    {"name": "fig_5_19_benchmark_legal", "route": "/benchmark", "params": {"domain": "legal"}, "wait_ms": 1500},
    # Рисунок 5.20 — benchmark, медичний домен
    {"name": "fig_5_20_benchmark_medical", "route": "/benchmark", "params": {"domain": "medical"}, "wait_ms": 1500},
    # Рисунок 5.21 — багатокритеріальний вибір, технічний
    {"name": "fig_5_21_selection_tech", "route": "/benchmark/selection", "params": {"domain": "tech"}, "wait_ms": 1500},
    # Рисунок 5.22 — багатокритеріальний вибір, юридичний
    {"name": "fig_5_22_selection_legal", "route": "/benchmark/selection", "params": {"domain": "legal"}, "wait_ms": 1500},
    # Рисунок 5.23 — багатокритеріальний вибір, медичний
    {"name": "fig_5_23_selection_medical", "route": "/benchmark/selection", "params": {"domain": "medical"}, "wait_ms": 1500},
]

# Прибираємо навігацію для більш чистих скріншотів сторінок системи
CLIP_CSS = """
    nav { display: none !important; }
    main { padding: 16px !important; max-width: 100% !important; }
    body { padding-top: 0 !important; margin-top: 0 !important; }
"""

VIEWPORT = {"width": 1280, "height": 900}


def wait_for_server(timeout: int = 60) -> bool:
    import urllib.request
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"{BASE_URL}/favicon.ico", timeout=2)
            return True
        except Exception:
            time.sleep(0.5)
    return False


def take_screenshots(page, out_dir: Path) -> None:
    import urllib.parse

    page.set_viewport_size(VIEWPORT)
    page.set_default_timeout(180_000)

    def shot(path: Path, url: str, wait_ms: int = 1000) -> None:
        page.goto(url, wait_until="networkidle")
        page.add_style_tag(content=CLIP_CSS)
        page.wait_for_timeout(wait_ms)
        path.parent.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(path), full_page=True)
        print(f"  OK  {path.name}")

    # Цільові скріншоти для звіту
    print("\n[1/2] Цільові скріншоти для звіту:")
    for s in THESIS_SHOTS:
        params = s.get("params") or {}
        qs = ("?" + urllib.parse.urlencode(params)) if params else ""
        url = f"{BASE_URL}{s['route']}{qs}"
        shot(out_dir / "thesis" / f"{s['name']}.png", url, wait_ms=s.get("wait_ms", 1000))

    # Загальні статичні сторінки по доменах (для архіву)
    print("\n[2/2] Огляд сторінок по доменах:")
    for domain in DOMAINS:
        d_dir = out_dir / "by_domain" / domain
        for page_id, route in STATIC_PAGES:
            sep = "&" if "?" in route else "?"
            url = f"{BASE_URL}{route}{sep}domain={domain}"
            shot(d_dir / f"{page_id}.png", url)


def main() -> None:
    parser = argparse.ArgumentParser(description="Автоскріни KB Search")
    parser.add_argument("--start",  action="store_true", help="запустити Flask-сервер")
    parser.add_argument("--port",   type=int, default=5000)
    parser.add_argument("--out",    default=str(OUT_DIR), help="папка для скріншотів")
    parser.add_argument("--headed", action="store_true", help="показати вікно браузера")
    args = parser.parse_args()

    global BASE_URL
    BASE_URL = f"http://127.0.0.1:{args.port}"
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = Path(args.out) / timestamp

    server_proc = None
    if args.start:
        print("Starting Flask server...")
        server_proc = subprocess.Popen(
            [sys.executable, str(ROOT_DIR / "src" / "app.py")],
            env={**os.environ, "FLASK_ENV": "development"},
            cwd=str(ROOT_DIR),
        )
        print("Waiting for server...", end="", flush=True)
        if not wait_for_server(60):
            print(" TIMEOUT")
            server_proc.terminate()
            sys.exit(1)
        print(" OK")

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("Playwright not installed. Run:\n  pip install playwright\n  playwright install chromium")
        if server_proc:
            server_proc.terminate()
        sys.exit(1)

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    print(f"\nSaving to: {out_dir}\n")

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=not args.headed)
        context = browser.new_context(viewport=VIEWPORT, locale="uk-UA")
        page = context.new_page()
        try:
            take_screenshots(page, out_dir)
        finally:
            context.close()
            browser.close()

    total = sum(1 for _ in out_dir.rglob("*.png"))
    print(f"\nDONE. {total} screenshots saved in: {out_dir}")

    if server_proc:
        server_proc.terminate()


if __name__ == "__main__":
    main()

"""
Автоматичні скріншоти всіх сторінок KB Search для звіту.

Використання:
    pip install playwright
    playwright install chromium
    python make_screenshots.py          # сервер вже запущений на :5000
    python make_screenshots.py --start  # script сам запускає сервер
"""

import argparse
import json
import subprocess
import sys
import time
import shutil
from pathlib import Path

BASE_URL = "http://127.0.0.1:5000"
OUT_DIR  = Path("screenshots")

DOMAINS = ["tech", "legal", "medical"]

# Сторінки без пошуку (знімаємо для кожного домену)
STATIC_PAGES = [
    ("home",      "/"),
    ("documents", "/documents"),
    ("raw",           "/raw"),
    ("build",         "/build"),
    ("benchmark",     "/benchmark"),
    ("selection",     "/benchmark/selection"),
]

# Пошукові запити + моделі для кожного домену.
# Структура: { domain: { model_id: [query, ...] } }
SEARCH_QUERIES: dict[str, dict[str, list[str]]] = {
    "tech": {
        "sbert": [
            "machine learning algorithms overview",
            "how does TCP/IP protocol work",
            "огляд алгоритмів машинного навчання",
            "як працює протокол TCP/IP",
            "нейронні мережі та глибоке навчання",
        ],
        "bert": [
            "REST API best practices",
            "database indexing performance",
            "найкращі практики розробки REST API",
            "індексування бази даних для підвищення продуктивності",
            "архітектура мікросервісів принципи",
        ],
        "tfidf": [
            "sorting algorithms complexity",
            "object-oriented programming principles",
            "складність алгоритмів сортування",
            "принципи об'єктно-орієнтованого програмування",
            "хмарні обчислення та інфраструктура",
        ],
        "word2vec": [
            "neural network training techniques",
            "методи навчання нейронних мереж",
            "рекомендаційні системи колаборативна фільтрація",
            "виявлення аномалій у фінансових транзакціях",
            "обробка природньої мови трансформери",
        ],
        "fasttext": [
            "cloud computing infrastructure",
            "хмарна інфраструктура та контейнеризація",
            "Docker Kubernetes оркестрація контейнерів",
            "безпека веб-застосунків OWASP",
            "системи контролю версій Git",
        ],
    },
    "legal": {
        "sbert": [
            "цивільна відповідальність за завдану шкоду",
            "порядок укладення господарського договору",
            "захист прав інтелектуальної власності",
            "конституційні права та свободи громадян",
            "міжнародне приватне право колізійні норми",
        ],
        "bert": [
            "права споживача при поверненні товару",
            "трудовий договір умови розірвання",
            "відповідальність роботодавця за травму на виробництві",
            "порядок оскарження адміністративного рішення",
            "договір оренди нерухомості умови та розірвання",
        ],
        "tfidf": [
            "кримінальна відповідальність неповнолітніх",
            "захист персональних даних законодавство",
            "судова експертиза у кримінальному провадженні",
            "медіація як альтернативне вирішення спорів",
            "аліменти розмір та порядок стягнення",
        ],
        "word2vec": [
            "спадщина та заповіт оформлення",
            "корпоративне управління акціонерне товариство",
            "земельні відносини право власності на землю",
            "банківська таємниця та фінансовий моніторинг",
            "антимонопольне законодавство та конкуренція",
        ],
        "fasttext": [
            "адміністративне правопорушення штраф",
            "митне оформлення імпорт експорт товарів",
            "авторські права у цифровому середовищі",
            "нотаріальне посвідчення правочинів",
            "виконавче провадження примусове виконання рішень",
        ],
    },
    "medical": {
        "sbert": [
            "симптоми та лікування пневмонії",
            "цукровий діабет типу 2 дієта",
            "серцево-судинні захворювання фактори ризику",
            "психічне здоров'я депресія та тривожність",
            "онкологічні захворювання методи ранньої діагностики",
        ],
        "bert": [
            "першочергова допомога при серцевому нападі",
            "антибіотики та резистентність бактерій",
            "хірургічне лікування жовчнокам'яної хвороби",
            "вплив стресу на імунну систему організму",
            "неврологічні розлади хвороба Паркінсона",
        ],
        "tfidf": [
            "вакцинація дітей календар щеплень",
            "артеріальний тиск норма гіпертонія",
            "остеопороз профілактика та лікування",
            "харчова алергія діагностика та дієта",
            "ендокринні захворювання щитоподібна залоза",
        ],
        "word2vec": [
            "реабілітація після інсульту",
            "лікування хронічного болю в спині",
            "захворювання органів дихання бронхіальна астма",
            "порушення обміну речовин ожиріння",
            "інфекційні захворювання профілактика та лікування",
        ],
        "fasttext": [
            "алергічна реакція лікування",
            "анестезія та знеболення під час операції",
            "педіатричні захворювання розвиток дітей",
            "офтальмологічні захворювання зір та катаракта",
            "стоматологічне лікування карієс та протезування",
        ],
    },
}

CLIP_CSS = """
    nav { display: none !important; }
    main { padding: 16px !important; max-width: 100% !important; }
    body { padding-top: 0 !important; margin-top: 0 !important; }
"""

VIEWPORT = {"width": 1200, "height": 900}


def wait_for_server(timeout: int = 30) -> bool:
    import urllib.request, urllib.error
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"{BASE_URL}/favicon.ico", timeout=2)
            return True
        except Exception:
            time.sleep(0.5)
    return False


def slug(text: str) -> str:
    import re
    return re.sub(r"[^a-z0-9а-яіїєґ]+", "_", text.lower()).strip("_")[:60]


def take_screenshots(page, out_dir: Path) -> None:
    """Основна логіка скріншотів — виконується всередині playwright context."""

    page.set_viewport_size(VIEWPORT)

    def shot(path: Path, url: str, wait_ms: int = 800) -> None:
        page.goto(url, wait_until="networkidle")
        page.add_style_tag(content=CLIP_CSS)
        page.wait_for_timeout(wait_ms)
        path.parent.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(path), full_page=True)
        print(f"  ✓  {path}")

    # ── Статичні сторінки для кожного домену ──────────────────────────────
    for domain in DOMAINS:
        d_dir = out_dir / domain
        d_dir.mkdir(parents=True, exist_ok=True)
        for page_id, route in STATIC_PAGES:
            sep = "&" if "?" in route else "?"
            url = f"{BASE_URL}{route}{sep}domain={domain}"
            shot(d_dir / f"{page_id}.png", url)

    # ── Пошукові запити × моделі ─────────────────────────────────────────
    import urllib.parse
    import urllib.request as _ureq

    # domain → { model_id → [ { query, screenshot, results: [...] } ] }
    domain_data: dict = {d: {} for d in SEARCH_QUERIES}

    for domain, models_queries in SEARCH_QUERIES.items():
        for model_id, queries in models_queries.items():
            m_dir = out_dir / domain / "search" / model_id
            domain_data[domain][model_id] = []
            for query in queries:
                name = slug(query)
                params = urllib.parse.urlencode({"q": query, "domain": domain, "model": model_id, "top_k": 5})
                url = f"{BASE_URL}/?{params}"
                shot(m_dir / f"{name}.png", url, wait_ms=1500)

                # ── збираємо результати пошуку (без full_text) ────────────
                api_url = f"{BASE_URL}/api/search?{params}"
                try:
                    with _ureq.urlopen(api_url, timeout=10) as resp:
                        data = json.loads(resp.read().decode("utf-8"))
                    results = [
                        {k: v for k, v in r.items() if k != "full_text"}
                        for r in data.get("results", [])
                    ]
                except Exception as exc:
                    results = [{"error": str(exc)}]
                domain_data[domain][model_id].append({
                    "query": query,
                    "screenshot": f"{domain}/search/{model_id}/{name}.png",
                    "results": results,
                })

    # ── Зберігаємо по одному JSON на домен ───────────────────────────────
    for domain, models in domain_data.items():
        json_path = out_dir / domain / "search_results.json"
        json_path.write_text(json.dumps({"domain": domain, "models": models}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  ·  {json_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Автоматичні скріни KB Search")
    parser.add_argument("--start",  action="store_true", help="запустити Flask-сервер")
    parser.add_argument("--port",   type=int, default=5000, help="порт сервера (default 5000)")
    parser.add_argument("--out",    default="screenshots", help="папка для скріншотів")
    parser.add_argument("--headed", action="store_true", help="показати вікно браузера")
    args = parser.parse_args()

    global BASE_URL
    BASE_URL = f"http://127.0.0.1:{args.port}"
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = Path(args.out) / timestamp

    # ── Запускаємо сервер якщо потрібно ───────────────────────────────────
    server_proc = None
    if args.start:
        print("▶  Запускаємо Flask-сервер…")
        server_proc = subprocess.Popen(
            [sys.executable, "src/app.py"],
            env={**__import__("os").environ, "FLASK_ENV": "development"},
        )
        print("   Чекаємо готовності…", end="", flush=True)
        if not wait_for_server():
            print(" TIMEOUT — сервер не відповів за 30 с")
            server_proc.terminate()
            sys.exit(1)
        print(" OK")

    # ── Перевіряємо, чи playwright встановлено ────────────────────────────
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("Playwright не встановлено. Виконайте:\n  pip install playwright\n  playwright install chromium")
        if server_proc:
            server_proc.terminate()
        sys.exit(1)

    # ── Очищаємо папку ────────────────────────────────────────────────────
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    print(f"\n📸 Зберігаємо скріншоти у «{out_dir}/»\n")

    # ── Playwright ────────────────────────────────────────────────────────
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=not args.headed)
        context = browser.new_context(
            viewport=VIEWPORT,
            locale="uk-UA",
        )
        page = context.new_page()
        try:
            take_screenshots(page, out_dir)
        finally:
            context.close()
            browser.close()

    total = sum(1 for _ in out_dir.rglob("*.png"))
    print(f"\n✅  Готово! Збережено {total} скріншотів у «{out_dir}/»")

    if server_proc:
        server_proc.terminate()


if __name__ == "__main__":
    main()

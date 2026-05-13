"""
Capture screenshots for the defense-demo video walkthrough.

These screenshots illustrate what the student will demonstrate while
recording the demo video. Each shot has a stable URL with explicit query
parameters so the demo can always be reproduced.

Output: thesis/demo_screenshots/*.png — referenced from demo_video_script.md.

Requires Flask running on :5000 (`.venv/Scripts/python.exe src/app.py`).
"""
import asyncio
import sys
import io
from pathlib import Path
from urllib.parse import urlencode

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = ROOT / "thesis" / "demo_screenshots"
BASE = "http://127.0.0.1:5000"

# Demo queries — proven by API probe to give semantic-vs-lexical contrast:
#   tech "churn" : BGE finds churn-analysis content, BM25 returns abstract pages
#   medical antibiotics: BGE perfect on pharmacology, BM25 returns anxiety/stroke
Q_TECH    = "прогноз відтоку клієнтів (churn prediction)"
Q_MEDICAL = "відмінність між бактерицидними та бактеріостатичними антибіотиками"
Q_LEGAL   = "застава та домашній арешт як альтернативи триманню під вартою"

# (out_filename, url, viewport_h, label)
SHOTS = [
    # -------- Step 1: hero / empty state --------
    ("01_hero_tech.png",
     f"{BASE}/?domain=tech",
     900,
     "Step 1: головна сторінка, домен Tech, порожній стан"),

    # -------- Step 2: tech BGE-M3 wins (the wow moment) --------
    ("02_tech_bge_churn.png",
     f"{BASE}/?" + urlencode({"domain": "tech", "model": "bge-m3", "top_k": 5, "q": Q_TECH}),
     1500,
     "Step 2: BGE-M3, запит про churn — знаходить релевантні фрагменти"),

    # -------- Step 3: same query, BM25 — degraded --------
    ("03_tech_bm25_churn.png",
     f"{BASE}/?" + urlencode({"domain": "tech", "model": "bm25", "top_k": 5, "q": Q_TECH}),
     1500,
     "Step 3: той самий запит, BM25 — повертає абстракт і бібліографію"),

    # -------- Step 4: medical BGE-M3 — pharmacology match --------
    ("04_medical_bge_antibiotics.png",
     f"{BASE}/?" + urlencode({"domain": "medical", "model": "bge-m3", "top_k": 5, "q": Q_MEDICAL}),
     1500,
     "Step 4: medical, BGE-M3 — фармакологічний матеріал про антибіотики"),

    # -------- Step 5: same query, BM25 — completely off-topic --------
    ("05_medical_bm25_antibiotics.png",
     f"{BASE}/?" + urlencode({"domain": "medical", "model": "bm25", "top_k": 5, "q": Q_MEDICAL}),
     1500,
     "Step 5: medical, BM25 — нерелевантні дисертації"),

    # -------- Step 6: documents page (corpus overview) --------
    ("06_documents_tech.png",
     f"{BASE}/documents?domain=tech",
     1400,
     "Step 6: /documents — перегляд корпусу чанків"),

    # -------- Step 7: benchmark page (the main metrics table) --------
    ("07_benchmark_tech.png",
     f"{BASE}/benchmark?domain=tech",
     1700,
     "Step 7: /benchmark — таблиця метрик для tech-домену"),

    # -------- Step 8: benchmark page legal (different leader) --------
    ("08_benchmark_legal.png",
     f"{BASE}/benchmark?domain=legal",
     1700,
     "Step 8: /benchmark — legal-домен, інший лідер (Qwen3)"),

    # -------- Step 9: MCDA selection page (Tech default profile) --------
    ("09_selection_tech.png",
     f"{BASE}/benchmark/selection?domain=tech",
     1800,
     "Step 9: /benchmark/selection Tech — рекомендує BGE-M3"),

    # -------- Step 9a: Legal profile — different recommendation --------
    ("09a_selection_legal.png",
     f"{BASE}/benchmark/selection?domain=legal",
     1800,
     "Step 9a: /benchmark/selection Legal — рекомендує Qwen3"),

    # -------- Step 9b: Medical profile --------
    ("09b_selection_medical.png",
     f"{BASE}/benchmark/selection?domain=medical",
     1800,
     "Step 9b: /benchmark/selection Medical — рекомендує BGE-M3"),

    # -------- Step 10: explorer index (per-query overview) --------
    ("10_explorer_tech.png",
     f"{BASE}/benchmark/explorer?domain=tech",
     1500,
     "Step 10: /benchmark/explorer — огляд усіх запитів"),

    # -------- Step 11: explorer drill-down on the churn query --------
    ("11_explorer_q6_churn.png",
     f"{BASE}/benchmark/explorer/q6?domain=tech",
     1800,
     "Step 11: /benchmark/explorer/q6 — деталі по churn-запиту"),
]

VIEWPORT_W = 1400


async def take_shots() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        for img, url, vh, label in SHOTS:
            ctx = await browser.new_context(
                viewport={"width": VIEWPORT_W, "height": vh},
                locale="uk-UA",
            )
            page = await ctx.new_page()
            try:
                await page.goto(url, wait_until="networkidle", timeout=120_000)
                # Hide cursor and any focus rings for clean shots
                await page.add_style_tag(content="""
                    *:focus { outline: none !important; box-shadow: none !important; }
                    ::-webkit-scrollbar { display: none; }
                """)
                await page.wait_for_timeout(800)
                target = OUT_DIR / img
                await page.screenshot(path=str(target), full_page=False)
                kb = target.stat().st_size // 1024
                print(f"  OK  {img:<35} ({kb:>4} KB)  {label}")
            except Exception as exc:
                print(f"  FAIL {img:<35}  {label}  -> {exc}")
            finally:
                await ctx.close()

        # ── Bonus: selection page with latency weight pushed up ──────────────
        # Demonstrates that the recommendation changes when user prioritises
        # speed over quality. On tech default, BGE-M3 wins. With latency
        # weight near max + nDCG reduced, E5-base or BM25 climbs to the top.
        ctx = await browser.new_context(
            viewport={"width": VIEWPORT_W, "height": 1800},
            locale="uk-UA",
        )
        page = await ctx.new_page()
        try:
            await page.goto(
                f"{BASE}/benchmark/selection?domain=tech",
                wait_until="networkidle", timeout=120_000,
            )
            await page.add_style_tag(content="""
                *:focus { outline: none !important; box-shadow: none !important; }
                ::-webkit-scrollbar { display: none; }
            """)
            await page.wait_for_timeout(500)

            # Bump latency weight to near max, drop nDCG/MRR
            for slider_id, target_val in [
                ("w-avg_latency_ms", "0.50"),
                ("w-ndcg_at_k",      "0.15"),
                ("w-mrr_at_k",       "0.10"),
                ("w-recall_at_k",    "0.15"),
                ("w-precision_at_k", "0.10"),
            ]:
                await page.evaluate(f"""
                    () => {{
                        const sl = document.getElementById('{slider_id}');
                        if (sl) {{
                            sl.value = '{target_val}';
                            sl.dispatchEvent(new Event('input', {{bubbles: true}}));
                        }}
                    }}
                """)
            # Wait for the debounced recompute (450ms in template) + network
            await page.wait_for_timeout(1500)

            target = OUT_DIR / "09c_selection_tech_speed_priority.png"
            await page.screenshot(path=str(target), full_page=False)
            kb = target.stat().st_size // 1024
            print(f"  OK  09c_selection_tech_speed_priority.png       ({kb:>4} KB)  Step 9c: те саме tech, але вага latency=0.50 → інша рекомендація")
        except Exception as exc:
            print(f"  FAIL 09c_selection_tech_speed_priority.png        -> {exc}")
        finally:
            await ctx.close()

        await browser.close()


def main() -> None:
    print(f"[demo] capturing {len(SHOTS)} screenshots to {OUT_DIR}/")
    asyncio.run(take_shots())
    print("\nDone.")


if __name__ == "__main__":
    main()

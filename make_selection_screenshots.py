"""
Take selection page screenshots with domain-specific weight profiles.

Usage:
    python make_selection_screenshots.py          # server already running on :5000
    python make_selection_screenshots.py --start  # auto-start server
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

BASE_URL = "http://127.0.0.1:5000"

VIEWPORT = {"width": 1200, "height": 900}

CLIP_CSS = """
    nav { display: none !important; }
    main { padding: 16px !important; max-width: 100% !important; }
    body { padding-top: 0 !important; margin-top: 0 !important; }
"""

# Domain-specific weight profiles with justification:
#   Legal  — MRR↑: first relevant result is critical for legal practice
#   Medical — Recall↑: missing a relevant document is dangerous
#   Tech   — Latency↑: developer search needs fast response + balanced quality
DOMAIN_WEIGHTS = {
    "legal": {
        "ndcg_at_k":      0.30,
        "mrr_at_k":       0.30,
        "recall_at_k":    0.20,
        "precision_at_k": 0.10,
        "avg_latency_ms": 0.10,
    },
    "medical": {
        "ndcg_at_k":      0.25,
        "mrr_at_k":       0.15,
        "recall_at_k":    0.35,
        "precision_at_k": 0.10,
        "avg_latency_ms": 0.15,
    },
    "tech": {
        "ndcg_at_k":      0.30,
        "mrr_at_k":       0.20,
        "recall_at_k":    0.25,
        "precision_at_k": 0.05,
        "avg_latency_ms": 0.20,
    },
}


def wait_for_server(timeout: int = 30) -> bool:
    import urllib.request
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"{BASE_URL}/favicon.ico", timeout=2)
            return True
        except Exception:
            time.sleep(0.5)
    return False


def take_selection_screenshots(out_dir: Path, headed: bool = False) -> None:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=not headed)
        context = browser.new_context(viewport=VIEWPORT, locale="uk-UA")
        page = context.new_page()
        page.set_default_timeout(60_000)

        for domain, weights in DOMAIN_WEIGHTS.items():
            url = f"{BASE_URL}/benchmark/selection?domain={domain}"
            print(f"\n▶  {domain}: navigating to selection page…")
            page.goto(url, wait_until="networkidle")
            page.add_style_tag(content=CLIP_CSS)
            page.wait_for_timeout(500)

            # Patch JS recompute to include domain in the fetch URL
            page.evaluate(f"""(() => {{
                const origFetch = window.fetch;
                window.fetch = function(url, opts) {{
                    if (typeof url === 'string' && url.includes('/benchmark/selection/compute')) {{
                        url = url.split('?')[0] + '?domain={domain}';
                    }}
                    return origFetch.call(this, url, opts);
                }};
            }})()""")

            # Set each slider value via JS and trigger input event
            for key, val in weights.items():
                page.evaluate(f"""(() => {{
                    const sl = document.getElementById('w-{key}');
                    if (!sl) return;
                    sl.value = {val};
                    sl.dispatchEvent(new Event('input', {{ bubbles: true }}));
                }})()""")
                print(f"   {key} = {val}")

            # Wait for recompute fetch to finish
            page.wait_for_timeout(2500)

            # Take screenshot
            fpath = out_dir / f"{domain}_selection.png"
            fpath.parent.mkdir(parents=True, exist_ok=True)
            page.screenshot(path=str(fpath), full_page=True)
            print(f"   ✓ {fpath}")

        context.close()
        browser.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Selection page screenshots")
    parser.add_argument("--start", action="store_true", help="auto-start Flask server")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--headed", action="store_true", help="show browser window")
    args = parser.parse_args()

    global BASE_URL
    BASE_URL = f"http://127.0.0.1:{args.port}"

    out_dir = Path("screenshots/2026-03-26_20-21-16/report")

    server_proc = None
    if args.start:
        print("▶  Starting Flask server…")
        server_proc = subprocess.Popen(
            [sys.executable, "src/app.py"],
            env={**__import__("os").environ, "FLASK_ENV": "development"},
        )
        if not wait_for_server():
            print("TIMEOUT — server did not respond")
            server_proc.terminate()
            sys.exit(1)
        print("   Server ready")

    try:
        take_selection_screenshots(out_dir, headed=args.headed)
    finally:
        if server_proc:
            server_proc.terminate()

    print(f"\n✅  Done! Selection screenshots saved to {out_dir}/")


if __name__ == "__main__":
    main()

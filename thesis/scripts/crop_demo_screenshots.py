"""
Crop trailing whitespace from demo screenshots.

Many pages (benchmark, selection) are short content captured at a tall
viewport, leaving 30–50% of the image as blank white space below the
useful content. This script scans each PNG from the bottom upward and
trims pure-white rows, leaving a small margin.

In-place edit — backup copies live in thesis/_backups/demo_screenshots/
the first time the script runs.
"""
import pathlib
import shutil
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from PIL import Image

ROOT = pathlib.Path("D:/repos/kb-semantic-search-benchmark")
SHOTS_DIR = ROOT / "thesis" / "demo_screenshots"
BACKUP_DIR = ROOT / "thesis" / "_backups" / "demo_screenshots_uncropped"

# A pixel is "background" if all RGB channels >= this threshold.
# Real UI elements (text, lines, badges) all have at least one channel
# under 240, so 240 is a safe cutoff.
WHITE_THRESHOLD = 240
# Keep this many empty rows below the last content row.
PADDING_BOTTOM = 24


def find_content_bottom(img: Image.Image) -> int:
    """Return the y-coordinate of the last row that contains non-white pixels."""
    rgb = img.convert("RGB")
    w, h = rgb.size
    pixels = rgb.load()
    # Scan from bottom up. Sample every other column for speed (we don't
    # need perfect precision — just to skip ~half the empty pixels).
    sample_step = max(1, w // 200)  # ~200 samples per row
    for y in range(h - 1, -1, -1):
        for x in range(0, w, sample_step):
            r, g, b = pixels[x, y]
            if r < WHITE_THRESHOLD or g < WHITE_THRESHOLD or b < WHITE_THRESHOLD:
                return y
    return 0


def crop_one(path: pathlib.Path) -> tuple[int, int, int]:
    """Crop trailing whitespace. Returns (old_h, new_h, saved_bytes)."""
    img = Image.open(path)
    old_h = img.height
    bottom = find_content_bottom(img)
    new_h = min(old_h, bottom + 1 + PADDING_BOTTOM)
    if new_h >= old_h:
        return old_h, old_h, 0
    old_size = path.stat().st_size
    cropped = img.crop((0, 0, img.width, new_h))
    cropped.save(path, optimize=True)
    saved = old_size - path.stat().st_size
    return old_h, new_h, saved


def main() -> None:
    if not SHOTS_DIR.exists():
        sys.exit(f"no screenshots dir: {SHOTS_DIR}")

    # First-run backup
    if not BACKUP_DIR.exists():
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        for p in SHOTS_DIR.glob("*.png"):
            shutil.copy2(p, BACKUP_DIR / p.name)
        print(f"[backup] {len(list(BACKUP_DIR.glob('*.png')))} files -> {BACKUP_DIR}")
    else:
        print(f"[backup] existing backups in {BACKUP_DIR} — not overwriting")

    print(f"[crop]  scanning {SHOTS_DIR}")
    total_saved = 0
    for png in sorted(SHOTS_DIR.glob("*.png")):
        old_h, new_h, saved = crop_one(png)
        total_saved += saved
        cut = old_h - new_h
        if cut > 0:
            print(f"  {png.name:<42} {old_h:>4}px -> {new_h:>4}px (−{cut:>4}px, −{saved//1024:>3} KB)")
        else:
            print(f"  {png.name:<42} {old_h:>4}px (no change)")
    print(f"\n[done] saved {total_saved // 1024} KB total")


if __name__ == "__main__":
    main()

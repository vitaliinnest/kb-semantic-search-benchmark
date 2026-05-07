"""
Fix images in chapter 5 of the thesis docx.

Background:
  The previous run of `update_thesis_screenshots.py` used a wrong mapping
  (assumed `imageN.png` = figure N). The actual document has SHARED image
  refs:
      image18 -> fig 5.18 only
      image19 -> fig 5.19 and 5.23
      image20 -> fig 5.20 and 5.25
      image21 -> fig 5.22 and 5.26

  Result: figures 5.18, 5.19, 5.20, 5.22 got overwritten with wrong
  content (benchmark / selection screenshots instead of search-results).

  Original state (.bak.docx):
      image18 = search tech bge en              (fig 5.18 — correct)
      image19 = search tech nomic               (fig 5.19 — correct;
                                                 fig 5.23 — semantically WRONG)
      image20 = search medical e5               (fig 5.20 — correct;
                                                 fig 5.25 — semantically WRONG)
      image21 = benchmark legal                 (fig 5.22 — correct;
                                                 fig 5.26 — semantically WRONG)

This script:
  1. Restores image18,19,20,21 from .bak.docx (fixes 5.18-5.22).
  2. Captures NEW screenshots for figures 5.23 (benchmark medical),
     5.25 (selection legal), 5.26 (selection medical), saves as
     image40, image41, image42.
  3. Registers new relationships in word/_rels/document.xml.rels.
  4. Rewrites the rId references for figures 5.23, 5.25, 5.26 in
     document.xml so they point to the new images.

Requires Flask running on http://127.0.0.1:5000.
"""
import asyncio
import io
import re
import shutil
import sys
import zipfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path("D:/repos/kb-semantic-search-benchmark")
DOCX = ROOT / "thesis" / "2026_M_PI_Nesterenko_VV.docx"
ORIG = ROOT / "thesis" / "2026_M_PI_Nesterenko_VV.bak.docx"
UNPACKED = ROOT / "thesis" / "unpacked_docx"
MEDIA = UNPACKED / "word" / "media"
RELS = UNPACKED / "word" / "_rels" / "document.xml.rels"
DOCXML = UNPACKED / "word" / "document.xml"
BASE = "http://127.0.0.1:5000"

# Figure -> (URL, viewport_height, label) — only for the NEW images
NEW_SHOTS = [
    ("5.23", "benchmark medical",
     f"{BASE}/benchmark?domain=medical", 1500),
    ("5.25", "selection legal",
     f"{BASE}/benchmark/selection?domain=legal", 1500),
    ("5.26", "selection medical",
     f"{BASE}/benchmark/selection?domain=medical", 1500),
]

# Step 1: unpack current docx
def unpack(src: Path):
    if UNPACKED.exists():
        shutil.rmtree(UNPACKED)
    with zipfile.ZipFile(src) as z:
        z.extractall(UNPACKED)
    print(f"[unpack] {src.name}")


# Step 2: restore broken images from .bak.docx
def restore_broken_images():
    # Read originals from .bak.docx
    with zipfile.ZipFile(ORIG) as z:
        for name in ("image18.png", "image19.png", "image20.png", "image21.png"):
            data = z.read(f"word/media/{name}")
            (MEDIA / name).write_bytes(data)
            print(f"[restore] {name} <- .bak.docx ({len(data)//1024} KB)")


# Step 3: take new screenshots
async def take_new_shots():
    from playwright.async_api import async_playwright

    out_files = {}  # fig -> filename
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        for i, (fig, label, url, vh) in enumerate(NEW_SHOTS):
            ctx = await browser.new_context(
                viewport={"width": 1400, "height": vh},
                locale="uk-UA",
            )
            page = await ctx.new_page()
            try:
                await page.goto(url, wait_until="networkidle", timeout=120_000)
                await page.wait_for_timeout(900)
                # save into media as imageNN.png — pick a free slot
                target_name = f"image{40 + i}.png"
                target = MEDIA / target_name
                await page.screenshot(path=str(target), full_page=False)
                size_kb = target.stat().st_size // 1024
                print(f"[shot] {fig} {label}: {target_name} ({size_kb} KB)")
                out_files[fig] = target_name
            finally:
                await ctx.close()
        await browser.close()
    return out_files


# Step 4: register new rels and rewrite document.xml refs
def patch_xml(new_files: dict):
    rels_xml = RELS.read_text(encoding="utf-8")
    doc_xml = DOCXML.read_text(encoding="utf-8")

    # Figure out next free rId
    used_rids = set(re.findall(r'Id="(rId\d+)"', rels_xml))
    def next_rid():
        i = 1
        while f"rId{i}" in used_rids:
            i += 1
        rid = f"rId{i}"
        used_rids.add(rid)
        return rid

    # For each figure 5.23/5.25/5.26, allocate new rId, add rel, rewrite the
    # corresponding <a:blip r:embed="OLD"/> at the figure's anchor.
    # We need to find the SECOND occurrence of the shared rId (the one
    # that's near the relevant figure caption).
    fig_to_old_rid = {"5.23": "rId29", "5.25": "rId30", "5.26": "rId31"}

    # Find figure caption positions to identify which blip belongs to which fig
    # The blip closest BEFORE each caption is the one we want to repoint.
    for fig in ("5.23", "5.25", "5.26"):
        target_filename = new_files[fig]
        new_rid = next_rid()

        # Add a Relationship entry for the new image
        new_rel = (
            f'<Relationship Id="{new_rid}" '
            f'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" '
            f'Target="media/{target_filename}"/>'
        )
        rels_xml = rels_xml.replace(
            "</Relationships>", new_rel + "</Relationships>"
        )

        # In document.xml, find the figure caption position and the LAST blip ref before it
        cap_match = re.search(
            re.escape(f"Рисунок {fig}"), doc_xml
        )
        if not cap_match:
            raise SystemExit(f"caption for fig {fig} not found")
        cap_pos = cap_match.start()

        old_rid = fig_to_old_rid[fig]
        # The blip can have children (extLst) so it may be a non-self-closing
        # element. Match just the open tag with the rId attribute.
        pattern = re.compile(rf'<a:blip\s+r:embed="{old_rid}"')
        candidates = [(m.start(), m.end()) for m in pattern.finditer(doc_xml) if m.start() < cap_pos]
        if not candidates:
            raise SystemExit(f"no <a:blip> for {old_rid} before fig {fig}")
        last_start, last_end = candidates[-1]
        last_blip = doc_xml[last_start:last_end]
        new_blip = re.sub(rf'r:embed="{old_rid}"', f'r:embed="{new_rid}"', last_blip)
        doc_xml = doc_xml[:last_start] + new_blip + doc_xml[last_end:]
        print(f"[xml] fig {fig}: {old_rid} -> {new_rid}  (-> {target_filename})")

    DOCXML.write_text(doc_xml, encoding="utf-8")
    RELS.write_text(rels_xml, encoding="utf-8")
    print("[xml] document.xml + rels updated")


# Step 5: repack
def repack():
    backup = DOCX.with_suffix(".bak3.docx")
    shutil.copy2(DOCX, backup)
    print(f"[backup] -> {backup.name}")
    import os
    with zipfile.ZipFile(DOCX, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(UNPACKED):
            for f in files:
                full = Path(root) / f
                rel = full.relative_to(UNPACKED).as_posix()
                zf.write(full, rel)
    print(f"[repack] {DOCX.name} ({DOCX.stat().st_size:,} bytes)")


def main():
    unpack(DOCX)
    restore_broken_images()
    new_files = asyncio.run(take_new_shots())
    patch_xml(new_files)
    repack()
    print("\nDone.")


if __name__ == "__main__":
    main()

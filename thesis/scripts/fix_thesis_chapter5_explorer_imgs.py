"""
Restore image15, image16, image17 from .bak.docx.

The earlier (broken) version of update_thesis_screenshots.py used a wrong
mapping:
    image15 <- search tech bge en  (should be explorer index)
    image16 <- search tech nomic   (should be explorer detail)
    image17 <- search medical e5   (should be explorer drill bge)

In the doc, fig 5.15→image15, fig 5.16→image16, fig 5.17→image17, so the
captions don't match the actual content. The original .bak.docx had the
correct explorer screenshots.
"""
import shutil, sys, zipfile, pathlib

sys.stdout.reconfigure(encoding="utf-8")

ROOT = pathlib.Path("D:/repos/kb-semantic-search-benchmark")
DOCX = ROOT / "thesis" / "2026_M_PI_Nesterenko_VV.docx"
ORIG = ROOT / "thesis" / "2026_M_PI_Nesterenko_VV.bak.docx"
UNPACKED = ROOT / "thesis" / "unpacked_docx"
MEDIA = UNPACKED / "word" / "media"


def main():
    if UNPACKED.exists():
        shutil.rmtree(UNPACKED)
    with zipfile.ZipFile(DOCX) as z:
        z.extractall(UNPACKED)
    print(f"[unpack] {DOCX.name}")

    with zipfile.ZipFile(ORIG) as z:
        for name in ("image15.png", "image16.png", "image17.png"):
            data = z.read(f"word/media/{name}")
            (MEDIA / name).write_bytes(data)
            print(f"[restore] {name} <- .bak.docx ({len(data)//1024} KB)")

    backup = DOCX.with_suffix(".bak5.docx")
    shutil.copy2(DOCX, backup)
    print(f"[backup] -> {backup.name}")
    import os
    with zipfile.ZipFile(DOCX, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(UNPACKED):
            for f in files:
                full = pathlib.Path(root) / f
                rel = full.relative_to(UNPACKED).as_posix()
                zf.write(full, rel)
    print(f"[repack] {DOCX.name} ({DOCX.stat().st_size:,} bytes)")
    print("\nDone.")


if __name__ == "__main__":
    main()

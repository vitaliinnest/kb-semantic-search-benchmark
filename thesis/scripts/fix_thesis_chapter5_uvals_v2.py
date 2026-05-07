"""
Final fix: align text U-values in chapter 5.4 with what the web-app
selection screenshots actually display.

The web-app multi-criteria selection includes ALL 5 models (BM25 + 4 neural),
which gives different U-values than neural-only MCDA:

  Domain    | text v2 | screenshot v3 (with BM25)
  ----------|---------|---------------------------
  Tech BGE  | 0.923   | 0.904
  Tech E5   | 0.863   | 0.832
  Legal BGE | 0.922   | 0.914
  Med BGE   | 0.927   | 0.912
  Med E5    | 0.836   | 0.808

Also adds BM25 to the Pareto-optimal sets in narrative (since web app
shows it as Pareto-optimal due to its lowest latency).
"""
import shutil, sys, zipfile, pathlib

sys.stdout.reconfigure(encoding="utf-8")

ROOT = pathlib.Path("D:/repos/kb-semantic-search-benchmark")
DOCX = ROOT / "thesis" / "2026_M_PI_Nesterenko_VV.docx"
UNPACKED = ROOT / "thesis" / "unpacked_docx"
DOC_XML = UNPACKED / "word" / "document.xml"


def expect(xml, old, new, *, count, label):
    found = xml.count(old)
    if found != count:
        raise SystemExit(f"FAIL [{label}]: expected {count} of {old!r}, found {found}")
    print(f"  ok  [{label}]: {old!r:80s} -> {new!r}  (x{count})")
    return xml.replace(old, new, count)


def main():
    if UNPACKED.exists():
        shutil.rmtree(UNPACKED)
    with zipfile.ZipFile(DOCX) as z:
        z.extractall(UNPACKED)
    print(f"[unpack] {DOCX.name}")

    xml = DOC_XML.read_text(encoding="utf-8")

    # Tech: 0.923 -> 0.904, 0.863 -> 0.832
    xml = expect(
        xml,
        "BGE-M3 з інтегральною оцінкою 0.923, тоді як E5-base отримала 0.863. "
        "До множини Парето увійшли BGE-M3 та E5-base: BGE-M3",
        "BGE-M3 з інтегральною оцінкою 0.904, тоді як E5-base отримала 0.832. "
        "До множини Парето увійшли BM25 (як швидкий лексичний базелайн), E5-base та BGE-M3: BGE-M3",
        count=1,
        label="Tech BGE/E5 + Pareto",
    )

    # Legal: 0.922 -> 0.914, добавити BM25 у Парето
    xml = expect(
        xml,
        "BGE-M3 з інтегральною оцінкою 0.922. До множини Парето увійшли "
        "BGE-M3 (краща якість), E5-base (краща швидкодія) та Qwen3-Embedding-0.6B "
        "(найкращі якісні метрики у юридичному домені).",
        "BGE-M3 з інтегральною оцінкою 0.914. До множини Парето увійшли "
        "BM25 (швидкий лексичний базелайн), E5-base, BGE-M3 (краща якість серед нейронних моделей) "
        "та Qwen3-Embedding-0.6B (найкращі якісні метрики у юридичному домені).",
        count=1,
        label="Legal BGE U + Pareto",
    )

    # Medical: 0.927 -> 0.912, 0.836 -> 0.808, додати BM25
    xml = expect(
        xml,
        "BGE-M3 з інтегральною оцінкою 0.927, E5-base посіла друге місце (0.836). "
        "До множини Парето увійшли BGE-M3 та E5-base.",
        "BGE-M3 з інтегральною оцінкою 0.912, E5-base посіла друге місце (0.808). "
        "До множини Парето увійшли BM25 (швидкий лексичний базелайн), E5-base та BGE-M3.",
        count=1,
        label="Medical BGE/E5 + Pareto",
    )

    # Summary: U=0.923/0.922/0.927 -> 0.904/0.914/0.912
    xml = expect(
        xml,
        "U=0.923/0.922/0.927",
        "U=0.904/0.914/0.912",
        count=1,
        label="summary U",
    )

    DOC_XML.write_text(xml, encoding="utf-8")

    backup = DOCX.with_suffix(".bak4.docx")
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

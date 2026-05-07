"""
Two clear citation errors in chapter 4:

1. Section 4.1.1 (BGE-M3) cites [13] (Mikolov, Word2Vec, ICLR 2013) instead
   of [25] (Chen et al., the actual BGE-M3 paper, arXiv:2402.03216).

2. Section 4.1.2 (E5-base) cites [14] (Pennington et al., GloVe, EMNLP 2014)
   instead of [24] (Wang et al., the actual E5 paper, arXiv:2212.03533).

Bibliography:
  [13] = Mikolov — Word2Vec
  [14] = Pennington — GloVe
  [24] = Wang — E5
  [25] = Chen — BGE-M3
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
    print(f"  ok  [{label}]: {old!r:60s} -> {new!r}  (x{count})")
    return xml.replace(old, new, count)


def main():
    if UNPACKED.exists():
        shutil.rmtree(UNPACKED)
    with zipfile.ZipFile(DOCX) as z:
        z.extractall(UNPACKED)
    print(f"[unpack] {DOCX.name}")

    xml = DOC_XML.read_text(encoding="utf-8")

    # Fix 1: BGE-M3 cite [13] -> [25] (anchored to BAAI mention to be unique)
    xml = expect(
        xml,
        "Пекінською академією штучного інтелекту (BAAI) [13]",
        "Пекінською академією штучного інтелекту (BAAI) [25]",
        count=1,
        label="BGE-M3 [13] -> [25]",
    )

    # Fix 2: E5 cite [14] -> [24] (anchored to "трансформерної архітектури" to be unique)
    xml = expect(
        xml,
        "моделей ембеддінгів на основі трансформерної архітектури [14]",
        "моделей ембеддінгів на основі трансформерної архітектури [24]",
        count=1,
        label="E5-base [14] -> [24]",
    )

    DOC_XML.write_text(xml, encoding="utf-8")

    backup = DOCX.with_suffix(".bak8.docx")
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

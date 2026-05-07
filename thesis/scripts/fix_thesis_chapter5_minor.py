"""
Minor consistency fixes in chapter 5.4:
  1. Tech-specific paragraph: replace cross-domain avg latency
     "72 мс/запит проти 229 мс" with tech-specific values
     "70.9 мс/запит проти 222.5 мс" (matches Table 5.1 + screenshot fig 5.24).
  2. Summary paragraph: clarify that E5-base is the "closest NEURAL
     alternative" (since BM25 is now in Pareto and is even faster).
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

    # 1) Tech paragraph: cross-domain avg -> tech-specific
    xml = expect(
        xml,
        "(72 мс/запит проти 229 мс для BGE-M3)",
        "(70.9 мс/запит проти 222.5 мс для BGE-M3)",
        count=1,
        label="tech-specific latency",
    )

    # 2) Summary: clarify E5-base is "closest NEURAL" alternative.
    # The text is split across runs by a lastRenderedPageBreak, so we
    # patch each run-text separately.
    xml = expect(
        xml,
        "E5-base є найближчою альтернативою з перевагою за швидкодією ",
        "E5-base є найближчою серед нейронних альтернатив з перевагою за швидкодією ",
        count=1,
        label="clarify E5-base neural alt (part 1)",
    )
    xml = expect(
        xml,
        "<w:t>(72 мс/запит), що робить її оптимальним вибором за умов жорстких вимог до затримки або ресурсних обмежень.</w:t>",
        '<w:t xml:space="preserve">(72 мс/запит у середньому по доменах), що робить її оптимальним вибором за умов жорстких вимог до затримки або ресурсних обмежень.</w:t>',
        count=1,
        label="clarify E5-base neural alt (part 2)",
    )

    DOC_XML.write_text(xml, encoding="utf-8")

    backup = DOCX.with_suffix(".bak6.docx")
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

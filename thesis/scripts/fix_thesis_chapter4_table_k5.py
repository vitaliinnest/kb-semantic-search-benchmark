"""
Two final consistency fixes:

1. Table 4.1 K5 column doesn't match the 1-5 bal scale defined in section 4.2.5:
       >1000 ms -> 1, 500-1000 -> 2, 200-500 -> 3, 100-200 -> 4, <100 -> 5
   Per scale, BGE-M3 (~229 ms) and Qwen3 (~445 ms) both fall into 200-500 ->
   3 bals, but Table 4.1 listed them as 2 and 1 respectively.

   Note: this only affects Table 4.1 (the in-bal-only view). The downstream
   Pareto/U-score computation runs on Table 4.2 (latency in ms) and is
   unaffected.

2. Latin-Cyrillic typo in section 5.4 legal paragraph:
       'Otриманий' -> 'Отриманий' (the 'O' was U+004F instead of U+041E).
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

    # ── Fix 1: Table 4.1 K5 column ─────────────────────────────────────────
    # Find the unique row for BGE-M3 in Table 4.1 (it has K3=5 → row "BGE-M3 4 4 5 1 2")
    # and Qwen3 in Table 4.1 (row "Qwen3-Embedding-0.6B 4 4 4 1 1").
    # In the XML, each cell is its own <w:tc><w:p>... so we look for the
    # specific cell containing just "2" or "1" within the BGE-M3/Qwen3 row.
    # Easier: directly target the unique numeric cell by anchor.
    #
    # The table layout cells contain raw <w:t>2</w:t> etc. for the K5 column.
    # We need to find specifically the K5=2 cell after BGE-M3, and K5=1 cell after Qwen3.
    # Strategy: find "BGE-M3" cell, walk to its row's last <w:t>N</w:t> before next row.

    import re
    # locate Table 4.1 region by anchoring on its caption
    cap_idx = xml.find("Таблиця 4.1 – Початковий векторний опис альтернатив")
    tbl_start = xml.find("<w:tbl>", cap_idx)
    tbl_end = xml.find("</w:tbl>", tbl_start) + len("</w:tbl>")
    tbl = xml[tbl_start:tbl_end]
    # Sanity: should contain BGE-M3 and Qwen3 rows
    assert "BGE-M3" in tbl and "Qwen3-Embedding-0.6B" in tbl, "Table 4.1 not located"

    # Each row has 6 cells (header: model | K1..K5).
    # Replace BGE-M3 row K5 (last cell value '2' -> '3') and Qwen3 K5 ('1' -> '3').
    # Use a regex that captures the full row of cells so we can rewrite the last cell.
    rows = re.split(r"(</w:tr>)", tbl)
    new_rows = []
    for r in rows:
        if "BGE-M3" in r and "Якість ранжування" not in r:
            # In the BGE-M3 row, the last <w:t>NUMBER</w:t> is K5 ('2')
            r = re.sub(r"(<w:t>2</w:t>)(?!.*<w:t>\d</w:t>)", "<w:t>3</w:t>", r, count=1, flags=re.DOTALL)
            print("  [tbl 4.1] BGE-M3 K5: 2 -> 3")
        elif "Qwen3-Embedding-0.6B" in r and "Якість ранжування" not in r:
            r = re.sub(r"(<w:t>1</w:t>)(?!.*<w:t>\d</w:t>)", "<w:t>3</w:t>", r, count=1, flags=re.DOTALL)
            print("  [tbl 4.1] Qwen3 K5: 1 -> 3")
        new_rows.append(r)
    new_tbl = "".join(new_rows)
    if new_tbl == tbl:
        raise SystemExit("FAIL: nothing changed in Table 4.1")
    xml = xml[:tbl_start] + new_tbl + xml[tbl_end:]

    # ── Fix 2: Latin-O typo "Otриманий" -> "Отриманий" ──────────────────────
    xml = expect(
        xml,
        "Otриманий",  # Latin O + Cyrillic тримани...
        "Отриманий",  # Cyrillic О + Cyrillic тримани...
        count=1,
        label="Latin-O typo fix",
    )

    DOC_XML.write_text(xml, encoding="utf-8")

    backup = DOCX.with_suffix(".bak7.docx")
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

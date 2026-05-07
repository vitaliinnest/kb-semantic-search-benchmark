"""
Update abstract source count "33 джерела" → "29 джерел".

The "33" digits are split into two <w:t> runs (one per character) and
"джерела" is split as "джерел" + "а". Need targeted edits:
  - First "3" run text  -> "2"
  - Second "3" run text -> "9"
  - Drop trailing "а" run (replace with empty text)
"""
import os, re, shutil, sys, zipfile, pathlib

sys.stdout.reconfigure(encoding="utf-8")

ROOT = pathlib.Path("D:/repos/kb-semantic-search-benchmark")
DOCX = ROOT / "thesis" / "2026_M_PI_Nesterenko_VV.docx"
UNPACKED = ROOT / "thesis" / "unpacked_docx"
DOC_XML = UNPACKED / "word" / "document.xml"


def main():
    if UNPACKED.exists():
        shutil.rmtree(UNPACKED)
    with zipfile.ZipFile(DOCX) as z:
        z.extractall(UNPACKED)
    print(f"[unpack] {DOCX.name}")

    xml = DOC_XML.read_text(encoding="utf-8")

    # Anchor on the precise unique sequence around "9 табл., 33 джерела."
    old = (
        '9 табл., </w:t></w:r>'
        '<w:r w:rsidR="00D2275C"><w:rPr><w:lang w:val="en-US"/></w:rPr><w:t>3</w:t></w:r>'
        '<w:r w:rsidR="00196DCB"><w:rPr><w:lang w:val="ru-RU"/></w:rPr><w:t>3</w:t></w:r>'
        '<w:r w:rsidRPr="004A753E"><w:t xml:space="preserve"> джерел</w:t></w:r>'
        '<w:r w:rsidR="00900144"><w:t>а</w:t></w:r>'
    )
    new = (
        '9 табл., </w:t></w:r>'
        '<w:r w:rsidR="00D2275C"><w:rPr><w:lang w:val="en-US"/></w:rPr><w:t>2</w:t></w:r>'
        '<w:r w:rsidR="00196DCB"><w:rPr><w:lang w:val="ru-RU"/></w:rPr><w:t>9</w:t></w:r>'
        '<w:r w:rsidRPr="004A753E"><w:t xml:space="preserve"> джерел.</w:t></w:r>'
    )
    if old not in xml:
        # Try a more permissive search
        idx = xml.find("9 табл., ")
        print(f"FAIL: anchor not found verbatim. Found '9 табл., ' at @{idx}, content:")
        print(repr(xml[idx:idx+800]))
        sys.exit(1)

    cnt = xml.count(old)
    if cnt != 1:
        print(f"FAIL: expected 1 match, found {cnt}")
        sys.exit(1)

    xml = xml.replace(old, new, 1)
    print(f"  ok  '33 джерела' -> '29 джерел'")

    # Need to also handle the trailing period — old structure had a separate "."
    # run after "а" that we need to make sure stays consistent
    # Look at next 200 chars after our patch
    idx = xml.find("джерел")
    print(f"  context after fix: ...{re.sub(chr(60)+r'[^>]+>', '', xml[max(0,idx-30):idx+80])}...")

    DOC_XML.write_text(xml, encoding="utf-8")

    backup = DOCX.with_suffix(".bak10.docx")
    shutil.copy2(DOCX, backup)
    print(f"[backup] -> {backup.name}")
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

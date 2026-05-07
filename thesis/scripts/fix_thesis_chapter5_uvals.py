"""
Update stale U-values in Chapter 5.4 (practical MCDA) so they match
the current benchmark JSON results.

Old (computed from f987aff benchmark with slower latencies):
  Tech:    BGE-M3 U=0.920, E5-base U=0.863
  Legal:   BGE-M3 U=0.927
  Medical: BGE-M3 U=0.901, E5-base U=0.837

Current (from results/benchmark_*.json + multi_criteria.run_selection):
  Tech:    BGE-M3 U=0.923, E5-base U=0.863
  Legal:   BGE-M3 U=0.922
  Medical: BGE-M3 U=0.927, E5-base U=0.836

Tables 5.1–5.3 already use the current (fresh) latencies, so the U-values
need to follow.
"""
import sys
import shutil
import zipfile
import pathlib

sys.stdout.reconfigure(encoding="utf-8")

ROOT = pathlib.Path("D:/repos/kb-semantic-search-benchmark")
DOCX = ROOT / "thesis" / "2026_M_PI_Nesterenko_VV.docx"
UNPACKED = ROOT / "thesis" / "unpacked_docx"
DOC_XML = UNPACKED / "word" / "document.xml"


def expect(xml: str, old: str, new: str, *, count: int, label: str) -> str:
    found = xml.count(old)
    if found != count:
        raise SystemExit(f"FAIL [{label}]: expected {count} of {old!r}, found {found}")
    print(f"  ok  [{label}]: {old!r} -> {new!r}  (x{count})")
    return xml.replace(old, new, count)


def main():
    # Unpack
    if UNPACKED.exists():
        shutil.rmtree(UNPACKED)
    with zipfile.ZipFile(DOCX) as z:
        z.extractall(UNPACKED)
    print(f"[unpack] {DOCX.name}")

    xml = DOC_XML.read_text(encoding="utf-8")

    # Tech BGE-M3 U: 0.920 -> 0.923  (ONLY in the per-domain paragraph, not the summary "0.920/0.927/0.901")
    xml = expect(
        xml,
        "BGE-M3 з інтегральною оцінкою 0.920",
        "BGE-M3 з інтегральною оцінкою 0.923",
        count=1,
        label="Tech BGE-M3 0.920->0.923",
    )

    # Legal BGE-M3 U: 0.927 -> 0.922  (per-domain)
    xml = expect(
        xml,
        "BGE-M3 з інтегральною оцінкою 0.927",
        "BGE-M3 з інтегральною оцінкою 0.922",
        count=1,
        label="Legal BGE-M3 0.927->0.922",
    )

    # Medical BGE-M3 U: 0.901 -> 0.927, E5-base 0.837 -> 0.836
    xml = expect(
        xml,
        "BGE-M3 з інтегральною оцінкою 0.901, E5-base посіла друге місце (0.837)",
        "BGE-M3 з інтегральною оцінкою 0.927, E5-base посіла друге місце (0.836)",
        count=1,
        label="Medical BGE 0.901->0.927, E5 0.837->0.836",
    )

    # Summary line "U=0.920/0.927/0.901" -> "U=0.923/0.922/0.927"
    xml = expect(
        xml,
        "U=0.920/0.927/0.901",
        "U=0.923/0.922/0.927",
        count=1,
        label="summary 0.920/0.927/0.901 -> 0.923/0.922/0.927",
    )

    DOC_XML.write_text(xml, encoding="utf-8")

    # Repack (overwrite docx; backup-aware)
    backup = DOCX.with_suffix(".bak2.docx")
    shutil.copy2(DOCX, backup)
    print(f"[backup] -> {backup.name}")
    with zipfile.ZipFile(DOCX, "w", zipfile.ZIP_DEFLATED) as zf:
        import os
        for root, _, files in os.walk(UNPACKED):
            for f in files:
                full = pathlib.Path(root) / f
                rel = full.relative_to(UNPACKED).as_posix()
                zf.write(full, rel)
    print(f"[repack] {DOCX.name} ({DOCX.stat().st_size:,} bytes)")
    print("\nDone.")


if __name__ == "__main__":
    main()

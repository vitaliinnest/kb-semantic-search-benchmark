"""
Pre-defense fixes for 2026_M_PI_Nesterenko_VV.docx (theoretical part).

Surgical XML edits — does not regenerate the document from scratch.

Changes:
  1.  "(~600 слів)"            -> "(~800 слів)"           [chapter 5.1, corpus stats]
  2.  "minK5=151мс"             -> "minK5=72мс"            [section 4.3.6, normalization narrative]
  3.  "maxK5=669мс"             -> "maxK5=445мс"           [section 4.3.6, normalization narrative]
  4.  "(~1500 мс/запит на CPU)" -> "(~445 мс/запит на CPU)"  [chapter 5.4, two occurrences]
  5.  "(~498 мс/запит на CPU)"  -> "(~436 мс/запит на CPU)"  [chapter 5.4, medical example]
  6.  "(~711 мс/запит)"         -> "(~437 мс/запит)"      [chapter 5.4, legal example]
  7.  Formula 4.11: A = {a1, a2, a3, a4, a5}  -> {a1, a2, a3, a4}
  8.  Formula 4.13: 5x5 matrix X -> 4x5 (drop row 5)
  9.  Formula 4.17: 5x5 matrix Y -> 4x5 (drop row 5)
  10. Cross-reference "як таблиця 5.3" -> "як таблиця 4.3"
  11. Tables renumbered 4.3->4.1, 4.4->4.2, 4.5->4.3, 4.6->4.4, 4.7->4.5
      (titles + body cross-references)
  12. Title of (renumbered) Table 4.1 disambiguated:
      "Векторний опис альтернатив" -> "Початковий векторний опис альтернатив у балах"
  13. K1 best-domain note appended near (renumbered) Table 4.2.
  14. K4 degenerate-criterion note appended near normalization in 4.3.6.
"""
import re
import sys
import shutil
import zipfile
import pathlib

sys.stdout.reconfigure(encoding="utf-8")

ROOT = pathlib.Path("D:/repos/kb-semantic-search-benchmark")
DOCX = ROOT / "thesis" / "2026_M_PI_Nesterenko_VV.docx"
UNPACKED = ROOT / "thesis" / "unpacked_docx"
DOC_XML = UNPACKED / "word" / "document.xml"


def unpack():
    if UNPACKED.exists():
        shutil.rmtree(UNPACKED)
    with zipfile.ZipFile(DOCX) as z:
        z.extractall(UNPACKED)
    print(f"[unpack] {DOCX.name} -> {UNPACKED}")


def repack():
    backup = DOCX.with_suffix(".bak.docx")
    shutil.copy2(DOCX, backup)
    print(f"[backup] -> {backup}")
    with zipfile.ZipFile(DOCX, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in __import__("os").walk(UNPACKED):
            for f in files:
                full = pathlib.Path(root) / f
                rel = full.relative_to(UNPACKED).as_posix()
                zf.write(full, rel)
    size = DOCX.stat().st_size
    print(f"[repack] {DOCX.name} ({size:,} bytes)")


def expect_replace(xml: str, old: str, new: str, *, count: int = 1, label: str = "") -> str:
    """str.replace() with assertion on number of replacements."""
    found = xml.count(old)
    if found != count:
        raise SystemExit(
            f"FAIL [{label}]: expected {count} occurrence(s) of {old!r}, found {found}"
        )
    out = xml.replace(old, new, count)
    print(f"  ok  [{label}]: {old!r:60s} -> {new!r}  (x{count})")
    return out


def main():
    unpack()

    xml = DOC_XML.read_text(encoding="utf-8")
    orig_len = len(xml)

    # ───── 1.  ~600 слів  ->  ~800 слів  ────────────────────────────────────
    xml = expect_replace(
        xml,
        "(~600 слів)",
        "(~800 слів)",
        label="600 -> 800 слів (corpus stats)",
    )

    # ───── 2.  min K5 = 151мс  ->  72мс  ─────────────────────────────────────
    xml = expect_replace(
        xml,
        "<m:t>=151мс;</m:t>",
        "<m:t>=72мс;</m:t>",
        label="min K5 151 -> 72",
    )

    # ───── 3.  max K5 = 669мс  ->  445мс  ────────────────────────────────────
    xml = expect_replace(
        xml,
        "<m:t>=669мс.</m:t>",
        "<m:t>=445мс.</m:t>",
        label="max K5 669 -> 445",
    )

    # ───── 4.  ~1500 мс  ->  ~445 мс  (Qwen3 latency, 2 places)  ────────────
    xml = expect_replace(
        xml,
        "(~1500 мс/запит на CPU)",
        "(~445 мс/запит на CPU)",
        count=2,
        label="Qwen3 ~1500 -> ~445 ms",
    )

    # ───── 5.  ~498 мс  ->  ~436 мс  (Qwen3 medical)  ────────────────────────
    xml = expect_replace(
        xml,
        "(~498 мс/запит на CPU)",
        "(~436 мс/запит на CPU)",
        label="Qwen3 medical ~498 -> ~436",
    )

    # ───── 6.  ~711 мс  ->  ~437 мс  (Qwen3 legal)  ──────────────────────────
    xml = expect_replace(
        xml,
        "(~711 мс/запит)",
        "(~437 мс/запит)",
        label="Qwen3 legal ~711 -> ~437",
    )

    # ───── 7.  Formula 4.11 — drop a5 ───────────────────────────────────────
    # Pattern: separator ", " + sSub(a, 5)  immediately before  m:r{">"}
    # Find "(4.11)" tag, then walk back to find/remove the a5 sSub + leading ", ".
    f411_idx = xml.find("(4.11)")
    if f411_idx < 0:
        raise SystemExit("FAIL: '(4.11)' tag not found")
    # Find the closing }-bracket of formula 4.11 (the m:t>}</m:t> just before the (4.11) tag)
    close_bracket = xml.rfind("<m:t>}</m:t>", 0, f411_idx)
    if close_bracket < 0:
        raise SystemExit("FAIL: closing } of formula 4.11 not found")
    # Find the previous sSub end (which contains a5)
    a5_end = xml.rfind("</m:sSub>", 0, close_bracket)
    a5_start = xml.rfind("<m:sSub>", 0, a5_end)
    # Check this sSub indeed contains 'a' subscript '5'
    a5_block = xml[a5_start:a5_end + len("</m:sSub>")]
    if "<m:t>a</m:t>" not in a5_block or "<m:t>5</m:t>" not in a5_block:
        raise SystemExit("FAIL: a5 sSub not as expected")
    # Find the ", " separator immediately before a5
    # It's an m:r ... <m:t>, </m:t>...</m:r> right before <m:sSub>
    sep_end = a5_start
    sep_start = xml.rfind("<m:r>", 0, sep_end)
    sep_block = xml[sep_start:sep_end]
    # Word uses U+2008 PUNCTUATION SPACE after the comma in formula separators.
    # Accept any m:t whose content starts with ',' (with or without trailing space variants).
    if not re.search(r"<m:t[^>]*>,[\s  ]*</m:t>", sep_block):
        raise SystemExit(
            f"FAIL: ', ' separator before a5 not found. sep_block={sep_block!r}"
        )
    # Remove [sep_start, a5_end+9)
    drop = a5_end + len("</m:sSub>") - sep_start
    xml = xml[:sep_start] + xml[a5_end + len("</m:sSub>"):]
    print(f"  ok  [4.11]: dropped {drop} chars (', a5' subscript)")

    # ───── 8 & 9.  Drop last m:mr from both 5x5 matrices  ───────────────────
    # There are exactly two <m:m> elements with 5 rows each (formulas 4.13 and 4.17)
    cursor = 0
    matrices_fixed = 0
    while True:
        m_open = xml.find("<m:m>", cursor)
        if m_open < 0:
            break
        m_close = xml.find("</m:m>", m_open)
        if m_close < 0:
            break
        matrix_xml = xml[m_open:m_close]
        rows = matrix_xml.count("<m:mr>")
        if rows == 5:
            # Find the last <m:mr> ... </m:mr> within this matrix
            last_mr_open = matrix_xml.rfind("<m:mr>")
            last_mr_close = matrix_xml.find("</m:mr>", last_mr_open) + len("</m:mr>")
            new_matrix = matrix_xml[:last_mr_open] + matrix_xml[last_mr_close:]
            xml = xml[:m_open] + new_matrix + xml[m_close:]
            removed = m_close - m_open - len(new_matrix)
            print(f"  ok  [matrix @{m_open}]: dropped row 5 ({removed} chars)")
            matrices_fixed += 1
            cursor = m_open + len(new_matrix) + len("</m:m>")
        else:
            cursor = m_close + len("</m:m>")
    if matrices_fixed != 2:
        raise SystemExit(f"FAIL: expected 2 matrices fixed, got {matrices_fixed}")

    # ───── 10.  "як таблиця 5.3" -> "як таблиця 4.3"  (split-run reference) ─
    # The "5" and ".3" are in two separate <w:t> elements. We replace the "5" only.
    needle = (
        "як таблиця </w:t></w:r>"
        '<w:r w:rsidR="006E795D"><w:t>5</w:t></w:r>'
        '<w:r w:rsidRPr="0012776F"><w:t>.3</w:t></w:r>'
    )
    new = (
        "як таблиця </w:t></w:r>"
        '<w:r w:rsidR="006E795D"><w:t>4</w:t></w:r>'
        '<w:r w:rsidRPr="0012776F"><w:t>.3</w:t></w:r>'
    )
    xml = expect_replace(xml, needle, new, label="cross-ref 5.3 -> 4.3")

    # ───── 11.  Renumber tables 4.3..4.7 -> 4.1..4.5  ───────────────────────
    # IMPORTANT: shift in DESCENDING order to avoid double-mapping.
    # Title transformations (literal "Таблиця 4.X" appears once each)
    # Cross-refs: "таблиці 4.X", "таблиць 4.X", "таблиця 4.X" (lowercase, body refs)
    # We use unique 'sentinel' tokens to avoid double-renumbering during the swap.
    title_pairs = [
        ("Таблиця 4.7", "Таблиця 4.5"),
        ("Таблиця 4.6", "Таблиця 4.4"),
        ("Таблиця 4.5", "Таблиця 4.3"),
        ("Таблиця 4.4", "Таблиця 4.2"),
        ("Таблиця 4.3", "Таблиця 4.1"),
    ]
    # We can't apply these directly: replacing 4.7->4.5 then 4.5->4.3 would clobber the result of step 1.
    # Use a two-pass approach with sentinels.
    sentinels = [
        ("Таблиця 4.7", "§§T47§§"),
        ("Таблиця 4.6", "§§T46§§"),
        ("Таблиця 4.5", "§§T45§§"),
        ("Таблиця 4.4", "§§T44§§"),
        ("Таблиця 4.3", "§§T43§§"),
    ]
    for old, sentinel in sentinels:
        xml = expect_replace(xml, old, sentinel, label=f"sentinel {old}")
    final_titles = [
        ("§§T47§§", "Таблиця 4.5"),
        ("§§T46§§", "Таблиця 4.4"),
        ("§§T45§§", "Таблиця 4.3"),
        ("§§T44§§", "Таблиця 4.2"),
        ("§§T43§§", "Таблиця 4.1"),
    ]
    for sentinel, new in final_titles:
        xml = expect_replace(xml, sentinel, new, label=f"finalize {new}")

    # Body cross-references: lowercase forms
    body_renumber = [
        # 4.7 -> 4.5  (3 occurrences total: "таблиці 4.7" x2 + "таблиці 4.7," check)
        ("таблиці 4.7", "§§t47§§"),
        ("таблиці 4.6", "§§t46§§"),
        ("таблиці 4.5", "§§t45§§"),
        ("таблиці 4.4", "§§t44§§"),
    ]
    body_finalize = [
        ("§§t47§§", "таблиці 4.5"),
        ("§§t46§§", "таблиці 4.4"),
        ("§§t45§§", "таблиці 4.3"),
        ("§§t44§§", "таблиці 4.2"),
    ]
    # apply with auto-count
    def replace_all(xml, old, new, label):
        cnt = xml.count(old)
        if cnt == 0:
            print(f"  -- [{label}]: no occurrences of {old!r}")
            return xml
        xml = xml.replace(old, new)
        print(f"  ok  [{label}]: {old!r:25s} -> {new!r:25s}  (x{cnt})")
        return xml

    for old, sentinel in body_renumber:
        xml = replace_all(xml, old, sentinel, f"body sentinel {old}")
    for sentinel, new in body_finalize:
        xml = replace_all(xml, sentinel, new, f"body finalize {new}")

    # Also handle "як таблиця 4.3" -> "як таблиця 4.1" since we renumbered 4.3 -> 4.1
    # Wait: cross-ref 10 set "як таблиця 4.3" (split runs), and now table 4.3 became 4.1.
    # Update this specific reference too.
    needle2 = (
        "як таблиця </w:t></w:r>"
        '<w:r w:rsidR="006E795D"><w:t>4</w:t></w:r>'
        '<w:r w:rsidRPr="0012776F"><w:t>.3</w:t></w:r>'
    )
    new2 = (
        "як таблиця </w:t></w:r>"
        '<w:r w:rsidR="006E795D"><w:t>4</w:t></w:r>'
        '<w:r w:rsidRPr="0012776F"><w:t>.1</w:t></w:r>'
    )
    xml = expect_replace(xml, needle2, new2, label="cross-ref 4.3 -> 4.1 (split-run)")

    # ───── 12.  Disambiguate title of (new) Table 4.1  ──────────────────────
    xml = expect_replace(
        xml,
        "Таблиця 4.1 – Векторний опис альтернатив (таблиця виконана самостійно)",
        "Таблиця 4.1 – Початковий векторний опис альтернатив у балах за п'ятирівневою шкалою (таблиця виконана самостійно)",
        label="rename Table 4.1 (was 4.3)",
    )

    # ───── 13.  K1 best-domain note  ────────────────────────────────────────
    # Append to the paragraph that introduces Table 4.2 (was 4.4).
    # That paragraph ends with "...подається у мілісекундах як критерій типу «за мінімумом»."
    xml = expect_replace(
        xml,
        "подається у мілісекундах як критерій типу «за мінімумом».",
        "подається у мілісекундах як критерій типу «за мінімумом». "
        "Бал за критерієм K₁ присвоюється на основі найкращого результату моделі "
        "серед трьох предметних доменів (технічного, юридичного та медичного).",
        label="K1 best-domain note",
    )

    # ───── 14.  K4 degenerate-criterion note  ────────────────
    # Original text is split across multiple <w:t> runs. Append the note to
    # the first fragment within the same run by replacing its closing tag.
    xml = expect_replace(
        xml,
        "<w:t>Аналогічно обчислюються нормалізовані значення для інших альтернатив.</w:t>",
        '<w:t xml:space="preserve">Аналогічно обчислюються нормалізовані значення '
        'для інших альтернатив. Зазначимо, що для критерію K₄ значення max K₄ = min K₄ = 1 '
        'призводить до невизначеності формули нормалізації; '
        'за конвенцією прийнято y_(i4) = 0, що означає виродженість критерію '
        'в межах розглянутого набору альтернатив і відсутність його впливу '
        'на впорядкування Парето-множини.</w:t>',
        label="K4 degenerate-criterion note",
    )

    # ───── Save  ────────────────────────────────────────────────────────────
    new_len = len(xml)
    DOC_XML.write_text(xml, encoding="utf-8")
    print(f"\n[write] document.xml: {orig_len:,} -> {new_len:,} bytes  (Δ{new_len-orig_len:+,})")

    repack()
    print("\nDone.")


if __name__ == "__main__":
    main()

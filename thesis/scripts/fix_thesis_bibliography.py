"""
Bibliography cleanup:

1. Remove duplicate entries: [21] (≡ [5] Manning IR), [22] (≡ [6] Karpukhin DPR).
2. Remove orphaned entries (in list but never cited):
   [18] Malkov HNSW, [33] GitHub.
3. Redirect body [22] citations to [6] (since they point to the same paper).
4. Renumber remaining entries [19], [20], [23]-[32] to fill gaps:
       [19] → [18]   Johnson FAISS GPU
       [20] → [19]   Guo Neural Ranking
       [23] → [20]   Järvelin DCG
       [24] → [21]   Wang E5
       [25] → [22]   Chen BGE-M3
       [26] → [23]   Nussbaum nomic
       [27] → [24]   Zhang Qwen3
       [28] → [25]   Robertson BM25
       [29] → [26]   Thakur BEIR
       [30] → [27]   Muennighoff MTEB
       [31] → [28]   Lewis RAG
       [32] → [29]   Gao RAG survey
5. Update body citations to match the new numbering (sentinel-based to avoid
   double-mapping).
6. Update abstract "33 джерела" → "29 джерел".

Note: secondary list entries [9]-[12] (supervisor's works) are unchanged.
"""
import os
import re
import shutil
import sys
import zipfile
import pathlib

sys.stdout.reconfigure(encoding="utf-8")

ROOT = pathlib.Path("D:/repos/kb-semantic-search-benchmark")
DOCX = ROOT / "thesis" / "2026_M_PI_Nesterenko_VV.docx"
UNPACKED = ROOT / "thesis" / "unpacked_docx"
DOC_XML = UNPACKED / "word" / "document.xml"


# Body citation remap (OLD → NEW). Includes redirect [22] → [6].
BODY_REMAP = {
    19: 18,
    20: 19,
    22: 6,    # REDIRECT (duplicate of [6])
    23: 20,
    24: 21,
    25: 22,
    26: 23,
    27: 24,
    28: 25,
    29: 26,
    30: 27,
    31: 28,
    32: 29,
}

# Bibliography entries to remove.
TO_REMOVE = {18, 21, 22, 33}

# Bibliography entries to renumber (OLD → NEW).
# Note: 22 is removed (not renumbered).
BIB_RENUMBER = {
    19: 18,
    20: 19,
    23: 20,
    24: 21,
    25: 22,
    26: 23,
    27: 24,
    28: 25,
    29: 26,
    30: 27,
    31: 28,
    32: 29,
}


def unpack():
    if UNPACKED.exists():
        shutil.rmtree(UNPACKED)
    with zipfile.ZipFile(DOCX) as z:
        z.extractall(UNPACKED)
    print(f"[unpack] {DOCX.name}")


def repack():
    backup = DOCX.with_suffix(".bak9.docx")
    shutil.copy2(DOCX, backup)
    print(f"[backup] -> {backup.name}")
    with zipfile.ZipFile(DOCX, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(UNPACKED):
            for f in files:
                full = pathlib.Path(root) / f
                rel = full.relative_to(UNPACKED).as_posix()
                zf.write(full, rel)
    print(f"[repack] {DOCX.name} ({DOCX.stat().st_size:,} bytes)")


def remap_body_citations(xml: str) -> str:
    """Two-pass sentinel substitution for body [N] citations."""
    # Phase 1: convert all OLD [N] to sentinels
    for old_n in BODY_REMAP:
        xml = xml.replace(f"[{old_n}]", f"§§B{old_n}§§")

    # Phase 2: replace sentinels with new numbers
    for old_n, new_n in BODY_REMAP.items():
        xml = xml.replace(f"§§B{old_n}§§", f"[{new_n}]")

    return xml


def find_bib_paragraphs(xml: str):
    """Return list of (n, p_start, p_end) for each bibliography entry.

    Walks <w:p> elements between bib region anchors.
    """
    # The actual bibliography heading is the second-to-last "Перелік джерел посилання" occurrence.
    # Use byte-position heuristic: bib region is around 1.22M..1.31M.
    bib_start = 1222000
    # The secondary list heading text might be split across runs. Walk
    # forward looking for "за науковими напрямами" near it.
    bib_end = xml.find("за науковими напрямами", bib_start)
    if bib_end < 0:
        # Fall back: find last occurrence of "Перелік джерел"
        bib_end = xml.rfind("Перелік джерел")
        if bib_end < bib_start:
            raise SystemExit("FAIL: secondary bib list anchor not found")

    para_starts = [m.start() for m in re.finditer(r"<w:p ", xml[bib_start:bib_end])]
    para_starts = [p + bib_start for p in para_starts]
    # Append synthetic end position
    para_ends = para_starts[1:] + [bib_end]

    entries = []
    for p_start, p_end in zip(para_starts, para_ends):
        para_xml = xml[p_start:p_end]
        plain = re.sub(r"<[^>]+>", "", para_xml).strip()
        m = re.match(r"^(\d{1,2})\.\s+", plain)
        if not m:
            continue
        n = int(m.group(1))
        # Only main list 1-33 (skip 9-12 = secondary list — those are in a different region)
        if n < 1 or n > 33:
            continue
        # Find actual <w:p ...> ... </w:p> closing tag
        close_match = re.search(r"</w:p>", xml[p_start:])
        if not close_match:
            continue
        actual_end = p_start + close_match.end()
        entries.append((n, p_start, actual_end, plain))
    return entries


def rewrite_bib(xml: str) -> str:
    """Remove + renumber bibliography paragraphs.

    Strategy: process from END to START to preserve byte offsets.
    For each paragraph:
      - if N in TO_REMOVE → drop the entire <w:p>...</w:p> block
      - if N in BIB_RENUMBER → replace leading "OLD." with "NEW." inside the
        paragraph; consolidate digit text into the first <w:t> for safety.
    """
    entries = find_bib_paragraphs(xml)
    print(f"[bib] {len(entries)} entries detected")

    # Sort descending by start so removals don't shift earlier offsets
    for n, p_start, p_end, plain in sorted(entries, key=lambda e: -e[1]):
        para_xml = xml[p_start:p_end]
        if n in TO_REMOVE:
            xml = xml[:p_start] + xml[p_end:]
            print(f"  REMOVE [{n:2d}]  ({p_end - p_start} bytes)")
        elif n in BIB_RENUMBER:
            new_n = BIB_RENUMBER[n]
            new_para_xml = renumber_para_xml(para_xml, n, new_n)
            xml = xml[:p_start] + new_para_xml + xml[p_end:]
            print(f"  RENUM  [{n:2d}] -> [{new_n:2d}]  (Δ{len(new_para_xml) - len(para_xml):+d} bytes)")
    return xml


def renumber_para_xml(para_xml: str, old_n: int, new_n: int) -> str:
    """Replace the leading "OLD_N." with "NEW_N." inside a paragraph's XML.

    Handles three observed structures:
      A) <w:t>O</w:t>...<w:t>L</w:t>...<w:t>D. ...   (digits split, e.g. "1"+"3")
      B) <w:t>OLD_N</w:t>...<w:t>. ...               (full number in one run)
      C) <w:t xml:space="preserve">OLD_N. </w:t>     (number + dot in one run)
    """
    # Find all <w:t ...>...</w:t> elements with positions
    t_pattern = re.compile(r"<w:t[^>]*>([^<]*)</w:t>")
    t_matches = list(t_pattern.finditer(para_xml))
    if not t_matches:
        return para_xml

    # Concatenate text and find which <w:t> elements collectively contain
    # the leading number "OLD_N" (digits only — no dot needed yet).
    old_str = str(old_n)
    new_str = str(new_n)
    accumulated = ""
    digit_run_indices = []
    for i, m in enumerate(t_matches):
        text = m.group(1)
        accumulated += text
        digit_run_indices.append(i)
        if accumulated.startswith(old_str) and len(accumulated) >= len(old_str):
            # We have the leading digits within accumulated.
            # Find which <w:t> elements they live in and replace.
            break
    else:
        raise SystemExit(f"FAIL: could not locate leading '{old_str}' in paragraph")

    # We have indices [0..k] where the accumulated text reaches/exceeds old_str.
    # Strategy: put new_str into the FIRST <w:t>, clear leading digits from the
    # subsequent <w:t> elements (keeping any non-digit suffix in the last one).
    chars_needed = len(old_str)

    # Reconstruct texts for runs 0..k
    new_texts = []
    chars_consumed = 0
    for i in digit_run_indices:
        original = t_matches[i].group(1)
        if i == digit_run_indices[0]:
            # First run gets the full new number
            # Consume chars_needed bytes from the OLD digits across all the digit runs
            # We need to figure out how many chars came from THIS run's text
            chars_in_this = min(len(original), chars_needed)
            # Replace those chars with new_str
            suffix = original[chars_in_this:]
            new_texts.append(new_str + suffix)
            chars_consumed += chars_in_this
        else:
            # Drop chars_needed - chars_consumed chars from start of this run
            remaining = chars_needed - chars_consumed
            chars_in_this = min(len(original), remaining)
            new_texts.append(original[chars_in_this:])
            chars_consumed += chars_in_this
        if chars_consumed >= chars_needed:
            break

    # Apply replacements to para_xml. Process in reverse so offsets stay valid.
    new_para = para_xml
    for idx, new_text in zip(digit_run_indices[: len(new_texts)], new_texts):
        if idx >= len(t_matches):
            continue
    # Easier: rebuild by replacing each <w:t>...</w:t> in order
    # Substitute in reverse order (high-to-low indices) so positions don't shift
    for j, idx in enumerate(reversed(digit_run_indices[: len(new_texts)])):
        new_text = new_texts[len(new_texts) - 1 - j]
        m = t_matches[idx]
        old_full = m.group(0)
        # Build new <w:t...>NEW</w:t>
        # Preserve original opening tag
        opening = re.match(r"<w:t[^>]*>", old_full).group(0)
        # If new_text is empty AND text contains only spaces in original → keep xml:space?
        new_full = f"{opening}{new_text}</w:t>"
        # Replace at position
        m_start = m.start()
        m_end = m.end()
        new_para = new_para[:m_start] + new_full + new_para[m_end:]
        # Adjust subsequent t_matches positions (we're going backwards, so no need)

    return new_para


def update_abstract_count(xml: str) -> str:
    """Update abstract '33 джерела' to '29 джерел'."""
    # Try most likely literal first
    for old, new in [
        ("33 джерела", "29 джерел"),
        ("33 джерел", "29 джерел"),
    ]:
        if old in xml:
            xml = xml.replace(old, new, 1)
            print(f"  [abstract] {old!r} -> {new!r}")
            return xml
    print("  [abstract] WARN: '33 джерела' not found verbatim — may be split across runs")
    return xml


def main():
    unpack()
    xml = DOC_XML.read_text(encoding="utf-8")

    print("\n=== Phase 1: body citations ===")
    xml = remap_body_citations(xml)

    print("\n=== Phase 2: bibliography ===")
    xml = rewrite_bib(xml)

    print("\n=== Phase 3: abstract count ===")
    xml = update_abstract_count(xml)

    DOC_XML.write_text(xml, encoding="utf-8")
    repack()
    print("\nDone.")


if __name__ == "__main__":
    main()

"""
Fix formula numbering in chapter 4 (Опис теоретичного дослідження).

Issues fixed:
1. Display formulas labeled (5.x) — renumbered to (4.x)
2. Inline formula labels "(5.x)" → "(4.x)"
3. Five IR-metric formulas (DCG, nDCG, MRR, Recall, P@k) — added numbers (4.6)–(4.10)
4. Text references "формула 5.x" / "формулою 5.x" / "форм. 5.x" → "4.y"
5. Added text references for IR-metric formulas

Renumbering map (formulas only):
  5.1→4.1, 5.2→4.2, 5.3→4.3, 5.5→4.4, 5.6→4.5,
  NEW: 4.6 (DCG), 4.7 (nDCG), 4.8 (MRR), 4.9 (Recall), 4.10 (P@k),
  5.8→4.11, 5.9→4.12, 5.10→4.13, 5.11→4.14, 5.12→4.15,
  5.13→4.16, 5.14→4.17, 5.15→4.18, 5.16→4.19, 5.17→4.20,
  5.18→4.21, 5.19→4.22, 5.20→4.23, 5.21→4.24, 5.22→4.25,
  5.23→4.26, 5.24→4.27, 5.25→4.28, 5.26→4.29, 5.27→4.30

Note: Table references (Таблиця 5.x) are NOT touched.
"""
import zipfile, xml.etree.ElementTree as ET, sys, copy, pathlib, re, shutil
sys.stdout.reconfigure(encoding='utf-8')

ROOT = pathlib.Path("D:/repos/kb-semantic-search-benchmark")
DOCX = ROOT / "2026_M_PI_Nesterenko_VV.docx"
UNPACKED = ROOT / "unpacked_docx"

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
M = "http://schemas.openxmlformats.org/officeDocument/2006/math"

def _w(tag): return f"{{{W}}}{tag}"
def _m(tag): return f"{{{M}}}{tag}"

def get_text(el):
    return "".join(r.text or "" for r in el.iter(_w("t")))

def get_math_text(el):
    return "".join(r.text or "" for r in el.iter(_m("t")))

# ── Renumbering map ────────────────────────────────────────────────────
RENUMBER = {
    "5.1": "4.1",   "5.2": "4.2",   "5.3": "4.3",
    "5.5": "4.4",   "5.6": "4.5",
    "5.8": "4.11",  "5.9": "4.12",  "5.10": "4.13",
    "5.11": "4.14", "5.12": "4.15", "5.13": "4.16",
    "5.14": "4.17", "5.15": "4.18", "5.16": "4.19",
    "5.17": "4.20", "5.18": "4.21", "5.19": "4.22",
    "5.20": "4.23", "5.21": "4.24", "5.22": "4.25",
    "5.23": "4.26", "5.24": "4.27", "5.25": "4.28",
    "5.26": "4.29", "5.27": "4.30",
}

# Sort keys so longer numbers (5.27, 5.26...) come first to avoid double-replacement
# (5.2 inside 5.27 etc.)
SORTED_OLD = sorted(RENUMBER.keys(), key=lambda k: -int(k.split(".")[1]))

# ── IR metric formulas (oMathPara, no number) ─────────────────────────
IR_FORMULAS = {
    370: "4.6",   # DCG@k
    371: "4.7",   # nDCG@k
    381: "4.8",   # MRR
    391: "4.9",   # Recall@k
    401: "4.10",  # P@k
}

# ── Body indices that MUST NOT be touched (table refs, ToC entries) ───
SKIP_INDICES = {
    169, 170, 171, 172,        # ToC entries (5.1-5.4 in chapter 5)
    440, 442,                  # Table 5.3
    585, 587,                  # Table 5.4
    613, 615,                  # Table 5.5
    624, 626,                  # Table 5.6
    641, 644, 647,             # Table 5.7
    970, 971, 972, 974, 975, 976, 987, 990,  # later refs
}

# ── Re-unpack docx fresh ──────────────────────────────────────────────
print("Re-unpacking docx...")
if UNPACKED.exists():
    shutil.rmtree(UNPACKED)
UNPACKED.mkdir()
with zipfile.ZipFile(DOCX, 'r') as z:
    z.extractall(UNPACKED)

print("Loading document...")
tree = ET.parse(UNPACKED / "word" / "document.xml")
root = tree.getroot()
body = root.find(_w("body"))
children = list(body)
print(f"Total body children: {len(children)}")

# ── HELPER: replace text in a paragraph (handles split runs) ──────────
def replace_text_in_para_anywhere(para, old, new):
    """
    Replace `old` with `new` anywhere in the paragraph text.

    Word often splits text across multiple w:t elements (e.g. "формул"|"ою 5.1"
    may be in two separate runs). To handle this, we collect ALL w:t / m:t
    elements, join their text, find the substring, then redistribute.

    This handles both w:t (regular text) and m:t (math text).
    """
    # Try simple per-element replacement first
    changed_any = False
    for tag in (_w("t"), _m("t")):
        for t in para.iter(tag):
            if t.text and old in t.text:
                t.text = t.text.replace(old, new)
                changed_any = True

    if changed_any:
        return True

    # Fall back: try to find substring across split text elements
    # Collect all text-bearing elements in document order
    elems_w = list(para.iter(_w("t")))
    elems_m = list(para.iter(_m("t")))
    elems_all = sorted(elems_w + elems_m, key=lambda e: -1)  # placeholder
    # Actually we need them in tree order — use iter
    elems_all = []
    for el in para.iter():
        if el.tag in (_w("t"), _m("t")):
            elems_all.append(el)

    if not elems_all:
        return False

    # Build full text + map of position → element
    full = ""
    pos_map = []  # list of (start_pos, end_pos, element)
    for el in elems_all:
        s = el.text or ""
        pos_map.append((len(full), len(full) + len(s), el))
        full += s

    if old not in full:
        return False

    # Replace once
    idx = full.find(old)
    new_full = full[:idx] + new + full[idx + len(old):]

    # Redistribute: clear all elements, then put `new_full` into the first one
    for el in elems_all:
        el.text = ""
    elems_all[0].text = new_full
    return True

def replace_in_para(para, old, new):
    """Convenience wrapper. Returns True if changed."""
    return replace_text_in_para_anywhere(para, old, new)

# ── Phase 1: Renumber formula labels in display formulas ──────────────
print("\n--- Phase 1: Renumber formula labels (5.x → 4.y) ---")
n_renumbered = 0
formula_indices_modified = set()

for i, ch in enumerate(children):
    if i in SKIP_INDICES:
        continue
    full = get_text(ch) + " || " + get_math_text(ch)
    # Try each old number, biggest first
    for old in SORTED_OLD:
        new = RENUMBER[old]
        # Match (5.X) or (5.X. or "5.X)" patterns to be safe
        # We replace "(5.X)" → "(4.Y)" globally in this paragraph
        if f"({old})" in full:
            if replace_in_para(ch, f"({old})", f"({new})"):
                n_renumbered += 1
                formula_indices_modified.add(i)
                print(f"  B{i}: ({old}) → ({new})")
                # Re-read full text to handle multiple matches in same para
                full = get_text(ch) + " || " + get_math_text(ch)
print(f"Renumbered {n_renumbered} formula labels")

# ── Phase 2: Update text references "формула 5.x" → "формула 4.y" ─────
print("\n--- Phase 2: Update text references ---")
n_text_refs = 0

# Patterns: "формула 5.X", "формули 5.X", "формулою 5.X", "формул 5.X",
#           "форм. 5.X", "форм.5.X"
# Note: do NOT match "таблиця 5.X" — the SKIP_INDICES filter handles those.

for i, ch in enumerate(children):
    if i in SKIP_INDICES:
        continue
    if ch.tag != _w("p"):
        continue
    # Skip pure formula paragraphs (already handled in Phase 1)
    if i in formula_indices_modified:
        continue

    full_text = get_text(ch)
    if not full_text:
        continue

    changes_made = []
    # IMPORTANT: iterate in DESCENDING order of formula number to avoid
    # substring conflicts (5.1 is a substring of 5.10, 5.11, ..., 5.27)
    for old in SORTED_OLD:
        new = RENUMBER[old]
        # Various patterns
        for pattern in [
            f"формула {old}", f"формули {old}", f"формулою {old}",
            f"формул {old}", f"форм. {old}", f"форм.{old}",
            f"Формула {old}", f"Формулою {old}",
        ]:
            new_pattern = pattern.replace(old, new)
            # Use lookahead-style check: pattern must NOT be followed by a digit
            # We do this by checking each occurrence
            idx = full_text.find(pattern)
            while idx >= 0:
                next_char = full_text[idx + len(pattern)] if idx + len(pattern) < len(full_text) else ""
                if next_char.isdigit():
                    # This is actually a longer number (e.g. 5.1 inside 5.10) — skip
                    idx = full_text.find(pattern, idx + 1)
                    continue
                # Real match — replace
                if replace_in_para(ch, pattern, new_pattern):
                    changes_made.append((pattern, new_pattern))
                    full_text = get_text(ch)
                break  # break inner while; outer loop continues

    if changes_made:
        n_text_refs += 1
        snippet = ", ".join(f"'{a}' → '{b}'" for a, b in changes_made)
        print(f"  B{i}: {snippet}")

print(f"Updated {n_text_refs} text reference paragraphs")

# ── Phase 3: Add numbers to IR-metric formulas ────────────────────────
print("\n--- Phase 3: Add numbers to IR-metric formulas ---")

def add_number_to_omath_para(para, formula_num):
    """Append '(4.X)' label inside oMathPara."""
    omath_para = para.find(_m("oMathPara"))
    if omath_para is None:
        return False
    # Last oMath in oMathPara
    omaths = omath_para.findall(_m("oMath"))
    if not omaths:
        return False
    last_omath = omaths[-1]
    # Append run with " ... (4.X)"
    r = ET.SubElement(last_omath, _m("r"))
    rPr = ET.SubElement(r, _m("rPr"))
    sty = ET.SubElement(rPr, _m("sty"))
    sty.set(_m("val"), "p")  # plain (non-italic) for the label
    wrPr = ET.SubElement(r, _w("rPr"))
    rFonts = ET.SubElement(wrPr, _w("rFonts"))
    rFonts.set(_w("ascii"), "Cambria Math")
    rFonts.set(_w("hAnsi"), "Cambria Math")
    t = ET.SubElement(r, _m("t"))
    t.text = f"     ({formula_num})"
    t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    return True

n_numbered = 0
for body_idx, num in IR_FORMULAS.items():
    if body_idx >= len(children):
        print(f"  WARNING: B{body_idx} out of range")
        continue
    ch = children[body_idx]
    math_text = get_math_text(ch)
    if not math_text:
        print(f"  WARNING: B{body_idx} has no math content (text='{get_text(ch)[:40]}')")
        continue
    # Verify this looks like an IR metric formula
    expected_keywords = {
        "4.6": "DCG@k",
        "4.7": "nDCG@k",
        "4.8": "MRR",
        "4.9": "Recall@k",
        "4.10": "P@k",
    }
    keyword = expected_keywords[num]
    if keyword not in math_text:
        print(f"  WARNING: B{body_idx} math '{math_text[:40]}' does not contain '{keyword}'")
        continue
    if add_number_to_omath_para(ch, num):
        n_numbered += 1
        print(f"  B{body_idx}: added ({num}) to '{math_text[:50]}'")
print(f"Added numbers to {n_numbered} IR-metric formulas")

# ── Phase 4: Add text references for IR-metric formulas ────────────────
print("\n--- Phase 4: Add text references for IR-metric formulas ---")

ir_ref_updates = [
    # (search_text_for_paragraph, old_substring, new_substring)
    # B369 (ranking quality) — references DCG (4.6) and nDCG (4.7)
    ("Якість ранжування результатів пошуку характеризує",
     "доцільно використовувати метрику nDCG@k, оскільки вона враховує",
     "доцільно використовувати метрику nDCG@k, що визначається за формулами (4.6) і (4.7), оскільки вона враховує"),
    # B380 (speed → MRR 4.8)
    ("Швидкість знаходження релевантного результату відображає",
     "доцільно використовувати метрику MRR@k, яка визначає",
     "доцільно використовувати метрику MRR@k (формула (4.8)), яка визначає"),
    # B390 (recall 4.9)
    ("Повнота пошуку показує, яку частину релевантних",
     "доцільно використовувати метрику Recall@k, яка характеризує",
     "доцільно використовувати метрику Recall@k (формула (4.9)), яка характеризує"),
    # B400 (precision 4.10)
    ("Точність пошукової видачі характеризує частку релевантних",
     "використовується метрика Precision@k, яка дозволяє",
     "використовується метрика Precision@k (формула (4.10)), яка дозволяє"),
]

n_ir_refs = 0
for search_text, old_sub, new_sub in ir_ref_updates:
    found = False
    for i, ch in enumerate(children):
        if ch.tag != _w("p"):
            continue
        t = get_text(ch)
        if search_text in t:
            if old_sub in t:
                if replace_in_para(ch, old_sub, new_sub):
                    print(f"  B{i}: added IR formula reference")
                    n_ir_refs += 1
                    found = True
                    break
    if not found:
        print(f"  WARNING: could not find '{search_text[:40]}' / '{old_sub[:40]}'")
print(f"Added {n_ir_refs} IR-metric text references")

# ── Save ──────────────────────────────────────────────────────────────
ET.register_namespace("w", W)
ET.register_namespace("m", M)
ET.register_namespace("r", "http://schemas.openxmlformats.org/officeDocument/2006/relationships")
ET.register_namespace("mc", "http://schemas.openxmlformats.org/markup-compatibility/2006")
ET.register_namespace("v", "urn:schemas-microsoft-com:vml")
ET.register_namespace("wp", "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing")
ET.register_namespace("w14", "http://schemas.microsoft.com/office/word/2010/wordml")
ET.register_namespace("w15", "http://schemas.microsoft.com/office/word/2012/wordml")
ET.register_namespace("wne", "http://schemas.microsoft.com/office/word/2006/wordml")
ET.register_namespace("wps", "http://schemas.microsoft.com/office/word/2010/wordprocessingShape")
ET.register_namespace("o", "urn:schemas-microsoft-com:office:office")
ET.register_namespace("w10", "urn:schemas-microsoft-com:office:word")
ET.register_namespace("wp14", "http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing")
ET.register_namespace("wpc", "http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas")
ET.register_namespace("wpg", "http://schemas.microsoft.com/office/word/2010/wordprocessingGroup")
ET.register_namespace("wpi", "http://schemas.microsoft.com/office/word/2010/wordprocessingInk")

xml_path = UNPACKED / "word" / "document.xml"
tree.write(str(xml_path), xml_declaration=True, encoding="UTF-8")
print("\nSaved document.xml")

with zipfile.ZipFile(DOCX, "w", zipfile.ZIP_DEFLATED) as zout:
    for fpath in sorted(UNPACKED.rglob("*")):
        if fpath.is_file():
            zout.write(fpath, fpath.relative_to(UNPACKED))
print(f"Repacked: {DOCX.name} ({DOCX.stat().st_size:,} bytes)")
print("DONE.")

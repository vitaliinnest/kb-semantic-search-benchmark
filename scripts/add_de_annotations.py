"""
Add 'де' annotations for the 5 IR-metric formulas (DCG, nDCG, MRR, Recall, P@k)
that were added without parameter explanations.

Pattern: insert paragraphs after each formula explaining its variables.
Each parameter on a new line, ending with semicolon (period for last line).
"""
import zipfile, xml.etree.ElementTree as ET, sys, copy, pathlib, shutil
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

print("Loading document...")
tree = ET.parse(UNPACKED / "word" / "document.xml")
root = tree.getroot()
body = root.find(_w("body"))
children = list(body)
print(f"Total body children: {len(children)}")

# ── Helper: create paragraph cloned from a reference paragraph ────────
def make_para_like(reference_para, text):
    new_p = ET.Element(_w("p"))
    pPr = reference_para.find(_w("pPr"))
    if pPr is not None:
        new_p.append(copy.deepcopy(pPr))
    first_r = reference_para.find(_w("r"))
    rPr = first_r.find(_w("rPr")) if first_r is not None else None
    r = ET.SubElement(new_p, _w("r"))
    if rPr is not None:
        r.append(copy.deepcopy(rPr))
    t = ET.SubElement(r, _w("t"))
    if text != text.strip():
        t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    t.text = text
    return new_p

def make_blank_para(reference_para):
    new_p = ET.Element(_w("p"))
    pPr = reference_para.find(_w("pPr"))
    if pPr is not None:
        new_p.append(copy.deepcopy(pPr))
    return new_p

# ── Find a reference paragraph for body style (use B312 — existing де annotation) ──
# B312 is "де E_q — векторне подання запиту..." — matches our needed style
ref_para = children[312]
ref_text = get_text(ref_para)
print(f"Using B312 as style ref: '{ref_text[:50]}...'")

# ── Find formulas by content rather than fixed indices ────────────────
def find_formula_by_content(keyword):
    for i, ch in enumerate(children):
        if i < 300 or i > 410:  # IR formulas are in this range
            continue
        mt = get_math_text(ch)
        if keyword in mt:
            return i
    return -1

formulas_to_annotate = [
    # (formula_keyword, paragraphs_to_insert_after_formula)
    # DCG@k formula
    ("DCG@k = i = 1k", [
        "де relᵢ — оцінка релевантності i-го документа в ранжуванні;",
        "      k — глибина пошуку (кількість результатів top-k);",
        "      log₂(i + 1) — логарифмічний фактор зменшення ваги, який знижує внесок документів на нижчих позиціях."
    ], "DCG"),
    # nDCG@k formula (right after DCG)
    ("nDCG@k = DCG@kIDCG@k", [
        "де DCG@k — дисконтований кумулятивний приріст, обчислюється за формулою (4.6);",
        "      IDCG@k — ідеальний DCG@k, який отримує запит при оптимальному впорядкуванні релевантних документів;",
        "      nDCG@k — нормалізований DCG@k у діапазоні [0, 1]."
    ], "nDCG"),
    # MRR formula
    ("MRR = 1|Q|", [
        "де |Q| — кількість запитів у benchmark-наборі;",
        "      rank_q — позиція першого релевантного документа у пошуковій видачі для запиту q;",
        "      MRR — середнє значення оберненого рангу першого релевантного результату."
    ], "MRR"),
    # Recall@k formula
    ("Recall@k = |Relret|", [
        "де |Rel_ret| — кількість релевантних документів серед перших k результатів пошуку;",
        "      |Rel| — загальна кількість релевантних документів для даного запиту;",
        "      Recall@k — частка релевантних документів, знайдених у видачі."
    ], "Recall"),
    # P@k formula
    ("P@k = |Relk|", [
        "де |Rel_k| — кількість релевантних документів у перших k позиціях пошукової видачі;",
        "      k — глибина пошуку;",
        "      P@k — частка релевантних результатів серед перших k."
    ], "P@k"),
]

# ── Process each formula: find it, then insert annotation paragraphs AFTER ──
# IMPORTANT: process in REVERSE body order so insertions don't shift indices
# we still need to work with

# First, find all positions
formula_positions = []
for keyword, annotation_lines, label in formulas_to_annotate:
    body_idx = find_formula_by_content(keyword)
    if body_idx < 0:
        print(f"  WARNING: formula '{keyword}' not found")
        continue
    formula_positions.append((body_idx, annotation_lines, label, children[body_idx]))
    print(f"  Found {label} at B{body_idx}")

# Sort by body_idx descending so later insertions don't affect earlier indices
formula_positions.sort(key=lambda x: -x[0])

# Insert annotations
for body_idx, annotation_lines, label, formula_el in formula_positions:
    # Find current position in body (it may have shifted due to earlier insertions,
    # but we sorted descending so this should be stable)
    current_pos = list(body).index(formula_el)

    # Build new paragraphs in order
    new_paras = []
    # Don't add a leading blank — formula already typically followed by blank
    # Add one blank para after formula (if the next isn't already blank)
    next_para = list(body)[current_pos + 1] if current_pos + 1 < len(list(body)) else None
    next_text = get_text(next_para) if next_para is not None else "non-empty"
    if next_text.strip():
        new_paras.append(make_blank_para(ref_para))
    # Add 'де' annotation paragraphs
    for line in annotation_lines:
        new_paras.append(make_para_like(ref_para, line))
    # Add trailing blank
    new_paras.append(make_blank_para(ref_para))

    # Insert all new paragraphs after the formula
    insert_pos = current_pos + 1
    for j, p in enumerate(new_paras):
        body.insert(insert_pos + j, p)

    print(f"  B{body_idx} ({label}): inserted {len(new_paras)} paragraphs after formula")

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
print(f"\nSaved document.xml. New body children count: {len(list(body))}")

with zipfile.ZipFile(DOCX, "w", zipfile.ZIP_DEFLATED) as zout:
    for fpath in sorted(UNPACKED.rglob("*")):
        if fpath.is_file():
            zout.write(fpath, fpath.relative_to(UNPACKED))
print(f"Repacked: {DOCX.name} ({DOCX.stat().st_size:,} bytes)")
print("DONE.")

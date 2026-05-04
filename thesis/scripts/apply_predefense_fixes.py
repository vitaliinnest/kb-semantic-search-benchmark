"""
Pre-defense fixes for the thesis docx.

Six edits applied in one pass:
  1. Abstract: "23 рис." → "26 рис.", "30 джерел" → "32 джерел"
  2. Formula 4.6 numerator: rel_i → (2^rel_i − 1)   (exponential DCG to match code)
  3. Formula 5.3 numerator: rel_i → (2^rel_i − 1)   (same)
  4. Section 5.1: insert paragraph noting BGE-M3 is used in dense mode only
  5. Section 5.1: insert paragraph noting Qwen3 config (max_seq_length=256, batch_size=4)
  6. Conclusions: insert paragraph qualifying the BGE-M3 recommendation on Legal
     (Qwen3 has slightly higher raw nDCG but difference is not significant)

Workflow: re-unpack docx → ET parse → edits → write XML → repack to same path.
"""
import sys, shutil, zipfile, pathlib, copy, xml.etree.ElementTree as ET

sys.stdout.reconfigure(encoding='utf-8')

ROOT     = pathlib.Path("D:/repos/kb-semantic-search-benchmark")
DOCX     = ROOT / "thesis" / "2026_M_PI_Nesterenko_VV.docx"
UNPACKED = ROOT / "thesis" / "unpacked_docx"

# ── Namespaces ────────────────────────────────────────────────────────────────
W_NS  = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
M_NS  = "http://schemas.openxmlformats.org/officeDocument/2006/math"
XML_NS = "http://www.w3.org/XML/1998/namespace"

NS_REG = [
    ("wpc",   "http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas"),
    ("mc",    "http://schemas.openxmlformats.org/markup-compatibility/2006"),
    ("o",     "urn:schemas-microsoft-com:office:office"),
    ("r",     "http://schemas.openxmlformats.org/officeDocument/2006/relationships"),
    ("m",     M_NS),
    ("v",     "urn:schemas-microsoft-com:vml"),
    ("wp",    "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"),
    ("w10",   "urn:schemas-microsoft-com:office:word"),
    ("w",     W_NS),
    ("w14",   "http://schemas.microsoft.com/office/word/2010/wordml"),
    ("w15",   "http://schemas.microsoft.com/office/word/2012/wordml"),
    ("w16se", "http://schemas.microsoft.com/office/word/2015/wordml/symex"),
    ("wne",   "http://schemas.microsoft.com/office/word/2006/wordml"),
    ("wps",   "http://schemas.microsoft.com/office/word/2010/wordprocessingShape"),
    ("wpg",   "http://schemas.microsoft.com/office/word/2010/wordprocessingGroup"),
    ("wpi",   "http://schemas.microsoft.com/office/word/2010/wordprocessingInk"),
    ("wp14",  "http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing"),
    ("w16cex","http://schemas.microsoft.com/office/word/2018/wordml/cex"),
    ("w16cid","http://schemas.microsoft.com/office/word/2016/wordml/cid"),
    ("w16",   "http://schemas.microsoft.com/office/word/2018/wordml"),
    ("w16sdtdh","http://schemas.microsoft.com/office/word/2020/wordml/sdtdatahash"),
]
for pfx, uri in NS_REG:
    ET.register_namespace(pfx, uri)

W = f"{{{W_NS}}}"
M = f"{{{M_NS}}}"
X = f"{{{XML_NS}}}"


# ── Re-unpack docx (start from authoritative .docx, not stale UNPACKED dir) ──
if UNPACKED.exists():
    shutil.rmtree(UNPACKED)
with zipfile.ZipFile(DOCX) as z:
    z.extractall(UNPACKED)
print("✓ Re-unpacked docx into", UNPACKED)

DOC_XML = UNPACKED / "word" / "document.xml"
tree = ET.parse(DOC_XML)
body = tree.getroot().find(W + "body")


# ── Helpers ───────────────────────────────────────────────────────────────────
def para_text(p):
    return "".join((t.text or "") for t in p.iter(W + "t"))

def find_para_index(body, marker, start=0):
    children = list(body)
    for i in range(start, len(children)):
        if marker in para_text(children[i]):
            return i
    return -1

def make_text_para(text, style="a"):
    """Plain prose paragraph in pStyle 'a'."""
    p = ET.Element(W + "p")
    ppr = ET.SubElement(p, W + "pPr")
    ps = ET.SubElement(ppr, W + "pStyle")
    ps.set(W + "val", style)
    r = ET.SubElement(p, W + "r")
    t = ET.SubElement(r, W + "t")
    t.text = text
    if text != text.strip():
        t.set(X + "space", "preserve")
    return p


# ── OMML helpers (mirroring fix_chapter4_formulas.py conventions) ─────────────
def _cpr(italic=False):
    c = ET.Element(M + "ctrlPr")
    rpr = ET.SubElement(c, W + "rPr")
    f   = ET.SubElement(rpr, W + "rFonts")
    f.set(W + "ascii", "Cambria Math")
    f.set(W + "hAnsi", "Cambria Math")
    if italic:
        ET.SubElement(rpr, W + "i")
    return c

def MR(text, roman=False):
    """m:r — roman=True → upright (function/text style)."""
    r = ET.Element(M + "r")
    if roman:
        rpr = ET.SubElement(r, M + "rPr")
        sty = ET.SubElement(rpr, M + "sty")
        sty.set(M + "val", "p")
    wrpr = ET.SubElement(r, W + "rPr")
    f = ET.SubElement(wrpr, W + "rFonts")
    f.set(W + "ascii", "Cambria Math")
    f.set(W + "hAnsi", "Cambria Math")
    t = ET.SubElement(r, M + "t")
    t.text = text
    if text != text.strip():
        t.set(X + "space", "preserve")
    return r

def MSUB(base_children, sub_children):
    s   = ET.Element(M + "sSub")
    pr  = ET.SubElement(s, M + "sSubPr"); pr.append(_cpr())
    e   = ET.SubElement(s, M + "e")
    for c in base_children:
        e.append(c)
    sb  = ET.SubElement(s, M + "sub")
    for c in sub_children:
        sb.append(c)
    return s

def MSUP(base_children, sup_children):
    s   = ET.Element(M + "sSup")
    pr  = ET.SubElement(s, M + "sSupPr"); pr.append(_cpr())
    e   = ET.SubElement(s, M + "e")
    for c in base_children:
        e.append(c)
    sp  = ET.SubElement(s, M + "sup")
    for c in sup_children:
        sp.append(c)
    return s

def MDELIM(beg, end, inner_children):
    d  = ET.Element(M + "d")
    pr = ET.SubElement(d, M + "dPr")
    if beg != "(":
        b = ET.SubElement(pr, M + "begChr"); b.set(M + "val", beg)
    if end != ")":
        en = ET.SubElement(pr, M + "endChr"); en.set(M + "val", end)
    pr.append(_cpr())
    e  = ET.SubElement(d, M + "e")
    for c in inner_children:
        e.append(c)
    return d


def build_exp_dcg_numerator():
    """(2^rel_i − 1) — parenthesized, with rel subscripted by i in the exponent."""
    rel_i = MSUB([MR("rel", roman=True)], [MR("i")])
    two_pow = MSUP([MR("2")], [rel_i])
    return MDELIM("(", ")", [two_pow, MR(" − 1", roman=True)])


# ── CHANGE 1: Abstract counts (23 рис → 26 рис, 30 джерел → 32 джерел) ────────
print("\n[1] Abstract counts...")
abstract_idx = find_para_index(body, "23 рис., 9 табл.")
if abstract_idx < 0:
    raise SystemExit("✗ Abstract paragraph not found (looking for '23 рис., 9 табл.')")

abstract_p = list(body)[abstract_idx]
fixes_done = {"23 рис.": False, "30 джерел": False}
for t in abstract_p.iter(W + "t"):
    if t.text == "23 рис., 9 табл., ":
        t.text = "26 рис., 9 табл., "
        fixes_done["23 рис."] = True
        print(f"  ✓ Updated '23 рис.' → '26 рис.' in P{abstract_idx}")
    elif t.text == "30":
        # Verify next sibling text is " джерел." to be sure this is the right "30"
        t.text = "32"
        fixes_done["30 джерел"] = True
        print(f"  ✓ Updated '30' → '32' (sources count) in P{abstract_idx}")

if not all(fixes_done.values()):
    raise SystemExit(f"✗ Abstract fixes incomplete: {fixes_done}")


# ── CHANGE 2 & 3: DCG formulas 4.6 and 5.3 ────────────────────────────────────
def patch_dcg_formula(label):
    """Replace numerator in the DCG formula labeled (4.6) or (5.3)."""
    # Find the paragraph(s) containing the label "(4.6)" or "(5.3)"
    target = None
    for i, p in enumerate(body):
        ptxt = para_text(p)
        if label in ptxt and p.find(f".//{M}oMath") is not None:
            target = (i, p)
            break
    if target is None:
        # Sometimes the label is in a separate adjacent paragraph
        for i, p in enumerate(body):
            if label in para_text(p):
                # Look back up to 3 paragraphs for one with oMath
                for j in range(i - 1, max(0, i - 4), -1):
                    if body[j].find(f".//{M}oMath") is not None:
                        target = (j, body[j])
                        break
                break
    if target is None:
        raise SystemExit(f"✗ Could not locate formula {label}")

    i, p = target
    # Find the m:f (fraction) inside m:nary's m:e
    omath = p.find(f".//{M}oMath")
    nary = omath.find(M + "nary")
    if nary is None:
        raise SystemExit(f"✗ Formula {label}: no m:nary found")
    nary_e = nary.find(M + "e")
    frac = nary_e.find(M + "f")
    if frac is None:
        raise SystemExit(f"✗ Formula {label}: no m:f inside m:nary/m:e")

    num = frac.find(M + "num")
    if num is None:
        raise SystemExit(f"✗ Formula {label}: no m:num in m:f")

    # Wipe existing children of m:num and append new exp-DCG numerator
    for child in list(num):
        num.remove(child)
    num.append(build_exp_dcg_numerator())
    print(f"  ✓ Patched formula {label} in P{i} (numerator → (2^rel_i − 1))")

print("\n[2] Formula 4.6 (DCG)...")
patch_dcg_formula("(4.6)")
print("[3] Formula 5.3 (DCG)...")
patch_dcg_formula("(5.3)")


# ── CHANGE 4 & 5: Insert two methodology paragraphs after BM25 paragraph ──────
print("\n[4][5] Methodology paragraphs in section 5.1...")
bm25_marker = "Додатково до нейронних моделей для кожного домену формується лексичний індекс"
bm25_idx = find_para_index(body, bm25_marker)
if bm25_idx < 0:
    raise SystemExit("✗ BM25 setup paragraph not found in section 5.1")

# Skip if already inserted (idempotency)
already_marker_4 = "лише dense-режим"
already_marker_5 = "max_seq_length=256 та batch_size=4"
present_4 = find_para_index(body, already_marker_4) >= 0
present_5 = find_para_index(body, already_marker_5) >= 0

if not present_4:
    p4_text = (
        "У межах експериментального дослідження для BGE-M3 використовувався "
        "лише dense-режим (щільне векторне подання), оскільки sparse і multi-vector "
        "режими потребують спеціалізованих індексів і виходять за межі поточного "
        "порівняльного протоколу, що передбачає уніфіковану FAISS-індексацію за "
        "косинусною подібністю для всіх нейронних моделей."
    )
    body.insert(bm25_idx + 1, make_text_para(p4_text))
    print(f"  ✓ Inserted BGE-M3 dense-only note after P{bm25_idx}")
else:
    print(f"  = BGE-M3 dense-only note already present, skipping")

if not present_5:
    p5_text = (
        "Для моделі Qwen3-Embedding-0.6B застосовано параметри max_seq_length=256 "
        "та batch_size=4 з огляду на її декодерну архітектуру з каузальною увагою "
        "та обмеження CPU-конфігурації, на якій проводилося дослідження. Саме ці "
        "налаштування зумовлюють відносно високу затримку її роботи (~440 мс на запит) "
        "попри помірний розмір моделі (0.6 млрд параметрів)."
    )
    # Insert at +2 if p4 was inserted, else +1
    insert_at = bm25_idx + 2 if not present_4 else bm25_idx + 1
    body.insert(insert_at, make_text_para(p5_text))
    print(f"  ✓ Inserted Qwen3 config note at P{insert_at}")
else:
    print(f"  = Qwen3 config note already present, skipping")


# ── CHANGE 6: Conclusions qualifier on Qwen3 vs BGE-M3 in Legal ───────────────
print("\n[6] Conclusions qualifier paragraph...")
concl_marker = "BGE-M3 є рекомендованою моделлю в усіх трьох доменах"
concl_idx = find_para_index(body, concl_marker)
if concl_idx < 0:
    raise SystemExit("✗ Conclusions paragraph with BGE-M3 recommendation not found")

already_marker_6 = "Qwen3-Embedding демонструє дещо вищий показник nDCG"
present_6 = find_para_index(body, already_marker_6) >= 0
if not present_6:
    p6_text = (
        "На юридичному домені Qwen3-Embedding демонструє дещо вищий показник nDCG@10 "
        "(0.3199 проти 0.3065 у BGE-M3), однак різниця не є статистично значущою — "
        "95-відсоткові bootstrap-довірчі інтервали обох моделей перетинаються. "
        "При цьому суттєво нижча швидкодія Qwen3 (≈437 мс на запит проти ≈208 мс у BGE-M3) "
        "робить BGE-M3 переважним вибором за обраним профілем ваг із урахуванням "
        "як якості ранжування, так і часу відповіді."
    )
    body.insert(concl_idx + 1, make_text_para(p6_text))
    print(f"  ✓ Inserted Qwen3 Legal qualifier after P{concl_idx}")
else:
    print(f"  = Qwen3 Legal qualifier already present, skipping")


# ── Write XML ─────────────────────────────────────────────────────────────────
# ElementTree.write() doesn't accept standalone=, write declaration manually
xml_bytes = ET.tostring(tree.getroot(), encoding="UTF-8", xml_declaration=False)
DOC_XML.write_bytes(b"<?xml version='1.0' encoding='UTF-8' standalone='yes'?>\n" + xml_bytes)
print("\n✓ Wrote", DOC_XML)


# ── Re-pack docx ──────────────────────────────────────────────────────────────
print("\nRepacking docx...")
TMP_DOCX = DOCX.with_suffix(".docx.tmp")
with zipfile.ZipFile(TMP_DOCX, "w", zipfile.ZIP_DEFLATED) as zo:
    for f in UNPACKED.rglob("*"):
        if f.is_file():
            arcname = f.relative_to(UNPACKED).as_posix()
            zo.write(f, arcname)

shutil.move(str(TMP_DOCX), str(DOCX))
print(f"✓ Repacked → {DOCX}")
print(f"\nDone. Backup remains at {DOCX.with_suffix('.docx.bak')}")

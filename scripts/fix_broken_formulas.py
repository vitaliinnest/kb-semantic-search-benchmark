"""
Replace 20 broken OMML formulas (4.11)–(4.30) in chapter 4
with LaTeX text inside proper m:oMathPara elements.

The broken formulas had all content stuffed into a single m:t element
inside a malformed m:sSub. This replaces each with a clean m:oMathPara
containing the LaTeX formula text — the user can then open each in
Word's equation editor (Linear mode) and convert.
"""
import zipfile, xml.etree.ElementTree as ET, sys, pathlib, shutil, copy
sys.stdout.reconfigure(encoding='utf-8')

ROOT = pathlib.Path("D:/repos/kb-semantic-search-benchmark")
DOCX = ROOT / "2026_M_PI_Nesterenko_VV.docx"
UNPACKED = ROOT / "unpacked_docx"

W  = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
M  = "http://schemas.openxmlformats.org/officeDocument/2006/math"
XML_NS = "http://www.w3.org/XML/1998/namespace"

def _w(tag): return f"{{{W}}}{tag}"
def _m(tag): return f"{{{M}}}{tag}"

# ── Re-unpack fresh ────────────────────────────────────────────────────
if UNPACKED.exists():
    shutil.rmtree(UNPACKED)
with zipfile.ZipFile(DOCX) as z:
    z.extractall(UNPACKED)
print("Re-unpacked docx.")

tree = ET.parse(UNPACKED / "word" / "document.xml")
root = tree.getroot()
body = root.find(_w("body"))
children = list(body)
print(f"Total body children: {len(children)}")

# ── LaTeX formula definitions ──────────────────────────────────────────
# Each entry: body_idx → (latex_string, formula_number)
# Indices determined by fresh-unpack scan on 2026-04-29.
BROKEN = {
    451: (r"A = \{a_{1}, a_{2}, a_{3}, a_{4}, a_{5}\}", "4.11"),
    457: (r"X_{i} = (x_{i1}, x_{i2}, x_{i3}, x_{i4}, x_{i5})", "4.12"),
    462: (
        r"X = \begin{pmatrix}"
        r" x_{11} & x_{12} & x_{13} & x_{14} & x_{15} \\"
        r" x_{21} & x_{22} & x_{23} & x_{24} & x_{25} \\"
        r" x_{31} & x_{32} & x_{33} & x_{34} & x_{35} \\"
        r" x_{41} & x_{42} & x_{43} & x_{44} & x_{45} \\"
        r" x_{51} & x_{52} & x_{53} & x_{54} & x_{55}"
        r" \end{pmatrix}",
        "4.13",
    ),
    475: (
        r"y_{ij} = \frac{x_{ij} - \min_{i}\,x_{ij}}{\max_{i}\,x_{ij} - \min_{i}\,x_{ij}}",
        "4.14",
    ),
    479: (
        r"y_{ij} = \frac{\max_{i}\,x_{ij} - x_{ij}}{\max_{i}\,x_{ij} - \min_{i}\,x_{ij}}",
        "4.15",
    ),
    491: (r"Y_{i} = (y_{i1}, y_{i2}, y_{i3}, y_{i4}, y_{i5})", "4.16"),
    498: (
        r"Y = \begin{pmatrix}"
        r" y_{11} & y_{12} & y_{13} & y_{14} & y_{15} \\"
        r" y_{21} & y_{22} & y_{23} & y_{24} & y_{25} \\"
        r" y_{31} & y_{32} & y_{33} & y_{34} & y_{35} \\"
        r" y_{41} & y_{42} & y_{43} & y_{44} & y_{45} \\"
        r" y_{51} & y_{52} & y_{53} & y_{54} & y_{55}"
        r" \end{pmatrix}",
        "4.17",
    ),
    508: (
        r"Y_{p} = (y_{p1}, y_{p2}, y_{p3}, y_{p4}, y_{p5}),\quad"
        r"Y_{q} = (y_{q1}, y_{q2}, y_{q3}, y_{q4}, y_{q5})",
        "4.18",
    ),
    516: (r"y_{pj} \geq y_{qj},\;\forall j = 1,\ldots,5", "4.19"),
    523: (r"y_{pj} > y_{qj}", "4.20"),
    530: (r"P = \{a_{i} \in A \mid \nexists\,a_{k} \in A : a_{k} \succ a_{i}\}", "4.21"),
    548: (r"U(a_{i}) = \sum_{j=1}^{m} w_{j} \cdot y_{ij}", "4.22"),
    553: (
        r"U(a_{i}) = w_{1}y_{i1} + w_{2}y_{i2} + w_{3}y_{i3} + w_{4}y_{i4} + w_{5}y_{i5}",
        "4.23",
    ),
    566: (r"K_{1} \succ K_{2} \succ K_{3} \succ K_{4} \succ K_{5}", "4.24"),
    570: (
        r"w_{j} = \frac{m - r_{j} + 1}{\displaystyle\sum_{k=1}^{m}(m - r_{k} + 1)}",
        "4.25",
    ),
    595: (r"U(a_{i}) = \sum_{j=1}^{5} w_{j} \cdot y_{ij}", "4.26"),
    600: (
        r"U(a_{i}) = 0{,}33\,y_{i1} + 0{,}27\,y_{i2} + 0{,}20\,y_{i3}"
        r" + 0{,}13\,y_{i4} + 0{,}07\,y_{i5}",
        "4.27",
    ),
    605: (r"a^{*} = \arg\max_{a_{i} \in P}\,U(a_{i})", "4.28"),
    616: (
        r"y_{ij} = \frac{x_{ij} - \min_{i}\,x_{ij}}{\max_{i}\,x_{ij} - \min_{i}\,x_{ij}}",
        "4.29",
    ),
    623: (
        r"y_{i5} = \frac{\max_{i}\,x_{i5} - x_{i5}}{\max_{i}\,x_{i5} - \min_{i}\,x_{i5}}",
        "4.30",
    ),
}

# ── Helper: build replacement paragraph ───────────────────────────────
def make_latex_formula_para(old_para, latex, num):
    """
    Return a new w:p that contains:
      - the same w:pPr as old_para (keeps style & right-justification)
      - m:oMathPara (centered) with m:oMath containing LaTeX text + formula number
    """
    p = ET.Element(_w("p"))

    # 1. Copy paragraph properties
    pPr_src = old_para.find(_w("pPr"))
    if pPr_src is not None:
        p.append(copy.deepcopy(pPr_src))
    else:
        pPr = ET.SubElement(p, _w("pPr"))
        pStyle = ET.SubElement(pPr, _w("pStyle"))
        pStyle.set(_w("val"), "a")
        jc = ET.SubElement(pPr, _w("jc"))
        jc.set(_w("val"), "right")

    # 2. Build m:oMathPara
    oMathPara = ET.SubElement(p, _m("oMathPara"))

    oMathParaPr = ET.SubElement(oMathPara, _m("oMathParaPr"))
    jc = ET.SubElement(oMathParaPr, _m("jc"))
    jc.set(_m("val"), "center")

    # 3. Build m:oMath
    oMath = ET.SubElement(oMathPara, _m("oMath"))

    # 4. Single run with LaTeX text + formula number
    r_el = ET.SubElement(oMath, _m("r"))
    wrPr = ET.SubElement(r_el, _w("rPr"))
    fonts = ET.SubElement(wrPr, _w("rFonts"))
    fonts.set(_w("ascii"), "Cambria Math")
    fonts.set(_w("hAnsi"), "Cambria Math")
    t_el = ET.SubElement(r_el, _m("t"))
    t_el.set(f"{{{XML_NS}}}space", "preserve")
    t_el.text = f"{latex}     ({num})"

    return p


# ── Replace broken formulas (descending order → stable indices) ────────
for idx in sorted(BROKEN.keys(), reverse=True):
    latex, num = BROKEN[idx]
    old_el = list(body)[idx]

    # Verify the element looks like a broken formula
    mt_texts = [t.text or "" for t in old_el.iter(_m("t"))]
    combined = "".join(mt_texts)
    if f"({num})" not in combined and f".{num.split('.')[1]}" not in combined:
        print(f"  WARNING B{idx}: expected ({num}) not found in math text: {combined[:80]}")

    new_el = make_latex_formula_para(old_el, latex, num)

    pos = list(body).index(old_el)
    body.remove(old_el)
    body.insert(pos, new_el)
    print(f"  Replaced B{idx} ({num}): {latex[:60]}...")

# ── Register namespaces & save ─────────────────────────────────────────
ET.register_namespace("w",   W)
ET.register_namespace("m",   M)
ET.register_namespace("r",   "http://schemas.openxmlformats.org/officeDocument/2006/relationships")
ET.register_namespace("mc",  "http://schemas.openxmlformats.org/markup-compatibility/2006")
ET.register_namespace("v",   "urn:schemas-microsoft-com:vml")
ET.register_namespace("wp",  "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing")
ET.register_namespace("w14", "http://schemas.microsoft.com/office/word/2010/wordml")
ET.register_namespace("w15", "http://schemas.microsoft.com/office/word/2012/wordml")
ET.register_namespace("wne", "http://schemas.microsoft.com/office/word/2006/wordml")
ET.register_namespace("wps", "http://schemas.microsoft.com/office/word/2010/wordprocessingShape")
ET.register_namespace("o",   "urn:schemas-microsoft-com:office:office")
ET.register_namespace("w10", "urn:schemas-microsoft-com:office:word")
ET.register_namespace("wp14","http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing")
ET.register_namespace("wpc", "http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas")
ET.register_namespace("wpg", "http://schemas.microsoft.com/office/word/2010/wordprocessingGroup")
ET.register_namespace("wpi", "http://schemas.microsoft.com/office/word/2010/wordprocessingInk")

xml_path = UNPACKED / "word" / "document.xml"
tree.write(str(xml_path), xml_declaration=True, encoding="UTF-8")
print(f"\nSaved document.xml. Body children count: {len(list(body))}")

with zipfile.ZipFile(DOCX, "w", zipfile.ZIP_DEFLATED) as zout:
    for fpath in sorted(UNPACKED.rglob("*")):
        if fpath.is_file():
            zout.write(fpath, fpath.relative_to(UNPACKED))
print(f"Repacked: {DOCX.name} ({DOCX.stat().st_size:,} bytes)")
print("DONE.")

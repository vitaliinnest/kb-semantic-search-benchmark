"""
Fix formula number placement for (4.1)–(4.10).

These use m:oMathPara with the number embedded inside m:oMath as the
last run (m:sty p). Word renders the number visually inside the equation
block, not at the right edge of the page line.

Fix: same tab-stop layout used for (4.11)–(4.30):
  [tab→center 4960]  [m:oMath formula]  [tab→right 9921]  (4.x)
The OMML formula content is preserved exactly; only the number run is
removed from m:oMath and placed as plain text at the right tab stop.
"""
import zipfile, xml.etree.ElementTree as ET, sys, pathlib, shutil, copy, re
sys.stdout.reconfigure(encoding='utf-8')

ROOT     = pathlib.Path("D:/repos/kb-semantic-search-benchmark")
DOCX     = ROOT / "2026_M_PI_Nesterenko_VV.docx"
UNPACKED = ROOT / "unpacked_docx"

W      = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
M      = "http://schemas.openxmlformats.org/officeDocument/2006/math"
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

# ── Page geometry ──────────────────────────────────────────────────────
# A4: page_w=11906, left=1418, right=567 → text_w=9921, center=4960
CENTER_TAB = 4960
RIGHT_TAB  = 9921

NUM_RE = re.compile(r'^\s+\(\d+\.\d+\)\s*$')


def fix_formula_para(old_para):
    """
    Given a w:p containing m:oMathPara, return a new w:p with:
      [tab→center] [m:oMath formula without number] [tab→right] (n.x)
    Returns None if the paragraph doesn't match the expected pattern.
    """
    omp = old_para.find(_m("oMathPara"))
    if omp is None:
        return None
    om = omp.find(_m("oMath"))
    if om is None:
        return None

    # Find and remove the last m:r (the number run with m:sty p)
    all_r = list(om.findall(_m("r")))
    if not all_r:
        return None
    last_r = all_r[-1]
    rPr = last_r.find(_m("rPr"))
    sty = rPr.find(_m("sty")) if rPr is not None else None
    if sty is None or sty.get(_m("val")) != "p":
        return None
    t_el = last_r.find(_m("t"))
    num_text = t_el.text if t_el is not None else ""
    if not NUM_RE.match(num_text):
        return None

    num_label = num_text.strip()   # e.g. "(4.1)"

    # Deep-copy the m:oMath, then remove the number run from the copy
    om_copy = copy.deepcopy(om)
    # Remove the LAST m:r child (the number run)
    all_r_copy = list(om_copy.findall(_m("r")))
    om_copy.remove(all_r_copy[-1])

    # ── Build new paragraph ────────────────────────────────────────────
    p = ET.Element(_w("p"))

    # Paragraph properties: style "a", left-align, no first-line indent,
    # center tab at 4960, right tab at 9921
    pPr = ET.SubElement(p, _w("pPr"))

    pStyle = ET.SubElement(pPr, _w("pStyle"))
    pStyle.set(_w("val"), "a")

    jc = ET.SubElement(pPr, _w("jc"))
    jc.set(_w("val"), "left")

    ind = ET.SubElement(pPr, _w("ind"))
    ind.set(_w("firstLine"), "0")

    tabs = ET.SubElement(pPr, _w("tabs"))
    tab_c = ET.SubElement(tabs, _w("tab"))
    tab_c.set(_w("val"), "center")
    tab_c.set(_w("pos"), str(CENTER_TAB))
    tab_r = ET.SubElement(tabs, _w("tab"))
    tab_r.set(_w("val"), "right")
    tab_r.set(_w("pos"), str(RIGHT_TAB))

    # Tab → center position
    r_tab1 = ET.SubElement(p, _w("r"))
    ET.SubElement(r_tab1, _w("tab"))

    # Inline m:oMath with the formula content (number run already removed)
    p.append(om_copy)

    # Tab → right edge
    r_tab2 = ET.SubElement(p, _w("r"))
    ET.SubElement(r_tab2, _w("tab"))

    # Formula number as plain text at the right margin
    r_num = ET.SubElement(p, _w("r"))
    t_num = ET.SubElement(r_num, _w("t"))
    t_num.text = num_label

    return p, num_label


# ── Scan and fix all matching paragraphs ──────────────────────────────
fixed = 0
for i in range(len(children) - 1, -1, -1):   # reverse → stable indices
    ch = children[i]
    result = fix_formula_para(ch)
    if result is None:
        continue
    new_para, num_label = result
    pos = list(body).index(ch)
    body.remove(ch)
    body.insert(pos, new_para)
    print(f"  Fixed B{i} {num_label}")
    fixed += 1

print(f"\nTotal fixed: {fixed}")

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
print(f"Saved document.xml. Body children: {len(list(body))}")

with zipfile.ZipFile(DOCX, "w", zipfile.ZIP_DEFLATED) as zout:
    for fpath in sorted(UNPACKED.rglob("*")):
        if fpath.is_file():
            zout.write(fpath, fpath.relative_to(UNPACKED))
print(f"Repacked: {DOCX.name} ({DOCX.stat().st_size:,} bytes)")
print("DONE.")

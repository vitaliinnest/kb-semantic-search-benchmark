"""
Виправляє форматування формул у практичному розділі 5:
  B718 BM25     → (5.1) з tab-stop layout
  B719 IDF      → (5.2) з tab-stop layout
  B722 DCG@k    → (5.3) з tab-stop layout
  B723 nDCG@k   → (5.4) з tab-stop layout
  B724 MRR      → (5.5) з tab-stop layout

Поточна структура: m:oMathPara (center), без номера.
Цільова структура: [w:r tab(center)] [m:oMath] [w:r tab(right)] [w:r (5.x)]
  з tabs (center=4960, right=9921) і w:ind firstLine=0, w:jc left
  — ідентично виправленим формулам розділу 4.
"""
import sys, shutil, zipfile, pathlib, xml.etree.ElementTree as ET
sys.stdout.reconfigure(encoding='utf-8')

ROOT    = pathlib.Path("D:/repos/kb-semantic-search-benchmark")
DOCX    = ROOT / "thesis" / "2026_M_PI_Nesterenko_VV.docx"
UNPACKED = ROOT / "thesis" / "unpacked_docx"

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
M = "http://schemas.openxmlformats.org/officeDocument/2006/math"
def _w(t): return f"{{{W}}}{t}"
def _m(t): return f"{{{M}}}{t}"

# ── Re-unpack ──────────────────────────────────────────────────────────
if UNPACKED.exists():
    shutil.rmtree(UNPACKED)
with zipfile.ZipFile(DOCX) as z:
    z.extractall(UNPACKED)
print("Re-unpacked docx.")

DOC_XML = UNPACKED / "word" / "document.xml"
tree = ET.parse(DOC_XML)
body = tree.getroot().find(_w("body"))
children = list(body)

# Formulas to fix: (body_index, formula_number)
FORMULAS = [
    (718, "5.1"),
    (719, "5.2"),
    (722, "5.3"),
    (723, "5.4"),
    (724, "5.5"),
]

for idx, num in FORMULAS:
    p = children[idx]

    # 1. Extract m:oMath from inside m:oMathPara
    omath_para = p.find(_m("oMathPara"))
    if omath_para is None:
        print(f"[!] B{idx}: no oMathPara found, skipping")
        continue
    omath = omath_para.find(_m("oMath"))
    if omath is None:
        print(f"[!] B{idx}: no oMath inside oMathPara, skipping")
        continue

    # 2. Remove oMathPara from paragraph
    p.remove(omath_para)

    # 3. Update w:pPr: remove jc, add tabs + ind firstLine=0 + jc left
    ppr = p.find(_w("pPr"))
    if ppr is None:
        ppr = ET.SubElement(p, _w("pPr"))

    # Remove existing jc
    for jc in ppr.findall(_w("jc")):
        ppr.remove(jc)

    # Remove existing tabs (if any)
    for tabs in ppr.findall(_w("tabs")):
        ppr.remove(tabs)

    # Remove existing ind (if any)
    for ind in ppr.findall(_w("ind")):
        ppr.remove(ind)

    # Build new tabs element
    tabs_el = ET.SubElement(ppr, _w("tabs"))
    tab_center = ET.SubElement(tabs_el, _w("tab"))
    tab_center.set(_w("val"), "center")
    tab_center.set(_w("pos"), "4960")
    tab_right = ET.SubElement(tabs_el, _w("tab"))
    tab_right.set(_w("val"), "right")
    tab_right.set(_w("pos"), "9921")

    # Add ind firstLine=0
    ind_el = ET.SubElement(ppr, _w("ind"))
    ind_el.set(_w("firstLine"), "0")

    # Add jc left
    jc_el = ET.SubElement(ppr, _w("jc"))
    jc_el.set(_w("val"), "left")

    # 4. Build body: [w:r tab] [m:oMath] [w:r tab] [w:r (5.x)]
    r_tab_center = ET.Element(_w("r"))
    ET.SubElement(r_tab_center, _w("tab"))

    r_tab_right = ET.Element(_w("r"))
    ET.SubElement(r_tab_right, _w("tab"))

    r_num = ET.Element(_w("r"))
    t_num = ET.SubElement(r_num, _w("t"))
    t_num.text = f"({num})"

    # Append in order after pPr
    p.append(r_tab_center)
    p.append(omath)
    p.append(r_tab_right)
    p.append(r_num)

    print(f"[OK] B{idx}: formula ({num}) — tab-stop layout applied")


# ── Save & repack ──────────────────────────────────────────────────────
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

tree.write(str(DOC_XML), xml_declaration=True, encoding="UTF-8")
print(f"Saved document.xml ({DOC_XML.stat().st_size:,} bytes)")

with zipfile.ZipFile(DOCX, "w", zipfile.ZIP_DEFLATED) as zout:
    for fpath in sorted(UNPACKED.rglob("*")):
        if fpath.is_file():
            zout.write(fpath, fpath.relative_to(UNPACKED))
print(f"Repacked: {DOCX.name} ({DOCX.stat().st_size:,} bytes)")
print("DONE.")

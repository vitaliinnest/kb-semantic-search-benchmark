"""
Нумерує формулу P = BGE-M3, E5-base як (4.31):
  1. Перебудовує B689 з m:oMathPara у tab-based layout (як решта формул)
  2. Додає w:r "(4.31)" після m:oMath
  3. Оновлює B687: "має вигляд:" → "має вигляд (4.31):"
"""
import sys, shutil, zipfile, pathlib, xml.etree.ElementTree as ET
sys.stdout.reconfigure(encoding='utf-8')

ROOT     = pathlib.Path("D:/repos/kb-semantic-search-benchmark")
DOCX     = ROOT / "2026_M_PI_Nesterenko_VV.docx"
UNPACKED = ROOT / "unpacked_docx"

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
M = "http://schemas.openxmlformats.org/officeDocument/2006/math"
def _w(t): return f"{{{W}}}{t}"
def _m(t): return f"{{{M}}}{t}"

if UNPACKED.exists():
    shutil.rmtree(UNPACKED)
with zipfile.ZipFile(DOCX) as z:
    z.extractall(UNPACKED)
print("Re-unpacked docx.")

DOC_XML = UNPACKED / "word" / "document.xml"
tree = ET.parse(DOC_XML)
body = tree.getroot().find(_w("body"))

def all_text(p):
    return "".join((t.text or "") for t in p.iter() if t.tag in (_w("t"), _m("t")))

children = list(body)

# ── 1. Знаходимо B689 (формула без номера) ────────────────────────────────
formula_idx = -1
for i, p in enumerate(children):
    if p.find(f".//{_m('oMathPara')}") is not None:
        t = all_text(p)
        if "BGE" in t and "E5" in t and "base" in t and "P" in t:
            formula_idx = i
            break

if formula_idx == -1:
    print("[!] Formula P=BGE-M3,E5-base NOT FOUND")
else:
    p = children[formula_idx]

    # Витягуємо m:oMath з m:oMathPara
    om_para = p.find(_m("oMathPara"))
    om = om_para.find(_m("oMath"))

    # Видаляємо m:oMathPara зі старого параграфа
    p.remove(om_para)

    # Перебудовуємо w:pPr — додаємо tabs і jc:left
    ppr = p.find(_w("pPr"))
    if ppr is None:
        ppr = ET.SubElement(p, _w("pPr"))

    # Стиль "a"
    ps = ppr.find(_w("pStyle"))
    if ps is None:
        ps = ET.SubElement(ppr, _w("pStyle"))
    ps.set(_w("val"), "a")

    # Додаємо w:tabs
    tabs_el = ET.SubElement(ppr, _w("tabs"))
    tab_center = ET.SubElement(tabs_el, _w("tab"))
    tab_center.set(_w("val"), "center")
    tab_center.set(_w("pos"), "4960")
    tab_right = ET.SubElement(tabs_el, _w("tab"))
    tab_right.set(_w("val"), "right")
    tab_right.set(_w("pos"), "9921")

    # w:ind firstLine=0
    ind = ET.SubElement(ppr, _w("ind"))
    ind.set(_w("firstLine"), "0")

    # w:jc left
    jc = ET.SubElement(ppr, _w("jc"))
    jc.set(_w("val"), "left")

    # Додаємо w:r > w:tab (для центрування)
    r_tab = ET.Element(_w("r"))
    ET.SubElement(r_tab, _w("tab"))
    p.append(r_tab)

    # Додаємо m:oMath
    p.append(om)

    # Додаємо w:r з "(4.31)"
    r_num = ET.Element(_w("r"))
    t_num = ET.SubElement(r_num, _w("t"))
    t_num.text = "(4.31)"
    p.append(r_num)

    print(f"[OK] Rebuilt B{formula_idx} as numbered formula (4.31).")

# ── 2. Оновлюємо B687: "має вигляд:" → "має вигляд (4.31):" ──────────────
anchor_idx = formula_idx - 2  # B687 = B689 - 2
intro_p = children[anchor_idx]
intro_txt = all_text(intro_p)
if "вигляд" in intro_txt and "(4.31)" not in intro_txt:
    # Знаходимо останній w:r і додаємо після нього ":" → " (4.31):"
    runs = intro_p.findall(_w("r"))
    if runs:
        last_r = runs[-1]
        t_el = last_r.find(_w("t"))
        if t_el is not None and t_el.text == ":":
            # Вставляємо "(4.31)" перед двокрапкою
            new_r = ET.Element(_w("r"))
            new_t = ET.SubElement(new_r, _w("t"))
            new_t.text = " (4.31)"
            new_t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
            idx_in_para = list(intro_p).index(last_r)
            intro_p.insert(idx_in_para, new_r)
            print(f"[OK] Updated B{anchor_idx}: added '(4.31)' before ':'")
        else:
            # Просто замінюємо текст останнього run
            if t_el is not None:
                old = t_el.text
                t_el.text = (old or "") + " (4.31):"
                t_el.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
                print(f"[OK] Updated B{anchor_idx}: appended '(4.31):' to last run")
else:
    print(f"[=] B{anchor_idx} already has (4.31) or 'вигляд' not found")

# ── Save & repack ──────────────────────────────────────────────────────────
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

"""
Додає явне формулювання про benchmark dataset на початку розділу 5.1
(одразу після заголовку "Опис експерименту", перед B711).

Текст містить: custom dataset, україномовний корпус, 3 домени,
числа (32 документи, 1491 чанк, 300 запитів), qrels.
"""
import sys, shutil, zipfile, pathlib, xml.etree.ElementTree as ET
sys.stdout.reconfigure(encoding='utf-8')

ROOT     = pathlib.Path("D:/repos/kb-semantic-search-benchmark")
DOCX     = ROOT / "thesis" / "2026_M_PI_Nesterenko_VV.docx"
UNPACKED = ROOT / "thesis" / "unpacked_docx"

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
M = "http://schemas.openxmlformats.org/officeDocument/2006/math"
def _w(t): return f"{{{W}}}{t}"

if UNPACKED.exists():
    shutil.rmtree(UNPACKED)
with zipfile.ZipFile(DOCX) as z:
    z.extractall(UNPACKED)
print("Re-unpacked docx.")

DOC_XML = UNPACKED / "word" / "document.xml"
tree = ET.parse(DOC_XML)
body = tree.getroot().find(_w("body"))

def all_text(p):
    return "".join((t.text or "") for t in p.iter(_w("t")))

def get_style(p):
    ppr = p.find(_w("pPr"))
    if ppr is not None:
        ps = ppr.find(_w("pStyle"))
        if ps is not None:
            return ps.get(_w("val"), "")
    return ""

def make_para(text, style="a"):
    p = ET.Element(_w("p"))
    ppr = ET.SubElement(p, _w("pPr"))
    ps = ET.SubElement(ppr, _w("pStyle"))
    ps.set(_w("val"), style)
    r = ET.SubElement(p, _w("r"))
    t = ET.SubElement(r, _w("t"))
    t.text = text
    if text != text.strip():
        t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    return p

# Знаходимо заголовок "Опис експерименту" в розділі 5
children = list(body)
anchor_idx = -1
for i, p in enumerate(children):
    if i < 700 or i > 730:
        continue
    txt = all_text(p).strip()
    style = get_style(p)
    if txt == "Опис експерименту" and style == "--1":
        anchor_idx = i
        break

if anchor_idx == -1:
    print("[!] Заголовок 'Опис експерименту' NOT FOUND — перевір діапазон")
else:
    # Перевіряємо, чи вже вставлено dataset statement
    next_p = children[anchor_idx + 1]
    next_txt = all_text(next_p)
    if "власноруч сформований" in next_txt or "domain-specific benchmark dataset" in next_txt:
        print(f"[=] Dataset statement вже існує після B{anchor_idx} — пропускаємо.")
    else:
        dataset_text = (
            "У дослідженні використано власноруч сформований domain-specific benchmark dataset, "
            "який складається з україномовного текстового корпусу документів у трьох предметних "
            "доменах (технічному, юридичному та медичному) — 32 документи та 1 491 чанк "
            "загалом, — а також наборів пошукових запитів (по 100 запитів на кожен домен, "
            "300 запитів сумарно) і еталонних релевантних відповідностей (qrels), "
            "сформованих на рівні чанків автором вручну."
        )
        new_para = make_para(dataset_text, style="a")
        body.insert(anchor_idx + 1, new_para)
        print(f"[OK] Inserted dataset statement after B{anchor_idx} ('{all_text(children[anchor_idx]).strip()}').")

# Save & repack
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

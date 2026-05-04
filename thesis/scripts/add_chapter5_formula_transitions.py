"""
Додає перехідні абзаци між формулами розділу 5, де їх немає:
  — перед (5.2) IDF: зв'язок BM25 → IDF
  — перед (5.4) nDCG: зв'язок DCG → nDCG
  — перед (5.5) MRR: зв'язок nDCG → MRR
Вставляємо знизу вгору, щоб зсуви не впливали на пошук.
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


def para_text(p):
    return "".join((t.text or "") for t in p.iter(_w("t")))


def has_math(p):
    return p.find(f".//{{{M}}}oMath") is not None


def find_formula_para(num_str, start=715, end=745):
    for i, ch in enumerate(list(body)):
        if i < start or i > end:
            continue
        if has_math(ch) and num_str in para_text(ch):
            return i, ch
    return -1, None


def insert_before(anchor_el, new_el):
    children = list(body)
    idx = children.index(anchor_el)
    body.insert(idx, new_el)


def make_para(text, style="a"):
    p = ET.Element(_w("p"))
    ppr = ET.SubElement(p, _w("pPr"))
    ps = ET.SubElement(ppr, _w("pStyle"))
    ps.set(_w("val"), style)
    r = ET.SubElement(p, _w("r"))
    t = ET.SubElement(r, _w("t"))
    t.text = text
    return p


# Перехідні тексти (вставляємо знизу вгору)
TRANSITIONS = [
    # (формула перед якою вставити, текст)
    ("(5.5)", (
        "Для оцінки позиції першого релевантного результату у видачі використовується "
        "метрика середнього взаємного рангу (MRR — Mean Reciprocal Rank), яка обчислюється "
        "за формулою (5.5):"
    )),
    ("(5.4)", (
        "Оскільки абсолютні значення DCG@k залежать від кількості релевантних документів і "
        "тому не є зіставними між різними запитами, вводиться нормалізований показник "
        "nDCG@k (Normalized DCG), який розраховується за формулою (5.4):"
    )),
    ("(5.2)", (
        "Ваговий коефіцієнт терміна у формулі BM25 визначається через обернену документну "
        "частоту (IDF — Inverse Document Frequency), яка враховує, наскільки рідкісним є "
        "термін у колекції загалом. IDF обчислюється за формулою (5.2):"
    )),
]

for num_str, text in TRANSITIONS:
    idx, p_formula = find_formula_para(num_str)
    if p_formula is None:
        print(f"[!] Formula {num_str} NOT FOUND")
        continue
    new_p = make_para(text)
    insert_before(p_formula, new_p)
    print(f"[OK] Inserted transition before formula {num_str} (was B{idx}).")


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

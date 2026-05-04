"""
Додає "де:" аннотації після кожної з 5 формул розділу 5.
Формат ідентичний розділу 4: w:pStyle="a", w:ind firstLine=0, w:jc left.
Перший рядок починається "де ...", наступні — з відступом (пробіли),
крапка з комою в кінці кожного рядка крім останнього (крапка).

Формули:
  (5.1) BM25    — 7 рядків
  (5.2) IDF     — 2 рядки
  (5.3) DCG@k   — 2 рядки
  (5.4) nDCG@k  — 2 рядки
  (5.5) MRR     — 2 рядки
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


def find_formula_para(num_str, start=715, end=740):
    """Знаходить параграф формули за номером '(5.x)'."""
    children = list(body)
    for i in range(start, min(end, len(children))):
        p = children[i]
        if has_math(p) and num_str in para_text(p):
            return i, p
    return -1, None


def insert_after(anchor_el, new_elements):
    """Вставляє список елементів одразу після anchor_el."""
    children = list(body)
    idx = children.index(anchor_el)
    for j, el in enumerate(new_elements):
        body.insert(idx + 1 + j, el)


def make_de_para(text):
    """Параграф стилю 'a' з w:ind firstLine=0 і w:jc left."""
    p = ET.Element(_w("p"))
    ppr = ET.SubElement(p, _w("pPr"))
    ps = ET.SubElement(ppr, _w("pStyle"))
    ps.set(_w("val"), "a")
    ind = ET.SubElement(ppr, _w("ind"))
    ind.set(_w("firstLine"), "0")
    jc = ET.SubElement(ppr, _w("jc"))
    jc.set(_w("val"), "left")
    r = ET.SubElement(p, _w("r"))
    t = ET.SubElement(r, _w("t"))
    t.text = text
    if text != text.strip():
        t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    return p


# Де-анотації для кожної формули
DE_ANNOTATIONS = {
    "(5.1)": [
        "де BM25(D, Q) — оцінка релевантності документа D щодо запиту Q;",
        "     IDF(qᵢ) — обернена документна частота терміна qᵢ;",
        "     f(qᵢ, D) — частота входжень терміна qᵢ у документі D;",
        "     |D| — довжина документа D (у словах);",
        "     avgdl — середня довжина документа в колекції;",
        "     k₁ — параметр насичення (типово k₁ = 1.2);",
        "     b — параметр нормалізації довжини (типово b = 0.75).",
    ],
    "(5.2)": [
        "де N — загальна кількість документів у колекції;",
        "     df(qᵢ) — кількість документів, що містять термін qᵢ.",
    ],
    "(5.3)": [
        "де relᵢ — оцінка релевантності результату на позиції i;",
        "     k — глибина видачі (у дослідженні k = 10).",
    ],
    "(5.4)": [
        "де DCG@k — дисконтований кумулятивний прибуток для поточної видачі;",
        "     IDCG@k — ідеальне значення DCG@k (для видачі з ідеальним ранжуванням).",
    ],
    "(5.5)": [
        "де |Q| — кількість запитів у benchmark-наборі;",
        "     rankᵤ — позиція першого релевантного результату для запиту q.",
    ],
}

# Обробляємо формули знизу вгору щоб вставки не зсували наступні
for num_str in ["(5.5)", "(5.4)", "(5.3)", "(5.2)", "(5.1)"]:
    idx, p_formula = find_formula_para(num_str, start=715, end=740)
    if p_formula is None:
        print(f"[!] Formula {num_str} NOT FOUND")
        continue
    lines = DE_ANNOTATIONS[num_str]
    de_paras = [make_de_para(line) for line in lines]
    insert_after(p_formula, de_paras)
    print(f"[OK] {num_str} (B{idx}): inserted {len(lines)} de-annotation lines.")


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

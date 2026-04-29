"""
Додає три блоки до практичного розділу 5:

1. Параграф із кількісною характеристикою корпусу (після опису доменів)
2. Параграф з методологією формування запитів і qrels (після опису qrels)
3. Зведена таблиця nDCG@10 по всіх моделях і доменах (Таблиця 5.11)
   + вступний параграф та підпис таблиці — після кросдоменного аналізу
"""
import sys, copy, shutil, zipfile, pathlib, xml.etree.ElementTree as ET
sys.stdout.reconfigure(encoding='utf-8')

ROOT     = pathlib.Path("D:/repos/kb-semantic-search-benchmark")
DOCX     = ROOT / "2026_M_PI_Nesterenko_VV.docx"
UNPACKED = ROOT / "unpacked_docx"

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


def body_children():
    return list(body)


def find_para_containing(substring, start=700, end=1100):
    for i, ch in enumerate(body_children()):
        if i < start or i > end:
            continue
        txt = "".join((t.text or "") for t in ch.iter(_w("t")))
        if substring in txt:
            return i, ch
    return -1, None


def para_text(p):
    return "".join((t.text or "") for t in p.iter(_w("t")))


def insert_after(anchor_el, new_el):
    """Вставляє new_el одразу після anchor_el у body."""
    children = body_children()
    idx = children.index(anchor_el)
    body.insert(idx + 1, new_el)


def make_text_para(text, style="a", ref_para=None):
    """Будує новий параграф стилю 'a' із заданим текстом, клонуючи стиль ref_para."""
    if ref_para is not None:
        p = copy.deepcopy(ref_para)
        # Clear all runs and text
        for r in list(p.findall(_w("r"))):
            p.remove(r)
        for r in list(p.findall(_w("hyperlink"))):
            p.remove(r)
    else:
        p = ET.Element(_w("p"))

    # Ensure pPr with correct style
    ppr = p.find(_w("pPr"))
    if ppr is None:
        ppr = ET.SubElement(p, _w("pPr"))
    ps = ppr.find(_w("pStyle"))
    if ps is None:
        ps = ET.SubElement(ppr, _w("pStyle"))
    ps.set(_w("val"), style)

    # Add run with text
    r = ET.SubElement(p, _w("r"))
    t = ET.SubElement(r, _w("t"))
    t.text = text
    if text != text.strip():
        t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    return p


# ── 1. Кількісна характеристика корпусу ──────────────────────────────
# Вставляємо після абзацу де описано домени: "...медичний домен включає..."
anchor_sub = "Медичний домен включає матеріали"
idx_anchor, p_anchor = find_para_containing(anchor_sub)
if p_anchor is None:
    print("[1] Anchor NOT FOUND, trying alt...")
    anchor_sub = "Медичний домен"
    idx_anchor, p_anchor = find_para_containing(anchor_sub)

corpus_text = (
    "Кількісна характеристика корпусу: технічний домен — 10 документів, 178 чанків; "
    "юридичний — 13 документів, 1046 чанків; медичний — 9 документів, 267 чанків. "
    "Загалом корпус охоплює 32 документи та 1491 чанк. Середній розмір чанка становить "
    "приблизно 5 900–6 100 символів (~600 слів), що забезпечує достатній контекст для "
    "формування якісних векторних представлень. Усі документи є матеріалами переважно "
    "українською мовою — наукові роботи, дисертації та навчальні видання відповідних "
    "предметних галузей."
)
if p_anchor is not None:
    new_p1 = make_text_para(corpus_text, style="a", ref_para=p_anchor)
    insert_after(p_anchor, new_p1)
    print(f"[1] Inserted corpus stats paragraph after B{idx_anchor}.")
else:
    print("[1] FAILED: corpus anchor not found.")


# ── 2. Методологія формування запитів і qrels ─────────────────────────
# Вставляємо після: "Набори qrels у chunk-level форматі"
anchor_sub2 = "Набори qrels у chunk-level форматі"
idx2, p2 = find_para_containing(anchor_sub2)
if p2 is None:
    anchor_sub2 = "одиницею індексації з одиницею оцінювання"
    idx2, p2 = find_para_containing(anchor_sub2)

methodology_text = (
    "Benchmark-запити та відповідні qrels сформовано автором вручну на основі реального "
    "змісту документів кожного домену. Для кожного запиту визначено один або кілька "
    "найбільш релевантних чанків, що безпосередньо відповідають на поставлене питання "
    "або містять ключову тематичну інформацію. Середня кількість релевантних чанків на "
    "запит: технічний домен — 2.15, юридичний — 2.86, медичний — 2.53. Запити охоплюють "
    "різноманітні типи інформаційних потреб — визначення понять, фактологічні питання, "
    "класифікації та порівняння — що зменшує ризик систематичної упередженості на "
    "користь будь-якого підходу до retrieval."
)
if p2 is not None:
    new_p2 = make_text_para(methodology_text, style="a", ref_para=p2)
    insert_after(p2, new_p2)
    print(f"[2] Inserted methodology paragraph after B{idx2}.")
else:
    print("[2] FAILED: qrels anchor not found.")


# ── 3. Зведена таблиця nDCG@10 (Таблиця 5.11) ────────────────────────
# Вставляємо після кросдоменного абзацу:
# "Порівняння результатів у трьох доменах показує..."
anchor_sub3 = "Порівняння результатів у трьох доменах показує"
idx3, p3 = find_para_containing(anchor_sub3)

if p3 is None:
    print("[3] FAILED: cross-domain anchor not found.")
else:
    # 3a. Вступний абзац
    intro_text = (
        "Для зручності зіставлення зведені значення nDCG@10 по всіх моделях і доменах "
        "наведено в таблиці 5.11. Жирним виділено найкращий результат у кожному домені."
    )
    p_intro = make_text_para(intro_text, style="a", ref_para=p3)

    # 3b. Підпис таблиці
    caption_text = (
        "Таблиця 5.11 – Зведені значення nDCG@10 для всіх моделей і доменів "
        "(таблиця виконана автором самостійно)"
    )
    p_caption = make_text_para(caption_text, style="a", ref_para=p3)

    # 3c. Будуємо таблицю
    ROWS_DATA = [
        # (model_name, tech, legal, medical)
        ("BGE-M3 (BAAI/bge-m3)",              "0.6722", "0.3065", "0.4339"),
        ("E5-base (multilingual-e5-base)",     "0.6121", "0.2567", "0.3909"),
        ("nomic-embed-text-v1.5",              "0.3765", "0.0951", "0.1668"),
        ("Qwen3-Embedding-0.6B",               "0.6325", "0.3199", "0.3629"),
        ("BM25 (Okapi) — базелайн",           "0.4861", "0.1875", "0.3222"),
    ]
    # Best per column (to bold)
    BEST = {
        "tech":    "0.6722",  # BGE-M3
        "legal":   "0.3199",  # Qwen3
        "medical": "0.4339",  # BGE-M3
    }
    COL_ORDER = ["tech", "legal", "medical"]

    def make_cell(text, style_val="-5", bold=False, is_header=False):
        tc = ET.Element(_w("tc"))
        tcpr = ET.SubElement(tc, _w("tcPr"))
        tcw = ET.SubElement(tcpr, _w("tcW"))
        tcw.set(_w("w"), "2500")
        tcw.set(_w("type"), "pct")
        hm = ET.SubElement(tcpr, _w("hideMark"))
        p = ET.SubElement(tc, _w("p"))
        ppr = ET.SubElement(p, _w("pPr"))
        ps = ET.SubElement(ppr, _w("pStyle"))
        ps.set(_w("val"), "-3" if is_header else "-5")
        r = ET.SubElement(p, _w("r"))
        rpr = ET.SubElement(r, _w("rPr"))
        if bold:
            ET.SubElement(rpr, _w("b"))
        t = ET.SubElement(r, _w("t"))
        t.text = text
        if text != text.strip():
            t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
        return tc

    tbl = ET.Element(_w("tbl"))
    tblpr = ET.SubElement(tbl, _w("tblPr"))
    tblstyle = ET.SubElement(tblpr, _w("tblStyle"))
    tblstyle.set(_w("val"), "TableGrid1")
    tblw = ET.SubElement(tblpr, _w("tblW"))
    tblw.set(_w("w"), "5000")
    tblw.set(_w("type"), "pct")
    tbllook = ET.SubElement(tblpr, _w("tblLook"))
    tbllook.set(_w("val"), "04A0")
    tbllook.set(_w("firstRow"), "1")
    tbllook.set(_w("lastRow"), "0")
    tbllook.set(_w("firstColumn"), "1")
    tbllook.set(_w("lastColumn"), "0")
    tbllook.set(_w("noHBand"), "0")
    tbllook.set(_w("noVBand"), "1")

    tblgrid = ET.SubElement(tbl, _w("tblGrid"))
    for _ in range(4):
        gc = ET.SubElement(tblgrid, _w("gridCol"))
        gc.set(_w("w"), "2500")

    # Header row
    tr_hdr = ET.SubElement(tbl, _w("tr"))
    tr_hdr.append(make_cell("Модель", is_header=True))
    tr_hdr.append(make_cell("Технічний", is_header=True))
    tr_hdr.append(make_cell("Юридичний", is_header=True))
    tr_hdr.append(make_cell("Медичний", is_header=True))

    # Data rows
    for model_name, tech_val, legal_val, med_val in ROWS_DATA:
        tr = ET.SubElement(tbl, _w("tr"))
        tr.append(make_cell(model_name))
        tr.append(make_cell(tech_val,  bold=(tech_val  == BEST["tech"])))
        tr.append(make_cell(legal_val, bold=(legal_val == BEST["legal"])))
        tr.append(make_cell(med_val,   bold=(med_val   == BEST["medical"])))

    # Insert: intro → caption → table  (in reverse since insert_after pushes down)
    # We insert after p3 (cross-domain paragraph), then each after previous
    insert_after(p3, p_intro)
    insert_after(p_intro, p_caption)
    insert_after(p_caption, tbl)

    print(f"[3] Inserted cross-domain nDCG@10 table (Таблиця 5.11) after B{idx3}.")


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

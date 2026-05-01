"""
Додає теоретичний опис RAG (~1 сторінка, 5 абзаців) у розділ 1.2
"Огляд існуючих підходів" — перед підсумковим абзацом "Таким чином..."
Також додає 2 нових джерела [31] і [32] до бібліографії.
"""
import sys, shutil, zipfile, pathlib, xml.etree.ElementTree as ET
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
    ps_el = ET.SubElement(ppr, _w("pStyle"))
    ps_el.set(_w("val"), style)
    if text:
        r = ET.SubElement(p, _w("r"))
        t_el = ET.SubElement(r, _w("t"))
        t_el.text = text
        if text != text.strip():
            t_el.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    return p

# ── 1. Вставляємо 5 абзаців про RAG перед "Таким чином, існуючі підходи" ──

RAG_PARAGRAPHS = [
    (
        "Окремим і принципово важливим підходом у сучасних інформаційних системах є парадигма "
        "Retrieval-Augmented Generation (RAG), запропонована Lewis та співавторами у роботі "
        "«Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks» [31]. RAG поєднує "
        "переваги інформаційного пошуку та генеративних великих мовних моделей (LLM) в єдиному "
        "конвеєрі: система отримує запит користувача, виконує пошук релевантних фрагментів у базі "
        "знань, а потім передає знайдені уривки як додатковий контекст до LLM, яка генерує кінцеву "
        "відповідь. Такий підхід дозволяє долати ключові обмеження чистих генеративних моделей — "
        "фіксований момент навчання (knowledge cutoff), схильність до галюцинацій і відсутність "
        "доступу до приватної або корпоративної документації — без необхідності дорогого повторного "
        "навчання моделі."
    ),
    (
        "Архітектурно RAG-система складається з двох ключових компонентів. Перший — ретривер "
        "(retriever) — відповідає за пошук релевантних текстових фрагментів: у сучасних реалізаціях "
        "це, як правило, dense retrieval на основі векторних ембеддінгів, де запит і документи "
        "кодуються в єдиному векторному просторі, а пошук здійснюється за косинусною подібністю або "
        "скалярним добутком. Другий компонент — генератор (generator) — є великою мовною моделлю, "
        "яка на основі запиту і набору знайдених фрагментів формує зв'язану, семантично точну "
        "відповідь природною мовою. Зв'язок між компонентами є критично важливим: якість ретривера "
        "безпосередньо визначає якість всієї RAG-системи, оскільки генератор може сформувати "
        "коректну відповідь лише за умови отримання справді релевантного контексту [31]."
    ),
    (
        "Якість ретривера є вузьким місцем RAG-конвеєра і безпосередньо залежить від embedding-моделі, "
        "що використовується для векторного пошуку. Дослідження показують [32], що покращення "
        "retrieval-метрик (зокрема Recall@k та nDCG@k) у standalone задачах пошуку безпосередньо "
        "транслюється у вищу точність відповідей на рівні RAG-системи. Вибір embedding-моделі для "
        "ретривера — за критеріями якості, мовної підтримки, швидкодії та вимог до обчислювальних "
        "ресурсів — є фундаментальним архітектурним рішенням при проєктуванні RAG-системи. Саме тому "
        "порівняльні дослідження embedding-моделей на реальних доменних колекціях, подібні до "
        "представленого у цій роботі, мають безпосередню практичну цінність не лише для задач "
        "окремого семантичного пошуку, а й для побудови повноцінних RAG-архітектур."
    ),
    (
        "У корпоративних системах управління знаннями RAG набуває особливої актуальності. Він "
        "дозволяє реалізовувати інтелектуальних асистентів, здатних відповідати на запити "
        "природною мовою на основі внутрішньої документації, нормативно-правових актів, технічних "
        "регламентів і медичних протоколів — без дорогого повторного навчання моделі на нових даних. "
        "При цьому система повертає не лише сформовану відповідь, а й посилання на вихідні документи, "
        "що забезпечує верифікованість і прозорість результатів. Такий підхід є закономірним "
        "продовженням розвитку корпоративного семантичного пошуку і суттєво перевищує можливості "
        "класичного keyword-пошуку з погляду інтелектуального опрацювання інформації."
    ),
    (
        "У дослідженнях останніх років виокремлюють кілька поколінь RAG-архітектур [32]. Naive RAG "
        "реалізує базову схему «ретривер — генератор» без додаткової обробки. Advanced RAG включає "
        "механізми попередньої обробки запитів (query rewriting, HyDE — hypothetical document "
        "embeddings), постобробки результатів (reranking) та оптимізацію формування контексту. "
        "Modular RAG передбачає гнучку комбінацію спеціалізованих компонентів пошуку, фільтрації, "
        "верифікації та генерації. Паралельно розвивається гібридний retrieval, що поєднує dense "
        "(векторний) і sparse (лексичний, зокрема BM25) підходи для підвищення повноти пошуку. "
        "Усі ці тенденції підкреслюють принципову роль якості retrieval-компонента і обраної "
        "embedding-моделі як основи ефективного RAG-конвеєра."
    ),
]

children = list(body)
anchor_idx = -1
for i, p in enumerate(children):
    if i < 219 or i > 230:
        continue
    txt = all_text(p)
    if txt.startswith("Таким чином, існуючі підходи до пошуку"):
        anchor_idx = i
        break

if anchor_idx == -1:
    print("[!] Anchor 'Таким чином, існуючі підходи' NOT FOUND")
else:
    # Перевіряємо, чи RAG вже вставлений
    prev_txt = all_text(children[anchor_idx - 1])
    if "RAG-конвеєра" in prev_txt or "Retrieval-Augmented Generation" in prev_txt:
        print(f"[=] RAG paragraphs already present before B{anchor_idx} — skipping.")
    else:
        new_paras = [make_para(text) for text in RAG_PARAGRAPHS]
        for j, para in enumerate(new_paras):
            body.insert(anchor_idx + j, para)
        print(f"[OK] Inserted {len(new_paras)} RAG paragraphs before B{anchor_idx}.")

# ── 2. Додаємо джерела [31] і [32] після запису 30 ──────────────────────

NEW_REFS = [
    (
        "31. Lewis P., Perez E., Piktus A., Petroni F., Karpukhin V., Goyal N., Küttler H., "
        "Lewis M., Yih W., Rocktäschel T., Riedel S., Kiela D. Retrieval-Augmented Generation "
        "for Knowledge-Intensive NLP Tasks // Advances in Neural Information Processing Systems "
        "(NeurIPS). 2020. URL: https://arxiv.org/abs/2005.11401 (дата звернення: 25.01.2026)."
    ),
    (
        "32. Gao Y., Xiong Y., Gao X., Jia K., Pan J., Bi Y., Dai Y., Sun J., Wang H. "
        "Retrieval-Augmented Generation for Large Language Models: A Survey. arXiv preprint "
        "arXiv:2312.10997. 2023. URL: https://arxiv.org/abs/2312.10997 "
        "(дата звернення: 25.01.2026)."
    ),
]

# Знаходимо запис 30 (MTEB)
children = list(body)
ref30_idx = -1
for i, p in enumerate(children):
    t = all_text(p)
    if "30." in t and "MTEB" in t:
        ref30_idx = i
        break

if ref30_idx == -1:
    print("[!] Entry 30 (MTEB) NOT FOUND in bibliography")
else:
    # Перевіряємо, чи вже є запис 31
    if "31." in all_text(children[ref30_idx + 1]) or "Lewis" in all_text(children[ref30_idx + 1]):
        print(f"[=] Ref [31] already exists after B{ref30_idx} — skipping.")
    else:
        new_ref_paras = [make_para(text) for text in NEW_REFS]
        for j, para in enumerate(new_ref_paras):
            body.insert(ref30_idx + 1 + j, para)
        print(f"[OK] Inserted 2 new bibliography entries [31, 32] after B{ref30_idx}.")

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

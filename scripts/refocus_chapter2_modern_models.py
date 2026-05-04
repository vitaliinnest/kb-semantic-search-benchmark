"""
Перефокусує розділ 2 з минулих моделей на сучасні моделі, що реально
використовуються в роботі. Робить:

  1. Видаляє 2 додані раніше абзаци (modern models intro + benchmarks)
  2. Замінює зміст 4 абзаців у "Моделі векторних ембеддінгів":
        Word2Vec       → E5 (Wang et al. 2022)
        FastText       → BGE-M3 (Chen et al. 2024)
        BERT           → nomic-embed-text (Nussbaum et al. 2024)
        Sentence-BERT  → Qwen3-Embedding (Zhang et al. 2025)
  3. Переписує вступний абзац секції (фокус на сучасних моделях)
  4. Переписує заключний абзац секції
  5. Оновлює листинг джерел у "Огляд основних джерел"
  6. Оновлює формулювання у "Оцінка актуальності та новизни"
     (приберає старі моделі з фокус-фрази)
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


def find_para_containing(substring, start=0, end=2000):
    for i, ch in enumerate(list(body)):
        if i < start or i > end:
            continue
        if substring in para_text(ch):
            return i, ch
    return -1, None


def set_paragraph_text(p, new_text):
    """Записує новий текст у перший w:t, інші очищає."""
    t_elements = []
    for r in p.findall(_w("r")):
        for t in r.findall(_w("t")):
            t_elements.append(t)
    if not t_elements:
        # Параграф не має w:t — додамо
        r = p.find(_w("r"))
        if r is None:
            r = ET.SubElement(p, _w("r"))
        t = ET.SubElement(r, _w("t"))
        t.text = new_text
        return True
    t_elements[0].text = new_text
    if new_text != new_text.strip():
        t_elements[0].set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    for t in t_elements[1:]:
        t.text = ""
    return True


# ── 1. Видаляємо 2 додані раніше абзаци ───────────────────────────────
removed = 0
markers = [
    "Подальший розвиток моделей векторних ембеддінгів пов'язаний із появою нового",
    "Для оцінювання якості сучасних моделей ембеддінгів у наукових працях використовуються",
]
for marker in markers:
    idx, p = find_para_containing(marker, start=240, end=280)
    if p is not None:
        body.remove(p)
        removed += 1
        print(f"[1] Removed previously inserted paragraph (was B{idx}).")
    else:
        print(f"[1] Marker not found: {marker[:50]}...")
print(f"[1] Total removed: {removed}/2")


# ── 2. Замінюємо 4 абзаци про моделі ──────────────────────────────────
REPLACEMENTS = [
    # (anchor substring, new text, label)
    (
        "Однією з базових моделей є Word2Vec",
        "Однією з ключових сучасних моделей мультилінгвального семантичного пошуку є "
        "E5, представлена у роботі Wang та співавторів «Text Embeddings by "
        "Weakly-Supervised Contrastive Pre-training» [24]. Модель побудована на "
        "трансформерній архітектурі і навчена контрастивним підходом на масштабному "
        "наборі з мільярда текстових пар, отриманих із відкритих джерел Інтернету. Її "
        "ключовою перевагою є здатність формувати високоякісні щільні векторні подання "
        "для задач retrieval без необхідності у великих маркованих датасетах. "
        "Multilingual-версія E5 поширює цей підхід на понад 100 мов, що робить її "
        "придатною для україномовних корпоративних колекцій.",
        "Word2Vec → E5",
    ),
    (
        "Подальший розвиток статичних моделей відображено у роботі Piotr Bojanowski",
        "Іншою сучасною моделлю, що реалізує принципово відмінний підхід, є BGE-M3, "
        "описана у роботі Chen та співавторів «BGE-M3: Multi-Lingual, "
        "Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge "
        "Distillation» [25]. Її особливістю є поєднання трьох режимів подання тексту "
        "в єдиній архітектурі: dense (щільне векторне подання), sparse (розріджене "
        "лексичне) та multi-vector (ColBERT-подібне). Модель навчена на понад 100 "
        "мовах із підтримкою документів довжиною до 8192 токенів. Така мультирежимність "
        "дозволяє адаптувати retrieval-стратегію до конкретної задачі: dense-режим — "
        "для семантичного пошуку, sparse — для точного збігу за термінами, "
        "multi-vector — для високоточного ранжування.",
        "FastText → BGE-M3",
    ),
    (
        "Якісно новий етап розвитку моделей ембеддінгів пов’язаний із трансформерною",
        "Окремий напрям сучасних моделей ембеддінгів представлений моделлю "
        "nomic-embed-text, описаною у роботі Nussbaum та співавторів «Nomic Embed: "
        "Training a Reproducible Long Context Text Embedder» [26]. Її ключовою "
        "відмінністю є повна відкритість — авторами оприлюднено не лише ваги моделі, "
        "а й увесь набір тренувальних даних і код навчання. Архітектурно модель "
        "підтримує контекст до 8192 токенів і використовує механізм Matryoshka "
        "representations, що дозволяє адаптивно зменшувати розмірність вектора без "
        "перенавчання моделі. Такий підхід корисний для оптимізації пам'яті та "
        "швидкості пошуку у великих корпусах. Водночас модель тренована переважно на "
        "англомовних корпусах, що обмежує її якість на текстах інших мов.",
        "BERT → nomic-embed-text",
    ),
    (
        "Більш придатним для задач семантичного пошуку є підхід, запропонований Nils Reimers",
        "Принципово відмінну архітектурну стратегію реалізує модель Qwen3-Embedding-0.6B, "
        "описана у роботі Zhang та співавторів «Qwen3 Embedding: Advancing Text "
        "Embedding and Reranking Through Foundation Models» [27]. На відміну від "
        "енкодерних моделей (BERT-подібних), вона побудована на декодерній архітектурі "
        "великої мовної моделі (LLM) і використовує знання, отримані під час pre-training "
        "на сотнях мільярдів токенів. Модель підтримує управління через інструкції "
        "(instruction-following), що дозволяє адаптувати поведінку до конкретного типу "
        "пошукової задачі. Таке поєднання забезпечує конкурентну якість retrieval, "
        "однак спричиняє вищі обчислювальні витрати порівняно з компактними "
        "енкодерними моделями.",
        "Sentence-BERT → Qwen3-Embedding",
    ),
]

for anchor, new_text, label in REPLACEMENTS:
    idx, p = find_para_containing(anchor, start=240, end=280)
    if p is None:
        print(f"[2] {label}: anchor NOT FOUND")
        continue
    set_paragraph_text(p, new_text)
    print(f"[2] {label}: replaced B{idx}")


# ── 3. Переписуємо вступний абзац секції "Моделі векторних ембеддінгів" ─
intro_anchor = "У сучасній літературі моделі векторних ембеддінгів розглядаються як основний"
idx, p = find_para_containing(intro_anchor, start=240, end=280)
if p is not None:
    new_text = (
        "У межах поточної роботи об'єктом експериментального дослідження обрано "
        "чотири актуальні моделі векторних ембеддінгів останнього покоління — E5, "
        "BGE-M3, nomic-embed-text-v1.5 та Qwen3-Embedding-0.6B. Усі вони побудовані на "
        "трансформерних архітектурах, мають мультилінгвальну підтримку та орієнтовані "
        "безпосередньо на задачі семантичного пошуку, що робить їх придатними для "
        "застосування в україномовних корпоративних базах знань. Нижче розглядаються "
        "ключові архітектурні особливості та результати, представлені авторами цих моделей."
    )
    set_paragraph_text(p, new_text)
    print(f"[3] Section intro rewritten (B{idx}).")
else:
    print("[3] Section intro anchor NOT FOUND")


# ── 4. Переписуємо заключний абзац секції ─────────────────────────────
concl_anchor = "Отже, аналіз основних джерел показує еволюцію моделей векторних ембеддінгів"
idx, p = find_para_containing(concl_anchor, start=240, end=290)
if p is not None:
    new_text = (
        "Отже, аналіз сучасних літературних джерел показує, що E5 [24], BGE-M3 [25], "
        "nomic-embed-text [26] та Qwen3-Embedding [27] є найбільш актуальним поколінням "
        "мультилінгвальних моделей ембеддінгів для задач семантичного пошуку. Кожна з "
        "них реалізує власний архітектурний підхід — від класичних щільних подань (E5) "
        "до мультирежимного представлення (BGE-M3), Matryoshka-навчання (nomic-embed) "
        "та декодерної LLM-архітектури (Qwen3). Така різноманітність архітектур у межах "
        "одного покоління моделей робить обґрунтований вибір серед них нетривіальною "
        "задачею і визначає необхідність порівняльного дослідження."
    )
    set_paragraph_text(p, new_text)
    print(f"[4] Section conclusion rewritten (B{idx}).")
else:
    print("[4] Section conclusion anchor NOT FOUND")


# ── 5. Оновлюємо "Огляд основних джерел" (перший абзац секції) ────────
overview_anchor = "Для аналізу обрано праці, присвячені трьом ключовим напрямам"
idx, p = find_para_containing(overview_anchor, start=240, end=280)
if p is not None:
    new_text = (
        "Для аналізу обрано праці двох категорій: фундаментальні дослідження, що "
        "визначили теоретичні основи векторного подання тексту (Mikolov та співавтори, "
        "Pennington та співавтори), та сучасні роботи, що описують моделі ембеддінгів "
        "останнього покоління — об'єкти експериментального порівняння в межах поточної "
        "роботи: Wang та співавтори (E5) [24], Chen та співавтори (BGE-M3) [25], "
        "Nussbaum та співавтори (nomic-embed-text) [26] і Zhang та співавтори "
        "(Qwen3-Embedding) [27]. Додатково розглянуто роботи з оцінювання якості "
        "семантичного пошуку, зокрема benchmark-набори BEIR [29] та MTEB [30], а також "
        "класичний базелайн-метод BM25 [28]. Обрані джерела дозволяють обґрунтовано "
        "порівняти сучасні моделі векторних ембеддінгів і визначити найбільш придатні "
        "рішення для семантичного пошуку в корпоративних базах знань."
    )
    set_paragraph_text(p, new_text)
    print(f"[5] Overview paragraph rewritten (B{idx}).")
else:
    print("[5] Overview anchor NOT FOUND")


# ── 6. Оновлюємо абзац у "Оцінка актуальності та новизни" ─────────────
# Замінюємо речення про "Word2Vec, TF-IDF, FastText, BERT і Sentence-BERT"
actuality_anchor = "Розглянуті джерела показують, що векторні ембеддінги стали базовим"
idx, p = find_para_containing(actuality_anchor, start=240, end=300)
if p is not None:
    old_text = para_text(p)
    new_text = old_text.replace(
        "Роботи, присвячені Word2Vec, TF-IDF, FastText, BERT і Sentence-BERT, а також "
        "новому поколінню моделей BGE-M3, E5-base, nomic-embed-text та Qwen3-Embedding, "
        "відображають еволюцію підходів від статистичних і статичних моделей слів до "
        "контекстних і мультилінгвальних моделей речень і документів",
        "Сучасне покоління моделей — BGE-M3 [25], E5 [24], nomic-embed-text [26] та "
        "Qwen3-Embedding [27] — забезпечує перехід від ранніх статичних моделей слів до "
        "контекстних мультилінгвальних подань речень і документів, орієнтованих "
        "безпосередньо на задачі retrieval"
    )
    if new_text != old_text:
        set_paragraph_text(p, new_text)
        print(f"[6] Actuality paragraph updated (B{idx}).")
    else:
        print(f"[6] Actuality paragraph unchanged (substring not found at B{idx}).")
else:
    print("[6] Actuality anchor NOT FOUND")


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

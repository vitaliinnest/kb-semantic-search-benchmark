"""
Оновлює розділ 2 (огляд літератури) і список джерел:

  1. Додає у розділ "Моделі векторних ембеддінгів у задачах семантичного
     пошуку" (після B256, Sentence-BERT) новий абзац про сучасне покоління
     мультилінгвальних моделей: E5, BGE-M3, nomic-embed-text, Qwen3-Embedding
     з посиланнями [24]–[27] на оригінальні роботи.

  2. Додає абзац про сучасні benchmark-набори (BEIR, MTEB) і базелайн BM25
     з посиланнями [28]–[30].

  3. Додає 7 нових записів у "Перелік джерел посилання" (24–30):
      24. E5 (Wang et al. 2022)
      25. BGE-M3 (Chen et al. 2024)
      26. nomic-embed (Nussbaum et al. 2024)
      27. Qwen3-Embedding (Zhang et al. 2025)
      28. BM25 (Robertson, Zaragoza 2009)
      29. BEIR (Thakur et al. 2021)
      30. MTEB (Muennighoff et al. 2023)
"""
import sys, copy, shutil, zipfile, pathlib, xml.etree.ElementTree as ET
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


def insert_after(anchor_el, new_elements):
    children = list(body)
    idx = children.index(anchor_el)
    for j, el in enumerate(new_elements):
        body.insert(idx + 1 + j, el)


def make_para(text, style="a"):
    p = ET.Element(_w("p"))
    ppr = ET.SubElement(p, _w("pPr"))
    ps = ET.SubElement(ppr, _w("pStyle"))
    ps.set(_w("val"), style)
    r = ET.SubElement(p, _w("r"))
    t = ET.SubElement(r, _w("t"))
    t.text = text
    return p


# ── 1. Додаємо абзац про сучасні моделі у розділ 2 ────────────────────
modern_models_text = (
    "Подальший розвиток моделей векторних ембеддінгів пов'язаний із появою нового "
    "покоління мультилінгвальних моделей, орієнтованих на високу якість семантичного "
    "пошуку у різних мовах і доменах. До них належать E5 [24], BGE-M3 [25], "
    "nomic-embed-text [26] та Qwen3-Embedding [27], які й стали основою для "
    "експериментального дослідження в межах поточної роботи. Модель E5 побудована на "
    "базі трансформерної архітектури і навчена контрастивним підходом на мільярді "
    "текстових пар, отриманих із відкритих джерел. BGE-M3 поєднує dense, sparse та "
    "multi-vector представлення в єдиній архітектурі, що забезпечує гнучкість для "
    "різних сценаріїв retrieval. Модель nomic-embed-text підтримує довгий контекст до "
    "8192 токенів і використовує механізм Matryoshka representations для адаптивної "
    "розмірності векторів. Qwen3-Embedding побудована на декодерній архітектурі великої "
    "мовної моделі та підтримує управління через інструкції. Усі ці моделі забезпечують "
    "одночасну роботу з понад 100 мовами, що критично для україномовних корпоративних "
    "колекцій."
)
benchmarks_text = (
    "Для оцінювання якості сучасних моделей ембеддінгів у наукових працях використовуються "
    "стандартизовані benchmark-набори. Найбільш поширеним є BEIR [29], який об'єднує 18 "
    "різнорідних датасетів для zero-shot оцінювання retrieval-моделей у різних доменах "
    "і типах задач. Узагальнюючим бенчмарком для моделей ембеддінгів є MTEB [30], що "
    "охоплює понад 50 завдань (retrieval, classification, clustering, semantic similarity) "
    "і де-факто став стандартом порівняння нових архітектур. Базелайн-методом для "
    "оцінювання retrieval-якості традиційно виступає алгоритм BM25 [28] — класичний "
    "імовірнісний підхід, що поєднує статистику частоти термінів з оберненою документною "
    "частотою. Незважаючи на значний вік, BM25 залишається складним до перевершення "
    "базелайном для багатьох доменно-специфічних задач і використовується як обов'язковий "
    "компонент порівняльних досліджень."
)

# Anchor: paragraph про Sentence-BERT (B256) — закінчується словом "корпоративних базах знань."
anchor_sub = "Sentence-BERT часто розглядається як одна з найбільш перспективних"
idx_sb, p_sb = find_para_containing(anchor_sub, start=240, end=280)
if p_sb is None:
    print("[1/2] Sentence-BERT anchor NOT FOUND")
else:
    p_modern = make_para(modern_models_text, style="a")
    p_bench  = make_para(benchmarks_text,    style="a")
    insert_after(p_sb, [p_modern, p_bench])
    print(f"[1/2] Inserted 2 paragraphs (modern models, benchmarks) after B{idx_sb}.")


# ── 3. Додаємо нові записи 24–30 у бібліографію ──────────────────────
NEW_ENTRIES = [
    "24. Wang L., Yang N., Huang X., Jiao B., Yang L., Jiang D., Majumder R., Wei F. "
    "Text Embeddings by Weakly-Supervised Contrastive Pre-training. arXiv preprint "
    "arXiv:2212.03533. 2022. URL: https://arxiv.org/abs/2212.03533 "
    "(дата звернення: 25.01.2026).",

    "25. Chen J., Xiao S., Zhang P., Luo K., Lian D., Liu Z. BGE-M3: Multi-Lingual, "
    "Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge "
    "Distillation. arXiv preprint arXiv:2402.03216. 2024. "
    "URL: https://arxiv.org/abs/2402.03216 (дата звернення: 25.01.2026).",

    "26. Nussbaum Z., Morris J. X., Duderstadt B., Mulyar A. Nomic Embed: Training a "
    "Reproducible Long Context Text Embedder. arXiv preprint arXiv:2402.01613. 2024. "
    "URL: https://arxiv.org/abs/2402.01613 (дата звернення: 25.01.2026).",

    "27. Zhang Y., Li M., Long D., Zhang X., Lin H., Yang B., Xie P., Yang A., Liu D., "
    "Lin J., Huang F., Zhou J. Qwen3 Embedding: Advancing Text Embedding and Reranking "
    "Through Foundation Models. arXiv preprint arXiv:2506.05176. 2025. "
    "URL: https://arxiv.org/abs/2506.05176 (дата звернення: 15.04.2026).",

    "28. Robertson S., Zaragoza H. The Probabilistic Relevance Framework: BM25 and "
    "Beyond // Foundations and Trends in Information Retrieval. 2009. Vol. 3, no. 4. "
    "P. 333–389. URL: https://doi.org/10.1561/1500000019 "
    "(дата звернення: 25.01.2026).",

    "29. Thakur N., Reimers N., Rücklé A., Srivastava A., Gurevych I. BEIR: A "
    "Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models. "
    "Proceedings of the 35th Conference on Neural Information Processing Systems "
    "(NeurIPS) Datasets and Benchmarks Track. 2021. "
    "URL: https://arxiv.org/abs/2104.08663 (дата звернення: 25.01.2026).",

    "30. Muennighoff N., Tazi N., Magne L., Reimers N. MTEB: Massive Text Embedding "
    "Benchmark. Proceedings of the 17th Conference of the European Chapter of the "
    "Association for Computational Linguistics (EACL). 2023. "
    "URL: https://arxiv.org/abs/2210.07316 (дата звернення: 25.01.2026).",
]

# Anchor: останнє джерело (B1064 — Järvelin [23])
idx_last, p_last = find_para_containing("Järvelin", start=1040, end=1080)
if p_last is None:
    print("[3] Bibliography anchor NOT FOUND")
else:
    new_paras = [make_para(entry, style="a") for entry in NEW_ENTRIES]
    insert_after(p_last, new_paras)
    print(f"[3] Inserted {len(NEW_ENTRIES)} new bibliography entries (24–30) after B{idx_last}.")


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

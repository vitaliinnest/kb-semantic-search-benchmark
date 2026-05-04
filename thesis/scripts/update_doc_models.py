"""
Update the thesis document XML to replace old model names with new ones
in sections 1-4 (abstract, abbreviations, sections 2-4).
"""
import sys
import io
import copy
import re
from pathlib import Path
import xml.etree.ElementTree as ET

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

DOC_XML = Path("D:/repos/kb-semantic-search-benchmark/thesis/unpacked_docx/word/document.xml")
NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
W = f"{{{NS}}}"

ET.register_namespace("wpc", "http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas")
ET.register_namespace("cx", "http://schemas.microsoft.com/office/drawing/2014/chartex")
ET.register_namespace("cx1", "http://schemas.microsoft.com/office/drawing/2015/9/8/chartex")
ET.register_namespace("cx2", "http://schemas.microsoft.com/office/drawing/2015/10/21/chartex")
ET.register_namespace("cx3", "http://schemas.microsoft.com/office/drawing/2016/5/9/chartex")
ET.register_namespace("cx4", "http://schemas.microsoft.com/office/drawing/2016/5/10/chartex")
ET.register_namespace("cx5", "http://schemas.microsoft.com/office/drawing/2016/5/11/chartex")
ET.register_namespace("cx6", "http://schemas.microsoft.com/office/drawing/2016/5/12/chartex")
ET.register_namespace("cx7", "http://schemas.microsoft.com/office/drawing/2016/5/13/chartex")
ET.register_namespace("cx8", "http://schemas.microsoft.com/office/drawing/2016/5/14/chartex")
ET.register_namespace("mc", "http://schemas.openxmlformats.org/markup-compatibility/2006")
ET.register_namespace("aink", "http://schemas.microsoft.com/office/drawing/2016/ink")
ET.register_namespace("am3d", "http://schemas.microsoft.com/office/drawing/2017/model3d")
ET.register_namespace("o", "urn:schemas-microsoft-com:office:office")
ET.register_namespace("oel", "http://schemas.microsoft.com/office/2019/extlst")
ET.register_namespace("r", "http://schemas.openxmlformats.org/officeDocument/2006/relationships")
ET.register_namespace("m", "http://schemas.openxmlformats.org/officeDocument/2006/math")
ET.register_namespace("v", "urn:schemas-microsoft-com:vml")
ET.register_namespace("wp14", "http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing")
ET.register_namespace("wp", "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing")
ET.register_namespace("w10", "urn:schemas-microsoft-com:office:word")
ET.register_namespace("w", "http://schemas.openxmlformats.org/wordprocessingml/2006/main")
ET.register_namespace("w14", "http://schemas.microsoft.com/office/word/2010/wordml")
ET.register_namespace("w15", "http://schemas.microsoft.com/office/word/2012/wordml")
ET.register_namespace("w16cex", "http://schemas.microsoft.com/office/word/2018/wordml/cex")
ET.register_namespace("w16cid", "http://schemas.microsoft.com/office/word/2016/wordml/cid")
ET.register_namespace("w16", "http://schemas.microsoft.com/office/word/2018/wordml")
ET.register_namespace("w16sdtdh", "http://schemas.microsoft.com/office/word/2020/wordml/sdtdatahash")
ET.register_namespace("w16se", "http://schemas.microsoft.com/office/word/2015/wordml/symex")
ET.register_namespace("wpg", "http://schemas.microsoft.com/office/word/2010/wordprocessingGroup")
ET.register_namespace("wpi", "http://schemas.microsoft.com/office/word/2010/wordprocessingInk")
ET.register_namespace("wne", "http://schemas.microsoft.com/office/word/2006/wordml")
ET.register_namespace("wps", "http://schemas.microsoft.com/office/word/2010/wordprocessingShape")


def get_para_text(para) -> str:
    """Return the combined text of all w:t children in the paragraph."""
    parts = []
    for t in para.iter(f"{W}t"):
        parts.append(t.text or "")
    return "".join(parts)


def set_para_text(para, new_text: str) -> None:
    """
    Replace all runs in the paragraph with a single run that carries new_text.
    Preserves paragraph properties (pPr) and run properties of the first run.
    """
    # Collect pPr and first rPr
    pPr = para.find(f"{W}pPr")
    first_rPr = None
    first_r = para.find(f"{W}r")
    if first_r is not None:
        first_rPr = first_r.find(f"{W}rPr")

    # Remove all child elements from para
    for child in list(para):
        para.remove(child)

    # Re-add pPr
    if pPr is not None:
        para.append(pPr)

    # Build a new run
    run = ET.SubElement(para, f"{W}r")
    if first_rPr is not None:
        run.append(copy.deepcopy(first_rPr))
    t_elem = ET.SubElement(run, f"{W}t")
    t_elem.text = new_text
    if new_text.startswith(" ") or new_text.endswith(" "):
        t_elem.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")


def replace_in_para(para, old: str, new: str) -> bool:
    """Replace old substring with new in the merged text of the paragraph."""
    text = get_para_text(para)
    if old not in text:
        return False
    new_text = text.replace(old, new)
    set_para_text(para, new_text)
    return True


# ── Load document ──────────────────────────────────────────────────────────────
tree = ET.parse(str(DOC_XML))
root = tree.getroot()
paras = root.findall(f".//{W}p")

print(f"Total paragraphs: {len(paras)}")

changes = []

# ── 1. Abstract keywords (P133, P139) ─────────────────────────────────────────
OLD_KEYWORDS_UK = "BERT, FASTTEXT, OPENAI EMBEDDINGS, SBERT."
NEW_KEYWORDS_UK = "BGE-M3, E5-BASE, QWEN3-EMBEDDING, NOMIC-EMBED-TEXT."
OLD_KEYWORDS_EN = "BERT, FASTTEXT, OPENAI EMBEDDINGS, SBERT."
NEW_KEYWORDS_EN = "BGE-M3, E5-BASE, QWEN3-EMBEDDING, NOMIC-EMBED-TEXT."

for idx in [133, 139]:
    p = paras[idx]
    text = get_para_text(p)
    if "BERT, FASTTEXT" in text:
        if idx == 133:
            ok = replace_in_para(p, OLD_KEYWORDS_UK, NEW_KEYWORDS_UK)
        else:
            ok = replace_in_para(p, OLD_KEYWORDS_EN, NEW_KEYWORDS_EN)
        changes.append(f"P{idx} keywords: {'OK' if ok else 'MISS'}")
    else:
        changes.append(f"P{idx} keywords: already updated or text mismatch")

# ── 2. Abbreviations ───────────────────────────────────────────────────────────
ABBREV_MAP = {
    201: ("BERT – Bidirectional Encoder Representations from Transformers",
          "BGE-M3 – BAAI General Embedding, мультилінгвальна модель ембеддінгів третього покоління"),
    204: ("FastText – Fast Text Representation and Classification",
          "E5 – Embeddings from bidirectional Encoder rEpresentations, модель Microsoft для семантичного пошуку"),
    214: ("SBERT – Sentence-BERT",
          "Nomic – платформа Nomic AI для відкритих мультилінгвальних моделей ембеддінгів"),
    215: ("TF-IDF – Term Frequency – Inverse Document Frequency",
          "Qwen3 – мультилінгвальна модель ембеддінгів від Alibaba Cloud"),
    218: ("Word2Vec – Word to Vector",
          "text-embedding-3-large – комерційна модель OpenAI для генерації текстових ембеддінгів через API"),
}

for idx, (old_abbr, new_abbr) in ABBREV_MAP.items():
    p = paras[idx]
    text = get_para_text(p)
    if old_abbr in text:
        ok = replace_in_para(p, old_abbr, new_abbr)
        changes.append(f"P{idx} abbrev: {'OK' if ok else 'MISS'}")
    else:
        changes.append(f"P{idx} abbrev: skip (text='{text[:60]}')")

# ── 3. Section 2 – domain analysis ────────────────────────────────────────────
# P244: Update mention of statistical models to add modern context
OLD_P244 = "Водночас ці підходи здебільшого не враховують контекст конкретного використання слова, що обмежує їхню ефективність у задачах глибокого семантичного пошуку"
NEW_P244 = ("Водночас ці підходи здебільшого не враховують контекст конкретного використання слова, що обмежує їхню ефективність у задачах глибокого семантичного пошуку. "
            "Подальший розвиток цього напряму привів до появи нових поколінь мультилінгвальних моделей ембеддінгів, зокрема BGE-M3, E5-base, nomic-embed-text та Qwen3-Embedding, "
            "які поєднують переваги трансформерних архітектур із ефективним векторним поданням документів і запитів")

ok = replace_in_para(paras[244], OLD_P244, NEW_P244)
changes.append(f"P244 domain: {'OK' if ok else 'MISS'}")

# P245: Update "BERT і Sentence-BERT" mention to include new generation
OLD_P245_SNIPPET = "зокрема BERT і Sentence-BERT. Такі моделі формують векторні подання з урахуванням контексту, в якому використовується слово, речення або документ"
NEW_P245_SNIPPET = ("зокрема BERT і Sentence-BERT, а також нові мультилінгвальні моделі, зокрема BGE-M3, E5-base, nomic-embed-text та Qwen3-Embedding. "
                    "Такі моделі формують векторні подання з урахуванням контексту, в якому використовується слово, речення або документ")
ok = replace_in_para(paras[245], OLD_P245_SNIPPET, NEW_P245_SNIPPET)
changes.append(f"P245 domain: {'OK' if ok else 'MISS'}")

# P252: Update "BERT і Sentence-BERT" in limitations section
OLD_P252 = "зокрема BERT і Sentence-BERT, забезпечують суттєво вищу якість семантичного аналізу тексту, однак також мають низку обмежень"
NEW_P252 = "зокрема BERT, Sentence-BERT, BGE-M3 та E5-base, забезпечують суттєво вищу якість семантичного аналізу тексту, однак також мають низку обмежень"
ok = replace_in_para(paras[252], OLD_P252, NEW_P252)
changes.append(f"P252 domain: {'OK' if ok else 'MISS'}")

# ── 4. Section 3 – literature review ──────────────────────────────────────────
# P279: Add new generation models to the evolution summary
OLD_P279 = "Отже, аналіз основних джерел показує еволюцію моделей векторних ембеддінгів від Word2Vec, TF-IDF і FastText до BERT і Sentence-BERT."
NEW_P279 = "Отже, аналіз основних джерел показує еволюцію моделей векторних ембеддінгів від Word2Vec, TF-IDF і FastText до BERT і Sentence-BERT, а далі — до нового покоління мультилінгвальних моделей, зокрема BGE-M3, E5-base, nomic-embed-text та Qwen3-Embedding."
ok = replace_in_para(paras[279], OLD_P279, NEW_P279)
changes.append(f"P279 lit-review: {'OK' if ok else 'MISS'}")

# P293: Update model list in literature assessment
OLD_P293 = ("Роботи, присвячені Word2Vec, TF-IDF, FastText, BERT і Sentence-BERT, відображають еволюцію підходів від статистичних і статичних моделей слів до контекстних моделей речень і документів")
NEW_P293 = ("Роботи, присвячені Word2Vec, TF-IDF, FastText, BERT і Sentence-BERT, а також новому поколінню моделей BGE-M3, E5-base, nomic-embed-text та Qwen3-Embedding, "
            "відображають еволюцію підходів від статистичних і статичних моделей слів до контекстних і мультилінгвальних моделей речень і документів")
ok = replace_in_para(paras[293], OLD_P293, NEW_P293)
changes.append(f"P293 lit-review: {'OK' if ok else 'MISS'}")

# P295: Add specific model names to novelty statement
OLD_P295 = ("Новизна дослідження полягає у комплексному розгляді моделей векторних ембеддінгів з позиції їх використання у задачах семантичного пошуку текстових даних у корпоративних базах знань")
NEW_P295 = ("Новизна дослідження полягає у комплексному розгляді сучасних моделей векторних ембеддінгів — BGE-M3, E5-base, nomic-embed-text, Qwen3-Embedding та text-embedding-3-large — "
            "з позиції їх використання у задачах семантичного пошуку текстових даних у корпоративних базах знань")
ok = replace_in_para(paras[295], OLD_P295, NEW_P295)
changes.append(f"P295 novelty: {'OK' if ok else 'MISS'}")

# ── 5. Section 4 – problem statement ─────────────────────────────────────────
# P301: Change "TF-IDF, Word2Vec, FastText, BERT і Sentence-BERT" to new models
OLD_P301 = ("як класичні підходи до представлення текстових даних, так і сучасні контекстні моделі, "
            "зокрема TF-IDF, Word2Vec, FastText, BERT і Sentence-BERT. "
            "Такий вибір дозволяє охопити основні типи моделей, що відрізняються принципом побудови векторних подань, рівнем урахування контексту та придатністю до практичного використання у системах пошуку")
NEW_P301 = ("сучасні нейронні моделі векторних ембеддінгів, зокрема BGE-M3, E5-base, nomic-embed-text, "
            "Qwen3-Embedding та text-embedding-3-large. "
            "Такий вибір дозволяє охопити репрезентативний набір актуальних рішень, що відрізняються архітектурою, способом побудови векторних подань, мультилінгвальними можливостями та придатністю до практичного використання у системах пошуку")
ok = replace_in_para(paras[301], OLD_P301, NEW_P301)
changes.append(f"P301 prob-stmt: {'OK' if ok else 'MISS'}")

# P308: Change model list in methods section
OLD_P308 = ("До розгляду включаються як класичні статистичні підходи, так і сучасні контекстні моделі, "
            "зокрема TF-IDF, Word2Vec, FastText, BERT та Sentence-BERT. "
            "Вибір саме таких моделей обумовлений тим, що вони представляють різні підходи до векторизації тексту: від лексико-статистичних до контекстно-залежних.")
NEW_P308 = ("До розгляду включаються сучасні нейронні моделі ембеддінгів, зокрема BGE-M3, E5-base, nomic-embed-text, "
            "Qwen3-Embedding та text-embedding-3-large. "
            "Вибір саме таких моделей обумовлений тим, що вони представляють провідні відкриті та комерційні рішення для семантичного пошуку, "
            "що різняться архітектурою, підходами до формування подань і мультилінгвальними можливостями.")
ok = replace_in_para(paras[308], OLD_P308, NEW_P308)
changes.append(f"P308 methods: {'OK' if ok else 'MISS'}")

# ── Save ───────────────────────────────────────────────────────────────────────
tree.write(str(DOC_XML), encoding="unicode", xml_declaration=False)
print("Saved document.xml")

for ch in changes:
    print(f"  {ch}")

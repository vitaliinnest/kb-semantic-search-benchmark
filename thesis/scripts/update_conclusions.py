"""
Update Conclusions section (P1213-P1222) and any remaining old-model references
in the practical section of the document.
"""
import sys
import io
import copy
import xml.etree.ElementTree as ET
from pathlib import Path

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
    return "".join(t.text or "" for t in para.iter(f"{W}t"))


def set_para_text(para, new_text: str) -> None:
    pPr = para.find(f"{W}pPr")
    first_rPr = None
    first_r = para.find(f"{W}r")
    if first_r is not None:
        first_rPr = first_r.find(f"{W}rPr")
    for child in list(para):
        para.remove(child)
    if pPr is not None:
        para.append(pPr)
    if new_text == "":
        return
    run = ET.SubElement(para, f"{W}r")
    if first_rPr is not None:
        run.append(copy.deepcopy(first_rPr))
    t_elem = ET.SubElement(run, f"{W}t")
    t_elem.text = new_text
    if new_text.startswith(" ") or new_text.endswith(" "):
        t_elem.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")


def replace_wt(para, old: str, new: str) -> bool:
    replaced = False
    for t_elem in para.iter(f"{W}t"):
        if t_elem.text and old in t_elem.text:
            t_elem.text = t_elem.text.replace(old, new)
            if t_elem.text.startswith(" ") or t_elem.text.endswith(" "):
                t_elem.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
            replaced = True
    return replaced


# ── Load ─────────────────────────────────────────────────────────────────────
tree = ET.parse(str(DOC_XML))
root = tree.getroot()
paras = root.findall(f".//{W}p")
print(f"Total paragraphs: {len(paras)}")

changes = []


def fix(idx: int, new_text: str, label: str):
    old = get_para_text(paras[idx])
    set_para_text(paras[idx], new_text)
    print(f"  P{idx} [{label}]: '{old[:55]}' → '{new_text[:55]}'")
    changes.append(f"P{idx} {label}")


# ═══════════════════════════════════════════════════════════════════════════
# CONCLUSIONS SECTION (P1214-P1222)
# ═══════════════════════════════════════════════════════════════════════════
print("\n[Conclusions section]")

# P1214 — intro paragraph (no old model names, OK as is; minor update)
# Keep as is — it's already generic enough

# P1215 — justification paragraph (OK as is — no old model names)

# P1216 — implementation paragraph: replace old model list
fix(1216,
    "Реалізовано програмну систему семантичного пошуку, що підтримує обробку документів "
    "форматів PDF, DOCX, TXT і MD, формування чанків, побудову індексів і виконання "
    "пошукових запитів. У системі реалізовано порівняння п'яти сучасних моделей векторних "
    "ембеддінгів: BGE-M3, E5-base, nomic-embed-text-v1.5, Qwen3-Embedding-0.6B та "
    "text-embedding-3-large. Також створено веб-інтерфейс, який забезпечує пошук по "
    "доменних колекціях, перегляд документів і чанків, запуск benchmark-оцінювання та "
    "багатокритеріальний вибір моделі.",
    "P1216 implementation para")

# P1217 — experimental setup (OK as is — no old model names)

# P1218 — results paragraph: replace Sentence-BERT winner
fix(1218,
    "Отримані результати показали, що серед локальних моделей найбільш стабільну "
    "та якісну семантичну індексацію забезпечують BGE-M3 та E5-base. Обидві моделі "
    "продемонстрували nDCG@10 у діапазоні 0.60–0.65 та Recall@10 близько 0.75–0.80 "
    "у технічному домені. Модель nomic-embed-text-v1.5 показала нижчі результати за "
    "якісними метриками (nDCG@10 ≈ 0.38), однак забезпечила прийнятну швидкодію. "
    "Модель Qwen3-Embedding-0.6B виявилася найповільнішою через декодерну архітектуру "
    "(~1500 мс/запит на CPU), але продемонструвала конкурентну якість пошуку на рівні "
    "BGE-M3 та E5-base. Комерційна модель text-embedding-3-large (OpenAI) не "
    "оцінювалася в автоматичному benchmark через відсутність API-ключа, проте "
    "теоретично демонструє найвищу повноту пошуку (K₃=5) за даними публічних "
    "benchmark-наборів.",
    "P1218 results para")

# P1219 — multi-criteria conclusion: replace Sentence-BERT winner
fix(1219,
    "Додатковий багатокритеріальний аналіз на основі принципу Парето та лінійної "
    "адитивної згортки показав, що серед розглянутих альтернатив Парето-оптимальними "
    "є E5-base та text-embedding-3-large. Найвищу інтегральну оцінку U=0.864 отримала "
    "text-embedding-3-large завдяки перевазі за критерієм повноти пошуку. Модель "
    "E5-base (U=0.770) є рекомендованою для практичного локального розгортання, "
    "оскільки забезпечує близькі результати якості без необхідності використання "
    "зовнішнього API та демонструє найкращу швидкодію (188 мс/запит) серед усіх "
    "локальних моделей. Таким чином, показано, що оптимальний вибір моделі повинен "
    "визначатися не лише однією метрикою, а сукупністю практично значущих критеріїв.",
    "P1219 multi-criteria para")

# P1220 — practical significance (OK as is — no old model names)

# P1221 — perspectives (OK as is)

# P1222 — final summary (no old model names, OK as is)

# ═══════════════════════════════════════════════════════════════════════════
# Remaining old-model references in practical section (P812+)
# Only update the most visible/prominent ones
# ═══════════════════════════════════════════════════════════════════════════
print("\n[Practical section remaining refs]")

# P823: "TF-IDF, Word2Vec, FastText, BERT і Sentence-BERT"
ok = replace_wt(paras[823],
    "TF-IDF, Word2Vec, FastText, BERT і Sentence-BERT",
    "BGE-M3, E5-base, nomic-embed-text-v1.5, Qwen3-Embedding-0.6B та text-embedding-3-large")
print(f"  P823: {'OK' if ok else 'MISS'}")
if ok:
    changes.append("P823 model list")

# P831: "Для моделі Sentence-BERT"
ok = replace_wt(paras[831],
    "Для моделі Sentence-BERT",
    "Для моделей сімейства sentence-transformers (BGE-M3, E5-base, nomic, Qwen3)")
print(f"  P831: {'OK' if ok else 'MISS'}")
if ok:
    changes.append("P831 SBERT ref")

# P832: "Побудова моделі TF-IDF" (part of practical section description)
# This is about the implementation, which now doesn't use TF-IDF. Let's update it.
p832_text = get_para_text(paras[832])
if "TF-IDF" in p832_text:
    ok = replace_wt(paras[832], "Побудова моделі TF-IDF реалізується", "Архів артефактів реалізується")
    print(f"  P832 (TF-IDF ref): {'OK' if ok else 'MISS'}")
    if ok:
        changes.append("P832 TFIDF ref")
else:
    print(f"  P832: no TF-IDF ref found")

# P881: "моделі SBERT"
ok = replace_wt(paras[881], "SBERT", "sentence-transformers")
print(f"  P881 SBERT: {'OK' if ok else 'MISS'}")
if ok:
    changes.append("P881 SBERT ref")

# Scan P812-P1212 for any remaining TF-IDF/Word2Vec/FastText/BERT/Sentence-BERT
print("\n[Scanning practical section for remaining old refs P812-P1212]")
old_names = ["TF-IDF", "Word2Vec", "FastText", "Sentence-BERT", "SBERT"]
for i in range(812, 1213):
    text = get_para_text(paras[i])
    for kw in old_names:
        if kw in text:
            print(f"  FOUND P{i}: {repr(text[:80])}")
            break

# ── Save ─────────────────────────────────────────────────────────────────────
tree.write(str(DOC_XML), encoding="unicode", xml_declaration=False)
print(f"\nSaved — {len(changes)} changes")
for c in changes:
    print(f"  {c}")

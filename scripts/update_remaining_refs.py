"""
Fix remaining old-model references in sections 1-5:
  - TOC entries P171-P175 (subsection titles)
  - P92  (task definition — вихідні дані)
  - P316 (limitations — listed old model names)
  - P451 (multi-criteria section — listed old model names)
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
    """Strip all runs/math from paragraph, add a single plain-text run."""
    pPr = para.find(f"{W}pPr")
    first_rPr = None
    first_r = para.find(f"{W}r")
    if first_r is not None:
        first_rPr = first_r.find(f"{W}rPr")
    for child in list(para):
        para.remove(child)
    if pPr is not None:
        para.append(pPr)
    run = ET.SubElement(para, f"{W}r")
    if first_rPr is not None:
        run.append(copy.deepcopy(first_rPr))
    t_elem = ET.SubElement(run, f"{W}t")
    t_elem.text = new_text
    if new_text.startswith(" ") or new_text.endswith(" "):
        t_elem.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")


def replace_in_para(para, old: str, new: str) -> bool:
    text = get_para_text(para)
    if old not in text:
        return False
    set_para_text(para, text.replace(old, new))
    return True


def replace_wt_in_para(para, old: str, new: str) -> bool:
    """Replace text in-place within w:t elements (preserves paragraph XML structure)."""
    replaced = False
    for t_elem in para.iter(f"{W}t"):
        if t_elem.text and old in t_elem.text:
            t_elem.text = t_elem.text.replace(old, new)
            # Set xml:space="preserve" if needed
            if t_elem.text.startswith(" ") or t_elem.text.endswith(" "):
                t_elem.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
            replaced = True
    return replaced


# ── Load ────────────────────────────────────────────────────────────────────────
tree = ET.parse(str(DOC_XML))
root = tree.getroot()
paras = root.findall(f".//{W}p")
print(f"Total paragraphs: {len(paras)}")

changes = []

# ═══════════════════════════════════════════════════════════════════════════════
# 1.  TOC entries  P171-P175  — update title text in-place (preserve hyperlinks)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[TOC subsection titles]")

TOC_UPDATES = {
    171: ("Використання TF-IDF у задачах обробки текстів",
          "BGE-M3 як засіб мультилінгвального семантичного пошуку"),
    172: ("Модель Word2Vec як засіб векторного подання слів",
          "Модель E5-base для контрастивного семантичного пошуку"),
    173: ("Модель FastText для врахування підсловної інформації",
          "nomic-embed-text-v1.5 із підтримкою довгого контексту та Matryoshka-подань"),
    174: ("Використання BERT у задачах семантичного аналізу тексту",
          "Qwen3-Embedding-0.6B як декодерна модель векторних ембеддінгів"),
    175: ("Застосування Sentence-BERT у семантичному пошуку",
          "text-embedding-3-large як комерційна хмарна модель ембеддінгів"),
}

for idx, (old_title, new_title) in TOC_UPDATES.items():
    ok = replace_wt_in_para(paras[idx], old_title, new_title)
    status = "OK" if ok else "MISS"
    print(f"  P{idx} TOC: [{status}] '{old_title[:50]}' → '{new_title[:50]}'")
    changes.append(f"P{idx} TOC: {status}")

# ═══════════════════════════════════════════════════════════════════════════════
# 2.  P92 — task definition (вихідні дані): replace old model list
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[P92 — task definition]")
OLD_P92 = "моделі векторного представлення текстової інформації TF-IDF, Word2Vec, FastText, BERT та Sentence-BERT, значення критеріїв оцінювання ефективності семантичного пошуку, вагові коефіцієнти критеріїв, а також результати нормалізації та зважування показників для виконання багатокритеріального вибору оптимальної моделі."
NEW_P92 = "моделі векторних ембеддінгів BGE-M3, E5-base, nomic-embed-text-v1.5, Qwen3-Embedding-0.6B та text-embedding-3-large, значення критеріїв оцінювання ефективності семантичного пошуку, вагові коефіцієнти критеріїв, а також результати нормалізації та зважування показників для виконання багатокритеріального вибору оптимальної моделі."
ok = replace_in_para(paras[92], OLD_P92, NEW_P92)
print(f"  P92: {'OK' if ok else 'MISS'}")
changes.append(f"P92 task-def: {'OK' if ok else 'MISS'}")

# ═══════════════════════════════════════════════════════════════════════════════
# 3.  P316 — limitations section: replace old model list
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[P316 — limitations]")
OLD_P316 = "набором моделей, включених до аналізу, а саме TF-IDF, Word2Vec, FastText, BERT і Sentence-BERT. Отже, підсумковий вибір оптимальної моделі здійснюється лише серед зазначених альтернатив і не охоплює всі можливі сучасні моделі ембеддінгів."
NEW_P316 = "набором моделей, включених до аналізу, а саме BGE-M3, E5-base, nomic-embed-text-v1.5, Qwen3-Embedding-0.6B та text-embedding-3-large. Отже, підсумковий вибір оптимальної моделі здійснюється лише серед зазначених альтернатив і не охоплює всі можливі сучасні моделі ембеддінгів."
ok = replace_in_para(paras[316], OLD_P316, NEW_P316)
print(f"  P316: {'OK' if ok else 'MISS'}")
changes.append(f"P316 limitations: {'OK' if ok else 'MISS'}")

# ═══════════════════════════════════════════════════════════════════════════════
# 4.  P451 — multi-criteria section: replace old model list
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[P451 — multi-criteria alternatives]")
OLD_P451 = "як альтернативи розглядаються моделі TF-IDF, Word2Vec, FastText, BERT і Sentence-BERT, які реалізують різні підходи до векторного подання тексту"
NEW_P451 = "як альтернативи розглядаються сучасні моделі векторних ембеддінгів BGE-M3, E5-base, nomic-embed-text-v1.5, Qwen3-Embedding-0.6B та text-embedding-3-large, які реалізують різні підходи до нейронного векторного подання тексту"
ok = replace_in_para(paras[451], OLD_P451, NEW_P451)
print(f"  P451: {'OK' if ok else 'MISS'}")
changes.append(f"P451 mc-section: {'OK' if ok else 'MISS'}")

# ── Save ────────────────────────────────────────────────────────────────────────
tree.write(str(DOC_XML), encoding="unicode", xml_declaration=False)
print(f"\nSaved document.xml — {len(changes)} changes")
for ch in changes:
    print(f"  {ch}")

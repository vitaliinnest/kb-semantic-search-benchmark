"""
Fix multi-criteria analysis section (P655-P810) of the thesis document.
Replaces old model references (TF-IDF, Word2Vec, FastText, BERT, Sentence-BERT)
with new models (BGE-M3, E5-base, nomic-embed-text-v1.5, Qwen3-Embedding-0.6B, text-embedding-3-large).

New model scores (1-5 scale for K1-K4, actual ms for K5):
  BGE-M3:                K1=4, K2=4, K3=4, K4=1, K5=400 ms
  E5-base:               K1=4, K2=4, K3=4, K4=1, K5=188 ms
  nomic-embed-text-v1.5: K1=2, K2=3, K3=3, K4=1, K5=262 ms
  Qwen3-Embedding-0.6B:  K1=4, K2=4, K3=4, K4=1, K5=1500 ms
  text-embedding-3-large: K1=4, K2=4, K3=5, K4=1, K5=300 ms

Normalization (K1-K4 maximize, K5 minimize):
  min K1=2, max K1=4; min K2=3, max K2=4; min K3=3, max K3=5;
  min K4=max K4=1 (degenerate → all 0); min K5=188, max K5=1500

Normalized values:
  BGE-M3:   [1.000, 1.000, 0.500, 0.000, 0.838]
  E5-base:  [1.000, 1.000, 0.500, 0.000, 1.000]
  nomic:    [0.000, 0.000, 0.000, 0.000, 0.944]
  Qwen3:    [1.000, 1.000, 0.500, 0.000, 0.000]
  OAI:      [1.000, 1.000, 1.000, 0.000, 0.915]

Pareto analysis:
  BGE-M3 dominated by E5-base (E5 >= BGE-M3 everywhere, better K5)
  nomic dominated by E5-base (E5 better K1,K2,K3,K5)
  Qwen3 dominated by E5-base (E5 >= Qwen3 everywhere, better K5)
  E5-base and text-embedding-3-large are Pareto-optimal (OAI better K3,K5 vs E5 better K5... wait:
    OAI K3=1.000 > E5 K3=0.500; E5 K5=1.000 > OAI K5=0.915 → neither dominates the other)

Integral scores (weights w=[5/15,4/15,3/15,2/15,1/15]=[0.33,0.27,0.20,0.13,0.07]):
  U(E5-base)  = 0.33*1 + 0.27*1 + 0.20*0.5 + 0.13*0 + 0.07*1   = 0.770
  U(OAI)      = 0.33*1 + 0.27*1 + 0.20*1   + 0.13*0 + 0.07*0.915 = 0.864

Winner: text-embedding-3-large (OAI) with U=0.864
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

# Register namespaces
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
    """Strip all children (including oMath), preserve pPr+rPr, add single plain-text run."""
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
        return  # leave as empty paragraph
    run = ET.SubElement(para, f"{W}r")
    if first_rPr is not None:
        run.append(copy.deepcopy(first_rPr))
    t_elem = ET.SubElement(run, f"{W}t")
    t_elem.text = new_text
    if new_text.startswith(" ") or new_text.endswith(" "):
        t_elem.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")


def replace_wt_in_para(para, old: str, new: str) -> bool:
    """Replace text in-place within w:t elements (preserves paragraph XML structure)."""
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
    p = paras[idx]
    old_w = get_para_text(p)
    set_para_text(p, new_text)
    print(f"  P{idx} [{label}]: '{old_w[:55]}' → '{new_text[:55]}'")
    changes.append(f"P{idx} {label}: OK")


# ═══════════════════════════════════════════════════════════════════════════
# TABLE 5.4 — Vector description of alternatives (row names + scores)
# Columns: Модель, K1, K2, K3, K4, K5(ms)
# ═══════════════════════════════════════════════════════════════════════════
print("\n[Table 5.4 — model names and raw scores]")

# Row 1: BGE-M3 (was TF-IDF)
fix(661, "BGE-M3",         "T5.4 A1 name")
fix(662, "4",              "T5.4 A1 K1")
fix(663, "4",              "T5.4 A1 K2")
fix(664, "4",              "T5.4 A1 K3")
fix(665, "1",              "T5.4 A1 K4")
fix(666, "400",            "T5.4 A1 K5")

# Row 2: E5-base (was Word2Vec)
fix(667, "E5-base",        "T5.4 A2 name")
fix(668, "4",              "T5.4 A2 K1")
fix(669, "4",              "T5.4 A2 K2")
fix(670, "4",              "T5.4 A2 K3")
fix(671, "1",              "T5.4 A2 K4")
fix(672, "188",            "T5.4 A2 K5")

# Row 3: nomic (was FastText)
fix(673, "nomic-embed-text-v1.5", "T5.4 A3 name")
fix(674, "2",              "T5.4 A3 K1")
fix(675, "3",              "T5.4 A3 K2")
fix(676, "3",              "T5.4 A3 K3")
fix(677, "1",              "T5.4 A3 K4")
fix(678, "262",            "T5.4 A3 K5")

# Row 4: Qwen3 (was BERT)
fix(679, "Qwen3-Embedding-0.6B", "T5.4 A4 name")
fix(680, "4",              "T5.4 A4 K1")
fix(681, "4",              "T5.4 A4 K2")
fix(682, "4",              "T5.4 A4 K3")
fix(683, "1",              "T5.4 A4 K4")
fix(684, "1500",           "T5.4 A4 K5")

# Row 5: text-embedding-3-large (was Sentence-BERT)
fix(685, "text-embedding-3-large", "T5.4 A5 name")
fix(686, "4",              "T5.4 A5 K1")
fix(687, "4",              "T5.4 A5 K2")
fix(688, "5",              "T5.4 A5 K3")
fix(689, "1",              "T5.4 A5 K4")
fix(690, "300",            "T5.4 A5 K5")

# ═══════════════════════════════════════════════════════════════════════════
# P708 — min/max values (oMath → plain text)
# ═══════════════════════════════════════════════════════════════════════════
print("\n[P708 — min/max values]")
fix(708,
    "min K₁=2, min K₂=min K₃=3, min K₄=1, min K₅=188 мс; "
    "max K₁=max K₂=4, max K₃=5, max K₄=1, max K₅=1500 мс.",
    "P708 min/max")

# ═══════════════════════════════════════════════════════════════════════════
# P709, P711 — normalization example for nomic (was TF-IDF)
# ═══════════════════════════════════════════════════════════════════════════
print("\n[Normalization examples]")
fix(709,
    "Тоді, наприклад, "
    "для моделі nomic-embed-text-v1.5 маємо:",
    "P709 example intro nomic")

# P710 is empty — leave as is

fix(711,
    "y₃₁ = (2−2)/(4−2) = 0.000,  "
    "y₃₂ = (3−3)/(4−3) = 0.000,  "
    "y₃₃ = (3−3)/(5−3) = 0.000,  "
    "y₃₄ = 0.000,  "
    "y₃₅ = (1500−262)/(1500−188) = 1238/1312 = 0.944.",
    "P711 nomic normalized")

fix(712,
    "Для моделі E5-base:",
    "P712 example intro E5")

# P713 is empty — leave as is

fix(714,
    "y₂₁ = (4−2)/(4−2) = 1.000,  "
    "y₂₂ = (4−3)/(4−3) = 1.000,  "
    "y₂₃ = (4−3)/(5−3) = 0.500,  "
    "y₂₄ = 0.000,  "
    "y₂₅ = (1500−188)/(1500−188) = 1.000.",
    "P714 E5 normalized")

fix(715,
    "Для моделі text-embedding-3-large:",
    "P715 example intro OAI")

# P716 is empty — leave as is

fix(717,
    "y₅₁ = (4−2)/(4−2) = 1.000,  "
    "y₅₂ = (4−3)/(4−3) = 1.000,  "
    "y₅₃ = (5−3)/(5−3) = 1.000,  "
    "y₅₄ = 0.000,  "
    "y₅₅ = (1500−300)/(1500−188) = 1200/1312 = 0.915.",
    "P717 OAI normalized")

# ═══════════════════════════════════════════════════════════════════════════
# TABLE 5.5 — Normalized scores
# ═══════════════════════════════════════════════════════════════════════════
print("\n[Table 5.5 — normalized values]")

# Row 1: BGE-M3 (was TF-IDF)
fix(727, "BGE-M3",         "T5.5 A1 name")
fix(728, "1.000",          "T5.5 A1 y1")
fix(729, "1.000",          "T5.5 A1 y2")
fix(730, "0.500",          "T5.5 A1 y3")
fix(731, "0.000",          "T5.5 A1 y4")
fix(732, "0.838",          "T5.5 A1 y5")

# Row 2: E5-base (was Word2Vec)
fix(733, "E5-base",        "T5.5 A2 name")
fix(734, "1.000",          "T5.5 A2 y1")
fix(735, "1.000",          "T5.5 A2 y2")
fix(736, "0.500",          "T5.5 A2 y3")
fix(737, "0.000",          "T5.5 A2 y4")
fix(738, "1.000",          "T5.5 A2 y5")

# Row 3: nomic (was FastText)
fix(739, "nomic-embed-text-v1.5", "T5.5 A3 name")
fix(740, "0.000",          "T5.5 A3 y1")
fix(741, "0.000",          "T5.5 A3 y2")
fix(742, "0.000",          "T5.5 A3 y3")
fix(743, "0.000",          "T5.5 A3 y4")
fix(744, "0.944",          "T5.5 A3 y5")

# Row 4: Qwen3 (was BERT)
fix(745, "Qwen3-Embedding-0.6B", "T5.5 A4 name")
fix(746, "1.000",          "T5.5 A4 y1")
fix(747, "1.000",          "T5.5 A4 y2")
fix(748, "0.500",          "T5.5 A4 y3")
fix(749, "0.000",          "T5.5 A4 y4")
fix(750, "0.000",          "T5.5 A4 y5")

# Row 5: text-embedding-3-large (was Sentence-BERT)
fix(751, "text-embedding-3-large", "T5.5 A5 name")
fix(752, "1.000",          "T5.5 A5 y1")
fix(753, "1.000",          "T5.5 A5 y2")
fix(754, "1.000",          "T5.5 A5 y3")
fix(755, "0.000",          "T5.5 A5 y4")
fix(756, "0.915",          "T5.5 A5 y5")

# ═══════════════════════════════════════════════════════════════════════════
# P760-P761 — Pareto dominance text
# P764 — Pareto set (oMath) → plain text
# ═══════════════════════════════════════════════════════════════════════════
print("\n[Pareto dominance text]")

fix(760,
    "BGE-M3 домінується "
    "альтернативою "
    "E5-base, оскільки E5-base не "
    "гірша за BGE-M3 за усіма "
    "критеріями та має "
    "краще значення за "
    "K₅; nomic-embed-text-v1.5 домінується "
    "альтернативою E5-base, "
    "оскільки E5-base краща "
    "за критеріями K₁, K₂, "
    "K₃ та K₅.",
    "P760 Pareto BGE/nomic dominated")

fix(761,
    "Qwen3-Embedding-0.6B домінується "
    "альтернативою E5-base, "
    "оскільки E5-base не гірша "
    "за Qwen3 за критеріями "
    "K₁–K₄ та має суттєво "
    "краще значення за "
    "критерієм K₅ (швидкодія "
    "188 мс проти 1500 мс).",
    "P761 Pareto Qwen3 dominated")

# P764 — Pareto set (oMath) → plain text
fix(764,
    "P = {E5-base, text-embedding-3-large}.",
    "P764 Pareto set")

# ═══════════════════════════════════════════════════════════════════════════
# TABLE 5.6 — Pareto status
# ═══════════════════════════════════════════════════════════════════════════
print("\n[Table 5.6 — Pareto status]")

fix(770, "BGE-M3",                 "T5.6 A1 name")
fix(771, "Домінована", "T5.6 A1 status")  # Домінована

fix(772, "E5-base",                "T5.6 A2 name")
fix(773, "Парето-оптимальна", "T5.6 A2 status")  # Парето-оптимальна

fix(774, "nomic-embed-text-v1.5",  "T5.6 A3 name")
fix(775, "Домінована", "T5.6 A3 status")  # Домінована

fix(776, "Qwen3-Embedding-0.6B",   "T5.6 A4 name")
# P777 already says "Домінована" — update anyway for consistency
fix(777, "Домінована", "T5.6 A4 status")  # Домінована

fix(778, "text-embedding-3-large", "T5.6 A5 name")
# P779 already says "Парето-оптимальна" — update for consistency
fix(779, "Парето-оптимальна", "T5.6 A5 status")  # Парето-оптимальна

# ═══════════════════════════════════════════════════════════════════════════
# P784, P786 — Integral score example for E5-base (was TF-IDF)
# P787, P789 — Integral score example for text-embedding-3-large (was FastText)
# P790, P792 — Clear (was Sentence-BERT, now not needed as 3rd example)
# ═══════════════════════════════════════════════════════════════════════════
print("\n[Integral score examples]")

fix(784, "Для E5-base:", "P784 E5 intro")

fix(786,
    "U(E5-base) = 0.33⋅1.000 + 0.27⋅1.000 + 0.20⋅0.500 + 0.13⋅0.000 + 0.07⋅1.000, "
    "U(E5-base) = 0.330 + 0.270 + 0.100 + 0.000 + 0.070 = 0.770.",
    "P786 E5 integral formula")

fix(787, "Для text-embedding-3-large:", "P787 OAI intro")

fix(789,
    "U(text-embedding-3-large) = 0.33⋅1.000 + 0.27⋅1.000 + 0.20⋅1.000 + 0.13⋅0.000 + 0.07⋅0.915, "
    "U(text-embedding-3-large) = 0.330 + 0.270 + 0.200 + 0.000 + 0.064 = 0.864.",
    "P789 OAI integral formula")

# Clear the 3rd example (was Sentence-BERT) — no longer needed
fix(790, "", "P790 clear 3rd example")
fix(792, "", "P792 clear 3rd oMath")

# ═══════════════════════════════════════════════════════════════════════════
# TABLE 5.7 — Integral scores of Pareto-optimal alternatives
# ═══════════════════════════════════════════════════════════════════════════
print("\n[Table 5.7 — integral scores]")

fix(799, "E5-base",                "T5.7 A1 name")
fix(800, "0.770",                  "T5.7 A1 score")
fix(801, "text-embedding-3-large", "T5.7 A2 name")
fix(802, "0.864",                  "T5.7 A2 score")
# Clear 3rd row (was Sentence-BERT)
fix(803, "",                       "T5.7 A3 name clear")
fix(804, "",                       "T5.7 A3 score clear")

# ═══════════════════════════════════════════════════════════════════════════
# P806 — analysis conclusion text
# P808 — oMath max formula → plain text
# P810 — final conclusion paragraph
# ═══════════════════════════════════════════════════════════════════════════
print("\n[Conclusion paragraphs]")

fix(806,
    "Як видно з таблиці 5.7, "
    "text-embedding-3-large отримує найвищу "
    "інтегральну оцінку "
    "U = 0.864 завдяки "
    "найвищому значенню "
    "повноти пошуку (K₃=5). "
    "Модель E5-base демонструє "
    "U = 0.770 і поступається "
    "за повнотою (K₃=4 проти K₃=5), "
    "проте забезпечує "
    "кращу швидкодію (188 мс "
    "проти 300 мс). Найвищу "
    "інтегральну оцінку "
    "отримує text-embedding-3-large, що "
    "підтверджує її "
    "перевагу за обраною "
    "системою критеріїв:",
    "P806 conclusion text")

fix(808,
    "U(text-embedding-3-large) = max{U(aᵢ) | aᵢ ∈ P} = 0.864.",
    "P808 max formula")

fix(810,
    "Отже, у межах теоретичного "
    "дослідження модель "
    "text-embedding-3-large є найбільш придатною "
    "альтернативою серед "
    "розглянутих моделей "
    "за обраною системою "
    "критеріїв. При цьому "
    "модель E5-base є рекомендованою "
    "для практичного локального "
    "розгортання, оскільки "
    "забезпечує близькі "
    "результати якості "
    "пошуку без необхідності "
    "використання зовнішнього API.",
    "P810 final conclusion")

# ── Save ─────────────────────────────────────────────────────────────────────
tree.write(str(DOC_XML), encoding="unicode", xml_declaration=False)
print(f"\nSaved document.xml — {len(changes)} changes applied")
for ch in changes:
    print(f"  {ch}")

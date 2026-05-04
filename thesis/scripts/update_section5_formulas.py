"""
Fix MATH paragraphs in Section 5 of the thesis document.
Replaces old-model formulas (TF-IDF, Word2Vec, FastText, BERT, Sentence-BERT)
with correct formulas for new models (BGE-M3, E5-base, nomic, Qwen3, text-embedding-3-large).
Also fills in table score cells and updates alternative labels A1-A5.
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

# Register all namespaces to avoid ns0/ns1/ns2 prefixes
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
    parts = []
    for t in para.iter(f"{W}t"):
        parts.append(t.text or "")
    return "".join(parts)


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


# ── Load document ─────────────────────────────────────────────────────────────
tree = ET.parse(str(DOC_XML))
root = tree.getroot()
paras = root.findall(f".//{W}p")
print(f"Total paragraphs: {len(paras)}")

changes = []


def fix(idx: int, new_text: str, label: str):
    p = paras[idx]
    old = get_para_text(p)
    set_para_text(p, new_text)
    print(f"  P{idx} [{label}]: '{old[:60]}' → '{new_text[:60]}'")
    changes.append(f"P{idx} {label}: OK")


# ═══════════════════════════════════════════════════════════════════════════════
# BGE-M3 formulas  (P331, P333, P334, P337, P338)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[BGE-M3 formulas]")

fix(331,
    "Семантична схожість між запитом та документом у режимі щільного пошуку BGE-M3 "
    "обчислюється як косинусна подібність їх ембеддінгів і описується формулою 5.1:",
    "BGE-M3 dense intro")

fix(333,
    "sim(q, d) = (E_q · E_d) / (‖E_q‖ · ‖E_d‖)    (5.1)",
    "BGE-M3 dense formula")

fix(334,
    "де E_q — векторне подання запиту (query embedding); "
    "E_d — векторне подання документа (document embedding); "
    "‖·‖ — евклідова норма вектора.",
    "BGE-M3 dense vars")

fix(337,
    "s_sparse(q, d) = Σᵢ wᵢ(q) · wᵢ(d)    (5.2)",
    "BGE-M3 sparse formula")

fix(338,
    "де wᵢ(q), wᵢ(d) — ваги i-го токена у розрідженому поданні запиту q "
    "та документа d відповідно.",
    "BGE-M3 sparse vars")

# ═══════════════════════════════════════════════════════════════════════════════
# E5-base formulas  (P352, P353)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[E5-base formulas]")

fix(352,
    "ℒ_InfoNCE = −log[ exp(sim(q, d⁺)/τ) / "
    "Σⱼ₌₁ᵎ exp(sim(q, dⱼ)/τ) ]    (5.3)",
    "E5-base InfoNCE formula")

fix(353,
    "де q — запит; d⁺ — позитивний документ для q; "
    "dⱼ — j-й документ у батчі розміром N; "
    "τ — температурний параметр; "
    "sim(q, d) — косинусна подібність між ембеддінгами запиту та документа.",
    "E5-base InfoNCE vars")

# ═══════════════════════════════════════════════════════════════════════════════
# nomic-embed-text formulas  (P361, P363, P364)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[nomic formulas]")

fix(361,
    "Векторне подання тексту у nomic-embed-text-v1.5 формується через mean pooling "
    "прихованих станів трансформера з RoPE-позиційним кодуванням (формула 5.5):",
    "nomic intro")

fix(363,
    "e(x) = (1/T) Σₜ₌₁ᵀ hₜ    (5.5)",
    "nomic mean pooling formula")

fix(364,
    "де hₜ — прихований стан t-го токена в останньому шарі трансформера; "
    "T — кількість токенів у послідовності; "
    "e(x) — результуючий ембеддінг тексту x.",
    "nomic vars")

# ═══════════════════════════════════════════════════════════════════════════════
# Qwen3-Embedding formulas  (P373, P375, P376)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[Qwen3 formulas]")

fix(373,
    "У декодерній архітектурі Qwen3-Embedding ембеддінг тексту отримується через "
    "last-token pooling — зчитування прихованого стану останнього (EOS) токена "
    "послідовності, що описується формулою 5.6:",
    "Qwen3 intro")

fix(375,
    "e(x) = h_T⁽ᴸ⁾    (5.6)",
    "Qwen3 last-token formula")

fix(376,
    "де h_T⁽ᴸ⁾ — прихований стан останнього (T-го) токена "
    "у L-му (фінальному) шарі декодера; "
    "e(x) — ембеддінг тексту x.",
    "Qwen3 vars")

# ═══════════════════════════════════════════════════════════════════════════════
# text-embedding-3-large formulas  (P384, P386, P387, P388, P389, P390, P391)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[text-embedding-3-large formulas]")

fix(384,
    "Семантична схожість між запитом q та документом d для text-embedding-3-large "
    "обчислюється як косинусна подібність їхніх векторних подань (формула 5.7):",
    "OAI intro")

fix(386,
    "sim(q, d) = (E_q · E_d) / (‖E_q‖ · ‖E_d‖)    (5.7)",
    "OAI cosine formula")

fix(387,
    "де q, d — запит та документ;",
    "OAI var1")

fix(388,
    "     E_q, E_d — їхні ембеддінги, сформовані моделлю text-embedding-3-large;",
    "OAI var2 (no Sentence-BERT)")

fix(389,
    "     (E_q · E_d) — скалярний добуток векторних подань;",
    "OAI var3")

fix(390,
    "     ‖E_q‖, ‖E_d‖ — евклідові норми відповідних ембеддінгів;",
    "OAI var4")

fix(391,
    "     sim(q, d) — значення косинусної подібності між запитом та документом.",
    "OAI var5")

# ═══════════════════════════════════════════════════════════════════════════════
# Alternative labels  (P460, P461)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[Alternative labels A1-A5]")

fix(460,
    "де A₁ — BGE-M3;",
    "A1 label")

fix(461,
    "     A₂ — E5-base;     A₃ — nomic-embed-text-v1.5;     "
    "A₄ — Qwen3-Embedding-0.6B;     A₅ — text-embedding-3-large.",
    "A2-A5 labels")

# ═══════════════════════════════════════════════════════════════════════════════
# Table score cells — 5-point scores per criterion
# Criteria: 1=nDCG@10, 2=MRR@10, 3=Recall@10, 4=P@10, 5=Latency (speed)
# Scoring scale for metrics 1-4: 0.00-0.20→1, 0.21-0.40→2, 0.41-0.60→3,
#                                 0.61-0.80→4, 0.81-1.00→5
# Scoring for latency (lower=better): see document criteria section
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[Table scores]")

SCORES = {
    # (paragraph_index): score_string
    # BGE-M3: nDCG=4, MRR=4, Recall=4, P@10=1, Latency=2
    482: "4", 483: "4", 484: "4", 485: "1", 486: "2",
    # E5-base: nDCG=4, MRR=4, Recall=4, P@10=1, Latency=4
    488: "4", 489: "4", 490: "4", 491: "1", 492: "4",
    # nomic: nDCG=2, MRR=3, Recall=3, P@10=1, Latency=3
    494: "2", 495: "3", 496: "3", 497: "1", 498: "3",
    # Qwen3: nDCG=4, MRR=4, Recall=4, P@10=1, Latency=1
    500: "4", 501: "4", 502: "4", 503: "1", 504: "1",
    # text-embedding-3-large: nDCG=4, MRR=4, Recall=5, P@10=1, Latency=3
    506: "4", 507: "4", 508: "5", 509: "1", 510: "3",
}

for idx, score in SCORES.items():
    p = paras[idx]
    old_mt = [e.text or "" for e in p.iter("{http://schemas.openxmlformats.org/officeDocument/2006/math}t")]
    set_para_text(p, score)
    print(f"  P{idx}: m:t={old_mt!r} → '{score}'")
    changes.append(f"P{idx} score={score}: OK")

# ── Save ───────────────────────────────────────────────────────────────────────
tree.write(str(DOC_XML), encoding="unicode", xml_declaration=False)
print(f"\nSaved document.xml — {len(changes)} changes applied")
for ch in changes:
    print(f"  {ch}")

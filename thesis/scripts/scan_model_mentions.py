"""
Сканує згадки embedding-моделей у документі: де згадуються E5, BGE, nomic,
Qwen3, BM25, sentence-bert, openai тощо. Допомагає зіставити теоретичну
частину (розділ 2) з практичною (розділи 5-7).
"""
import sys, pathlib, xml.etree.ElementTree as ET
sys.stdout.reconfigure(encoding='utf-8')

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
M = "http://schemas.openxmlformats.org/officeDocument/2006/math"
def _w(t): return f"{{{W}}}{t}"
def _m(t): return f"{{{M}}}{t}"

UNPACKED = pathlib.Path("D:/repos/kb-semantic-search-benchmark/thesis/unpacked_docx")
tree = ET.parse(UNPACKED / "word" / "document.xml")
body = tree.getroot().find(_w("body"))

def all_text(p):
    parts = []
    for t in p.iter():
        if t.tag in (_w("t"), _m("t")):
            parts.append(t.text or "")
    return "".join(parts)

def get_style(p):
    ppr = p.find(_w("pPr"))
    if ppr is not None:
        ps = ppr.find(_w("pStyle"))
        if ps is not None:
            return ps.get(_w("val"), "")
    return ""

# Models to track
MODELS = {
    "E5":          ["E5", "intfloat", "multilingual-e5"],
    "BGE":         ["BGE", "bge-m3", "BAAI"],
    "nomic":       ["nomic", "Nomic"],
    "Qwen3":       ["Qwen3", "Qwen-3"],
    "BM25":        ["BM25"],
    "OpenAI":      ["OpenAI", "text-embedding-3", "text-embedding-ada"],
    "Word2Vec":    ["Word2Vec", "word2vec"],
    "GloVe":       ["GloVe"],
    "FastText":    ["FastText", "fastText"],
    "BERT":        ["BERT"],
    "Sentence-BERT": ["Sentence-BERT", "SBERT", "sentence-bert"],
    "ELMo":        ["ELMo"],
    "RoBERTa":     ["RoBERTa"],
    "DistilBERT":  ["DistilBERT"],
    "GPT":         ["GPT-3", "GPT-4"],
    "TF-IDF":      ["TF-IDF", "TFIDF", "tfidf"],
    "ColBERT":     ["ColBERT"],
    "DPR":         ["DPR"],
}

# Identify chapter boundaries by heading style/text
chapter_starts = {}
for i, p in enumerate(body):
    style = get_style(p)
    txt = all_text(p).strip()
    if style.startswith("1") or style == "Heading1" or style == "10":
        # heading-level paragraphs
        if txt.startswith("РОЗДІЛ"):
            chapter_starts[i] = txt[:60]

print("=== Розділи (за стилем) ===")
for i, t in sorted(chapter_starts.items()):
    print(f"  B{i}: {t}")

# Collect mentions
print()
print("=== Згадки моделей (B-індекс → модель) ===")
mentions = {model: [] for model in MODELS}
for i, p in enumerate(body):
    txt = all_text(p)
    if not txt:
        continue
    for model, keywords in MODELS.items():
        for kw in keywords:
            if kw in txt:
                mentions[model].append(i)
                break

for model, idxs in mentions.items():
    if not idxs:
        continue
    # group by chapter
    print(f"\n{model}: {len(idxs)} згадок")
    print(f"  індекси: {idxs[:30]}{'...' if len(idxs)>30 else ''}")

# Find chapter 2 mentions vs chapter 5+ mentions
print()
print("=== Розподіл по розділах ===")
ch_starts = sorted(chapter_starts.keys())
def chapter_of(idx):
    for j, s in enumerate(ch_starts):
        next_s = ch_starts[j+1] if j+1 < len(ch_starts) else 999999
        if s <= idx < next_s:
            return j+1
    return 0

for model, idxs in mentions.items():
    if not idxs:
        continue
    by_ch = {}
    for idx in idxs:
        ch = chapter_of(idx)
        by_ch.setdefault(ch, []).append(idx)
    print(f"{model:20s} " + " ".join(f"Р{ch}={len(v)}" for ch, v in sorted(by_ch.items())))

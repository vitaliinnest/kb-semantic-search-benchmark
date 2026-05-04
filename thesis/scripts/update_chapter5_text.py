"""
Тонкі правки тексту і таблиць у розділі 5 під актуальний застосунок:

  • B732: прибираємо речення про модель BERT (її в системі немає).
  • B900–B917: оновлюємо JSON-приклад benchmark-результатів — використано
    BGE-M3 з реальними числами, додано поля ndcg_ci_lo/ndcg_ci_hi.
  • B949: переформульовуємо «характерну проблему BERT» як загальну
    проблему щільних embedding-моделей.
  • Таблиці 5.8/5.9/5.10 (B962/B973/B985): оновлюємо колонку ms/query,
    щоб відповідала фактичним результатам останнього benchmark-запуску.
  • Після таблиці 5.10 додаємо абзац про bootstrap 95% CI як індикатор
    статистичної значущості переваги BGE-M3 над BM25.
"""
import sys, copy, shutil, zipfile, pathlib, xml.etree.ElementTree as ET
sys.stdout.reconfigure(encoding='utf-8')

ROOT = pathlib.Path("D:/repos/kb-semantic-search-benchmark")
DOCX = ROOT / "thesis" / "2026_M_PI_Nesterenko_VV.docx"
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
children = list(body)


# ── helper: write text into a paragraph, collapsing into first w:t ─────
def set_paragraph_text(p, new_text):
    t_elements = []
    for r in p.findall(_w("r")):
        for t in r.findall(_w("t")):
            t_elements.append(t)
    if not t_elements:
        return False
    t_elements[0].text = new_text
    if new_text != new_text.strip():
        t_elements[0].set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    for t in t_elements[1:]:
        t.text = ""
    return True


# ── 1. B732: прибираємо "Для моделі BERT…" ───────────────────────────
B732 = children[732]
old = ''.join((t.text or '') for t in B732.iter(_w('t')))
new = old.replace(
    " Для моделі BERT використовуються бібліотеки transformers і torch, які забезпечують завантаження трансформерних моделей та обчислення контекстних векторних представлень тексту.",
    " Лексична модель BM25 реалізована за допомогою бібліотеки rank_bm25, що забезпечує токенізацію і обчислення TF-IDF-подібних оцінок без необхідності у векторних поданнях."
)
if new != old:
    set_paragraph_text(B732, new)
    print("[1] B732: BERT mention replaced with BM25/rank_bm25 description.")
else:
    print("[1] B732: NOT MATCHED")


# ── 2. B900–B917: оновлюємо JSON-приклад ──────────────────────────────
JSON_LINES = [
    '{',
    '  "timestamp": "2026-04-27T13:55:12",',
    '  "top_k": 10,',
    '  "queries_count": 100,',
    '  "queries_file": "data/domains/tech/benchmark/queries.jsonl",',
    '  "qrels_file": "data/domains/tech/benchmark/qrels.jsonl",',
    '  "models": [',
    '    {',
    '      "model_name": "BGE-M3 (BAAI/bge-m3)",',
    '      "artifacts_dir": "artifacts/tech/bge-m3",',
    '      "queries_evaluated": 100,',
    '      "recall_at_k": 0.8150,',
    '      "mrr_at_k": 0.6993,',
    '      "ndcg_at_k": 0.6722,',
    '      "ndcg_ci_lo": 0.6059,',
    '      "ndcg_ci_hi": 0.7367,',
    '      "precision_at_k": 0.1430,',
    '      "avg_latency_ms": 2959.8',
    '    }',
    '  ]',
    '}',
]
# B900..B918 = 19 paragraphs (some empty separator). Let's update mapping:
# B900='{' B901..B915='lines' B916='}' B917=']' B918='}'
# Actually checking dump: B900..B918 has 19 paras. Our new content is 21 lines.
# So we either: (a) add new w:p or (b) compress. Let me compress.
# Strategy: rewrite each existing paragraph to a new line, append extras if needed.

# First find the exact range. B900..B918 => indices in current body.
json_start = 900
json_end_excl = 919  # 919 is empty separator after }
# Get list of <w:p> children in this range
paras = [children[i] for i in range(json_start, json_end_excl)]
# Take a reference paragraph from a non-empty one for cloning style
ref_p = next((p for p in paras if any(((t.text or '').strip()) for t in p.iter(_w('t')))), None)
if ref_p is None:
    print("[2] JSON: no reference paragraph found")
else:
    # Update each existing para's text in-order; if more lines than paras, clone.
    n_existing = len(paras)
    n_new = len(JSON_LINES)
    for i, line in enumerate(JSON_LINES):
        if i < n_existing:
            set_paragraph_text(paras[i], line)
        else:
            # Append a clone of ref_p with the new line
            new_p = copy.deepcopy(ref_p)
            set_paragraph_text(new_p, line)
            # Insert just after the last paragraph in this range
            anchor = paras[-1]
            anchor_pos = list(body).index(anchor)
            body.insert(anchor_pos + 1 + (i - n_existing), new_p)
    # If there are MORE existing paragraphs than new lines, clear the extras
    if n_existing > n_new:
        for p in paras[n_new:]:
            set_paragraph_text(p, "")
    print(f"[2] JSON: updated example block ({n_existing} -> {n_new} lines)")


# ── 3. B949: переформульовуємо BERT-проблему ──────────────────────────
# Note: indices may have shifted by JSON extension. Re-find by text.
def find_para_by_text_substring(substring, start=700, end=1020):
    body_now = list(body)
    for i in range(start, min(end, len(body_now))):
        text = ''.join((t.text or '') for t in body_now[i].iter(_w('t')))
        if substring in text:
            return i, body_now[i]
    return -1, None

idx_949, p_949 = find_para_by_text_substring("характерну проблему BERT")
if p_949 is not None:
    text = ''.join((t.text or '') for t in p_949.iter(_w('t')))
    text = text.replace(
        "характерну проблему BERT у межах даної системи",
        "характерну проблему щільних embedding-моделей у межах даної системи"
    )
    set_paragraph_text(p_949, text)
    print(f"[3] B{idx_949}: BERT framing reframed to general embedding issue.")
else:
    print("[3] BERT framing: NOT FOUND")


# ── 4. Оновлюємо latency у таблицях 5.8/5.9/5.10 ──────────────────────
# Поточні справжні значення з benchmark_results_2026-04-27_01-00-00.json
LATENCIES = {
    "tech":    {"BM25": "1.5", "E5-base": "225.1", "BGE-M3": "2959.8", "nomic": "407.4", "Qwen3": "365.4"},
    "legal":   {"BM25": "2.9", "E5-base": "67.0",  "BGE-M3": "202.9",  "nomic": "131.6", "Qwen3": "399.1"},
    "medical": {"BM25": "0.6", "E5-base": "80.3",  "BGE-M3": "236.3",  "nomic": "150.4", "Qwen3": "465.9"},
}

def find_table_after(caption_substring):
    body_now = list(body)
    for i, ch in enumerate(body_now):
        text = ''.join((t.text or '') for t in ch.iter(_w('t')))
        if caption_substring in text:
            # Next sibling element should be the table
            for j in range(i + 1, min(i + 5, len(body_now))):
                if body_now[j].tag == _w("tbl"):
                    return j, body_now[j]
    return -1, None

def update_latency_in_table(tbl, latency_map):
    rows = list(tbl.findall(_w("tr")))
    updated = 0
    for r in rows[1:]:  # skip header
        cells = list(r.findall(_w("tc")))
        if len(cells) < 6:
            continue
        first_cell = ''.join((t.text or '') for t in cells[0].iter(_w('t')))
        # Match model by substring
        model_key = None
        for k in latency_map:
            if k in first_cell:
                model_key = k
                break
        if model_key is None:
            continue
        # Update last cell (ms/query) — collapse into first w:t
        last_cell = cells[-1]
        ts = list(last_cell.iter(_w('t')))
        if not ts:
            continue
        new_val = latency_map[model_key]
        ts[0].text = new_val
        for t in ts[1:]:
            t.text = ""
        updated += 1
    return updated

for domain, caption in [
    ("tech",    "Таблиця 5.8 – Результати benchmark-оцінювання моделей у технічному"),
    ("legal",   "Таблиця 5.9 – Результати benchmark-оцінювання моделей у юридичному"),
    ("medical", "Таблиця 5.10 – Результати benchmark-оцінювання моделей у медичному"),
]:
    idx, tbl = find_table_after(caption)
    if tbl is None:
        print(f"[4] {domain}: table NOT FOUND")
        continue
    n = update_latency_in_table(tbl, LATENCIES[domain])
    print(f"[4] {domain}: updated {n} latency cells in table {idx}.")


# ── 5. Додаємо абзац про bootstrap 95% CI після таблиці 5.10 ─────────
# Вставляємо після опису медичного домену (B987 за оригіналом — "У медичному домені також лідирують").
# Простіше: знаходимо абзац, що починається з "Порівняння результатів у трьох доменах показує".
ci_text = (
    "Окремо для метрики nDCG@10 у розрахунках використано bootstrap-оцінку 95-відсоткового "
    "довірчого інтервалу (n = 2000). Це дозволяє оцінити статистичну значущість відмінностей "
    "між моделями. Так, у технічному домені довірчий інтервал nDCG@10 для BGE-M3 становить "
    "[0.6059; 0.7367], а для BM25 — [0.4312; 0.5412]; інтервали не перетинаються, що підтверджує "
    "статистично значущу перевагу нейронних embedding-моделей над лексичним базелайном. Аналогічна "
    "ситуація спостерігається у медичному домені, тоді як у юридичному довірчі інтервали для "
    "BGE-M3 і Qwen3 частково перекриваються між собою, що свідчить про порівнянну якість цих "
    "моделей у даній доменній області."
)

idx_porivn, p_porivn = find_para_by_text_substring("Порівняння результатів у трьох доменах показує")
if p_porivn is not None:
    # Clone style from existing paragraph
    new_p = copy.deepcopy(p_porivn)
    set_paragraph_text(new_p, ci_text)
    pos = list(body).index(p_porivn)
    body.insert(pos, new_p)   # вставити перед "Порівняння результатів..."
    print(f"[5] B{idx_porivn}: inserted bootstrap CI paragraph before it.")
else:
    print("[5] CI paragraph: anchor NOT FOUND")


# ── Save & repack ─────────────────────────────────────────────────────
ET.register_namespace("w", W)
ET.register_namespace("m", M)
ET.register_namespace("r", "http://schemas.openxmlformats.org/officeDocument/2006/relationships")
ET.register_namespace("mc", "http://schemas.openxmlformats.org/markup-compatibility/2006")
ET.register_namespace("v", "urn:schemas-microsoft-com:vml")
ET.register_namespace("wp", "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing")
ET.register_namespace("w14", "http://schemas.microsoft.com/office/word/2010/wordml")
ET.register_namespace("w15", "http://schemas.microsoft.com/office/word/2012/wordml")
ET.register_namespace("wne", "http://schemas.microsoft.com/office/word/2006/wordml")
ET.register_namespace("wps", "http://schemas.microsoft.com/office/word/2010/wordprocessingShape")
ET.register_namespace("o", "urn:schemas-microsoft-com:office:office")
ET.register_namespace("w10", "urn:schemas-microsoft-com:office:word")
ET.register_namespace("wp14", "http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing")
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

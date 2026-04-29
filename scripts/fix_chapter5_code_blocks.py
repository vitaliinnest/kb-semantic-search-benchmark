"""
Оновлює застарілі фрагменти коду в розділі 5:

1. B811: SentenceTransformer(args.model) → SentenceTransformer(args.model_name, trust_remote_code=True)
         + додає рядок model.max_seq_length = args.max_seq_length

2. B821-B827: замінює гілку "tfidf" (видалена з кодової бази) на гілку "e5"/"nomic"
              з уніфікованим інтерфейсом encode_documents()

3. Додає порожній параграф між кінцем блоку queries.jsonl (B869) і текстом B870
   (qrels.jsonl), де бракує пустого рядка після фрагмента коду.
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


def para_text(p):
    return "".join((t.text or "") for t in p.iter(_w("t")))


def get_style(p):
    ppr = p.find(_w("pPr"))
    if ppr is not None:
        ps = ppr.find(_w("pStyle"))
        if ps is not None:
            return ps.get(_w("val"), "")
    return ""


def find_para_exact(text, start=700, end=960):
    children = list(body)
    for i in range(start, min(end, len(children))):
        if para_text(children[i]) == text:
            return i, children[i]
    return -1, None


def find_para_containing(substring, start=700, end=960):
    children = list(body)
    for i in range(start, min(end, len(children))):
        if substring in para_text(children[i]):
            return i, children[i]
    return -1, None


def set_para_text(p, new_text):
    """Замінює текст параграфа, залишаючи форматування runs."""
    first_r = None
    for r in list(p.findall(_w("r"))):
        if first_r is None:
            first_r = r
            t = r.find(_w("t"))
            if t is None:
                t = ET.SubElement(r, _w("t"))
            t.text = new_text
            if new_text != new_text.strip():
                t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
        else:
            p.remove(r)
    if first_r is None:
        r = ET.SubElement(p, _w("r"))
        t = ET.SubElement(r, _w("t"))
        t.text = new_text
        if new_text != new_text.strip():
            t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")


def make_code_para(text):
    """Параграф стилю a9 (код)."""
    p = ET.Element(_w("p"))
    ppr = ET.SubElement(p, _w("pPr"))
    ps = ET.SubElement(ppr, _w("pStyle"))
    ps.set(_w("val"), "a9")
    if text:
        r = ET.SubElement(p, _w("r"))
        t = ET.SubElement(r, _w("t"))
        t.text = text
        if text != text.strip():
            t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    return p


def make_empty_para(style="a"):
    p = ET.Element(_w("p"))
    ppr = ET.SubElement(p, _w("pPr"))
    ps = ET.SubElement(ppr, _w("pStyle"))
    ps.set(_w("val"), style)
    return p


def insert_after(anchor_el, new_elements):
    children = list(body)
    idx = children.index(anchor_el)
    for j, el in enumerate(new_elements):
        body.insert(idx + 1 + j, el)


def remove_elements(elements):
    for el in elements:
        body.remove(el)


# ── 1. Оновити рядок SentenceTransformer(args.model) ─────────────────
idx_st, p_st = find_para_exact("    model = SentenceTransformer(args.model)", start=818, end=835)
if p_st is None:
    print("[1] SentenceTransformer line NOT FOUND")
else:
    set_para_text(p_st, "    model = SentenceTransformer(args.model_name, trust_remote_code=True)")
    # Додаємо рядок max_seq_length після
    p_max = make_code_para("    model.max_seq_length = args.max_seq_length")
    insert_after(p_st, [p_max])
    print(f"[1] Updated SentenceTransformer line at B{idx_st}, inserted max_seq_length line.")


# ── 2. Замінити гілку tfidf на e5/nomic ───────────────────────────────
idx_tfidf, p_tfidf = find_para_exact('elif args.model_type == "tfidf":', start=830, end=850)
if p_tfidf is None:
    print("[2] tfidf branch NOT FOUND")
else:
    children = list(body)
    # Знайти кінець гілки tfidf — рядок "    vectors_np = model.encode_documents(texts)"
    end_idx = -1
    for i in range(idx_tfidf, min(idx_tfidf + 10, len(children))):
        if para_text(children[i]) == "    vectors_np = model.encode_documents(texts)":
            end_idx = i
            break
    if end_idx == -1:
        print("[2] tfidf branch end NOT FOUND")
    else:
        # Видалити параграфи від idx_tfidf до end_idx включно
        to_remove = children[idx_tfidf : end_idx + 1]
        # Нові рядки для гілки e5/nomic
        new_lines = [
            'elif args.model_type in ("e5", "nomic"):',
            '    from embedding_models import E5EmbeddingModel, NomicEmbeddingModel',
            '',
            '    model_cls = E5EmbeddingModel if args.model_type == "e5" else NomicEmbeddingModel',
            '    model = model_cls(args.model_name, max_seq_length=args.max_seq_length)',
            '    vectors_np = model.encode_documents(texts)',
        ]
        new_paras = [make_code_para(line) for line in new_lines]
        # Вставляємо нові після останнього видаленого (спочатку вставляємо, потім видаляємо)
        insert_after(to_remove[-1], new_paras)
        remove_elements(to_remove)
        print(f"[2] Replaced tfidf branch (B{idx_tfidf}–B{end_idx}) with e5/nomic branch ({len(new_paras)} lines).")


# ── 3. Додати порожній рядок після блоку queries.jsonl ────────────────
# Остання строчка queries-блоку: {"query_id":"q3","query":"комбінація Autoencoder..."}
idx_q3, p_q3 = find_para_containing(
    'комбінація Autoencoder та Isolation Forest для fraud detection',
    start=870, end=910
)
if p_q3 is None:
    print("[3] queries block last line NOT FOUND")
else:
    # Перевіряємо, чи наступний параграф порожній
    children = list(body)
    next_idx = idx_q3 + 1
    if next_idx < len(children):
        next_text = para_text(children[next_idx])
        next_style = get_style(children[next_idx])
        if next_text == "" and next_style in ("a", ""):
            print(f"[3] Empty para already exists after B{idx_q3} — skipping.")
        else:
            p_empty = make_empty_para("a")
            insert_after(p_q3, [p_empty])
            print(f"[3] Inserted empty paragraph after queries block (B{idx_q3}).")
    else:
        print("[3] No next element to check.")


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

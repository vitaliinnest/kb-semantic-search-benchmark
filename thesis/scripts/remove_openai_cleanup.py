"""Second pass: remove remaining text-embedding-3-large mentions."""
import zipfile, xml.etree.ElementTree as ET, sys, copy, pathlib
sys.stdout.reconfigure(encoding='utf-8')

ROOT = pathlib.Path("D:/repos/kb-semantic-search-benchmark")
DOCX = next(p for p in ROOT.glob("2026_*.docx") if "bak" not in p.name)
UNPACKED = ROOT / "thesis" / "unpacked_docx"
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

def _w(tag): return f"{{{W}}}{tag}"

def get_text(el):
    return "".join(r.text or "" for r in el.iter(_w("t")))

def set_single_run_text(para, new_text):
    rPr_src = None
    first_r = para.find(_w("r"))
    if first_r is not None:
        rPr_src = first_r.find(_w("rPr"))
    for ch in list(para):
        if ch.tag == _w("r"):
            para.remove(ch)
    r = ET.SubElement(para, _w("r"))
    if rPr_src is not None:
        r.append(copy.deepcopy(rPr_src))
    t = ET.SubElement(r, _w("t"))
    if new_text != new_text.strip():
        t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    t.text = new_text

def remove_sentence(text, sentence_fragment):
    """Remove a sentence from text that contains sentence_fragment."""
    import re
    # Try to find and remove the sentence ending with period
    idx = text.find(sentence_fragment)
    if idx < 0:
        return text, False
    # Find start of sentence (after previous period+space or start)
    start = text.rfind('. ', 0, idx)
    if start < 0:
        start = 0
    else:
        start += 2  # skip '. '
    # Find end of sentence
    end = text.find('. ', idx)
    if end < 0:
        end = text.find('.\n', idx)
    if end < 0:
        end = len(text) - 1
    else:
        end += 1  # include the period

    # Remove the sentence
    removed_sentence = text[start:end+1]
    new_text = text[:start] + text[end+1:].lstrip()
    return new_text.strip(), True

print("Loading document...")
tree = ET.parse(UNPACKED / "word" / "document.xml")
root = tree.getroot()
body = root.find(_w("body"))
children = list(body)
print(f"Total body children: {len(children)}")

# ── B359: Remove "та text-embedding-3-large" from alternatives list ──
idx = 359
t = get_text(children[idx])
if 'text-embedding-3-large' in t:
    new_t = t.replace(" та text-embedding-3-large", "")
    new_t = new_t.replace(", які реалізують різні підходи до нейронного векторного подання тексту",
                          ", які реалізують різні підходи до нейронного векторного подання тексту")
    set_single_run_text(children[idx], new_t)
    print(f"Updated B{idx}: removed text-embedding from alternatives")

# ── B841: Remove last sentence about OpenAI ──────────────────────────
idx = 841
t = get_text(children[idx])
if 'text-embedding-3-large' in t:
    # Remove: "Комерційна модель text-embedding-3-large...тестовому середовищі."
    new_t = t.replace(
        " Комерційна модель text-embedding-3-large (OpenAI) не оцінювалася в автоматичному benchmark через відсутність API-ключа у тестовому середовищі.",
        "")
    set_single_run_text(children[idx], new_t)
    print(f"Updated B{idx}: removed OpenAI sentence from benchmark intro")

# ── B876: Remove OpenAI sentence from cross-domain comparison ─────────
idx = 876
t = get_text(children[idx])
if 'text-embedding-3-large' in t:
    new_t = t.replace(
        " Комерційна модель text-embedding-3-large теоретично демонструє найкращі показники повноти, але потребує зовнішнього API і не була оцінена у локальному benchmark.",
        "")
    set_single_run_text(children[idx], new_t)
    print(f"Updated B{idx}: removed OpenAI from cross-domain comparison")

# ── B892: Remove API-key sentence ─────────────────────────────────────
idx = 892
t = get_text(children[idx])
if 'text-embedding-3-large' in t:
    new_t = t.replace(
        " За умови доступності API-ключа оптимальним вибором для максимальної якості є text-embedding-3-large (OpenAI).",
        "")
    set_single_run_text(children[idx], new_t)
    print(f"Updated B{idx}: removed API-key sentence from multicriteria summary")

# ── B895: Remove OpenAI reference ─────────────────────────────────────
idx = 895
t = get_text(children[idx])
if 'text-embedding-3-large' in t:
    new_t = t.replace(
        ", тоді як для максимальної теоретичної якості слід розглянути text-embedding-3-large (OpenAI)",
        "")
    set_single_run_text(children[idx], new_t)
    print(f"Updated B{idx}: removed OpenAI from multicriteria conclusions")

# ── B902: Remove OpenAI sentence from main conclusions ────────────────
idx = 902
t = get_text(children[idx])
if 'text-embedding-3-large' in t:
    new_t = t.replace(
        " Комерційна модель text-embedding-3-large (OpenAI) не оцінювалася в автоматичному benchmark через відсутність API-ключа, проте теоретично демонструє найвищу повноту пошуку (K₃=5) за даними публічних benchmark-наборів.",
        "")
    set_single_run_text(children[idx], new_t)
    print(f"Updated B{idx}: removed OpenAI from main conclusions")

# ── Remove OpenAI rows from benchmark result tables ────────────────────
for bi in [849, 860, 872]:
    tbl = children[bi]
    rows_to_del = []
    for tr in tbl.findall(_w("tr")):
        t = get_text(tr)
        if 'text-embedding' in t.lower() or 'openai' in t.lower():
            rows_to_del.append(tr)
    for row in rows_to_del:
        tbl.remove(row)
    print(f"Table B{bi}: removed {len(rows_to_del)} OpenAI row(s)")

# ── Save ──────────────────────────────────────────────────────────────
ET.register_namespace("wpc", "http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas")
ET.register_namespace("cx", "http://schemas.microsoft.com/office/drawing/2014/chartex")
ET.register_namespace("mc", "http://schemas.openxmlformats.org/markup-compatibility/2006")
ET.register_namespace("r", "http://schemas.openxmlformats.org/officeDocument/2006/relationships")
ET.register_namespace("m", "http://schemas.openxmlformats.org/officeDocument/2006/math")
ET.register_namespace("v", "urn:schemas-microsoft-com:vml")
ET.register_namespace("wp", "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing")
ET.register_namespace("w", "http://schemas.openxmlformats.org/wordprocessingml/2006/main")
ET.register_namespace("w14", "http://schemas.microsoft.com/office/word/2010/wordml")
ET.register_namespace("w15", "http://schemas.microsoft.com/office/word/2012/wordml")
ET.register_namespace("wne", "http://schemas.microsoft.com/office/word/2006/wordml")
ET.register_namespace("wps", "http://schemas.microsoft.com/office/word/2010/wordprocessingShape")

xml_path = UNPACKED / "word" / "document.xml"
tree.write(str(xml_path), xml_declaration=True, encoding="UTF-8")
print("Saved document.xml")

with zipfile.ZipFile(DOCX, "w", zipfile.ZIP_DEFLATED) as zout:
    for fpath in sorted(UNPACKED.rglob("*")):
        if fpath.is_file():
            zout.write(fpath, fpath.relative_to(UNPACKED))
print(f"Repacked: {DOCX.name} ({DOCX.stat().st_size:,} bytes)")

# Final check
with zipfile.ZipFile(DOCX) as z:
    xml2 = z.read('word/document.xml')
root2 = ET.fromstring(xml2)
body2 = root2.find(_w("body"))
remaining = [(i, "".join(r.text or "" for r in ch.iter(_w("t")))[:100])
             for i, ch in enumerate(list(body2))
             if 'text-embedding' in "".join(r.text or "" for r in ch.iter(_w("t"))).lower()
             or 'openai' in "".join(r.text or "" for r in ch.iter(_w("t"))).lower()]
print(f"\nFinal check: {len(remaining)} remaining mentions")
for i, t in remaining:
    print(f"  B{i}: {t}")

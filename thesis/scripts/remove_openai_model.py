"""
Remove text-embedding-3-large (OpenAI) from thesis document.

Changes:
1. Delete section 5.1.5 (body B303-B318)
2. Delete ToC entry B95, abbreviations entry B138, reference B937
3. Update model lists in ~10 paragraphs
4. Update multi-criteria analysis:
   - Remove A5 from alternatives list
   - Pareto: {E5-base, text-embedding-3-large} → {BGE-M3, E5-base}
   - Add BGE-M3 MAUT calculation (U=0.842)
   - Update conclusion (BGE-M3 wins, not OpenAI)
5. Remove text-embedding-3-large rows from tables 5.3-5.7
6. Update Table 5.6: BGE-M3 status → Парето-оптимальна
7. Update Table 5.7: BGE-M3=0.842, E5-base=0.770
"""

import zipfile, xml.etree.ElementTree as ET, sys, copy, pathlib
sys.stdout.reconfigure(encoding='utf-8')

ROOT = pathlib.Path("D:/repos/kb-semantic-search-benchmark")
DOCX = next(p for p in ROOT.glob("2026_*.docx") if "bak" not in p.name)
UNPACKED = ROOT / "thesis" / "unpacked_docx"

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
M = "http://schemas.openxmlformats.org/officeDocument/2006/math"

def _w(tag): return f"{{{W}}}{tag}"
def _m(tag): return f"{{{M}}}{tag}"

def get_text(el):
    return "".join(r.text or "" for r in el.iter(_w("t")))

def set_single_run_text(para, new_text):
    """Replace all runs in a paragraph with a single run containing new_text."""
    # Find rPr from first run if exists
    rPr_src = None
    first_r = para.find(_w("r"))
    if first_r is not None:
        rPr_src = first_r.find(_w("rPr"))

    # Remove all existing runs and bookmarks
    to_remove = []
    for ch in list(para):
        tag = ch.tag.split("}")[-1]
        if tag in ("r", "hyperlink", "bookmarkStart", "bookmarkEnd", "del", "ins"):
            to_remove.append(ch)
    # Actually keep bookmarks but remove runs
    for ch in list(para):
        if ch.tag in (_w("r"),):
            para.remove(ch)

    # Add new single run
    r = ET.SubElement(para, _w("r"))
    if rPr_src is not None:
        r.append(copy.deepcopy(rPr_src))
    t = ET.SubElement(r, _w("t"))
    if new_text != new_text.strip():
        t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    t.text = new_text

def replace_text_in_para(para, old_sub, new_sub):
    """Replace substring in all text runs of a paragraph."""
    # Collect all text
    full = get_text(para)
    if old_sub not in full:
        return False
    new_full = full.replace(old_sub, new_sub)
    set_single_run_text(para, new_full)
    return True

def make_para_like(reference_para, text):
    """Create a new paragraph with same style as reference, containing text."""
    new_p = ET.Element(_w("p"))
    pPr = reference_para.find(_w("pPr"))
    if pPr is not None:
        new_p.append(copy.deepcopy(pPr))

    # Copy rPr from first run
    first_r = reference_para.find(_w("r"))
    rPr = None
    if first_r is not None:
        rPr = first_r.find(_w("rPr"))

    r = ET.SubElement(new_p, _w("r"))
    if rPr is not None:
        r.append(copy.deepcopy(rPr))
    t = ET.SubElement(r, _w("t"))
    if text != text.strip():
        t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    t.text = text
    return new_p

def make_blank_para(reference_para):
    """Create an empty paragraph with same style."""
    new_p = ET.Element(_w("p"))
    pPr = reference_para.find(_w("pPr"))
    if pPr is not None:
        new_p.append(copy.deepcopy(pPr))
    return new_p

def find_table_rows_with_text(tbl, search_text):
    """Find all w:tr rows in a table that contain search_text."""
    rows = []
    for tr in tbl.findall(_w("tr")):
        if search_text in get_text(tr):
            rows.append(tr)
    return rows

def find_cell_with_text(tbl, search_text):
    """Find first w:tc cell containing exactly search_text (stripped)."""
    for tr in tbl.findall(_w("tr")):
        for tc in tr.findall(_w("tc")):
            if get_text(tc).strip() == search_text:
                return tr, tc
    return None, None

def set_cell_text(tc, new_text):
    """Replace text in a table cell."""
    for p in tc.findall(_w("p")):
        for r in p.findall(_w("r")):
            p.remove(r)
        r = ET.SubElement(p, _w("r"))
        t = ET.SubElement(r, _w("t"))
        t.text = new_text
        return  # Only update first paragraph


print("Loading document...")
tree = ET.parse(UNPACKED / "word" / "document.xml")
root = tree.getroot()
body = root.find(_w("body"))
children = list(body)

print(f"Total body children: {len(children)}")

# ── Helper: get child by searching for known text ──────────────────────
def find_child(text_contains, start=0, end=None):
    """Find first body child index whose text contains the given string."""
    if end is None:
        end = len(children)
    for i in range(start, end):
        if text_contains in get_text(children[i]):
            return i
    return -1

# ── 1. DELETE SECTION 5.1.5 (B303-B318) ───────────────────────────────
# Find start by heading text, find end by next section heading
sec_start = find_child("text-embedding-3-large як комерційна хмарна модель ембеддінгів", 290, 320)
sec_end   = find_child("Критерії оцінки моделей", 290, 340)
print(f"Section 5.1.5: B{sec_start}–B{sec_end-1}")

to_delete = set()
for i in range(sec_start, sec_end):
    to_delete.add(id(children[i]))

# ── 2. DELETE TOC ENTRY B95 ───────────────────────────────────────────
toc_idx = find_child("5.1.5", 85, 105)
if toc_idx >= 0 and "text-embedding" in get_text(children[toc_idx]).lower():
    to_delete.add(id(children[toc_idx]))
    print(f"ToC entry: B{toc_idx}")

# ── 3. DELETE ABBREVIATIONS ENTRY B138 ───────────────────────────────
abbr_idx = find_child("text-embedding-3-large", 130, 145)
if abbr_idx >= 0:
    to_delete.add(id(children[abbr_idx]))
    print(f"Abbreviations entry: B{abbr_idx}")

# ── 4. DELETE OPENAI REFERENCE ────────────────────────────────────────
ref_idx = find_child("OpenAI. Introducing text and code embeddings", 920, 960)
if ref_idx >= 0:
    to_delete.add(id(children[ref_idx]))
    print(f"OpenAI reference: B{ref_idx}")

# ── 5. TEXT UPDATES IN PARAGRAPHS ─────────────────────────────────────
updates = [
    # (body_idx_approx, old_substring, new_substring)
    # B35 - inputs list
    (35, " та text-embedding-3-large", ""),
    # B215 - novelty
    (215, ", Qwen3-Embedding та text-embedding-3-large —", " та Qwen3-Embedding —"),
    # B221 - research topic
    (221, ", Qwen3-Embedding та text-embedding-3-large.", " та Qwen3-Embedding."),
    # B228 - methods
    (228, ", Qwen3-Embedding та text-embedding-3-large.", " та Qwen3-Embedding."),
    # B228 second part
    (228, "відкриті та комерційні рішення", "відкриті рішення"),
    # B236 - limitations
    (236, ", Qwen3-Embedding-0.6B та text-embedding-3-large.", " та Qwen3-Embedding-0.6B."),
    # B246 - models list
    (246, ", Qwen3-Embedding-0.6B та text-embedding-3-large.", " та Qwen3-Embedding-0.6B."),
    (246, "encoder-based та decoder-based", "encoder-based та decoder-based"),  # keep as is
    (246, "відкриті та комерційні рішення", "відкриті рішення"),
    # B248 - model descriptions
    (248, " та комерційних хмарних API-сервісів (text-embedding-3-large)", ""),
    # B629 - index building list
    (629, " та text-embedding-3-large", ""),
    # B922 - conclusions (five → four models)
    (922, "п'яти сучасних моделей векторних ембеддінгів: BGE-M3, E5-base, nomic-embed-text-v1.5, Qwen3-Embedding-0.6B та text-embedding-3-large",
          "чотирьох моделей векторних ембеддінгів: BGE-M3, E5-base, nomic-embed-text-v1.5 та Qwen3-Embedding-0.6B"),
]

# Build a lookup from text fragment → body index for approximate search
def apply_updates(updates):
    for (approx_idx, old, new) in updates:
        # Search in ±30 range around approx_idx
        found = False
        for delta in range(0, 35):
            for sign in ([0, 1, -1] if delta > 0 else [0]):
                i = approx_idx + sign * delta
                if 0 <= i < len(children):
                    if old in get_text(children[i]):
                        replaced = replace_text_in_para(children[i], old, new)
                        if replaced:
                            print(f"Updated B{i}: '{old[:50]}...' → '{new[:50]}...'")
                            found = True
                            break
            if found:
                break
        if not found:
            print(f"WARNING: Could not find '{old[:60]}' near B{approx_idx}")

apply_updates(updates)

# ── 6. UPDATE A5 ALTERNATIVES LIST ────────────────────────────────────
# B387: Remove A₅ — text-embedding-3-large
idx = find_child("A₅ — text-embedding-3-large", 380, 400)
if idx >= 0:
    old_text = get_text(children[idx])
    # Remove the A₅ part (it's part of a combined paragraph)
    new_text = old_text.replace("     A₅ — text-embedding-3-large.", "").strip()
    set_single_run_text(children[idx], new_text)
    print(f"Updated B{idx}: removed A₅ — text-embedding-3-large")

# Also need to update the matrix size description (5x5 → 4x5)
# Find paragraph that says "5 альтернатив" near B383
for i in range(375, 400):
    t = get_text(children[i])
    if "п'яти альтернатив" in t or "A = {A₁, A₂, A₃, A₄, A₅}" in t:
        new_t = t.replace("п'яти альтернатив", "чотирьох альтернатив")
        new_t = new_t.replace("A = {A₁, A₂, A₃, A₄, A₅}", "A = {A₁, A₂, A₃, A₄}")
        set_single_run_text(children[i], new_t)
        print(f"Updated B{i}: alternatives count")

# ── 7. UPDATE PARETO ANALYSIS TEXT ────────────────────────────────────
# B581: BGE-M3 dominated by text-embedding-3-large → new Pareto analysis
idx_pareto = find_child("BGE-M3 домінується альтернативою text-embedding-3-large", 575, 590)
if idx_pareto >= 0:
    new_text = ("nomic-embed-text-v1.5 домінується альтернативою E5-base, оскільки E5-base "
                "перевершує nomic за критеріями K₁, K₂, K₃ та K₅ одночасно. "
                "Qwen3-Embedding-0.6B домінується альтернативою BGE-M3, оскільки BGE-M3 "
                "не гірша за Qwen3 за критеріями K₁–K₄ та має краще значення за критерієм K₅ "
                "(0.602 > 0.000). Між BGE-M3 та E5-base домінування відсутнє: BGE-M3 "
                "переважає за K₃ (1.000 > 0.500), натомість E5-base — за K₅ (1.000 > 0.602).")
    set_single_run_text(children[idx_pareto], new_text)
    print(f"Updated B{idx_pareto}: Pareto analysis text")

# B582: Qwen3 dominated by E5-base → delete (covered by new B581 text)
idx_qwen3_pareto = find_child("Qwen3-Embedding-0.6B домінується альтернативою E5-base", 576, 590)
if idx_qwen3_pareto >= 0:
    to_delete.add(id(children[idx_qwen3_pareto]))
    print(f"Deleted B{idx_qwen3_pareto}: old Qwen3 Pareto sentence")

# ── 8. UPDATE PARETO SET ──────────────────────────────────────────────
idx_pset = find_child("P = {E5-base, text-embedding-3-large}", 580, 595)
if idx_pset >= 0:
    set_single_run_text(children[idx_pset], "P = {BGE-M3, E5-base}.")
    print(f"Updated B{idx_pset}: P set")

# ── 9. DELETE TEXT-EMBEDDING NORMALIZATION ────────────────────────────
# B571: "Для моделі text-embedding-3-large:"
# B572: blank
# B573: calculation
# B574: "Аналогічно обчислюються..."
# After deletion, B574 stays (it's about "all alternatives")
# Actually keep B574 ("Аналогічно...") since it refers to all alternatives
idx_oai_norm = find_child("Для моделі text-embedding-3-large:", 565, 580)
if idx_oai_norm >= 0:
    to_delete.add(id(children[idx_oai_norm]))
    print(f"Marked delete B{idx_oai_norm}: text-embedding-3-large normalization header")
    # Also delete the next paragraph (blank) and the one after (calculation)
    calc_idx = find_child("y₅₁ = (4−2)/(4−2)", idx_oai_norm, idx_oai_norm + 4)
    if calc_idx >= 0:
        to_delete.add(id(children[calc_idx]))
        # Delete blank between header and calculation
        if idx_oai_norm + 1 != calc_idx:
            for j in range(idx_oai_norm + 1, calc_idx):
                if not get_text(children[j]).strip():
                    to_delete.add(id(children[j]))
        print(f"Marked delete B{calc_idx}: text-embedding-3-large normalization values")

# ── 10. ADD BGE-M3 MAUT CALCULATION ──────────────────────────────────
# Find "Для E5-base:" paragraph
idx_e5_header = find_child("Для E5-base:", 588, 600)
print(f"E5-base MAUT header: B{idx_e5_header}")

# We'll insert BGE-M3 calculation BEFORE E5-base
# Using reference paragraph for style
ref_para = children[idx_e5_header]
bge_header_para = make_para_like(ref_para, "Для BGE-M3:")
bge_blank_para  = make_blank_para(ref_para)
bge_calc_para   = make_para_like(ref_para,
    "U(BGE-M3) = 0.33⋅1.000 + 0.27⋅1.000 + 0.20⋅1.000 + 0.13⋅0.000 + 0.07⋅0.602, "
    "U(BGE-M3) = 0.330 + 0.270 + 0.200 + 0.000 + 0.042 = 0.842.")
bge_blank2_para = make_blank_para(ref_para)

# Mark them for insertion before E5-base header
insert_before_e5 = [bge_header_para, bge_blank_para, bge_calc_para, bge_blank2_para]

# ── 11. DELETE TEXT-EMBEDDING MAUT PARAGRAPHS ─────────────────────────
idx_oai_maut = find_child("Для text-embedding-3-large:", 592, 605)
if idx_oai_maut >= 0:
    to_delete.add(id(children[idx_oai_maut]))
    print(f"Marked delete B{idx_oai_maut}: text-embedding-3-large MAUT header")
    calc2_idx = find_child("U(text-embedding-3-large) =", idx_oai_maut, idx_oai_maut + 4)
    if calc2_idx >= 0:
        to_delete.add(id(children[calc2_idx]))
        for j in range(idx_oai_maut + 1, calc2_idx):
            if not get_text(children[j]).strip():
                to_delete.add(id(children[j]))
        # Also delete blank after calc2
        if calc2_idx + 1 < len(children) and not get_text(children[calc2_idx + 1]).strip():
            to_delete.add(id(children[calc2_idx + 1]))
        print(f"Marked delete B{calc2_idx}: text-embedding-3-large MAUT values")

# ── 12. UPDATE MAUT CONCLUSION ────────────────────────────────────────
idx_maut_concl = find_child("text-embedding-3-large отримує найвищу інтегральну оцінку", 603, 618)
if idx_maut_concl >= 0:
    new_text = ("Як видно з таблиці 5.7, BGE-M3 отримує найвищу інтегральну оцінку U = 0.842 "
                "завдяки перевазі за критерієм повноти пошуку (K₃=1.000). Модель E5-base "
                "демонструє U = 0.770 і поступається BGE-M3 за критерієм K₃, але має перевагу "
                "за швидкодією (K₅=1.000).")
    set_single_run_text(children[idx_maut_concl], new_text)
    print(f"Updated B{idx_maut_concl}: MAUT conclusion")

idx_maut_formula = find_child("U(text-embedding-3-large) = max", 608, 618)
if idx_maut_formula >= 0:
    set_single_run_text(children[idx_maut_formula],
        "U(BGE-M3) = max{U(aᵢ) | aᵢ ∈ P} = 0.842.")
    print(f"Updated B{idx_maut_formula}: MAUT max formula")

idx_final_concl = find_child("у межах теоретичного дослідження модель text-embedding-3-large є найбільш", 610, 620)
if idx_final_concl >= 0:
    new_text = ("Отже, у межах теоретичного дослідження модель BGE-M3 є найбільш придатною "
                "альтернативою серед розглянутих моделей за обраною системою критеріїв "
                "(U = 0.842). При цьому модель E5-base є перспективним рішенням (U = 0.770) "
                "та рекомендується для сценаріїв, де критична мінімальна затримка пошуку.")
    set_single_run_text(children[idx_final_concl], new_text)
    print(f"Updated B{idx_final_concl}: final conclusion")

# ── 13. UPDATE CONCLUSIONS SECTION ───────────────────────────────────
idx_concl_pareto = find_child("Парето-оптимальними є E5-base та text-embedding-3-large", 920, 935)
if idx_concl_pareto >= 0:
    new_text = ("Додатковий багатокритеріальний аналіз на основі принципу Парето та лінійної "
                "адитивної згортки показав, що серед розглянутих альтернатив Парето-оптимальними "
                "є BGE-M3 та E5-base. Найвищу інтегральну оцінку U = 0.842 отримала модель "
                "BGE-M3 завдяки перевазі за критерієм повноти пошуку. Модель E5-base (U = 0.770) "
                "є рекомендованою для сценаріїв, де важлива мінімальна затримка відповіді, "
                "оскільки демонструє найкращу швидкодію серед усіх моделей. Таким чином, "
                "теоретичний вибір збігається з практичним: обидва методи визначають BGE-M3 "
                "як найбільш збалансовану модель за сукупністю критеріїв.")
    set_single_run_text(children[idx_concl_pareto], new_text)
    print(f"Updated B{idx_concl_pareto}: conclusions Pareto paragraph")

# ── 14. UPDATE TABLES ─────────────────────────────────────────────────
def remove_oai_rows_from_table(tbl, table_name):
    """Remove all rows containing 'text-embedding-3-large' from a table."""
    rows_to_del = find_table_rows_with_text(tbl, "text-embedding-3-large")
    for row in rows_to_del:
        tbl.remove(row)
    print(f"Table {table_name}: removed {len(rows_to_del)} OpenAI row(s)")

# Find tables by adjacent paragraph content
tables_found = []
for i, ch in enumerate(children):
    if ch.tag == _w("tbl"):
        tables_found.append((i, ch))

def find_table_near(text_before, search_range=10):
    """Find table element near a paragraph containing text_before."""
    for i, ch in enumerate(children):
        if text_before in get_text(ch):
            for j in range(i, min(i + search_range, len(children))):
                if children[j].tag == _w("tbl"):
                    return children[j]
    return None

# Table 5.3 – Векторний опис альтернатив
tbl_53 = find_table_near("Таблиця 5.3")
if tbl_53 is not None:
    remove_oai_rows_from_table(tbl_53, "5.3")

# Table 5.4 – Векторний опис альтернатив за критеріями
tbl_54 = find_table_near("Таблиця 5.4")
if tbl_54 is not None:
    remove_oai_rows_from_table(tbl_54, "5.4")

# Table 5.5 – Нормалізований векторний опис
tbl_55 = find_table_near("Таблиця 5.5")
if tbl_55 is not None:
    remove_oai_rows_from_table(tbl_55, "5.5")

# Table 5.6 – Парето-оптимальні альтернативи
tbl_56 = find_table_near("Таблиця 5.6")
if tbl_56 is not None:
    remove_oai_rows_from_table(tbl_56, "5.6")
    # Change BGE-M3 status from Домінована → Парето-оптимальна
    for tr in tbl_56.findall(_w("tr")):
        row_text = get_text(tr)
        if "BGE-M3" in row_text and "Домінована" in row_text:
            for tc in tr.findall(_w("tc")):
                if "Домінована" in get_text(tc):
                    set_cell_text(tc, "Парето-оптимальна")
                    print("Table 5.6: BGE-M3 status → Парето-оптимальна")

# Table 5.7 – Розрахунок інтегральної оцінки
tbl_57 = find_table_near("Таблиця 5.7")
if tbl_57 is not None:
    # Remove text-embedding-3-large row
    remove_oai_rows_from_table(tbl_57, "5.7")
    # Update E5-base row (stays 0.770 – unchanged)
    # Add BGE-M3 row before E5-base
    # Find header row and E5-base row
    rows = tbl_57.findall(_w("tr"))
    e5_row_idx = None
    for idx_r, tr in enumerate(rows):
        if "E5-base" in get_text(tr):
            e5_row_idx = idx_r
            break

    if e5_row_idx is not None:
        e5_row = rows[e5_row_idx]
        # Create BGE-M3 row by copying E5-base row structure
        bge_row = copy.deepcopy(e5_row)
        cells = bge_row.findall(_w("tc"))
        if len(cells) >= 2:
            set_cell_text(cells[0], "BGE-M3")
            set_cell_text(cells[1], "0.842")
        # Insert before E5 row
        e5_row_elem = tbl_57.findall(_w("tr"))[e5_row_idx]
        tbl_57.insert(list(tbl_57).index(e5_row_elem), bge_row)
        print("Table 5.7: inserted BGE-M3=0.842 row before E5-base")

# ── 15. PERFORM INSERTIONS ────────────────────────────────────────────
# Insert BGE-M3 MAUT paragraphs before "Для E5-base:" paragraph
e5_header_elem = children[idx_e5_header]
e5_header_pos = list(body).index(e5_header_elem)
for j, p in enumerate(insert_before_e5):
    body.insert(e5_header_pos + j, p)
print(f"Inserted {len(insert_before_e5)} BGE-M3 MAUT paragraphs before E5-base at body pos {e5_header_pos}")

# ── 16. PERFORM DELETIONS ─────────────────────────────────────────────
n_deleted = 0
for ch in list(body):
    if id(ch) in to_delete:
        body.remove(ch)
        n_deleted += 1

print(f"Deleted {n_deleted} body children")
print(f"Total body children after: {len(list(body))}")

# ── 17. SAVE ──────────────────────────────────────────────────────────
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

xml_path = UNPACKED / "word" / "document.xml"
tree.write(str(xml_path), xml_declaration=True, encoding="UTF-8")
print("Saved document.xml")

# Repack DOCX
import shutil
bak = DOCX.parent / (DOCX.stem + "_bak_pre_oai_removal.docx")
shutil.copy2(DOCX, bak)
print(f"Backup saved: {bak.name}")

with zipfile.ZipFile(DOCX, "w", zipfile.ZIP_DEFLATED) as zout:
    for fpath in sorted(UNPACKED.rglob("*")):
        if fpath.is_file():
            zout.write(fpath, fpath.relative_to(UNPACKED))
print(f"Repacked: {DOCX.name} ({DOCX.stat().st_size:,} bytes)")
print("DONE.")

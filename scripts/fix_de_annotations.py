"""
Fix «де» annotations for all formulas (4.1)–(4.30).

Rules enforced:
- Starts with «де» lowercase, no first-line paragraph indent
- Each new parameter on a new line (aligned ~6 spaces under first param)
- Lines end with «;», last line ends with «.»

Cases handled:
  (4.1),(4.3)-(4.5): single-line annotation → split into multiple lines
  (4.2):             already single-param with period → leave unchanged
  (4.6)-(4.10):      already correct multi-line → leave unchanged
  (4.11):            B452 OK, B453 has 3 params on one line → split
  (4.12)-(4.30):     OMML placeholder paragraphs → replace with text «де»
"""
import zipfile, xml.etree.ElementTree as ET, sys, pathlib, shutil, copy
sys.stdout.reconfigure(encoding='utf-8')

ROOT     = pathlib.Path("D:/repos/kb-semantic-search-benchmark")
DOCX     = ROOT / "2026_M_PI_Nesterenko_VV.docx"
UNPACKED = ROOT / "unpacked_docx"

W      = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
M      = "http://schemas.openxmlformats.org/officeDocument/2006/math"
XML_NS = "http://www.w3.org/XML/1998/namespace"

def _w(tag): return f"{{{W}}}{tag}"
def _m(tag): return f"{{{M}}}{tag}"

# ── Re-unpack fresh ────────────────────────────────────────────────────
if UNPACKED.exists():
    shutil.rmtree(UNPACKED)
with zipfile.ZipFile(DOCX) as z:
    z.extractall(UNPACKED)
print("Re-unpacked docx.")

tree = ET.parse(UNPACKED / "word" / "document.xml")
root = tree.getroot()
body = root.find(_w("body"))
children = list(body)
print(f"Total body children: {len(children)}")

# B312 is the existing «де» annotation for (4.1) — use as style reference
# (has firstLine=0 and style «a»)
REF_PARA_IDX = 312

def make_de_para(ref_para, text):
    """
    Create a paragraph for a «де» annotation line.
    Copies pPr from ref_para (firstLine=0, style «a»).
    """
    p = ET.Element(_w("p"))
    pPr_src = ref_para.find(_w("pPr"))
    if pPr_src is not None:
        p.append(copy.deepcopy(pPr_src))
    else:
        pPr = ET.SubElement(p, _w("pPr"))
        ps = ET.SubElement(pPr, _w("pStyle"))
        ps.set(_w("val"), "a")
        ind = ET.SubElement(pPr, _w("ind"))
        ind.set(_w("firstLine"), "0")

    first_r = ref_para.find(_w("r"))
    rPr_src = first_r.find(_w("rPr")) if first_r is not None else None

    r = ET.SubElement(p, _w("r"))
    if rPr_src is not None:
        r.append(copy.deepcopy(rPr_src))
    t = ET.SubElement(r, _w("t"))
    if text != text.strip():
        t.set(f"{{{XML_NS}}}space", "preserve")
    t.text = text
    return p


# ── Define all fixes ───────────────────────────────────────────────────
# Each fix: {
#   'remove': [idx, ...],   # body indices to remove (absolute, pre-edit)
#   'insert_at': idx,       # body index to insert new paragraphs at
#   'lines': [str, ...]     # new paragraph texts to insert
# }
# Process in DESCENDING insert_at order → stable indices.

FIXES = [
    # ── (4.30) B623: remove B624-B626 (OMML), insert 4 lines ──────────
    {
        'remove': [624, 625, 626],
        'insert_at': 624,
        'lines': [
            "де xᵢ₅ — вихідна оцінка i-ї альтернативи за критерієм K₅;",
            "      min xᵢ₅ — мінімальне значення критерію K₅ серед усіх альтернатив;",
            "      max xᵢ₅ — максимальне значення критерію K₅ серед усіх альтернатив;",
            "      yᵢ₅ — нормалізована оцінка за критерієм K₅ у діапазоні [0, 1].",
        ],
    },
    # ── (4.29) B616: remove B617-B619 (OMML), insert 4 lines ──────────
    {
        'remove': [617, 618, 619],
        'insert_at': 617,
        'lines': [
            "де xᵢⱼ — вихідна оцінка i-ї альтернативи за j-м критерієм;",
            "      min xᵢⱼ — мінімальне значення j-го критерію серед усіх альтернатив;",
            "      max xᵢⱼ — максимальне значення j-го критерію серед усіх альтернатив;",
            "      yᵢⱼ — нормалізована оцінка у діапазоні [0, 1].",
        ],
    },
    # ── (4.28) B605: remove B606 (OMML), insert 3 lines ───────────────
    {
        'remove': [606],
        'insert_at': 606,
        'lines': [
            "де P — множина Парето-оптимальних альтернатив;",
            "      U(aᵢ) — інтегральна оцінка i-ї альтернативи;",
            "      a* — оптимальна альтернатива з найбільшою зведеною оцінкою.",
        ],
    },
    # ── (4.27) B600: remove B601 (OMML), insert 2 lines ───────────────
    {
        'remove': [601],
        'insert_at': 601,
        'lines': [
            "де yᵢ₁, yᵢ₂, yᵢ₃, yᵢ₄, yᵢ₅ — нормалізовані оцінки i-ї альтернативи за критеріями K₁–K₅;",
            "      0,33; 0,27; 0,20; 0,13; 0,07 — вагові коефіцієнти відповідних критеріїв.",
        ],
    },
    # ── (4.26) B595: remove B596 (OMML), insert 3 lines ───────────────
    {
        'remove': [596],
        'insert_at': 596,
        'lines': [
            "де U(aᵢ) — інтегральна оцінка i-ї альтернативи;",
            "      wⱼ — ваговий коефіцієнт j-го критерію (j = 1, 2, 3, 4, 5);",
            "      yᵢⱼ — нормалізована оцінка i-ї альтернативи за j-м критерієм.",
        ],
    },
    # ── (4.25) B570: remove B571 (OMML), insert 3 lines ───────────────
    {
        'remove': [571],
        'insert_at': 571,
        'lines': [
            "де wⱼ — ваговий коефіцієнт j-го критерію;",
            "      rⱼ — ранг j-го критерію (1 — найважливіший, m — найменш важливий);",
            "      m — кількість критеріїв.",
        ],
    },
    # ── (4.24) B566: remove B567 (OMML), insert 2 lines ───────────────
    {
        'remove': [567],
        'insert_at': 567,
        'lines': [
            "де K₁, K₂, K₃, K₄, K₅ — критерії оцінювання, впорядковані за спаданням важливості;",
            "      ≻ — відношення переваги (важливіший за).",
        ],
    },
    # ── (4.23) B553: remove B554-B556 (OMML), insert 3 lines ──────────
    {
        'remove': [554, 555, 556],
        'insert_at': 554,
        'lines': [
            "де w₁, w₂, w₃, w₄, w₅ — вагові коефіцієнти критеріїв K₁–K₅ відповідно;",
            "      yᵢ₁, yᵢ₂, yᵢ₃, yᵢ₄, yᵢ₅ — нормалізовані оцінки i-ї альтернативи;",
            "      U(aᵢ) — інтегральна оцінка альтернативи aᵢ.",
        ],
    },
    # ── (4.22) B548: remove B549 (OMML), insert 4 lines ───────────────
    {
        'remove': [549],
        'insert_at': 549,
        'lines': [
            "де U(aᵢ) — інтегральна оцінка i-ї альтернативи;",
            "      wⱼ — ваговий коефіцієнт j-го критерію;",
            "      yᵢⱼ — нормалізована оцінка i-ї альтернативи за j-м критерієм;",
            "      m — кількість критеріїв.",
        ],
    },
    # ── (4.21) B530: remove B531-B535 (OMML), insert 5 lines ──────────
    {
        'remove': [531, 532, 533, 534, 535],
        'insert_at': 531,
        'lines': [
            "де P — множина Парето-оптимальних альтернатив;",
            "      A — вся множина альтернатив;",
            "      aᵢ — i-та альтернатива;",
            "      aₖ — будь-яка інша альтернатива з A;",
            "      ≻ — відношення суворого домінування (aₖ ≻ aᵢ означає, що aₖ домінує aᵢ).",
        ],
    },
    # ── (4.20) B523: remove B524-B525 (OMML), insert 2 lines ──────────
    {
        'remove': [524, 525],
        'insert_at': 524,
        'lines': [
            "де yₚⱼ — нормалізована оцінка альтернативи aₚ за j-м критерієм;",
            "      yqⱼ — нормалізована оцінка альтернативи aq за j-м критерієм.",
        ],
    },
    # ── (4.19) B516: remove B517-B519 (OMML), insert 2 lines ──────────
    {
        'remove': [517, 518, 519],
        'insert_at': 517,
        'lines': [
            "де yₚⱼ — нормалізована оцінка альтернативи aₚ за j-м критерієм;",
            "      yqⱼ — нормалізована оцінка альтернативи aq за j-м критерієм.",
        ],
    },
    # ── (4.18) B508: remove B509-B512 (OMML), insert 3 lines ──────────
    {
        'remove': [509, 510, 511, 512],
        'insert_at': 509,
        'lines': [
            "де yₚⱼ — нормалізована оцінка альтернативи aₚ за j-м критерієм;",
            "      yqⱼ — нормалізована оцінка альтернативи aq за j-м критерієм;",
            "      p, q — індекси порівнюваних альтернатив (p ≠ q).",
        ],
    },
    # ── (4.17) B498: remove B499-B502 (OMML), insert 3 lines ──────────
    {
        'remove': [499, 500, 501, 502],
        'insert_at': 499,
        'lines': [
            "де yᵢⱼ — нормалізована оцінка i-ї альтернативи за j-м критерієм;",
            "      i — рядок матриці (індекс альтернативи, i = 1, 2, 3, 4);",
            "      j — стовпець матриці (індекс критерію, j = 1, 2, 3, 4, 5).",
        ],
    },
    # ── (4.16) B491: remove B492-B494 (OMML), insert 2 lines ──────────
    {
        'remove': [492, 493, 494],
        'insert_at': 492,
        'lines': [
            "де yᵢⱼ — нормалізована оцінка i-ї альтернативи за j-м критерієм;",
            "      i — індекс альтернативи (i = 1, 2, 3, 4).",
        ],
    },
    # ── (4.15) B479: remove B480 (OMML), insert 4 lines ───────────────
    {
        'remove': [480],
        'insert_at': 480,
        'lines': [
            "де xᵢⱼ — вихідна оцінка i-ї альтернативи за j-м критерієм;",
            "      min xᵢⱼ — мінімальне значення j-го критерію серед усіх альтернатив;",
            "      max xᵢⱼ — максимальне значення j-го критерію серед усіх альтернатив;",
            "      yᵢⱼ — нормалізована оцінка у діапазоні [0, 1].",
        ],
    },
    # ── (4.14) B475: remove B476 (OMML), insert 4 lines ───────────────
    {
        'remove': [476],
        'insert_at': 476,
        'lines': [
            "де xᵢⱼ — вихідна оцінка i-ї альтернативи за j-м критерієм;",
            "      min xᵢⱼ — мінімальне значення j-го критерію серед усіх альтернатив;",
            "      max xᵢⱼ — максимальне значення j-го критерію серед усіх альтернатив;",
            "      yᵢⱼ — нормалізована оцінка у діапазоні [0, 1].",
        ],
    },
    # ── (4.13) B462: INSERT 3 lines at 463 (before blank), no remove ───
    {
        'remove': [],
        'insert_at': 463,
        'lines': [
            "де xᵢⱼ — оцінка i-ї альтернативи (i = 1, 2, 3, 4) за j-м критерієм (j = 1, 2, 3, 4, 5).",
        ],
    },
    # ── (4.12) B457: remove B458 (OMML), insert 3 lines ───────────────
    {
        'remove': [458],
        'insert_at': 458,
        'lines': [
            "де xᵢⱼ — оцінка i-ї альтернативи за j-м критерієм;",
            "      i — індекс альтернативи (i = 1, 2, 3, 4);",
            "      j — індекс критерію (j = 1, 2, 3, 4, 5).",
        ],
    },
    # ── (4.11) B451: remove B453 only, insert 3 lines at 453 ──────────
    # B452 "де A₁ — BGE-M3;" stays; B453 has 3 params on one line
    {
        'remove': [453],
        'insert_at': 453,
        'lines': [
            "      A₂ — E5-base;",
            "      A₃ — nomic-embed-text-v1.5;",
            "      A₄ — Qwen3-Embedding-0.6B.",
        ],
    },
    # ── (4.5) B354: remove B355 (one-line), insert 2 lines ────────────
    {
        'remove': [355],
        'insert_at': 355,
        'lines': [
            "де h_T⁽ᴸ⁾ — прихований стан останнього (T-го) токена у L-му (фінальному) шарі декодера;",
            "      e(x) — ембеддінг тексту x.",
        ],
    },
    # ── (4.4) B342: remove B343 (one-line), insert 3 lines ────────────
    {
        'remove': [343],
        'insert_at': 343,
        'lines': [
            "де hₜ — прихований стан t-го токена в останньому шарі трансформера;",
            "      T — кількість токенів у послідовності;",
            "      e(x) — результуючий ембеддінг тексту x.",
        ],
    },
    # ── (4.3) B331: remove B332 (one-line), insert 5 lines ────────────
    {
        'remove': [332],
        'insert_at': 332,
        'lines': [
            "де q — запит;",
            "      d⁺ — позитивний документ для q;",
            "      dⱼ — j-й документ у батчі розміром N;",
            "      τ — температурний параметр;",
            "      sim(q, d) — косинусна подібність між ембеддінгами запиту та документа.",
        ],
    },
    # ── (4.1) B311: remove B312 (one-line), insert 3 lines ────────────
    {
        'remove': [312],
        'insert_at': 312,
        'lines': [
            "де E_q — векторне подання запиту (query embedding);",
            "      E_d — векторне подання документа (document embedding);",
            "      ‖·‖ — евклідова норма вектора.",
        ],
    },
]

# Sort by insert_at descending → stable indices during editing
FIXES.sort(key=lambda f: -f['insert_at'])

# ── Execute fixes ──────────────────────────────────────────────────────
ref_para = list(body)[REF_PARA_IDX]  # will be remapped after edits start

for fix in FIXES:
    # Re-read ref_para by searching for it (it stays in body)
    # Since we remove and insert, its position may shift — but content stays
    # Re-fetch from body each iteration (safe because we process descending)
    current_body = list(body)
    ref_para = current_body[REF_PARA_IDX]  # B312 index stays stable since we
    # only touch indices >= 312 in descending order; 312 itself IS removed
    # in the last iteration (4.1 fix). So for the 4.1 fix, ref_para = B312
    # before it's removed. That's fine — we build all new paragraphs first.

    remove_idxs = sorted(fix['remove'], reverse=True)
    insert_at   = fix['insert_at']
    lines       = fix['lines']

    # Build new paragraphs using current ref_para
    new_paras = [make_de_para(ref_para, line) for line in lines]

    # Remove old paragraphs (descending so indices stay valid)
    current_body = list(body)
    elements_to_remove = [current_body[i] for i in remove_idxs]
    for el in elements_to_remove:
        body.remove(el)

    # Insert new paragraphs at insert_at position
    # (after removals, insert_at may need adjustment)
    # Since we removed elements with idx >= insert_at (they're directly after the formula),
    # and insert_at is the position of the first removed element (or the insertion point),
    # we insert at insert_at. But if removes were at insert_at..insert_at+N,
    # after removing them, the element that was at insert_at is gone, so
    # inserting at insert_at places us right after the formula. Correct.
    # For the 4.13 case (no removes), insert_at=463 shifts naturally.
    current_body = list(body)
    # Find actual current insert position
    # Since we removed fix['remove'] elements (all >= insert_at),
    # the element that was at insert_at-1 (the formula) is now at insert_at-1
    # and we insert at insert_at.
    for j, para in enumerate(new_paras):
        body.insert(insert_at + j, para)

    removed_str = str(fix['remove']) if fix['remove'] else 'none'
    print(f"  insert_at={insert_at}, removed={removed_str}, +{len(lines)} lines: {lines[0][:50]!r}")

print(f"\nBody children after all fixes: {len(list(body))}")

# ── Register namespaces & save ─────────────────────────────────────────
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

xml_path = UNPACKED / "word" / "document.xml"
tree.write(str(xml_path), xml_declaration=True, encoding="UTF-8")
print(f"Saved document.xml")

with zipfile.ZipFile(DOCX, "w", zipfile.ZIP_DEFLATED) as zout:
    for fpath in sorted(UNPACKED.rglob("*")):
        if fpath.is_file():
            zout.write(fpath, fpath.relative_to(UNPACKED))
print(f"Repacked: {DOCX.name} ({DOCX.stat().st_size:,} bytes)")
print("DONE.")

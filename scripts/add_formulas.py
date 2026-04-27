"""
add_formulas.py
Insert OMML mathematical formulas into the thesis XML at
unpacked_docx/word/document.xml, then repack the DOCX.

Locations:
  Section 5 (criteria descriptions):
    - After "Якiсть ранжування..." -> DCG@k, nDCG@k
    - After "Швидкiсть знаходження..." -> MRR
    - After "Повнота пошуку..." -> Recall@k
    - After "Точнiсть пошукової видачi..." -> P@k

  Section 6 (BM25):
    - After "Додатково до нейронних моделей..." -> BM25, IDF

  Section 6 (metrics analysis):
    - After "Для кiлькiсного аналiзу..." -> DCG@k, nDCG@k, MRR
"""

import sys
import io
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
XML_PATH = ROOT / "unpacked_docx" / "word" / "document.xml"
UNPACKED = ROOT / "unpacked_docx"
DOCX_PATH = next(ROOT.glob("2026_*.docx"))

# ---------------------------------------------------------------------------
# Namespace constants
# ---------------------------------------------------------------------------
NS_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS_M = "http://schemas.openxmlformats.org/officeDocument/2006/math"
NS_W14 = "http://schemas.microsoft.com/office/word/2010/wordml"

# Register namespaces so ET doesn't mangle them.
# NOTE: prefixes starting with "ns" are reserved by ElementTree for internal
# use and cannot be registered; those URIs will get auto-assigned ns0/ns1/…
# prefixes on serialization, but since we only *add* new formula elements
# (which use w: and m: only) that is acceptable — Word accepts any prefix.
_NS_MAP = {
    "w":   NS_W,
    "m":   NS_M,
    "w14": NS_W14,
    "mc":  "http://schemas.openxmlformats.org/markup-compatibility/2006",
    "r":   "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "wp":  "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
    "w15": "http://schemas.microsoft.com/office/word/2012/wordml",
}
for prefix, uri in _NS_MAP.items():
    ET.register_namespace(prefix, uri)

# ---------------------------------------------------------------------------
# Helper: tag shortcuts
# ---------------------------------------------------------------------------
def _w(tag):
    return f"{{{NS_W}}}{tag}"

def _m(tag):
    return f"{{{NS_M}}}{tag}"

# ---------------------------------------------------------------------------
# OMML builder helpers
# ---------------------------------------------------------------------------

def mr(text, italic=False):
    """Create m:r element with Cambria Math font."""
    r = ET.Element(_m("r"))
    rPr = ET.SubElement(r, _m("rPr"))
    if not italic:
        sty = ET.SubElement(rPr, _m("sty"))
        sty.set(_m("val"), "p")
    # w:rPr with Cambria Math font
    wrPr = ET.SubElement(r, _w("rPr"))
    rFonts = ET.SubElement(wrPr, _w("rFonts"))
    rFonts.set(_w("ascii"), "Cambria Math")
    rFonts.set(_w("hAnsi"), "Cambria Math")
    t = ET.SubElement(r, _m("t"))
    # Use xml:space="preserve" if text has leading/trailing whitespace
    if text != text.strip():
        t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    t.text = text
    return r


def mfrac(num_list, den_list):
    """Create m:f fraction element."""
    f = ET.Element(_m("f"))
    fPr = ET.SubElement(f, _m("fPr"))
    ET.SubElement(fPr, _m("ctrlPr"))
    num = ET.SubElement(f, _m("num"))
    for el in num_list:
        num.append(el)
    den = ET.SubElement(f, _m("den"))
    for el in den_list:
        den.append(el)
    return f


def mssub(base_list, sub_list):
    """Create m:sSub subscript element."""
    ss = ET.Element(_m("sSub"))
    ssPr = ET.SubElement(ss, _m("sSubPr"))
    ET.SubElement(ssPr, _m("ctrlPr"))
    e = ET.SubElement(ss, _m("e"))
    for el in base_list:
        e.append(el)
    sub = ET.SubElement(ss, _m("sub"))
    for el in sub_list:
        sub.append(el)
    return ss


def mssup(base_list, sup_list):
    """Create m:sSup superscript element."""
    ss = ET.Element(_m("sSup"))
    ssPr = ET.SubElement(ss, _m("sSupPr"))
    ET.SubElement(ssPr, _m("ctrlPr"))
    e = ET.SubElement(ss, _m("e"))
    for el in base_list:
        e.append(el)
    sup = ET.SubElement(ss, _m("sup"))
    for el in sup_list:
        sup.append(el)
    return ss


def mssubsup(base_list, sub_list, sup_list):
    """Create m:sSubSup sub+superscript element."""
    ss = ET.Element(_m("sSubSup"))
    ssPr = ET.SubElement(ss, _m("sSubSupPr"))
    ET.SubElement(ssPr, _m("ctrlPr"))
    e = ET.SubElement(ss, _m("e"))
    for el in base_list:
        e.append(el)
    sub = ET.SubElement(ss, _m("sub"))
    for el in sub_list:
        sub.append(el)
    sup = ET.SubElement(ss, _m("sup"))
    for el in sup_list:
        sup.append(el)
    return ss


def mnary(char, sub_list, sup_list, body_list):
    """Create m:nary n-ary operator element (sum, product, etc.)."""
    nary = ET.Element(_m("nary"))
    naryPr = ET.SubElement(nary, _m("naryPr"))
    chr_el = ET.SubElement(naryPr, _m("chr"))
    chr_el.set(_m("val"), char)
    limLoc = ET.SubElement(naryPr, _m("limLoc"))
    limLoc.set(_m("val"), "subSup")
    grow = ET.SubElement(naryPr, _m("grow"))
    grow.set(_m("val"), "0")
    subHide = ET.SubElement(naryPr, _m("subHide"))
    subHide.set(_m("val"), "0")
    supHide = ET.SubElement(naryPr, _m("supHide"))
    supHide.set(_m("val"), "0")
    ET.SubElement(naryPr, _m("ctrlPr"))
    sub_el = ET.SubElement(nary, _m("sub"))
    for el in sub_list:
        sub_el.append(el)
    sup_el = ET.SubElement(nary, _m("sup"))
    for el in sup_list:
        sup_el.append(el)
    e_el = ET.SubElement(nary, _m("e"))
    for el in body_list:
        e_el.append(el)
    return nary


def mfunc(name_list, arg_list):
    """Create m:func function application element."""
    func = ET.Element(_m("func"))
    funcPr = ET.SubElement(func, _m("funcPr"))
    ET.SubElement(funcPr, _m("ctrlPr"))
    fName = ET.SubElement(func, _m("fName"))
    for el in name_list:
        fName.append(el)
    e = ET.SubElement(func, _m("e"))
    for el in arg_list:
        e.append(el)
    return func


def mdel(begin_char, end_char, content_list):
    """Create m:d delimiter element."""
    d = ET.Element(_m("d"))
    dPr = ET.SubElement(d, _m("dPr"))
    begChr = ET.SubElement(dPr, _m("begChr"))
    begChr.set(_m("val"), begin_char)
    endChr = ET.SubElement(dPr, _m("endChr"))
    endChr.set(_m("val"), end_char)
    ET.SubElement(dPr, _m("ctrlPr"))
    e = ET.SubElement(d, _m("e"))
    for el in content_list:
        e.append(el)
    return d


def momath_para(*elements):
    """
    Create a full w:p with oMathPara wrapper (centered, style 'a', right-justified).
    elements: all m:* elements to include inside m:oMath.
    """
    p = ET.Element(_w("p"))
    pPr = ET.SubElement(p, _w("pPr"))
    pStyle = ET.SubElement(pPr, _w("pStyle"))
    pStyle.set(_w("val"), "a")
    jc = ET.SubElement(pPr, _w("jc"))
    jc.set(_w("val"), "right")

    oMathPara = ET.SubElement(p, _m("oMathPara"))
    oMathParaPr = ET.SubElement(oMathPara, _m("oMathParaPr"))
    jc_m = ET.SubElement(oMathParaPr, _m("jc"))
    jc_m.set(_m("val"), "center")

    oMath = ET.SubElement(oMathPara, _m("oMath"))
    for el in elements:
        oMath.append(el)

    return p


# ---------------------------------------------------------------------------
# Formula definitions
# ---------------------------------------------------------------------------

def make_dcg_formula():
    """
    DCG@k = sum_{i=1}^{k} rel_i / log2(i+1)
    """
    # DCG@k =
    lhs = mssub([mr("DCG", italic=False)], [mr("@k", italic=False)])
    eq = mr(" = ", italic=False)
    # sum from i=1 to k
    nary_sum = mnary(
        "∑",
        [mr("i", italic=True), mr(" = 1", italic=False)],
        [mr("k", italic=True)],
        [
            mfrac(
                [mssub([mr("rel", italic=False)], [mr("i", italic=True)])],
                [
                    mssub(
                        [mr("log", italic=False)],
                        [mr("2", italic=False)]
                    ),
                    mr("(i+1)", italic=True),
                ]
            )
        ]
    )
    return momath_para(lhs, eq, nary_sum)


def make_ndcg_formula():
    """
    nDCG@k = DCG@k / IDCG@k
    """
    lhs = mssub([mr("nDCG", italic=False)], [mr("@k", italic=False)])
    eq = mr(" = ", italic=False)
    frac = mfrac(
        [mssub([mr("DCG", italic=False)], [mr("@k", italic=False)])],
        [mssub([mr("IDCG", italic=False)], [mr("@k", italic=False)])]
    )
    return momath_para(lhs, eq, frac)


def make_mrr_formula():
    """
    MRR = (1/|Q|) * sum_{q=1}^{|Q|} (1/rank_q)
    """
    lhs = mr("MRR", italic=False)
    eq = mr(" = ", italic=False)
    coeff = mfrac([mr("1", italic=False)], [mr("|Q|", italic=False)])
    nary_sum = mnary(
        "∑",
        [mr("q", italic=True), mr(" = 1", italic=False)],
        [mr("|Q|", italic=False)],
        [
            mfrac(
                [mr("1", italic=False)],
                [mssub([mr("rank", italic=False)], [mr("q", italic=True)])]
            )
        ]
    )
    dot = mr(" · ", italic=False)
    return momath_para(lhs, eq, coeff, dot, nary_sum)


def make_recall_formula():
    """
    Recall@k = |Rel_ret| / |Rel|
    """
    lhs = mssub([mr("Recall", italic=False)], [mr("@k", italic=False)])
    eq = mr(" = ", italic=False)
    frac = mfrac(
        [mr("|", italic=False),
         mssub([mr("Rel", italic=False)], [mr("ret", italic=False)]),
         mr("|", italic=False)],
        [mr("|Rel|", italic=False)]
    )
    return momath_para(lhs, eq, frac)


def make_precision_formula():
    """
    P@k = |Rel_k| / k
    """
    lhs = mssub([mr("P", italic=False)], [mr("@k", italic=False)])
    eq = mr(" = ", italic=False)
    frac = mfrac(
        [mr("|", italic=False),
         mssub([mr("Rel", italic=False)], [mr("k", italic=False)]),
         mr("|", italic=False)],
        [mr("k", italic=True)]
    )
    return momath_para(lhs, eq, frac)


def make_bm25_formula():
    """
    BM25(D, Q) = sum_{i=1}^{n} IDF(q_i) * (f(q_i,D)*(k1+1)) / (f(q_i,D) + k1*(1-b+b*|D|/avgdl))
    """
    # LHS: BM25(D, Q) =
    lhs_func = mfunc(
        [mr("BM25", italic=False)],
        [mr("D, Q", italic=True)]
    )
    eq = mr(" = ", italic=False)

    # sum_{i=1}^{n}
    nary_sum = mnary(
        "∑",
        [mr("i", italic=True), mr(" = 1", italic=False)],
        [mr("n", italic=True)],
        [
            # IDF(q_i) *
            mfunc(
                [mr("IDF", italic=False)],
                [mssub([mr("q", italic=True)], [mr("i", italic=True)])]
            ),
            mr(" · ", italic=False),
            # fraction: numerator = f(q_i, D) * (k1+1)
            # denominator = f(q_i, D) + k1*(1-b+b*|D|/avgdl)
            mfrac(
                [
                    mfunc(
                        [mr("f", italic=True)],
                        [mssub([mr("q", italic=True)], [mr("i", italic=True)]),
                         mr(", D", italic=True)]
                    ),
                    mr(" · (", italic=False),
                    mssub([mr("k", italic=True)], [mr("1", italic=False)]),
                    mr("+1)", italic=False),
                ],
                [
                    mfunc(
                        [mr("f", italic=True)],
                        [mssub([mr("q", italic=True)], [mr("i", italic=True)]),
                         mr(", D", italic=True)]
                    ),
                    mr(" + ", italic=False),
                    mssub([mr("k", italic=True)], [mr("1", italic=False)]),
                    mr(" · (1−b+b·", italic=False),
                    mfrac(
                        [mr("|D|", italic=False)],
                        [mr("avgdl", italic=False)]
                    ),
                    mr(")", italic=False),
                ]
            )
        ]
    )
    return momath_para(lhs_func, eq, nary_sum)


def make_idf_formula():
    """
    IDF(q_i) = ln( (N - df(q_i) + 0.5) / (df(q_i) + 0.5) + 1 )
    """
    lhs = mfunc(
        [mr("IDF", italic=False)],
        [mssub([mr("q", italic=True)], [mr("i", italic=True)])]
    )
    eq = mr(" = ", italic=False)
    inner_frac = mfrac(
        [
            mr("N − ", italic=False),
            mfunc([mr("df", italic=False)],
                  [mssub([mr("q", italic=True)], [mr("i", italic=True)])]),
            mr(" + 0.5", italic=False),
        ],
        [
            mfunc([mr("df", italic=False)],
                  [mssub([mr("q", italic=True)], [mr("i", italic=True)])]),
            mr(" + 0.5", italic=False),
        ]
    )
    ln_func = mfunc(
        [mr("ln", italic=False)],
        [inner_frac, mr(" + 1", italic=False)]
    )
    return momath_para(lhs, eq, ln_func)


# ---------------------------------------------------------------------------
# Find-and-insert logic
# ---------------------------------------------------------------------------

def get_para_text(p):
    """Get all text from a paragraph element."""
    return "".join(t.text or "" for t in p.iter(_w("t")) if t.text)


def has_omath_para_next(children, idx):
    """Check if the next sibling already has an oMathPara (guard)."""
    if idx + 1 >= len(children):
        return False
    next_el = children[idx + 1]
    return next_el.find(f".//{_m('oMathPara')}") is not None


def insert_after(body, children, idx, new_paragraphs):
    """
    Insert new_paragraphs into body after children[idx].
    Returns number inserted.
    """
    if has_omath_para_next(children, idx):
        print(f"  SKIP: next sibling already has oMathPara at index {idx+1}")
        return 0
    # Find position in body
    # body children list may not match children if we've inserted previously;
    # we use the live body list each time
    live = list(body)
    target = children[idx]
    pos = list(body).index(target)
    for offset, para in enumerate(new_paragraphs):
        body.insert(pos + 1 + offset, para)
    return len(new_paragraphs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading XML...")
    tree = ET.parse(str(XML_PATH))
    root = tree.getroot()
    body = root.find(_w("body"))

    children = list(body)
    print(f"Body children before: {len(children)}")

    # Search targets (substring match)
    search_targets = [
        # (substring, [formula_factory_callables], label)
        (
            "Якість ранжування "
            "результатів пошуку "
            "характеризує здатність моделі",
            [make_dcg_formula, make_ndcg_formula],
            "Section5: DCG + nDCG after ranking quality"
        ),
        (
            "Швидкість знаходження "
            "релевантного результату "
            "відображає",
            [make_mrr_formula],
            "Section5: MRR after speed paragraph"
        ),
        (
            "Повнота пошуку "
            "показує, яку частину "
            "релевантних документів "
            "система здатна",
            [make_recall_formula],
            "Section5: Recall@k after completeness paragraph"
        ),
        (
            "Точність пошукової "
            "видачі характеризує "
            "частку релевантних "
            "документів серед",
            [make_precision_formula],
            "Section5: P@k after precision paragraph"
        ),
        (
            "Додатково до "
            "нейронних моделей "
            "для кожного домену "
            "формується лексичний "
            "інде",
            [make_bm25_formula, make_idf_formula],
            "Section6: BM25 + IDF after lexical index paragraph"
        ),
        (
            "Для кількісного "
            "аналізу використовуються "
            "метрики nDCG@10",
            [make_dcg_formula, make_ndcg_formula, make_mrr_formula],
            "Section6: DCG + nDCG + MRR after metrics paragraph"
        ),
    ]

    total_inserted = 0
    # We rebuild the children list each time after insertions
    for search_text, formula_factories, label in search_targets:
        # Rebuild live children
        live_children = list(body)
        found_idx = None
        for i, child in enumerate(live_children):
            text = get_para_text(child)
            if search_text in text:
                found_idx = i
                break

        if found_idx is None:
            print(f"WARNING: Could not find paragraph for: {label}")
            continue

        print(f"Found paragraph for: {label} at index {found_idx}")

        # Guard: check if next sibling(s) already have oMathPara
        if has_omath_para_next(live_children, found_idx):
            print(f"  SKIP: formula(s) already present after index {found_idx}")
            continue

        # Build formula paragraphs
        new_paras = [factory() for factory in formula_factories]

        # Insert after found_idx
        pos = found_idx  # position in live body
        for offset, para in enumerate(new_paras):
            body.insert(pos + 1 + offset, para)

        total_inserted += len(new_paras)
        print(f"  Inserted {len(new_paras)} formula paragraph(s) after index {found_idx}")

    print(f"\nTotal formula paragraphs inserted: {total_inserted}")
    print(f"Body children after: {len(list(body))}")

    # Write back
    print("Writing XML...")
    tree.write(str(XML_PATH), encoding="utf-8", xml_declaration=True)
    print(f"XML written to: {XML_PATH}")

    # Repack DOCX
    print(f"Repacking DOCX: {DOCX_PATH}")
    with zipfile.ZipFile(str(DOCX_PATH), 'w', zipfile.ZIP_DEFLATED) as zout:
        for fpath in sorted(UNPACKED.rglob('*')):
            if fpath.is_file():
                arcname = fpath.relative_to(UNPACKED)
                zout.write(str(fpath), str(arcname))
    print("DOCX repacked successfully.")


if __name__ == "__main__":
    main()

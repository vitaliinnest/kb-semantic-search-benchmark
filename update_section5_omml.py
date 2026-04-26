"""
Convert the 6 plain-text display formula paragraphs in section 5 back to
proper Word OMML (Office Math Markup Language) equations.

Paragraphs targeted:
  P333 - sim(q,d) = (E_q · E_d)/(‖E_q‖ · ‖E_d‖)   (5.1)  BGE-M3 dense
  P337 - s_sparse(q,d) = Σᵢ wᵢ(q)·wᵢ(d)             (5.2)  BGE-M3 sparse
  P352 - ℒ_InfoNCE = −log[exp(…)/Σ exp(…)]            (5.3)  E5-base
  P363 - e(x) = (1/T)Σ hₜ                             (5.5)  nomic
  P375 - e(x) = h_T^(L)                               (5.6)  Qwen3
  P386 - sim(q,d) = (E_q·E_d)/(‖E_q‖·‖E_d‖)          (5.7)  text-emb-3-large
"""
import sys
import io
import copy
import xml.etree.ElementTree as ET
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# ── Namespace setup ────────────────────────────────────────────────────────────
W_NS  = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
M_NS  = "http://schemas.openxmlformats.org/officeDocument/2006/math"
XML_NS = "http://www.w3.org/XML/1998/namespace"
W  = f"{{{W_NS}}}"
M  = f"{{{M_NS}}}"

for pfx, uri in [
    ("wpc","http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas"),
    ("mc","http://schemas.openxmlformats.org/markup-compatibility/2006"),
    ("r","http://schemas.openxmlformats.org/officeDocument/2006/relationships"),
    ("m","http://schemas.openxmlformats.org/officeDocument/2006/math"),
    ("v","urn:schemas-microsoft-com:vml"),
    ("wp","http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"),
    ("w10","urn:schemas-microsoft-com:office:word"),
    ("w","http://schemas.openxmlformats.org/wordprocessingml/2006/main"),
    ("w14","http://schemas.microsoft.com/office/word/2010/wordml"),
    ("w15","http://schemas.microsoft.com/office/word/2012/wordml"),
    ("w16","http://schemas.microsoft.com/office/word/2018/wordml"),
    ("w16se","http://schemas.microsoft.com/office/word/2015/wordml/symex"),
    ("wne","http://schemas.microsoft.com/office/word/2006/wordml"),
    ("wps","http://schemas.microsoft.com/office/word/2010/wordprocessingShape"),
    ("o","urn:schemas-microsoft-com:office:office"),
    ("wpg","http://schemas.microsoft.com/office/word/2010/wordprocessingGroup"),
    ("wpi","http://schemas.microsoft.com/office/word/2010/wordprocessingInk"),
    ("wp14","http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing"),
    ("w16cex","http://schemas.microsoft.com/office/word/2018/wordml/cex"),
    ("w16cid","http://schemas.microsoft.com/office/word/2016/wordml/cid"),
    ("w16sdtdh","http://schemas.microsoft.com/office/word/2020/wordml/sdtdatahash"),
]:
    ET.register_namespace(pfx, uri)


# ── OMML builder helpers ───────────────────────────────────────────────────────

def _rPr_math(italic: bool = True) -> ET.Element:
    """m:rPr — 'p'=plain/upright for text, default=italic for variables."""
    rPr = ET.Element(f"{M}rPr")
    if not italic:
        sty = ET.SubElement(rPr, f"{M}sty")
        sty.set(f"{M}val", "p")
    return rPr

def _wRPr_cambria() -> ET.Element:
    """w:rPr inside m:r — set Cambria Math font."""
    rPr = ET.Element(f"{W}rPr")
    rFonts = ET.SubElement(rPr, f"{W}rFonts")
    rFonts.set(f"{W}ascii", "Cambria Math")
    rFonts.set(f"{W}hAnsi", "Cambria Math")
    return rPr

def mr(text: str, italic: bool = True) -> ET.Element:
    """Math run element: <m:r><m:rPr>…</m:rPr><m:t>text</m:t></m:r>."""
    r = ET.Element(f"{M}r")
    r.append(_rPr_math(italic))
    r.append(_wRPr_cambria())
    t = ET.SubElement(r, f"{M}t")
    t.text = text
    if text.startswith(" ") or text.endswith(" "):
        t.set(f"{{{XML_NS}}}space", "preserve")
    return r

def msub(base_elems, sub_elems) -> ET.Element:
    """Subscript: base_sub."""
    sSub = ET.Element(f"{M}sSub")
    sSubPr = ET.SubElement(sSub, f"{M}sSubPr")
    ET.SubElement(sSubPr, f"{M}ctrlPr")
    e = ET.SubElement(sSub, f"{M}e")
    for x in base_elems: e.append(x)
    sub = ET.SubElement(sSub, f"{M}sub")
    for x in sub_elems: sub.append(x)
    return sSub

def msup(base_elems, sup_elems) -> ET.Element:
    """Superscript: base^sup."""
    sSup = ET.Element(f"{M}sSup")
    sSupPr = ET.SubElement(sSup, f"{M}sSupPr")
    ET.SubElement(sSupPr, f"{M}ctrlPr")
    e = ET.SubElement(sSup, f"{M}e")
    for x in base_elems: e.append(x)
    sup = ET.SubElement(sSup, f"{M}sup")
    for x in sup_elems: sup.append(x)
    return sSup

def msubsup(base_elems, sub_elems, sup_elems) -> ET.Element:
    """Subscript+superscript: base_sub^sup."""
    sSubSup = ET.Element(f"{M}sSubSup")
    ET.SubElement(ET.SubElement(sSubSup, f"{M}sSubSupPr"), f"{M}ctrlPr")
    e = ET.SubElement(sSubSup, f"{M}e")
    for x in base_elems: e.append(x)
    sub = ET.SubElement(sSubSup, f"{M}sub")
    for x in sub_elems: sub.append(x)
    sup = ET.SubElement(sSubSup, f"{M}sup")
    for x in sup_elems: sup.append(x)
    return sSubSup

def mfrac(num_elems, den_elems) -> ET.Element:
    """Standard fraction bar."""
    f = ET.Element(f"{M}f")
    ET.SubElement(f, f"{M}fPr")           # default = bar fraction
    num = ET.SubElement(f, f"{M}num")
    den = ET.SubElement(f, f"{M}den")
    for x in num_elems: num.append(x)
    for x in den_elems: den.append(x)
    return f

def mnary(chr_sym: str, sub_elems, sup_elems, body_elems,
          hide_sup: bool = False) -> ET.Element:
    """N-ary operator (∑, ∏, …)."""
    nary = ET.Element(f"{M}nary")
    naryPr = ET.SubElement(nary, f"{M}naryPr")
    ET.SubElement(naryPr, f"{M}chr").set(f"{M}val", chr_sym)
    ET.SubElement(naryPr, f"{M}limLoc").set(f"{M}val", "subSup")
    ET.SubElement(naryPr, f"{M}grow").set(f"{M}val", "0")
    ET.SubElement(naryPr, f"{M}subHide").set(f"{M}val", "0")
    ET.SubElement(naryPr, f"{M}supHide").set(f"{M}val", "1" if hide_sup else "0")
    sub = ET.SubElement(nary, f"{M}sub")
    for x in sub_elems: sub.append(x)
    sup = ET.SubElement(nary, f"{M}sup")
    for x in sup_elems: sup.append(x)
    body = ET.SubElement(nary, f"{M}e")
    for x in body_elems: body.append(x)
    return nary

def mfunc(name_elems, arg_elems) -> ET.Element:
    """Function application: name(arg)."""
    func = ET.Element(f"{M}func")
    ET.SubElement(ET.SubElement(func, f"{M}funcPr"), f"{M}ctrlPr")
    fName = ET.SubElement(func, f"{M}fName")
    for x in name_elems: fName.append(x)
    e = ET.SubElement(func, f"{M}e")
    for x in arg_elems: e.append(x)
    return func

def mdelim(open_chr: str, close_chr: str, body_elems) -> ET.Element:
    """Delimiters: (body), [body], etc."""
    d = ET.Element(f"{M}d")
    dPr = ET.SubElement(d, f"{M}dPr")
    ET.SubElement(dPr, f"{M}begChr").set(f"{M}val", open_chr)
    ET.SubElement(dPr, f"{M}endChr").set(f"{M}val", close_chr)
    e = ET.SubElement(d, f"{M}e")
    for x in body_elems: e.append(x)
    return d

def set_para_omml(para, oMath_children: list) -> None:
    """
    Replace paragraph content with a centred oMathPara display equation.
    Preserves existing pPr.
    """
    pPr = para.find(f"{W}pPr")
    for child in list(para):
        para.remove(child)
    if pPr is not None:
        para.append(pPr)

    oMathPara = ET.SubElement(para, f"{M}oMathPara")
    oMathParaPr = ET.SubElement(oMathPara, f"{M}oMathParaPr")
    jc = ET.SubElement(oMathParaPr, f"{M}jc")
    jc.set(f"{M}val", "center")

    oMath = ET.SubElement(oMathPara, f"{M}oMath")
    for child in oMath_children:
        oMath.append(child)


# ── Helper: E_q subscript element ─────────────────────────────────────────────
def E_sub(letter: str):
    """E_letter subscript."""
    return msub([mr("E")], [mr(letter)])

def norm_E_sub(letter: str):
    """‖E_letter‖ as math sequence."""
    return [mr("‖"), E_sub(letter), mr("‖")]


# ════════════════════════════════════════════════════════════════════════════════
# Load document
# ════════════════════════════════════════════════════════════════════════════════
DOC_XML = Path("D:/repos/kb-semantic-search-benchmark/unpacked_docx/word/document.xml")
tree = ET.parse(str(DOC_XML))
root = tree.getroot()
paras = root.findall(f".//{W}p")
print(f"Total paragraphs: {len(paras)}")


# ════════════════════════════════════════════════════════════════════════════════
# P333: sim(q, d) = (E_q · E_d) / (‖E_q‖ · ‖E_d‖)     (5.1)
# ════════════════════════════════════════════════════════════════════════════════
cosine_formula_51 = [
    mfunc([mr("sim", italic=False)],
          [mr("q, d")]),
    mr(" = ", italic=False),
    mfrac(
        num_elems=[E_sub("q"), mr(" · "), E_sub("d")],
        den_elems=norm_E_sub("q") + [mr(" · ")] + norm_E_sub("d"),
    ),
    mr("    (5.1)", italic=False),
]
set_para_omml(paras[333], cosine_formula_51)
print("P333 ✓  cosine sim (5.1)")


# ════════════════════════════════════════════════════════════════════════════════
# P337: s_sparse(q, d) = Σᵢ wᵢ(q) · wᵢ(d)    (5.2)
# ════════════════════════════════════════════════════════════════════════════════
w_sub_i_q = [msub([mr("w")], [mr("i")]), mr("(q)")]
w_sub_i_d = [msub([mr("w")], [mr("i")]), mr("(d)")]

sparse_formula_52 = [
    msub([mr("s", italic=False)],
         [mr("sparse", italic=False)]),
    mr("(q, d) = ", italic=False),
    mnary("∑",
          sub_elems=[mr("i")],
          sup_elems=[],
          body_elems=w_sub_i_q + [mr(" · ")] + w_sub_i_d,
          hide_sup=True),
    mr("    (5.2)", italic=False),
]
set_para_omml(paras[337], sparse_formula_52)
print("P337 ✓  sparse retrieval (5.2)")


# ════════════════════════════════════════════════════════════════════════════════
# P352: ℒ_InfoNCE = −log[ exp(sim(q,d⁺)/τ) / Σⱼ₌₁ᴺ exp(sim(q,dⱼ)/τ) ]  (5.3)
# ════════════════════════════════════════════════════════════════════════════════
# numerator of the log-argument fraction
def sim_frac(q_arg_elems, tau_char="τ"):
    """exp( sim(q, arg) / τ ) as m:func(exp, m:frac(...))."""
    sim_call = mfunc([mr("sim", italic=False)],
                     [mr("q, ")] + q_arg_elems)
    inner_frac = mfrac([sim_call], [mr(tau_char)])
    return mfunc([mr("exp", italic=False)], [inner_frac])

d_plus  = msup([mr("d")], [mr("+")])
d_j     = msub([mr("d")], [mr("j")])

log_num = [sim_frac([d_plus])]

log_den = [
    mnary("∑",
          sub_elems=[mr("j=1")],
          sup_elems=[mr("N")],
          body_elems=[sim_frac([d_j])],
          hide_sup=False),
]

log_arg_frac = mfrac(log_num, log_den)

infonce_formula_53 = [
    msub([mr("ℒ")], [mr("InfoNCE", italic=False)]),
    mr(" = −", italic=False),
    mfunc([mr("log", italic=False)],
          [mdelim("[", "]", [log_arg_frac])]),
    mr("    (5.3)", italic=False),
]
set_para_omml(paras[352], infonce_formula_53)
print("P352 ✓  InfoNCE (5.3)")


# ════════════════════════════════════════════════════════════════════════════════
# P363: e(x) = (1/T) Σₜ₌₁ᵀ hₜ    (5.5)
# ════════════════════════════════════════════════════════════════════════════════
mean_pool_55 = [
    mfunc([mr("e", italic=True)], [mr("x")]),
    mr(" = ", italic=False),
    mfrac([mr("1")], [mr("T")]),
    mnary("∑",
          sub_elems=[mr("t=1")],
          sup_elems=[mr("T")],
          body_elems=[msub([mr("h")], [mr("t")])]),
    mr("    (5.5)", italic=False),
]
set_para_omml(paras[363], mean_pool_55)
print("P363 ✓  mean pooling (5.5)")


# ════════════════════════════════════════════════════════════════════════════════
# P375: e(x) = h_T^(L)    (5.6)
# ════════════════════════════════════════════════════════════════════════════════
h_T_L = msubsup(
    base_elems=[mr("h")],
    sub_elems=[mr("T")],
    sup_elems=[mdelim("(", ")", [mr("L")])],
)
last_token_56 = [
    mfunc([mr("e", italic=True)], [mr("x")]),
    mr(" = ", italic=False),
    h_T_L,
    mr("    (5.6)", italic=False),
]
set_para_omml(paras[375], last_token_56)
print("P375 ✓  last-token pooling (5.6)")


# ════════════════════════════════════════════════════════════════════════════════
# P386: sim(q, d) = (E_q · E_d) / (‖E_q‖ · ‖E_d‖)    (5.7)
# ════════════════════════════════════════════════════════════════════════════════
cosine_formula_57 = [
    mfunc([mr("sim", italic=False)],
          [mr("q, d")]),
    mr(" = ", italic=False),
    mfrac(
        num_elems=[E_sub("q"), mr(" · "), E_sub("d")],
        den_elems=norm_E_sub("q") + [mr(" · ")] + norm_E_sub("d"),
    ),
    mr("    (5.7)", italic=False),
]
set_para_omml(paras[386], cosine_formula_57)
print("P386 ✓  cosine sim (5.7)")


# ── Save ────────────────────────────────────────────────────────────────────────
tree.write(str(DOC_XML), encoding="unicode", xml_declaration=False)
print("\nSaved document.xml with proper OMML formulas.")

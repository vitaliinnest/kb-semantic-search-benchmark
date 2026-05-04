import sys, pathlib, xml.etree.ElementTree as ET
sys.stdout.reconfigure(encoding='utf-8')

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
M = "http://schemas.openxmlformats.org/officeDocument/2006/math"
def _w(t): return f"{{{W}}}{t}"

def para_text(p):
    return "".join((t.text or "") for t in p.iter(_w("t")))

def get_style(p):
    ppr = p.find(_w("pPr"))
    if ppr is not None:
        ps = ppr.find(_w("pStyle"))
        if ps is not None:
            return ps.get(_w("val"), "")
    return ""

def get_jc(p):
    ppr = p.find(_w("pPr"))
    if ppr is not None:
        jc = ppr.find(_w("jc"))
        if jc is not None:
            return jc.get(_w("val"), "")
    return ""

UNPACKED = pathlib.Path("D:/repos/kb-semantic-search-benchmark/thesis/unpacked_docx")
tree = ET.parse(UNPACKED / "word" / "document.xml")
body = tree.getroot().find(_w("body"))

LATEX_MARKERS = [r"\;", r"\cdot", r"\times", r"\frac", r"\sum", r"\sqrt",
                 r"\left", r"\right", r"\text{", r"\mathrm{", r"\mathbf{",
                 r"\leq", r"\geq", r"\neq", r"\approx", r"\ln", r"\log",
                 r"\infty", r"\alpha", r"\beta", r"\lambda", r"\sigma",
                 r"\mu", r"\pi", r"\Delta", r"\nabla", r"\in", r"\cup",
                 r"\cap", r"\forall", r"\exists", r"\rightarrow", r"\to"]

print("=== Paragraphs with LaTeX artifacts ===")
for i, p in enumerate(body):
    txt = para_text(p)
    hits = [lx for lx in LATEX_MARKERS if lx in txt]
    if hits:
        style = get_style(p)
        jc = get_jc(p)
        has_m = p.find(f".//{{{M}}}oMath") is not None
        print(f"B{i} style={style!r} jc={jc!r} math={has_m}: {repr(txt[:120])}")
        print(f"     hits: {hits}")

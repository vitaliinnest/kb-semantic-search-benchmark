import sys, pathlib, xml.etree.ElementTree as ET
sys.stdout.reconfigure(encoding='utf-8')

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
M = "http://schemas.openxmlformats.org/officeDocument/2006/math"
def _w(t): return f"{{{W}}}{t}"
def _m(t): return f"{{{M}}}{t}"

def para_text(p):
    return "".join((t.text or "") for t in p.iter(_w("t")))

def math_text(p):
    return "".join((t.text or "") for t in p.iter(_m("t")))

def all_text(p):
    parts = []
    for t in p.iter():
        if t.tag in (_w("t"), _m("t")):
            parts.append(t.text or "")
    return "".join(parts)

UNPACKED = pathlib.Path("D:/repos/kb-semantic-search-benchmark/unpacked_docx")
tree = ET.parse(UNPACKED / "word" / "document.xml")
body = tree.getroot().find(_w("body"))

BAD_IN_MATH = [r"\;", r"\cdot", r"\times", r"\frac", r"\sum", r"\sqrt",
               r"\left", r"\right", r"\text", r"\mathrm", r"\mathbf",
               r"\leq", r"\geq", r"\neq", r"\approx"]

print("=== Math elements with LaTeX artifacts ===")
found = 0
for i, p in enumerate(body):
    mt = math_text(p)
    if not mt:
        continue
    hits = [b for b in BAD_IN_MATH if b in mt]
    if hits:
        found += 1
        print(f"B{i}: math_text={repr(mt[:200])}")
        print(f"     hits: {hits}")
        print()

print(f"Total: {found} paragraphs with LaTeX in math")

# Also look for paragraphs that are centered/equation-style but have NO math and contain backslash
print()
print("=== Centered paragraphs with backslash (possible broken display formulas) ===")
for i, p in enumerate(body):
    txt = para_text(p)
    if "\\" not in txt:
        continue
    ppr = p.find(_w("pPr"))
    jc = ""
    if ppr is not None:
        jc_el = ppr.find(_w("jc"))
        if jc_el is not None:
            jc = jc_el.get(_w("val"), "")
    if jc == "center" or "=" in txt or "(" in txt:
        print(f"B{i} jc={jc!r}: {repr(txt[:150])}")

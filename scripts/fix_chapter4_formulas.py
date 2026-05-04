"""
Виправляє всі побиті формули розділу 4.

BAD_CHARS (B313, B372, B377, B392, B407, B422, B584):
  — LaTeX-мотлох у тексті замість OMML → повна заміна m:oMath

EMPTY_FUNC_ARG (B326, B481, B488, B629, B642, B650):
  — m:func з порожнім m:e → спеціалізовані виправлення

EMPTY_SSUB_BASE (B326 — L_InfoNCE):
  — r"L" + func(sSub(base={}, sub="InfoNCE"), e={}) → sSub(L, InfoNCE)
"""
import sys, shutil, zipfile, pathlib, xml.etree.ElementTree as ET
sys.stdout.reconfigure(encoding='utf-8')

ROOT     = pathlib.Path("D:/repos/kb-semantic-search-benchmark")
DOCX     = ROOT / "thesis" / "2026_M_PI_Nesterenko_VV.docx"
UNPACKED = ROOT / "thesis" / "unpacked_docx"

W   = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
M   = "http://schemas.openxmlformats.org/officeDocument/2006/math"
XNS = "http://www.w3.org/XML/1998/namespace"
def _w(t): return f"{{{W}}}{t}"
def _m(t): return f"{{{M}}}{t}"
def _x(t): return f"{{{XNS}}}{t}"

if UNPACKED.exists():
    shutil.rmtree(UNPACKED)
with zipfile.ZipFile(DOCX) as z:
    z.extractall(UNPACKED)
print("Re-unpacked docx.")

DOC_XML = UNPACKED / "word" / "document.xml"
tree    = ET.parse(DOC_XML)
body    = tree.getroot().find(_w("body"))

# ── OMML builder helpers ──────────────────────────────────────────────────────

def _cpr(italic=False):
    c = ET.Element(_m("ctrlPr"))
    rpr = ET.SubElement(c, _w("rPr"))
    f   = ET.SubElement(rpr, _w("rFonts"))
    f.set(_w("ascii"),  "Cambria Math")
    f.set(_w("hAnsi"), "Cambria Math")
    if italic: ET.SubElement(rpr, _w("i"))
    return c

def R(text, roman=False):
    """m:r  —  roman=True → upright / function style"""
    r = ET.Element(_m("r"))
    if roman:
        rpr = ET.SubElement(r, _m("rPr"))
        sty = ET.SubElement(rpr, _m("sty"))
        sty.set(_m("val"), "p")
    wrpr = ET.SubElement(r, _w("rPr"))
    f = ET.SubElement(wrpr, _w("rFonts"))
    f.set(_w("ascii"),  "Cambria Math")
    f.set(_w("hAnsi"), "Cambria Math")
    t = ET.SubElement(r, _m("t"))
    t.text = text
    if text != text.strip():
        t.set(_x("space"), "preserve")
    return r

def SUB(base, sub):
    """m:sSub(base, sub)  — base and sub are lists of elements"""
    s   = ET.Element(_m("sSub"))
    pr  = ET.SubElement(s, _m("sSubPr")); pr.append(_cpr())
    e   = ET.SubElement(s, _m("e"));  [e.append(c) for c in base]; e.append(_cpr())
    sb  = ET.SubElement(s, _m("sub")); [sb.append(c) for c in sub]
    return s

def SUP(base, sup):
    """m:sSup"""
    s   = ET.Element(_m("sSup"))
    pr  = ET.SubElement(s, _m("sSupPr")); pr.append(_cpr())
    e   = ET.SubElement(s, _m("e"));  [e.append(c) for c in base]
    sp  = ET.SubElement(s, _m("sup")); [sp.append(c) for c in sup]
    return s

def FRAC(num, den):
    """m:f"""
    f  = ET.Element(_m("f"))
    pr = ET.SubElement(f, _m("fPr")); pr.append(_cpr())
    n  = ET.SubElement(f, _m("num")); [n.append(c) for c in num]
    d  = ET.SubElement(f, _m("den")); [d.append(c) for c in den]
    return f

def NARY(char, sub, sup, e):
    """m:nary (∑, ∏ …)"""
    n   = ET.Element(_m("nary"))
    pr  = ET.SubElement(n, _m("naryPr"))
    ch  = ET.SubElement(pr, _m("chr")); ch.set(_m("val"), char)
    if not sup:
        h = ET.SubElement(pr, _m("supHide")); h.set(_m("val"), "1")
    pr.append(_cpr())
    s   = ET.SubElement(n, _m("sub")); [s.append(c) for c in sub]
    sp  = ET.SubElement(n, _m("sup")); [sp.append(c) for c in sup]
    ev  = ET.SubElement(n, _m("e"));  [ev.append(c) for c in e]; ev.append(_cpr(italic=True))
    return n

def DELIM(beg, end, inner):
    """m:d"""
    d  = ET.Element(_m("d"))
    pr = ET.SubElement(d, _m("dPr"))
    if beg != "(":
        b = ET.SubElement(pr, _m("begChr")); b.set(_m("val"), beg)
    if end != ")":
        en = ET.SubElement(pr, _m("endChr")); en.set(_m("val"), end)
    pr.append(_cpr())
    e  = ET.SubElement(d, _m("e")); [e.append(c) for c in inner]
    return d

def ABS(inner):
    return DELIM("|", "|", inner)

def FUNC(fname, arg):
    """m:func"""
    fn  = ET.Element(_m("func"))
    pr  = ET.SubElement(fn, _m("funcPr")); pr.append(_cpr())
    fnm = ET.SubElement(fn, _m("fName")); [fnm.append(c) for c in fname]; fnm.append(_cpr(italic=True))
    e   = ET.SubElement(fn, _m("e")); [e.append(c) for c in arg]
    return fn

def LIML(base, lim):
    """m:limLow"""
    ll  = ET.Element(_m("limLow"))
    pr  = ET.SubElement(ll, _m("limLowPr")); pr.append(_cpr(italic=True))
    e   = ET.SubElement(ll, _m("e")); [e.append(c) for c in base]
    lv  = ET.SubElement(ll, _m("lim")); [lv.append(c) for c in lim]; lv.append(_cpr())
    return ll

def OM(*ch):
    """m:oMath"""
    om = ET.Element(_m("oMath"))
    for c in ch: om.append(c)
    return om

# ── Formulas (fresh OMML) ────────────────────────────────────────────────────

def build_4_2():
    """s_sparse(q, d) = Σᵢ wᵢ(q) · wᵢ(d)"""
    return OM(
        SUB([R("s")], [R("sparse", roman=True)]),
        DELIM("(", ")", [R("q"), R(", "), R("d")]),
        R(" = ", roman=True),
        NARY("∑", [R("i")], [],
             [SUB([R("w")], [R("i")]),
              DELIM("(", ")", [R("q")]),
              R(" ⋅ "),
              SUB([R("w")], [R("i")]),
              DELIM("(", ")", [R("d")])])
    )

def build_4_6():
    """DCG@k = Σᵢ₌₁ᵏ relᵢ / log₂(i+1)"""
    return OM(
        SUB([R("DCG", roman=True)], [R("@k", roman=True)]),
        R(" = ", roman=True),
        NARY("∑", [R("i = 1")], [R("k")],
             [FRAC(
                 [SUB([R("rel", roman=True)], [R("i")])],
                 [FUNC([SUB([R("log", roman=True)], [R("2")])],
                       [DELIM("(", ")", [R("i"), R(" + "), R("1")])])]
             )])
    )

def build_4_7():
    """nDCG@k = DCG@k / IDCG@k"""
    return OM(
        SUB([R("nDCG", roman=True)], [R("@k", roman=True)]),
        R(" = ", roman=True),
        FRAC(
            [SUB([R("DCG",  roman=True)], [R("@k", roman=True)])],
            [SUB([R("IDCG", roman=True)], [R("@k", roman=True)])]
        )
    )

def build_4_8():
    """MRR = (1/|Q|) · Σ_{q=1}^|Q| (1/rank_q)"""
    return OM(
        R("MRR", roman=True),
        R(" = ", roman=True),
        FRAC([R("1")], [ABS([R("Q")])]),
        R(" · "),
        NARY("∑", [R("q = 1")], [ABS([R("Q")])],
             [FRAC([R("1")],
                   [SUB([R("rank", roman=True)], [R("q")])])])
    )

def build_4_9():
    """Recall@k = |Rel_ret| / |Rel|"""
    return OM(
        SUB([R("Recall", roman=True)], [R("@k", roman=True)]),
        R(" = ", roman=True),
        FRAC(
            [ABS([SUB([R("Rel", roman=True)], [R("ret", roman=True)])])],
            [ABS([R("Rel", roman=True)])]
        )
    )

def build_4_10():
    """P@k = |Rel_k| / k"""
    return OM(
        SUB([R("P", roman=True)], [R("@k", roman=True)]),
        R(" = ", roman=True),
        FRAC(
            [ABS([SUB([R("Rel", roman=True)], [R("k")])])],
            [R("k")]
        )
    )

def build_4_25():
    """w_j = (m − r_j + 1) / Σ_{k=1}^{m} (m − r_k + 1)"""
    return OM(
        SUB([R("w")], [R("j")]),
        R(" = ", roman=True),
        FRAC(
            [R("m"), R(" − "), SUB([R("r")], [R("j")]), R(" + "), R("1")],
            [NARY("∑", [R("k = 1")], [R("m")],
                  [DELIM("(", ")", [R("m"), R(" − "), SUB([R("r")], [R("k")]), R(" + "), R("1")])])]
        )
    )

# ── Patch helpers ─────────────────────────────────────────────────────────────

def para_text(p):
    return "".join((t.text or "") for t in p.iter(_w("t")))

def math_text(p):
    return "".join((t.text or "") for t in p.iter(_m("t")))

def has_math(p):
    return p.find(f".//{{{M}}}oMath") is not None

def find_formula(body, num_str, start=250, end=680):
    children = list(body)
    for i in range(start, min(end, len(children))):
        p = children[i]
        if has_math(p) and num_str in para_text(p):
            return i, p
    return -1, None

def replace_omath(para, new_omath):
    """Replace the m:oMath element inside para with new_omath."""
    # Build parent map for this paragraph
    parent_map = {child: parent for parent in para.iter() for child in parent}
    old = para.find(f".//{{{M}}}oMath")
    if old is None:
        return False
    parent = parent_map.get(old, para)
    children = list(parent)
    idx = children.index(old)
    parent.remove(old)
    parent.insert(idx, new_omath)
    return True

def is_empty_e(e_el):
    return all(ch.tag == _m("ctrlPr") for ch in e_el)

def is_space_run(el):
    if el.tag != _m("r"):
        return False
    return all((t.text or "").strip() == "" for t in el.iter(_m("t")))

def fix_empty_func_args_recursive(parent):
    """Scan parent's children; for each m:func with empty m:e, move the next
    non-whitespace sibling into m:e (removes the skipped spaces).
    Recurse into all non-func children."""
    changed = True
    while changed:
        changed = False
        children = list(parent)
        for i, el in enumerate(children):
            if el.tag == _m("func"):
                e_el = el.find(_m("e"))
                if e_el is not None and is_empty_e(e_el):
                    j = i + 1
                    while j < len(children) and is_space_run(children[j]):
                        j += 1
                    if j < len(children) and children[j].tag != _m("ctrlPr"):
                        target = children[j]
                        spaces  = children[i + 1 : j]
                        e_el.append(target)
                        parent.remove(target)
                        for sp in spaces:
                            parent.remove(sp)
                        changed = True
                        break
    # Recurse
    for ch in list(parent):
        fix_empty_func_args_recursive(ch)

# ── Apply fixes ───────────────────────────────────────────────────────────────

# 1. BAD_CHARS: replace entire oMath
REBUILDS = [
    ("(4.2)",  build_4_2),
    ("(4.6)",  build_4_6),
    ("(4.7)",  build_4_7),
    ("(4.8)",  build_4_8),
    ("(4.9)",  build_4_9),
    ("(4.10)", build_4_10),
    ("(4.25)", build_4_25),
]
for num_str, builder in REBUILDS:
    idx, para = find_formula(body, num_str)
    if para is None:
        print(f"[!] Formula {num_str}: NOT FOUND")
        continue
    new_om = builder()
    ok = replace_omath(para, new_om)
    print(f"[OK] {num_str} at B{idx}: oMath replaced." if ok else f"[!] {num_str}: replace failed")

# 2. Formula (4.3): fix L_InfoNCE
#    Current: r"L" + func(fName=[sSub(e={}, sub=[InfoNCE])], e={ctrlPr})
#    Target:  sSub(e=[r"L"], sub=[r"InfoNCE"])
idx3, para3 = find_formula(body, "(4.3)")
if para3 is None:
    print("[!] (4.3): NOT FOUND")
else:
    om3 = para3.find(f".//{{{M}}}oMath")
    children3 = list(om3)
    # Find r"L" and the broken func that follows
    L_idx = -1
    for i, ch in enumerate(children3):
        mt = "".join((t.text or "") for t in ch.iter(_m("t")))
        if ch.tag == _m("r") and mt.strip() == "L":
            L_idx = i
            break
    if L_idx >= 0 and L_idx + 1 < len(children3):
        broken_func = children3[L_idx + 1]
        if broken_func.tag == _m("func"):
            new_sub = SUB([R("L")], [R("InfoNCE", roman=True)])
            om3.insert(L_idx, new_sub)
            om3.remove(children3[L_idx])   # remove old r"L"
            om3.remove(broken_func)
            print(f"[OK] (4.3) at B{idx3}: L_InfoNCE fixed.")
        else:
            print(f"[!] (4.3): broken_func not found after L")
    else:
        print(f"[!] (4.3): L run not found")

# 3. EMPTY_FUNC_ARG formulas: recursive fix
for num_str, start, end in [
    ("(4.14)", 470, 495),
    ("(4.15)", 478, 502),
    ("(4.29)", 632, 655),
    ("(4.30)", 640, 665),
]:
    idx_f, para_f = find_formula(body, num_str, start=start, end=end)
    if para_f is None:
        print(f"[!] {num_str}: NOT FOUND")
        continue
    om_f = para_f.find(f".//{{{M}}}oMath")
    fix_empty_func_args_recursive(om_f)
    print(f"[OK] {num_str} at B{idx_f}: empty func args fixed.")

# 4. Formula (4.28): argmax — move U(a_i) into inner func's e
idx28, para28 = find_formula(body, "(4.28)", start=620, end=645)
if para28 is None:
    print("[!] (4.28): NOT FOUND")
else:
    om28 = para28.find(f".//{{{M}}}oMath")
    # Find the outer func (fName="arg") in oMath's children
    outer_func = None
    outer_func_idx = -1
    for i, ch in enumerate(list(om28)):
        if ch.tag == _m("func"):
            fname = ch.find(_m("fName"))
            if fname is not None:
                fname_texts = "".join((t.text or "") for t in fname.iter(_m("t")))
                if "arg" in fname_texts:
                    outer_func = ch
                    outer_func_idx = i
                    break
    if outer_func is None:
        print("[!] (4.28): outer func 'arg' not found")
    else:
        # Find inner func inside outer func's e
        outer_e = outer_func.find(_m("e"))
        inner_func = None
        if outer_e is not None:
            for ch in outer_e:
                if ch.tag == _m("func"):
                    e_ch = ch.find(_m("e"))
                    if e_ch is not None and is_empty_e(e_ch):
                        inner_func = ch
                        break
        if inner_func is None:
            print("[!] (4.28): inner func with empty e not found")
        else:
            inner_e = inner_func.find(_m("e"))
            # Collect elements after outer_func in oMath (skip space runs)
            om28_children = list(om28)
            after_elements = om28_children[outer_func_idx + 1:]
            # Move non-empty elements into inner_func's e
            for el in after_elements:
                om28.remove(el)
                inner_e.append(el)
            print(f"[OK] (4.28) at B{idx28}: moved {len(after_elements)} element(s) into inner func e.")


# ── Save & repack ──────────────────────────────────────────────────────────────
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

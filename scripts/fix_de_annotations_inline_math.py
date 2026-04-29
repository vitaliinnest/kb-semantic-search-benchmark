"""
Перетворює plain-text символи у "де:" абзацах (після формул розділів 4 і 5)
на inline OMML (m:oMath).

Кожен рядок має структуру:
  "де [symbol] — [description]"   або
  "     [symbol] — [description]"

Скрипт розбиває кожен такий рядок на 3 частини і перебудовує параграф:
  w:r("де " | "     ")  +  m:oMath(symbol)  +  w:r(" — description...")
"""
import sys, re, shutil, zipfile, pathlib, xml.etree.ElementTree as ET
sys.stdout.reconfigure(encoding='utf-8')

ROOT     = pathlib.Path("D:/repos/kb-semantic-search-benchmark")
DOCX     = ROOT / "2026_M_PI_Nesterenko_VV.docx"
UNPACKED = ROOT / "unpacked_docx"

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
    if italic:
        ET.SubElement(rpr, _w("i"))
    return c

def R(text, roman=False):
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
    s  = ET.Element(_m("sSub"))
    pr = ET.SubElement(s, _m("sSubPr")); pr.append(_cpr())
    e  = ET.SubElement(s, _m("e"));  [e.append(c) for c in base]; e.append(_cpr())
    sb = ET.SubElement(s, _m("sub")); [sb.append(c) for c in sub]
    return s

def SUP(base, sup):
    s  = ET.Element(_m("sSup"))
    pr = ET.SubElement(s, _m("sSupPr")); pr.append(_cpr())
    e  = ET.SubElement(s, _m("e"));  [e.append(c) for c in base]
    sp = ET.SubElement(s, _m("sup")); [sp.append(c) for c in sup]
    return s

def SUBSUP(base, sub, sup):
    s  = ET.Element(_m("sSubSup"))
    pr = ET.SubElement(s, _m("sSubSupPr")); pr.append(_cpr())
    e  = ET.SubElement(s, _m("e"));  [e.append(c) for c in base]
    sb = ET.SubElement(s, _m("sub")); [sb.append(c) for c in sub]
    sp = ET.SubElement(s, _m("sup")); [sp.append(c) for c in sup]
    return s

def FRAC(num, den):
    f  = ET.Element(_m("f"))
    pr = ET.SubElement(f, _m("fPr")); pr.append(_cpr())
    n  = ET.SubElement(f, _m("num")); [n.append(c) for c in num]
    d  = ET.SubElement(f, _m("den")); [d.append(c) for c in den]
    return f

def DELIM(beg, end, inner):
    d  = ET.Element(_m("d"))
    pr = ET.SubElement(d, _m("dPr"))
    if beg != "(": b = ET.SubElement(pr, _m("begChr")); b.set(_m("val"), beg)
    if end != ")": e = ET.SubElement(pr, _m("endChr")); e.set(_m("val"), end)
    pr.append(_cpr())
    ev = ET.SubElement(d, _m("e")); [ev.append(c) for c in inner]
    return ev  # wrong — fix below

def DELIM(beg, end, inner):
    d  = ET.Element(_m("d"))
    pr = ET.SubElement(d, _m("dPr"))
    if beg != "(":
        b = ET.SubElement(pr, _m("begChr")); b.set(_m("val"), beg)
    if end != ")":
        en = ET.SubElement(pr, _m("endChr")); en.set(_m("val"), end)
    pr.append(_cpr())
    ev = ET.SubElement(d, _m("e")); [ev.append(c) for c in inner]
    return d

def ABS(inner):
    return DELIM("|", "|", inner)

def FUNC(fname_elems, arg_elems):
    fn  = ET.Element(_m("func"))
    pr  = ET.SubElement(fn, _m("funcPr")); pr.append(_cpr())
    fnm = ET.SubElement(fn, _m("fName")); [fnm.append(c) for c in fname_elems]
    ev  = ET.SubElement(fn, _m("e")); [ev.append(c) for c in arg_elems]
    return fn

def OM(*children):
    om = ET.Element(_m("oMath"))
    for c in children: om.append(c)
    return om

# ── Symbol parser ─────────────────────────────────────────────────────────────

# Unicode subscript → base char
SUBS = {
    'ᵢ':'i', 'ⱼ':'j', 'ₜ':'t', 'ₖ':'k', 'ₗ':'l', 'ₘ':'m',
    'ₙ':'n', 'ₚ':'p', 'ᵤ':'u', 'ᵥ':'v', 'ₐ':'a', 'ₑ':'e', 'ₒ':'o',
    '₀':'0', '₁':'1', '₂':'2', '₃':'3', '₄':'4',
    '₅':'5', '₆':'6', '₇':'7', '₈':'8', '₉':'9',
}
# Unicode superscript → base char
SUPS = {
    '⁺':'+', '⁻':'−', '⁰':'0', '¹':'1', '²':'2', '³':'3',
    '⁴':'4', '⁵':'5', '⁶':'6', '⁷':'7', '⁸':'8', '⁹':'9',
    '⁽':'(', '⁾':')', 'ᴸ':'L', '⁎':'*',
}

# Words/abbreviations that should be roman (upright) in math
ROMAN_SET = {
    'sim', 'exp', 'log', 'BM25', 'IDF', 'DCG', 'nDCG', 'IDCG', 'MRR',
    'Recall', 'avgdl', 'min', 'max', 'argmax', 'rank', 'df',
    'Rel', 'U', 'E',
}

def is_roman(word):
    if word in ROMAN_SET:
        return True
    # Multi-char all-uppercase abbreviations (MRR, IDF, etc.)
    pure = re.sub(r'[@₀-₉ᵢⱼₜₖₚ]', '', word)
    if len(pure) > 1 and pure.isupper():
        return True
    return False

def match_func_call(sym):
    """Return (fname, args) if entire sym is a single function call, else None."""
    depth = 0
    paren_start = -1
    for i, c in enumerate(sym):
        if c == '(':
            if depth == 0:
                paren_start = i
            depth += 1
        elif c == ')':
            depth -= 1
            if depth == 0:
                if i == len(sym) - 1 and paren_start > 0:
                    fname = sym[:paren_start]
                    # fname must not contain spaces, commas, |
                    if ' ' not in fname and ',' not in fname and '|' not in fname:
                        return fname, sym[paren_start+1:i]
                return None
    return None

def split_comma_args(s):
    """Split args by ', ' respecting paren depth."""
    depth, parts, cur = 0, [], []
    i = 0
    while i < len(s):
        c = s[i]
        if c == '(':
            depth += 1; cur.append(c)
        elif c == ')':
            depth -= 1; cur.append(c)
        elif c == ',' and depth == 0 and i + 1 < len(s) and s[i+1] == ' ':
            parts.append(''.join(cur).strip())
            cur = []; i += 1  # skip space
        else:
            cur.append(c)
        i += 1
    if cur:
        parts.append(''.join(cur).strip())
    return [p for p in parts if p]

def tokenize(s):
    """Tokenize string into [('b'|'s'|'S', chars), ...] runs."""
    tokens, cur_type, cur = [], None, []
    for c in s:
        ct = 's' if c in SUBS else ('S' if c in SUPS else 'b')
        if ct == cur_type:
            cur.append(c)
        else:
            if cur: tokens.append((cur_type, ''.join(cur)))
            cur_type, cur = ct, [c]
    if cur: tokens.append((cur_type, ''.join(cur)))
    return tokens

def decode(chars, typ):
    if typ == 's': return ''.join(SUBS.get(c, c) for c in chars)
    if typ == 'S': return ''.join(SUPS.get(c, c) for c in chars)
    return chars

def build_tokens(tokens):
    """Build OMML elements from token list."""
    elems, i = [], 0
    while i < len(tokens):
        t, chars = tokens[i]
        if t != 'b':
            # orphan sub/sup — just output as regular text
            elems.append(R(decode(chars, t)))
            i += 1; continue
        base_txt = chars
        roman = is_roman(base_txt)
        base_el = [R(base_txt, roman=roman)]
        nxt1 = tokens[i+1] if i+1 < len(tokens) else None
        nxt2 = tokens[i+2] if i+2 < len(tokens) else None
        if nxt1 and nxt1[0] == 's' and nxt2 and nxt2[0] == 'S':
            elems.append(SUBSUP(base_el,
                                [R(decode(nxt1[1], 's'))],
                                [R(decode(nxt2[1], 'S'))]))
            i += 3
        elif nxt1 and nxt1[0] == 'S' and nxt2 and nxt2[0] == 's':
            # sup before sub — treat as subsup anyway
            elems.append(SUBSUP(base_el,
                                [R(decode(nxt2[1], 's'))],
                                [R(decode(nxt1[1], 'S'))]))
            i += 3
        elif nxt1 and nxt1[0] == 's':
            elems.append(SUB(base_el, [R(decode(nxt1[1], 's'))]))
            i += 2
        elif nxt1 and nxt1[0] == 'S':
            elems.append(SUP(base_el, [R(decode(nxt1[1], 'S'))]))
            i += 2
        else:
            elems.extend(base_el)
            i += 1
    return elems

def parse_sym(sym):
    """Parse a symbol string → list of m:oMath child elements."""
    sym = sym.strip()
    if not sym:
        return []

    # |...|
    if sym.startswith('|') and sym.endswith('|') and sym[1:-1].count('|') == 0:
        return [ABS(parse_sym(sym[1:-1]))]

    # function call: name(args) — entire sym
    fc = match_func_call(sym)
    if fc:
        fname, fargs = fc
        fname_elems = parse_sym(fname)
        arg_parts = split_comma_args(fargs)
        inner = []
        for k, part in enumerate(arg_parts):
            if k > 0: inner.append(R(', '))
            inner.extend(parse_sym(part))
        return fname_elems + [DELIM('(', ')', inner)]

    # @k notation: NAME@k → SUB(NAME, @k)
    if '@' in sym and not sym.startswith('@') and '(' not in sym:
        parts = sym.split('@', 1)
        return [SUB([R(parts[0], roman=True)], [R('@' + parts[1], roman=True)])]

    # spaces: "min xᵢⱼ" → parse each token
    if ' ' in sym:
        parts = sym.split(' ')
        elems = []
        for k, part in enumerate(parts):
            if k > 0: elems.append(R(' '))  # thin space
            elems.extend(parse_sym(part))
        return elems

    # comma list: "w₁, w₂, w₃"
    if ', ' in sym:
        parts = split_comma_args(sym)
        elems = []
        for k, part in enumerate(parts):
            if k > 0: elems.append(R(', '))
            elems.extend(parse_sym(part))
        return elems

    # underscore: "E_q", "rank_q", "h_T⁽ᴸ⁾"
    if '_' in sym and not sym.startswith('_'):
        base, rest = sym.split('_', 1)
        return [SUB(parse_sym(base), parse_sym(rest))]

    # tokenize by Unicode sub/sup
    tokens = tokenize(sym)
    return build_tokens(tokens)


# ── Paragraph helpers ─────────────────────────────────────────────────────────

def para_text(p):
    return "".join((t.text or "") for t in p.iter(_w("t")))

def get_style(p):
    ppr = p.find(_w("pPr"))
    if ppr is not None:
        ps = ppr.find(_w("pStyle"))
        if ps is not None:
            return ps.get(_w("val"), "")
    return ""

def is_de_anno(p):
    """True if this is a 'де:' annotation paragraph."""
    if get_style(p) != "a":
        return False
    ppr = p.find(_w("pPr"))
    if ppr is None:
        return False
    has_left = ppr.find(_w("jc")) is not None and ppr.find(_w("jc")).get(_w("val")) == "left"
    has_fl0  = ppr.find(_w("ind")) is not None and ppr.find(_w("ind")).get(_w("firstLine")) == "0"
    if not (has_left or has_fl0):
        return False
    txt = para_text(p)
    return txt.startswith("де ") or txt.startswith("     ") or txt.startswith("     ")

def make_word_run(text):
    """Plain w:r for non-math text."""
    r = ET.Element(_w("r"))
    t = ET.SubElement(r, _w("t"))
    t.text = text
    if text != text.strip():
        t.set(_x("space"), "preserve")
    return r

def rebuild_de_para(p, leading, sym_str, description):
    """Rebuild paragraph: keep pPr, replace content with leading+oMath+description."""
    # Remove all w:r, m:oMath, w:hyperlink children (keep pPr)
    to_remove = [ch for ch in list(p) if ch.tag != _w("pPr")]
    for ch in to_remove:
        p.remove(ch)

    # Leading text
    p.append(make_word_run(leading))

    # Symbol → m:oMath
    sym_elems = parse_sym(sym_str)
    om = OM(*sym_elems)
    p.append(om)

    # Description (with separator already included)
    p.append(make_word_run(description))


# ── Main pass ─────────────────────────────────────────────────────────────────

children = list(body)
count = 0
skipped = 0

for i, p in enumerate(children):
    if i < 300 or i > 760:
        continue
    if not is_de_anno(p):
        continue

    txt = para_text(p)

    # Split: "де sym — description"  or  "     sym — description"
    SEP = " — "
    if SEP not in txt:
        skipped += 1
        continue

    idx_sep = txt.index(SEP)

    # Leading: "де " or "     "
    if txt.startswith("де "):
        leading = "де "
        sym_str = txt[3:idx_sep]
    else:
        # Count leading spaces
        n_spaces = len(txt) - len(txt.lstrip(" "))
        leading = " " * n_spaces
        sym_str = txt[n_spaces:idx_sep]

    sym_str = sym_str.strip()
    description = SEP + txt[idx_sep + len(SEP):]  # includes " — "

    if not sym_str:
        skipped += 1
        continue

    rebuild_de_para(p, leading, sym_str, description)
    count += 1
    print(f"  B{i}: [{sym_str}]")

print(f"\n[DONE] Rebuilt {count} de-annotation paragraphs ({skipped} skipped).")


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

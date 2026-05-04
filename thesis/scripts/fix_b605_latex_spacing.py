"""
Fixes B605: removes leftover LaTeX \\; spacing runs from oMath.
The paragraph W=(0.33,\\;0.27,...) has literal \\; and trailing-space runs
that were unconverted LaTeX thin-space commands.
"""
import sys, shutil, zipfile, pathlib, xml.etree.ElementTree as ET
sys.stdout.reconfigure(encoding='utf-8')

ROOT     = pathlib.Path("D:/repos/kb-semantic-search-benchmark")
DOCX     = ROOT / "thesis" / "2026_M_PI_Nesterenko_VV.docx"
UNPACKED = ROOT / "thesis" / "unpacked_docx"

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
M = "http://schemas.openxmlformats.org/officeDocument/2006/math"
def _w(t): return f"{{{W}}}{t}"
def _m(t): return f"{{{M}}}{t}"

if UNPACKED.exists():
    shutil.rmtree(UNPACKED)
with zipfile.ZipFile(DOCX) as z:
    z.extractall(UNPACKED)
print("Re-unpacked docx.")

DOC_XML = UNPACKED / "word" / "document.xml"
tree = ET.parse(DOC_XML)
body = tree.getroot().find(_w("body"))
children = list(body)

p = children[605]

# Find the m:e element inside m:d inside oMath
om = p.find(f".//{_m('oMath')}")
d_el = om.find(_m("d"))
e_el = d_el.find(_m("e"))

# Remove all m:r children whose m:t text is "\;" or whitespace-only (m:nor runs)
to_remove = []
for r in list(e_el):
    if r.tag != _m("r"):
        continue
    nor = r.find(_m("rPr"))
    if nor is not None and nor.find(_m("nor")) is not None:
        t_el = r.find(_m("t"))
        if t_el is not None:
            txt = (t_el.text or "").strip()
            # Remove \; runs and the lone-space runs that follow them
            if txt == "\\;" or txt == "":
                to_remove.append(r)

for r in to_remove:
    e_el.remove(r)

print(f"Removed {len(to_remove)} \\; / space runs from B605.")

# Verify remaining m:t texts
remaining = [r.find(_m("t")).text for r in e_el if r.tag == _m("r") and r.find(_m("t")) is not None]
print(f"Remaining values in m:d: {remaining}")

# Save & repack
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

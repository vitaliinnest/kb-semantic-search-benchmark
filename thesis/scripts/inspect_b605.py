import sys, pathlib, xml.etree.ElementTree as ET
sys.stdout.reconfigure(encoding='utf-8')

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
M = "http://schemas.openxmlformats.org/officeDocument/2006/math"
def _w(t): return f"{{{W}}}{t}"
def _m(t): return f"{{{M}}}{t}"

UNPACKED = pathlib.Path("D:/repos/kb-semantic-search-benchmark/thesis/unpacked_docx")
tree = ET.parse(UNPACKED / "word" / "document.xml")
body = tree.getroot().find(_w("body"))
children = list(body)

p = children[605]
ET.indent(p)
print(ET.tostring(p, encoding='unicode'))

"""
Замінює підсумкове речення у Висновках, щоб формулювання досягнутої мети
відповідало меті, заявленій у Вступі (B204):
  "визначення доцільності застосування сучасних моделей векторних ембеддінгів
   для підвищення ефективності семантичного пошуку текстових документів у
   корпоративних базах знань шляхом їх порівняльного експериментального дослідження"
"""
import sys, shutil, zipfile, pathlib, xml.etree.ElementTree as ET
sys.stdout.reconfigure(encoding='utf-8')

ROOT     = pathlib.Path("D:/repos/kb-semantic-search-benchmark")
DOCX     = ROOT / "thesis" / "2026_M_PI_Nesterenko_VV.docx"
UNPACKED = ROOT / "thesis" / "unpacked_docx"

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
M = "http://schemas.openxmlformats.org/officeDocument/2006/math"
def _w(t): return f"{{{W}}}{t}"

if UNPACKED.exists():
    shutil.rmtree(UNPACKED)
with zipfile.ZipFile(DOCX) as z:
    z.extractall(UNPACKED)
print("Re-unpacked docx.")

DOC_XML = UNPACKED / "word" / "document.xml"
tree = ET.parse(DOC_XML)
body = tree.getroot().find(_w("body"))

def all_text(p):
    return "".join((t.text or "") for t in p.iter(_w("t")))

def set_para_text(p, new_text):
    """Замінює весь текст параграфа, зберігаючи pPr."""
    for r in list(p.findall(_w("r"))):
        p.remove(r)
    r = ET.SubElement(p, _w("r"))
    t_el = ET.SubElement(r, _w("t"))
    t_el.text = new_text

NEW_TEXT = (
    "Таким чином, поставлену мету досягнуто: визначено доцільність застосування "
    "сучасних моделей векторних ембеддінгів (BGE-M3, E5-base, nomic-embed-text-v1.5, "
    "Qwen3-Embedding-0.6B) для підвищення ефективності семантичного пошуку текстових "
    "документів у корпоративних базах знань шляхом їх порівняльного експериментального "
    "дослідження на трьох предметних доменах — технічному, юридичному та медичному."
)

found = False
for i, p in enumerate(body):
    if "поставлену мету досягнуто" in all_text(p):
        old = all_text(p)
        set_para_text(p, NEW_TEXT)
        print(f"[OK] Replaced B{i}:")
        print(f"  OLD: {old[:150]}")
        print(f"  NEW: {NEW_TEXT[:150]}")
        found = True
        break

if not found:
    print("[!] Paragraph 'поставлену мету досягнуто' NOT FOUND")

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

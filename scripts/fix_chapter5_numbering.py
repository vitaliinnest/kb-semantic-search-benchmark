"""
Виправляє нумерацію рисунків (6.x → 5.x) і таблиць у розділі 5.
Текст у Word розбито на кілька <w:t> рядків (каса/нижній індекс/розрив),
тому шукаємо за послідовністю запусків (runs), а не лише в одному w:t.

Алгоритм:
  • збираємо повний текст параграфа з усіх w:t елементів зі збереженням
    «карти» (run_index, t_index, char_offset);
  • виконуємо replace на повному тексті;
  • переписуємо w:t-вміст так, щоб збігалося з результатом, без втрати
    структури runs (просто пишемо новий текст у відповідні t елементи).

У випадку, коли довжина тексту змінюється, переносимо весь новий текст
у перший w:t, а решту t-елементів очищаємо. Це безпечно для звичайних
параграфів — формула не зачіпається.
"""
import sys, re, shutil, zipfile, pathlib, xml.etree.ElementTree as ET
sys.stdout.reconfigure(encoding='utf-8')

ROOT = pathlib.Path("D:/repos/kb-semantic-search-benchmark")
DOCX = ROOT / "2026_M_PI_Nesterenko_VV.docx"
UNPACKED = ROOT / "unpacked_docx"

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
M = "http://schemas.openxmlformats.org/officeDocument/2006/math"
def _w(t): return f"{{{W}}}{t}"
def _m(t): return f"{{{M}}}{t}"


# Re-unpack
if UNPACKED.exists():
    shutil.rmtree(UNPACKED)
with zipfile.ZipFile(DOCX) as z:
    z.extractall(UNPACKED)
print("Re-unpacked docx.")

DOC_XML = UNPACKED / "word" / "document.xml"
tree = ET.parse(DOC_XML)
body = tree.getroot().find(_w("body"))
children = list(body)


# Patterns for renumbering. Apply in order. Substring replace on combined paragraph text.
RENUMBER = []
for n in range(1, 24):
    RENUMBER.append((f"Рисунок 6.{n} ",  f"Рисунок 5.{n} "))
    RENUMBER.append((f"Рисунок 6.{n}–", f"Рисунок 5.{n}–"))
    RENUMBER.append((f"рисунку 6.{n} ",  f"рисунку 5.{n} "))
    RENUMBER.append((f"рисунку 6.{n}.",  f"рисунку 5.{n}."))
    RENUMBER.append((f"рисунку 6.{n},",  f"рисунку 5.{n},"))
    RENUMBER.append((f"рисунка 6.{n} ",  f"рисунка 5.{n} "))
    RENUMBER.append((f"рисунках 6.{n} ", f"рисунках 5.{n} "))
# benchmark-таблиці 6.2..6.4 → 5.8..5.10
RENUMBER += [
    ("Таблиця 6.2 ", "Таблиця 5.8 "),
    ("Таблиця 6.3 ", "Таблиця 5.9 "),
    ("Таблиця 6.4 ", "Таблиця 5.10 "),
    ("Таблиця 6.2–", "Таблиця 5.8–"),
    ("Таблиця 6.3–", "Таблиця 5.9–"),
    ("Таблиця 6.4–", "Таблиця 5.10–"),
    ("таблиці 6.2 ", "таблиці 5.8 "),
    ("таблиці 6.3 ", "таблиці 5.9 "),
    ("таблиці 6.4 ", "таблиці 5.10 "),
    ("таблиці 6.2.", "таблиці 5.8."),
    ("таблиці 6.3.", "таблиці 5.9."),
    ("таблиці 6.4.", "таблиці 5.10."),
    ("таблицю 6.2 ", "таблицю 5.8 "),
    ("таблицю 6.3 ", "таблицю 5.9 "),
    ("таблицю 6.4 ", "таблицю 5.10 "),
]
# BERT → E5-base (specific)
RENUMBER += [
    ("модель BERT повертає", "модель E5-base повертає"),
    ("видачі BERT для медичного", "видачі E5-base для медичного"),
]


def patch_paragraph(p) -> bool:
    """Збирає весь w:t-текст параграфа, замінює, перерозподіляє назад.
    Не чіпає math-runs (m:t) — лише звичайні w:t.
    Повертає True, якщо щось змінилось.
    """
    # Беремо лише w:t всередині w:r-послідовностей, які НЕ всередині math.
    t_elements = []
    for r in p.findall(_w("r")):  # лише прямі w:r дітей параграфа
        for t in r.findall(_w("t")):
            t_elements.append(t)
    if not t_elements:
        return False

    combined = "".join((t.text or "") for t in t_elements)
    new_combined = combined
    for old, new in RENUMBER:
        new_combined = new_combined.replace(old, new)
    if new_combined == combined:
        return False

    # Записуємо новий текст у перший w:t, інші очищаємо (текст там був).
    t_elements[0].text = new_combined
    # Зберегти xml:space="preserve" якщо текст починається/закінчується пробілом
    if new_combined != new_combined.strip():
        t_elements[0].set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    for t in t_elements[1:]:
        t.text = ""
    return True


SECTION5_START = 707
SECTION5_END   = 1010

n_changed = 0
for i in range(SECTION5_START, min(SECTION5_END + 1, len(children))):
    ch = children[i]
    if patch_paragraph(ch):
        n_changed += 1

print(f"Patched paragraphs: {n_changed}")


# Save & repack
ET.register_namespace("w", W)
ET.register_namespace("m", M)
ET.register_namespace("r", "http://schemas.openxmlformats.org/officeDocument/2006/relationships")
ET.register_namespace("mc", "http://schemas.openxmlformats.org/markup-compatibility/2006")
ET.register_namespace("v", "urn:schemas-microsoft-com:vml")
ET.register_namespace("wp", "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing")
ET.register_namespace("w14", "http://schemas.microsoft.com/office/word/2010/wordml")
ET.register_namespace("w15", "http://schemas.microsoft.com/office/word/2012/wordml")
ET.register_namespace("wne", "http://schemas.microsoft.com/office/word/2006/wordml")
ET.register_namespace("wps", "http://schemas.microsoft.com/office/word/2010/wordprocessingShape")
ET.register_namespace("o", "urn:schemas-microsoft-com:office:office")
ET.register_namespace("w10", "urn:schemas-microsoft-com:office:word")
ET.register_namespace("wp14", "http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing")
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

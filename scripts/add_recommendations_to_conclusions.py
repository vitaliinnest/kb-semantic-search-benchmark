"""
Додає практичні рекомендації всередину Висновки (після останнього абзацу,
перед "Перелік джерел посилання"). 7 пунктів bullet-списком стилю "a".
"""
import sys, shutil, zipfile, pathlib, xml.etree.ElementTree as ET
sys.stdout.reconfigure(encoding='utf-8')

ROOT     = pathlib.Path("D:/repos/kb-semantic-search-benchmark")
DOCX     = ROOT / "2026_M_PI_Nesterenko_VV.docx"
UNPACKED = ROOT / "unpacked_docx"

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

def get_style(p):
    ppr = p.find(_w("pPr"))
    if ppr is not None:
        ps = ppr.find(_w("pStyle"))
        if ps is not None:
            return ps.get(_w("val"), "")
    return ""

def make_para(text, style="a"):
    p = ET.Element(_w("p"))
    ppr = ET.SubElement(p, _w("pPr"))
    ps = ET.SubElement(ppr, _w("pStyle"))
    ps.set(_w("val"), style)
    if text:
        r = ET.SubElement(p, _w("r"))
        t = ET.SubElement(r, _w("t"))
        t.text = text
        if text != text.strip():
            t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    return p

# Знаходимо кінець Висновки — перед "Перелік джерел посилання"
children = list(body)
sources_idx = -1
for i, p in enumerate(children):
    if get_style(p) == "-" and "Перелік джерел" in all_text(p):
        sources_idx = i
        break

if sources_idx == -1:
    print("[!] 'Перелік джерел посилання' NOT FOUND")
else:
    # Якщо перед sources_idx є пустий абзац, вставляємо перед ним
    insert_at = sources_idx
    prev = children[sources_idx - 1]
    if all_text(prev).strip() == "":
        insert_at = sources_idx - 1  # вставити перед пустим рядком

    # Перевіряємо, чи рекомендації вже є
    if "сформульовано практичні рекомендації" in all_text(children[insert_at - 1]):
        print("[=] Рекомендації вже присутні — пропускаємо.")
    else:
        RECS = [
            # Вступний абзац
            (
                "На основі отриманих результатів сформульовано практичні рекомендації "
                "щодо впровадження семантичного пошуку в корпоративних системах:"
            ),
            # Bullet 1
            (
                "– для корпоративних баз знань доцільно переходити від класичного "
                "keyword-пошуку (типу BM25) до семантичного пошуку на основі векторних "
                "ембеддінгів, оскільки нейронні моделі забезпечують суттєво вищу якість "
                "ранжування при різній термінології та перефразуванні запитів; при цьому "
                "BM25 може залишатись конкурентним базелайном у доменах із жорсткою "
                "термінологією, де нейронні моделі не мають значної переваги;"
            ),
            # Bullet 2
            (
                "– як основну модель для впровадження рекомендується BGE-M3, оскільки "
                "вона демонструє найкращий баланс між якістю результатів і стабільністю "
                "роботи в різних типах документів;"
            ),
            # Bullet 3
            (
                "– якщо система має високі вимоги до швидкості (велика кількість запитів "
                "або інтерактивний пошук), варто розглядати E5-base як більш легку та швидку "
                "альтернативу з прийнятним рівнем якості;"
            ),
            # Bullet 4
            (
                "– модель Qwen3-Embedding варто використовувати лише за наявності "
                "GPU-інфраструктури, оскільки на CPU вона працює значно повільніше і може "
                "бути непридатною для роботи в реальному часі;"
            ),
            # Bullet 5 — чесна оцінка nomic
            (
                "– модель nomic-embed-text-v1.5 у проведеному дослідженні показала "
                "найнижчу якість retrieval серед нейронних моделей і в окремих доменах "
                "поступалась навіть лексичному базелайну BM25; її застосування може бути "
                "виправданим у демонстраційних чи дослідницьких сценаріях, де пріоритетом "
                "є відкритість та відтворюваність процесу навчання моделі;"
            ),
            # Bullet 6
            (
                "– при впровадженні системи важливо приділити увагу обробці документів "
                "(чанкінг, розмір фрагментів, перекриття), оскільки ці параметри безпосередньо "
                "впливають на якість пошуку незалежно від обраної моделі;"
            ),
            # Bullet 7
            (
                "– для кожної організації бажано формувати власний набір тестових запитів, "
                "щоб перевірити якість пошуку саме на корпоративних даних і реальних сценаріях "
                "використання;"
            ),
            # Bullet 8
            (
                "– оптимальним підходом є диференційований вибір моделі залежно від задачі: "
                "BGE-M3 — для аналітичних запитів, де важлива повнота; E5-base — для "
                "інтерактивного пошуку, де критичним є час відповіді."
            ),
        ]

        new_paras = [make_para(text) for text in RECS]
        for j, para in enumerate(new_paras):
            body.insert(insert_at + j, para)

        print(f"[OK] Inserted {len(new_paras)} paragraphs (intro + 8 bullets) before B{insert_at}.")

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

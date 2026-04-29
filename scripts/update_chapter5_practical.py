"""
Оновлення практичного розділу (5) докcа під актуальний застосунок:
  1. Замінює embedded-зображення-скріншоти (image4, 6, 7, 8, 9, 10, 12, 14–23)
     на свіжі скріншоти, зроблені скриптом make_screenshots.py.
  2. Виправляє нумерацію рисунків (Рисунок 6.x → 5.x) і таблиць у розділі 5
     (Таблиця 6.x → 5.x).
  3. Замінює згадку моделі BERT (вже відсутня в системі) на E5-base
     у тексті прикладу пошукової видачі для медичного домену.

Скріншоти не торкаються схем (image1, 2, 3, 5, 11, 13) і додатків
(image24+) — їх лишаємо як є.
"""
import sys, re, shutil, zipfile, pathlib, xml.etree.ElementTree as ET
sys.stdout.reconfigure(encoding='utf-8')

ROOT = pathlib.Path("D:/repos/kb-semantic-search-benchmark")
DOCX = ROOT / "2026_M_PI_Nesterenko_VV.docx"
UNPACKED = ROOT / "unpacked_docx"

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
def _w(t): return f"{{{W}}}{t}"


# ── 1. Re-unpack ──────────────────────────────────────────────────────
if UNPACKED.exists():
    shutil.rmtree(UNPACKED)
with zipfile.ZipFile(DOCX) as z:
    z.extractall(UNPACKED)
print("Re-unpacked docx.")


# ── 2. Знаходимо найновішу папку зі скріншотами ───────────────────────
SHOTS_ROOT = ROOT / "screenshots"
shot_dirs = sorted([p for p in SHOTS_ROOT.iterdir() if p.is_dir() and (p / "thesis").exists()])
if not shot_dirs:
    print("ERROR: no thesis screenshots found in screenshots/.")
    sys.exit(1)
SHOTS = shot_dirs[-1] / "thesis"
print(f"Using screenshots: {SHOTS}")


# ── 3. Замапимо image-файли → скріншоти ──────────────────────────────
# Якщо скріншоту немає — лишаємо існуюче зображення без змін.
IMAGE_REPLACEMENTS = {
    "image4.png":  "fig_5_04_documents_tech.png",
    "image6.png":  "fig_5_06_build_tech.png",
    "image7.png":  "fig_5_07_search_home_tech.png",
    "image8.png":  "fig_5_08_search_tech_bge.png",
    "image9.png":  "fig_5_09_search_legal_bge.png",
    "image10.png": "fig_5_10_search_medical_bge.png",
    "image12.png": "fig_5_12_benchmark_tech.png",
    "image14.png": "fig_5_14_selection_tech.png",
    "image15.png": "fig_5_15_search_tech_bge_en.png",
    "image16.png": "fig_5_16_search_tech_nomic.png",
    "image17.png": "fig_5_17_search_medical_e5.png",   # колись було BERT, тепер E5-base
    "image18.png": "fig_5_18_benchmark_tech.png",
    "image19.png": "fig_5_19_benchmark_legal.png",
    "image20.png": "fig_5_20_benchmark_medical.png",
    "image21.png": "fig_5_21_selection_tech.png",
    "image22.png": "fig_5_22_selection_legal.png",
    "image23.png": "fig_5_23_selection_medical.png",
}

media_dir = UNPACKED / "word" / "media"
replaced = 0
missing = []
for target, source in IMAGE_REPLACEMENTS.items():
    src = SHOTS / source
    dst = media_dir / target
    if not src.exists():
        missing.append(source)
        continue
    shutil.copyfile(src, dst)
    replaced += 1
    print(f"  IMG  {target} <- {source}")

print(f"Replaced {replaced} images. Missing screenshots: {missing or 'none'}")


# ── 4. Виправляємо нумерацію в тексті розділу 5 ───────────────────────
# Розділ 5 у тілі починається з B707 (5.1). Оновлюємо лише ці параграфи.
DOC_XML = UNPACKED / "word" / "document.xml"
tree = ET.parse(DOC_XML)
body = tree.getroot().find(_w("body"))
children = list(body)
print(f"Body children: {len(children)}")

# Карта замін: рисунки 6.1..6.23 → 5.1..5.23
fig_repl_full   = [(f"Рисунок 6.{i}", f"Рисунок 5.{i}") for i in range(1, 24)]
fig_repl_short  = [(f"рисунку 6.{i}",  f"рисунку 5.{i}")  for i in range(1, 24)]
fig_repl_short2 = [(f"рисунка 6.{i}",  f"рисунка 5.{i}")  for i in range(1, 24)]
fig_repl_short3 = [(f"рисунках 6.{i}", f"рисунках 5.{i}") for i in range(1, 24)]
# Таблиці 6.2/6.3/6.4 → 5.8/5.9/5.10
# (5.1..5.7 вже зайняті в розділі 4 — про векторний опис альтернатив тощо;
#  тому benchmark-таблиці отримують номери 5.8, 5.9, 5.10)
tbl_repl = [
    ("Таблиця 6.2",  "Таблиця 5.8"),
    ("Таблиця 6.3",  "Таблиця 5.9"),
    ("Таблиця 6.4",  "Таблиця 5.10"),
    ("таблиці 6.2",  "таблиці 5.8"),
    ("таблиці 6.3",  "таблиці 5.9"),
    ("таблиці 6.4",  "таблиці 5.10"),
    ("таблицю 6.2",  "таблицю 5.8"),
    ("таблицю 6.3",  "таблицю 5.9"),
    ("таблицю 6.4",  "таблицю 5.10"),
]

# Окрема заміна BERT → E5-base у двох параграфах (B949, B951)
text_repl_specific = [
    ("модель BERT повертає", "модель E5-base повертає"),
    ("видачі BERT для медичного", "видачі E5-base для медичного"),
]

ALL_REPL = (fig_repl_full + fig_repl_short + fig_repl_short2 + fig_repl_short3
            + tbl_repl + text_repl_specific)


def patch_runs_in_paragraph(p):
    """In-place заміна тексту в усіх w:t елементах параграфа.
    Повертає True, якщо було щось замінено."""
    changed = False
    for t in p.iter(_w("t")):
        if t.text is None:
            continue
        new_text = t.text
        for old, new in ALL_REPL:
            if old in new_text:
                new_text = new_text.replace(old, new)
        if new_text != t.text:
            t.text = new_text
            changed = True
    return changed


# Застосовуємо тільки до розділу 5 (B707..кінець розділу) і до глобального TOC
SECTION5_START = 707
SECTION5_END   = 1010   # до Висновків (B1010)

n_changed = 0
for i, ch in enumerate(children):
    in_section5 = SECTION5_START <= i <= SECTION5_END
    # У TOC (B0..B200) теж можуть бути рядки на кшталт "Таблиця 6.2..." — їх теж патчимо,
    # але насправді у TOC ми бачили лише пункти 5.1, 5.2 — тут немає що чіпати.
    if not in_section5:
        continue
    if patch_runs_in_paragraph(ch):
        n_changed += 1

print(f"Patched {n_changed} paragraphs in section 5.")


# ── 5. Save & repack ──────────────────────────────────────────────────
ET.register_namespace("w", W)
ET.register_namespace("m", "http://schemas.openxmlformats.org/officeDocument/2006/math")
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

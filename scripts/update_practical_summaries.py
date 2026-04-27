"""
Update key summary/analysis paragraphs in the practical section (P1192, P1208-P1211)
that still reference old models as winners.
Also update P1097 (tech domain analysis) and P1143, P1190 (legal/medical domain analysis).
These paragraphs describe aggregate conclusions — not figure captions or specific table data.
"""
import sys
import io
import copy
import xml.etree.ElementTree as ET
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

DOC_XML = Path("D:/repos/kb-semantic-search-benchmark/unpacked_docx/word/document.xml")
NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
W = f"{{{NS}}}"

ET.register_namespace("wpc", "http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas")
ET.register_namespace("cx", "http://schemas.microsoft.com/office/drawing/2014/chartex")
ET.register_namespace("cx1", "http://schemas.microsoft.com/office/drawing/2015/9/8/chartex")
ET.register_namespace("cx2", "http://schemas.microsoft.com/office/drawing/2015/10/21/chartex")
ET.register_namespace("cx3", "http://schemas.microsoft.com/office/drawing/2016/5/9/chartex")
ET.register_namespace("cx4", "http://schemas.microsoft.com/office/drawing/2016/5/10/chartex")
ET.register_namespace("cx5", "http://schemas.microsoft.com/office/drawing/2016/5/11/chartex")
ET.register_namespace("cx6", "http://schemas.microsoft.com/office/drawing/2016/5/12/chartex")
ET.register_namespace("cx7", "http://schemas.microsoft.com/office/drawing/2016/5/13/chartex")
ET.register_namespace("cx8", "http://schemas.microsoft.com/office/drawing/2016/5/14/chartex")
ET.register_namespace("mc", "http://schemas.openxmlformats.org/markup-compatibility/2006")
ET.register_namespace("aink", "http://schemas.microsoft.com/office/drawing/2016/ink")
ET.register_namespace("am3d", "http://schemas.microsoft.com/office/drawing/2017/model3d")
ET.register_namespace("o", "urn:schemas-microsoft-com:office:office")
ET.register_namespace("oel", "http://schemas.microsoft.com/office/2019/extlst")
ET.register_namespace("r", "http://schemas.openxmlformats.org/officeDocument/2006/relationships")
ET.register_namespace("m", "http://schemas.openxmlformats.org/officeDocument/2006/math")
ET.register_namespace("v", "urn:schemas-microsoft-com:vml")
ET.register_namespace("wp14", "http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing")
ET.register_namespace("wp", "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing")
ET.register_namespace("w10", "urn:schemas-microsoft-com:office:word")
ET.register_namespace("w", "http://schemas.openxmlformats.org/wordprocessingml/2006/main")
ET.register_namespace("w14", "http://schemas.microsoft.com/office/word/2010/wordml")
ET.register_namespace("w15", "http://schemas.microsoft.com/office/word/2012/wordml")
ET.register_namespace("w16cex", "http://schemas.microsoft.com/office/word/2018/wordml/cex")
ET.register_namespace("w16cid", "http://schemas.microsoft.com/office/word/2016/wordml/cid")
ET.register_namespace("w16", "http://schemas.microsoft.com/office/word/2018/wordml")
ET.register_namespace("w16sdtdh", "http://schemas.microsoft.com/office/word/2020/wordml/sdtdatahash")
ET.register_namespace("w16se", "http://schemas.microsoft.com/office/word/2015/wordml/symex")
ET.register_namespace("wpg", "http://schemas.microsoft.com/office/word/2010/wordprocessingGroup")
ET.register_namespace("wpi", "http://schemas.microsoft.com/office/word/2010/wordprocessingInk")
ET.register_namespace("wne", "http://schemas.microsoft.com/office/word/2006/wordml")
ET.register_namespace("wps", "http://schemas.microsoft.com/office/word/2010/wordprocessingShape")


def get_para_text(para) -> str:
    return "".join(t.text or "" for t in para.iter(f"{W}t"))


def set_para_text(para, new_text: str) -> None:
    pPr = para.find(f"{W}pPr")
    first_rPr = None
    first_r = para.find(f"{W}r")
    if first_r is not None:
        first_rPr = first_r.find(f"{W}rPr")
    for child in list(para):
        para.remove(child)
    if pPr is not None:
        para.append(pPr)
    if new_text == "":
        return
    run = ET.SubElement(para, f"{W}r")
    if first_rPr is not None:
        run.append(copy.deepcopy(first_rPr))
    t_elem = ET.SubElement(run, f"{W}t")
    t_elem.text = new_text
    if new_text.startswith(" ") or new_text.endswith(" "):
        t_elem.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")


# ── Load ─────────────────────────────────────────────────────────────────────
tree = ET.parse(str(DOC_XML))
root = tree.getroot()
paras = root.findall(f".//{W}p")
print(f"Total paragraphs: {len(paras)}")

changes = []


def fix(idx: int, new_text: str, label: str):
    old = get_para_text(paras[idx])
    set_para_text(paras[idx], new_text)
    print(f"  P{idx} [{label}]:\n    WAS: {old[:80]}\n    NOW: {new_text[:80]}")
    changes.append(f"P{idx} {label}")


# ═══════════════════════════════════════════════════════════════════════════
# P1097 — tech domain analysis
# ═══════════════════════════════════════════════════════════════════════════
print("\n[P1097 — tech domain analysis]")
fix(1097,
    "У технічному домені результати є найбільш конкурентними. "
    "Моделі BGE-M3 та E5-base продемонстрували найкращі значення nDCG@10 та MRR@10, "
    "тобто краще ранжують результати і частіше виводять перший релевантний фрагмент "
    "на верхні позиції. Модель nomic-embed-text-v1.5 поступається за якістю ранжування, "
    "але забезпечує прийнятну швидкодію. Qwen3-Embedding-0.6B демонструє конкурентну "
    "якість, однак є найповільнішою через декодерну архітектуру. Саме тому технічний "
    "домен виявився найбільш цікавим з точки зору балансу між якістю та швидкодією.",
    "tech domain analysis")

# ═══════════════════════════════════════════════════════════════════════════
# P1143 — legal domain analysis
# ═══════════════════════════════════════════════════════════════════════════
print("\n[P1143 — legal domain analysis]")
fix(1143,
    "У юридичному домені найкращі результати за основними retrieval-метриками "
    "продемонстрували моделі BGE-M3 та E5-base. Обидві моделі забезпечили суттєво "
    "вищі значення nDCG@10 та Recall@10 порівняно з nomic-embed-text-v1.5, яка "
    "поступилася за якісними метриками. Qwen3-Embedding-0.6B також показала "
    "конкурентні результати за метриками якості при значно вищому часі відповіді. "
    "Результати підтверджують перевагу сучасних мультилінгвальних ембеддінгів для "
    "юридичних текстів зі спеціалізованою термінологією.",
    "legal domain analysis")

# ═══════════════════════════════════════════════════════════════════════════
# P1190 — medical domain analysis
# ═══════════════════════════════════════════════════════════════════════════
print("\n[P1190 — medical domain analysis]")
fix(1190,
    "У медичному домені також лідирують моделі BGE-M3 та E5-base за метриками "
    "nDCG@10, MRR@10 та Recall@10. Їхня перевага пояснюється здатністю до "
    "семантичного зіставлення запитів і медичних текстів, що містять специфічну "
    "термінологію. Модель nomic-embed-text-v1.5 показала нижчі результати, тоді як "
    "Qwen3-Embedding-0.6B забезпечила конкурентну якість при значному часі відповіді.",
    "medical domain analysis")

# ═══════════════════════════════════════════════════════════════════════════
# P1192 — cross-domain comparison summary
# ═══════════════════════════════════════════════════════════════════════════
print("\n[P1192 — cross-domain comparison]")
fix(1192,
    "Порівняння результатів у трьох доменах показує, що моделі BGE-M3 та E5-base "
    "є найбільш стабільними з точки зору retrieval-якості. Вони посідають перші "
    "місця в усіх трьох доменах за метриками nDCG@10, MRR@10 та Recall@10. "
    "Модель nomic-embed-text-v1.5 виступає як більш швидкий, але менш точний "
    "варіант. Qwen3-Embedding-0.6B демонструє конкурентну якість при значно вищому "
    "часі відповіді (~1500 мс/запит на CPU). Комерційна модель text-embedding-3-large "
    "теоретично демонструє найкращі показники повноти, але потребує зовнішнього API "
    "і не була оцінена у локальному benchmark.",
    "cross-domain comparison")

# ═══════════════════════════════════════════════════════════════════════════
# P1208-P1211 — multi-criteria summary
# ═══════════════════════════════════════════════════════════════════════════
print("\n[P1208-P1211 — multi-criteria summary]")
fix(1208,
    "Узагальнення результатів багатокритеріального вибору показує, що E5-base є "
    "рекомендованою моделлю для локального розгортання в усіх трьох доменах. "
    "Для технічного домену було використано профіль ваг із акцентом на nDCG@k та "
    "Recall@k при значущій ролі швидкодії, за якого E5-base (188 мс/запит) отримала "
    "перевагу над більш повільними альтернативами. Модель BGE-M3 показала близькі "
    "результати якості, але вищий час відповіді (~400 мс). За умови доступності "
    "API-ключа оптимальним вибором для максимальної якості є text-embedding-3-large.",
    "multi-criteria summary P1208")

fix(1209,
    "Для юридичного домену профіль ваг було зміщено у бік MRR@k та nDCG@k, "
    "що відображає важливість точного ранжування у правових запитах. За цих умов "
    "E5-base та BGE-M3 показали найкращі результати завдяки мультилінгвальним "
    "семантичним можливостям. Для медичного домену найбільша вага надавалась "
    "критерію повноти Recall@k, де також лідирували E5-base та BGE-M3. "
    "Модель nomic-embed-text-v1.5 у всіх доменах поступалася за якісними метриками, "
    "але виграє за співвідношенням швидкодії до якості.",
    "multi-criteria summary P1209")

fix(1210,
    "Для всіх трьох доменів Qwen3-Embedding-0.6B продемонструвала конкурентну "
    "якість ранжування (порівнянну з E5-base та BGE-M3), проте надзвичайно низьку "
    "швидкодію (~1500 мс/запит на CPU), що робить її непрактичною для інтерактивних "
    "сценаріїв використання. Тому, попри якісні переваги, ця модель не є "
    "рекомендованою для виробничого розгортання без GPU-прискорення.",
    "multi-criteria summary P1210")

fix(1211,
    "Таким чином, багатокритеріальний вибір не дублює benchmark-оцінювання, а "
    "доповнює його. Benchmark дозволяє окремо оцінити retrieval-якість і швидкодію "
    "моделей, тоді як багатокритеріальний підхід дає можливість формалізовано "
    "врахувати різну важливість критеріїв у різних предметних областях. "
    "У юридичному та медичному доменах найкращою локальною альтернативою виявилася "
    "E5-base, тоді як для максимальної теоретичної якості слід розглянути "
    "text-embedding-3-large (OpenAI). Це свідчить про те, що фінальний вибір "
    "моделі доцільно робити не лише на основі однієї метрики, а з урахуванням "
    "практичних вимог конкретного домену та обмежень інфраструктури.",
    "multi-criteria summary P1211")

# ── Save ─────────────────────────────────────────────────────────────────────
tree.write(str(DOC_XML), encoding="unicode", xml_declaration=False)
print(f"\nSaved — {len(changes)} changes")
for c in changes:
    print(f"  {c}")

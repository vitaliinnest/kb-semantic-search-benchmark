"""
Update Section 5 of the thesis document XML to replace old model descriptions
(TF-IDF, Word2Vec, FastText, BERT, Sentence-BERT) with new model descriptions
(BGE-M3, E5-base, nomic-embed-text-v1.5, Qwen3-Embedding-0.6B, text-embedding-3-large).
"""
import sys
import io
import copy
from pathlib import Path
import xml.etree.ElementTree as ET

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

DOC_XML = Path("D:/repos/kb-semantic-search-benchmark/unpacked_docx/word/document.xml")
NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
W = f"{{{NS}}}"
MATH_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"

# Register all namespaces to preserve them on write
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
    """Return the combined text of all w:t children in the paragraph."""
    parts = []
    for t in para.iter(f"{W}t"):
        parts.append(t.text or "")
    return "".join(parts)


def set_para_text(para, new_text: str) -> None:
    """
    Replace all runs in the paragraph with a single run that carries new_text.
    Preserves paragraph properties (pPr) and run properties of the first run.
    """
    # Collect pPr and first rPr
    pPr = para.find(f"{W}pPr")
    first_rPr = None
    first_r = para.find(f"{W}r")
    if first_r is not None:
        first_rPr = first_r.find(f"{W}rPr")

    # Remove all child elements from para
    for child in list(para):
        para.remove(child)

    # Re-add pPr
    if pPr is not None:
        para.append(pPr)

    # Build a new run
    run = ET.SubElement(para, f"{W}r")
    if first_rPr is not None:
        run.append(copy.deepcopy(first_rPr))
    t_elem = ET.SubElement(run, f"{W}t")
    t_elem.text = new_text
    if new_text.startswith(" ") or new_text.endswith(" "):
        t_elem.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")


def is_math_para(para) -> bool:
    """Return True if this paragraph contains math namespace elements."""
    return (
        para.find(f"{{{MATH_NS}}}oMath") is not None
        or para.find(f"{{{MATH_NS}}}oMathPara") is not None
    )


def apply_change(paras, idx, new_text, changes, label):
    """Apply set_para_text to a paragraph if it is not a MATH paragraph."""
    p = paras[idx]
    if is_math_para(p):
        changes.append(f"P{idx} [{label}]: SKIPPED (MATH paragraph)")
        return
    old_text = get_para_text(p)
    set_para_text(p, new_text)
    changes.append(f"P{idx} [{label}]: OK (was: '{old_text[:60]}')")


# ── Load document ──────────────────────────────────────────────────────────────
tree = ET.parse(str(DOC_XML))
root = tree.getroot()
paras = root.findall(f".//{W}p")

print(f"Total paragraphs: {len(paras)}")

changes = []

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION INTRO (P326–P328)
# ═══════════════════════════════════════════════════════════════════════════════

apply_change(
    paras, 326,
    (
        "У межах даного дослідження розглядаються сучасні нейронні моделі векторних ембеддінгів: "
        "BGE-M3, E5-base, nomic-embed-text-v1.5, Qwen3-Embedding-0.6B та text-embedding-3-large. "
        "Їх вибір обумовлений тим, що вони представляють провідні відкриті та комерційні рішення для мультилінгвального "
        "семантичного пошуку, що відрізняються архітектурою (encoder-based та decoder-based), методами навчання та "
        "особливостями формування векторних подань [13]."
    ),
    changes, "section-intro P326",
)

# P327 — leave unchanged (not in the task list)

apply_change(
    paras, 328,
    (
        "Послідовний аналіз обраних моделей дозволяє охопити різні сучасні підходи до побудови ембеддінгів для "
        "семантичного пошуку: від спеціалізованих мультирежимних encoder-based моделей (BGE-M3) до bi-encoder "
        "архітектур з контрастивним навчанням (E5-base), відкритих моделей з Matryoshka-представленням "
        "(nomic-embed-text-v1.5), декодерних архітектур (Qwen3-Embedding-0.6B) та комерційних хмарних "
        "API-сервісів (text-embedding-3-large)."
    ),
    changes, "section-intro P328",
)

# ═══════════════════════════════════════════════════════════════════════════════
# BGE-M3 subsection (formerly TF-IDF, P329–P343)
# ═══════════════════════════════════════════════════════════════════════════════

apply_change(
    paras, 329,
    "BGE-M3 як засіб мультилінгвального семантичного пошуку",
    changes, "BGE-M3 heading P329",
)

apply_change(
    paras, 330,
    (
        "Модель BGE-M3 (BAAI General Embedding, version 3) є мультилінгвальною моделлю векторних ембеддінгів, "
        "розробленою Пекінською академією штучного інтелекту (BAAI) [13]. Назва моделі відображає три ключові "
        "властивості: Multi-Lingual (підтримка більш ніж 100 мов), Multi-Functionality (три режими подання: dense, "
        "sparse та multi-vector) та Multi-Granularity (можливість роботи з текстами різного розміру — від речень до "
        "8192 токенів). Модель базується на архітектурі XLM-RoBERTa і є однією з найбільш потужних відкритих моделей "
        "для семантичного пошуку."
    ),
    changes, "BGE-M3 intro P330",
)

# P331–P334 are MATH — skipped automatically by apply_change

apply_change(
    paras, 335,
    (
        "BGE-M3 також підтримує розріджений режим (sparse retrieval), при якому замість щільного вектора формується "
        "зважений набір ключових термінів. Вага кожного терміна обчислюється на основі активаційних значень "
        "відповідного токена після трансформерного кодування, що дозволяє поєднувати переваги нейронного та "
        "лексичного пошуку."
    ),
    changes, "BGE-M3 sparse P335",
)

# P336–P339 — not in the change list, leave as-is

apply_change(
    paras, 340,
    (
        "Режим multi-vector (ColBERT-подібне подання) дозволяє зберігати вектор для кожного токена документа та "
        "виконувати пізнє взаємодоповнення між запитом і документом. Це забезпечує вищу точність порівняння, але "
        "потребує більшого обсягу пам'яті для зберігання векторного індексу."
    ),
    changes, "BGE-M3 multi-vector P340",
)

apply_change(
    paras, 341,
    (
        "Перевагою BGE-M3 є висока якість мультилінгвального пошуку, підтверджена результатами на публічних "
        "бенчмарках MTEB і BEIR. Модель одночасно підтримує кілька режимів подання, що дозволяє обрати оптимальний "
        "компроміс між якістю та швидкодією. Модель є відкритою (MIT-ліцензія) і може бути розгорнута локально без "
        "залежності від зовнішніх API."
    ),
    changes, "BGE-M3 advantages P341",
)

apply_change(
    paras, 342,
    (
        "Разом із тим BGE-M3 має певні обмеження. Загальний обсяг параметрів (~570M) є значним, що призводить до "
        "підвищених вимог до оперативної пам'яті при використанні на CPU. Для досягнення прийнятної швидкодії "
        "рекомендується обмежувати максимальну довжину послідовності (max_seq_length=256) або використовувати "
        "квантизацію."
    ),
    changes, "BGE-M3 limitations P342",
)

apply_change(
    paras, 343,
    (
        "Отже, BGE-M3 доцільно розглядати як потужну мультирежимну мультилінгвальну модель для семантичного пошуку, "
        "яка поєднує переваги щільного, розрідженого та multi-vector подань. Вона є одним із провідних відкритих "
        "рішень для корпоративного пошуку, але потребує відповідного апаратного забезпечення або оптимізації для "
        "ефективного розгортання на CPU."
    ),
    changes, "BGE-M3 conclusion P343",
)

# ═══════════════════════════════════════════════════════════════════════════════
# E5-base subsection (formerly Word2Vec, P347–P358)
# ═══════════════════════════════════════════════════════════════════════════════

apply_change(
    paras, 347,
    "Модель E5-base для контрастивного семантичного пошуку",
    changes, "E5-base heading P347",
)

apply_change(
    paras, 348,
    (
        "Модель E5-base (Embeddings from bidirectional Encoder rEpresentations) розроблена командою Microsoft Research "
        "та є представником родини мультилінгвальних моделей ембеддінгів на основі трансформерної архітектури [14]. "
        "Модель навчена методом weakly supervised contrastive learning на великих колекціях різнорідних текстових пар "
        "і оптимізована для задач bi-encoder retrieval."
    ),
    changes, "E5-base intro P348",
)

apply_change(
    paras, 349,
    (
        "Особливістю E5-base є система prefix-based encoding: для пошукових запитів до тексту додається префікс "
        "'query: ', а для індексованих документів — 'passage: '. Цей підхід дозволяє моделі розрізняти семантичну "
        "роль тексту ще на рівні входу та формувати більш точні парні подання для задач semantic retrieval."
    ),
    changes, "E5-base prefix P349",
)

apply_change(
    paras, 350,
    (
        "Навчання моделі ґрунтується на контрастивній функції втрат InfoNCE, яка максимізує схожість між вектором "
        "запиту та вектором позитивного документа і мінімізує схожість з негативними прикладами. Формально цільова "
        "функція може бути подана за формулою 5.3:"
    ),
    changes, "E5-base formula intro P350",
)

# P351–P354 are MATH — skipped automatically by apply_change

apply_change(
    paras, 355,
    (
        "У результаті навчання модель формує щільні вектори розмірністю 768, що відображають семантичні зв'язки між "
        "запитами та документами. Перевагою E5-base є стабільна якість на різних мовах і доменах, а також відносно "
        "компактний розмір (~278M параметрів)."
    ),
    changes, "E5-base result P355",
)

apply_change(
    paras, 356,
    (
        "Водночас E5-base потребує строгого дотримання prefix-формату вводу: пропуск префікса 'query:' або 'passage:' "
        "призводить до суттєвого погіршення якості пошуку. Крім того, модель має максимальну довжину послідовності "
        "512 токенів, що обмежує її застосування для дуже довгих документів."
    ),
    changes, "E5-base limitations P356",
)

apply_change(
    paras, 357,
    (
        "Ще одним аспектом є те, що E5-base, як і всі bi-encoder моделі, не виконує перехресну увагу між запитом і "
        "документом під час пошуку — вектори кодуються незалежно. Для задач, де потрібна максимальна точність "
        "ранжування, додатково рекомендується використовувати cross-encoder re-ranker."
    ),
    changes, "E5-base bi-encoder note P357",
)

apply_change(
    paras, 358,
    (
        "Отже, E5-base є ефективним рішенням для мультилінгвального семантичного пошуку в корпоративних системах, "
        "яке поєднує компактний розмір моделі з хорошою якістю retrieval завдяки контрастивному навчанню на "
        "різнорідних текстових даних."
    ),
    changes, "E5-base conclusion P358",
)

# ═══════════════════════════════════════════════════════════════════════════════
# nomic-embed-text-v1.5 subsection (formerly FastText, P359–P369)
# ═══════════════════════════════════════════════════════════════════════════════

apply_change(
    paras, 359,
    "nomic-embed-text-v1.5 із підтримкою довгого контексту та Matryoshka-подань",
    changes, "nomic heading P359",
)

apply_change(
    paras, 360,
    (
        "Модель nomic-embed-text-v1.5, розроблена компанією Nomic AI, є відкритою мультилінгвальною моделлю "
        "ембеддінгів, яка відзначається повною прозорістю: як архітектура, так і набір навчальних даних (понад "
        "235 мільйонів текстових пар) є публічно доступними. Модель реалізована на основі архітектури nomic-bert "
        "із застосуванням RoPE (Rotary Position Embedding) замість абсолютних позиційних ембеддінгів."
    ),
    changes, "nomic intro P360",
)

# P361 — MATH paragraph, skipped automatically by apply_change
# (FastText formula intro; becomes nomic formula intro but must not be touched)

# P362–P364 are MATH — skipped automatically

apply_change(
    paras, 365,
    (
        "Унікальною особливістю nomic-embed-text-v1.5 є підтримка Matryoshka Representation Learning (MRL). Ця "
        "технологія дозволяє варіювати розмірність ембеддінгу від 64 до 768 без перенавчання моделі, що надає "
        "гнучкість при виборі між якістю та обчислювальними витратами на зберігання індексу."
    ),
    changes, "nomic MRL P365",
)

apply_change(
    paras, 366,
    (
        "Порівняно з іншими відкритими моделями, nomic-embed-text-v1.5 підтримує контекстне вікно до 8192 токенів. "
        "Це перевищує можливості стандартних BERT-подібних моделей (512 токенів) і є суттєвою перевагою при пошуку "
        "по довгих корпоративних документах."
    ),
    changes, "nomic context P366",
)

apply_change(
    paras, 367,
    (
        "Водночас модель має технічні особливості при розгортанні: вимагає параметра trust_remote_code=True при "
        "завантаженні з Hugging Face Hub, а також додаткового Python-пакету einops. Це ускладнює інтеграцію в "
        "середовища з суворими вимогами безпеки."
    ),
    changes, "nomic deployment P367",
)

apply_change(
    paras, 368,
    (
        "Ще одним обмеженням є відносно менша точність на деяких вузькоспеціалізованих задачах порівняно з більшими "
        "моделями (BGE-M3, Qwen3-Embedding) при роботі з технічною або юридичною термінологією."
    ),
    changes, "nomic limitations P368",
)

apply_change(
    paras, 369,
    (
        "Отже, nomic-embed-text-v1.5 є цінним рішенням для корпоративного пошуку завдяки повній відкритості, "
        "підтримці довгого контексту та можливості варіювання розмірності ембеддінгів. Модель особливо підходить "
        "для застосувань, де важлива прозорість і відтворюваність методу."
    ),
    changes, "nomic conclusion P369",
)

# ═══════════════════════════════════════════════════════════════════════════════
# Qwen3-Embedding-0.6B subsection (formerly BERT, P370–P381)
# ═══════════════════════════════════════════════════════════════════════════════

apply_change(
    paras, 370,
    "Qwen3-Embedding-0.6B як декодерна модель векторних ембеддінгів",
    changes, "Qwen3 heading P370",
)

apply_change(
    paras, 371,
    (
        "Qwen3-Embedding-0.6B — мультилінгвальна модель ембеддінгів від компанії Alibaba Cloud, що є частиною "
        "родини моделей Qwen3. На відміну від більшості моделей ембеддінгів на основі BERT-подібної "
        "encoder-архітектури, Qwen3-Embedding використовує decoder-only архітектуру (GPT-подібну). Це є "
        "принциповою архітектурною відмінністю, оскільки декодерні моделі традиційно застосовуються для "
        "генеративних задач, а не для побудови векторних подань."
    ),
    changes, "Qwen3 intro P371",
)

apply_change(
    paras, 372,
    (
        "Ключовим підходом до отримання ембеддінгу у декодерній моделі є last-token pooling: вектор ембеддінгу "
        "відповідає стану останнього (EOS) токена після пропуску всього тексту через декодер. Такий підхід дозволяє "
        "агрегувати контекст усіх попередніх токенів у єдине щільне векторне подання завдяки механізму каузальної "
        "(автонапрямної) уваги."
    ),
    changes, "Qwen3 last-token pooling P372",
)

# P373 — MATH paragraph (BERT formula intro), skipped automatically by apply_change
# The task note says: "У спрощеному вигляді процес отримання ембеддінгу через last-token pooling описується формулою 5.6:"
# but since the current P373 contains an oMath element it must be skipped per rules.

# P374–P377 are MATH — skipped automatically

apply_change(
    paras, 378,
    (
        "Перевагою Qwen3-Embedding є висока якість мультилінгвального пошуку при кількості параметрів лише 0.6B. "
        "Модель підтримує понад 100 мов і максимальну довжину контексту до 32768 токенів, що робить її придатною "
        "для різноманітних корпоративних задач. За результатами бенчмарку MTEB модель демонструє конкурентоспроможну "
        "ефективність порівняно з набагато більшими encoder-based моделями."
    ),
    changes, "Qwen3 advantages P378",
)

apply_change(
    paras, 379,
    (
        "Разом із тим декодерна архітектура має свої особливості. Авто-напрямна (causal) увага обчислювально менш "
        "ефективна для задач кодування порівняно з двонапрямною (bidirectional) увагою BERT-подібних моделей. Це "
        "означає дещо більший час обробки при рівній кількості токенів."
    ),
    changes, "Qwen3 causal attn P379",
)

apply_change(
    paras, 380,
    (
        "У практичному застосуванні для досягнення прийнятної швидкодії на CPU рекомендується обмежувати максимальну "
        "довжину послідовності (max_seq_length=256). При цьому якість пошуку залишається прийнятною для більшості "
        "корпоративних документів."
    ),
    changes, "Qwen3 practical note P380",
)

apply_change(
    paras, 381,
    (
        "Отже, Qwen3-Embedding-0.6B є перспективною моделлю нового покоління, яка демонструє конкурентоспроможну "
        "якість пошуку при компактному розмірі, але потребує більшого часу для обробки запитів на системах без GPU "
        "порівняно з легшими encoder-based моделями."
    ),
    changes, "Qwen3 conclusion P381",
)

# ═══════════════════════════════════════════════════════════════════════════════
# text-embedding-3-large subsection (formerly Sentence-BERT, P382–P397)
# ═══════════════════════════════════════════════════════════════════════════════

apply_change(
    paras, 382,
    "text-embedding-3-large як комерційна хмарна модель ембеддінгів",
    changes, "te3l heading P382",
)

apply_change(
    paras, 383,
    (
        "Модель text-embedding-3-large є комерційною моделлю ембеддінгів від компанії OpenAI, доступною виключно "
        "через API. На відміну від решти розглянутих у даній роботі моделей, text-embedding-3-large не розгортається "
        "локально — для її використання необхідний API-ключ та постійне інтернет-підключення. Розмірність ембеддінгів "
        "складає 3072 вимірності."
    ),
    changes, "te3l intro P383",
)

# P384–P392 — MATH paragraphs or not in change list; only P393 onward are changed

apply_change(
    paras, 393,
    (
        "Важливою особливістю text-embedding-3-large є підтримка Matryoshka Representation Learning (MRL): розмірність "
        "ембеддінгу може бути зменшена до 256 або 512 без повторного навчання, що дозволяє знижувати витрати на "
        "зберігання індексу без критичної втрати якості."
    ),
    changes, "te3l MRL P393",
)

apply_change(
    paras, 394,
    (
        "Перевагою text-embedding-3-large є висока якість ембеддінгів на численних публічних бенчмарках. Модель не "
        "потребує локального розгортання, що спрощує інтеграцію в хмарну корпоративну інфраструктуру. Стабільна "
        "якість підтримується OpenAI незалежно від апаратних ресурсів замовника."
    ),
    changes, "te3l advantages P394",
)

apply_change(
    paras, 395,
    (
        "Ще однією перевагою є простота інтеграції через Python SDK (openai.embeddings.create). Це дозволяє швидко "
        "розгорнути пошукову систему без налаштування GPU-інфраструктури."
    ),
    changes, "te3l SDK P395",
)

apply_change(
    paras, 396,
    (
        "Разом із тим text-embedding-3-large має принципові обмеження для корпоративного застосування. По-перше, "
        "передача текстових даних на сервери OpenAI може суперечити вимогам конфіденційності даних та GDPR. "
        "По-друге, вартість API-запитів ($0.13 за 1M токенів) може бути суттєвою при великих обсягах індексації. "
        "По-третє, модель недоступна у мережево ізольованих середовищах."
    ),
    changes, "te3l limitations P396",
)

apply_change(
    paras, 397,
    (
        "Отже, text-embedding-3-large є оптимальним вибором для хмарних корпоративних систем з гнучкими вимогами до "
        "конфіденційності, де важливі висока якість пошуку та простота розгортання. Для систем із суворими вимогами "
        "до локального зберігання даних перевагу слід надавати відкритим локальним моделям."
    ),
    changes, "te3l conclusion P397",
)

# ═══════════════════════════════════════════════════════════════════════════════
# Table model names (P481, P487, P493, P499, P505)
# ═══════════════════════════════════════════════════════════════════════════════

apply_change(paras, 481, "BGE-M3",                  changes, "table model P481")
apply_change(paras, 487, "E5-base",                 changes, "table model P487")
apply_change(paras, 493, "nomic-embed-text-v1.5",   changes, "table model P493")
apply_change(paras, 499, "Qwen3-Embedding-0.6B",    changes, "table model P499")
apply_change(paras, 505, "text-embedding-3-large",  changes, "table model P505")

# ── Save ───────────────────────────────────────────────────────────────────────
tree.write(str(DOC_XML), encoding="unicode", xml_declaration=False)
print("Saved document.xml")
print()
print("=== Change summary ===")
for ch in changes:
    print(f"  {ch}")
print(f"\nTotal changes attempted: {len(changes)}")
skipped = sum(1 for c in changes if "SKIPPED" in c)
applied = sum(1 for c in changes if "OK" in c)
print(f"  Applied: {applied}")
print(f"  Skipped (MATH): {skipped}")

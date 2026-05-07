/**
 * Builds the master's thesis defense presentation:
 *   "Дослідження моделей векторних ембеддінгів для семантичного пошуку
 *    текстових даних у корпоративних базах знань"
 *
 * Author: Нестеренко В.В., ІПЗм-24-1, ХНУРЕ, 2026
 *
 * Output: thesis/Nesterenko_Presentation.pptx (18 slides, 16:9)
 *
 * Design highlights:
 *   - Deep-tech academic palette (navy + cyan + gold accent)
 *   - Native PowerPoint charts (column, bar, scatter, doughnut)
 *   - Iconography rasterised from react-icons
 *   - Title + conclusion slides on dark background ("sandwich" structure)
 *   - All numeric values verified against results/benchmark_*.json
 */

// pptxgenjs is installed globally; resolve via the global node_modules path.
const path = require("path");
const globalRoot = "C:/Users/vital/AppData/Roaming/npm/node_modules";

const pptxgen = require(path.join(globalRoot, "pptxgenjs"));
const React = require(path.join(globalRoot, "react"));
const ReactDOMServer = require(path.join(globalRoot, "react-dom/server"));
const sharp = require(path.join(globalRoot, "sharp"));

const fa = require(path.join(globalRoot, "react-icons/fa"));
const md = require(path.join(globalRoot, "react-icons/md"));
const hi = require(path.join(globalRoot, "react-icons/hi"));
const bi = require(path.join(globalRoot, "react-icons/bi"));

const SCRIPTS_DIR = path.dirname(__filename);
const OUTPUT = path.join(SCRIPTS_DIR, "..", "Nesterenko_Presentation.pptx");
const LOGO_HEX = path.join(SCRIPTS_DIR, "_logo_hex.png");
const LOGO_XNURE = path.join(SCRIPTS_DIR, "_logo_xnure.png");
const LOGO_KAFEDRA = path.join(SCRIPTS_DIR, "_logo_kafedra.png");

// ============================================================================
// PALETTE & TYPOGRAPHY
// ============================================================================

const C = {
  // surfaces
  bgDark: "0F172A",      // deep navy (title / conclusion)
  bgWhite: "FFFFFF",
  surfaceSoft: "F8FAFC", // slate-50
  surfaceCool: "EFF6FF", // sky-50
  surfaceGold: "FEF3C7", // amber-50

  // brand
  primary: "1E40AF",     // indigo-700
  primaryDark: "1E3A8A", // indigo-800
  primaryLight: "3B82F6",// blue-500
  secondary: "0891B2",   // cyan-600
  secondaryLight: "06B6D4",// cyan-500

  // accents
  gold: "F59E0B",        // amber-500
  goldDark: "D97706",    // amber-600
  emerald: "059669",     // emerald-600
  rose: "E11D48",        // rose-600
  violet: "7C3AED",      // violet-600

  // text
  text: "0F172A",        // near-black
  textStrong: "1E293B",  // slate-800
  textMid: "475569",     // slate-600
  textMuted: "64748B",   // slate-500
  textOnDark: "F1F5F9",  // slate-100
  textOnDarkMuted: "94A3B8", // slate-400
  border: "CBD5E1",      // slate-300
  borderSoft: "E2E8F0",  // slate-200
};

const F = {
  serif: "Cambria",
  sans: "Calibri",
  mono: "Consolas",
};

// ============================================================================
// ICON HELPERS
// ============================================================================

async function iconPng(IconComponent, color = "#1E40AF", size = 256) {
  const svg = ReactDOMServer.renderToStaticMarkup(
    React.createElement(IconComponent, { color, size: String(size) })
  );
  const png = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + png.toString("base64");
}

const hash = (s) => "#" + s; // colour helper for react-icons

// ============================================================================
// LAYOUT HELPERS
// ============================================================================

const W = 10.0;     // slide width
const H = 5.625;    // slide height

// Outer margins
const M_LEFT = 0.45;
const M_RIGHT = 0.45;

const TITLE_Y = 0.32;
const TITLE_H = 0.55;
const CONTENT_TOP = 1.05;
const CONTENT_BOTTOM = 5.20;
const CONTENT_H = CONTENT_BOTTOM - CONTENT_TOP;

function addPageNumber(slide, n, total, isDark = false) {
  slide.addText(`${String(n).padStart(2, "0")} / ${total}`, {
    x: W - 1.30, y: H - 0.32, w: 1.05, h: 0.25,
    fontFace: F.sans, fontSize: 9,
    color: isDark ? C.textOnDarkMuted : C.textMuted,
    align: "right", valign: "middle", margin: 0,
  });
}

// Department-of-Software-Engineering logo, anchored bottom-left on every
// content slide. Aspect ratio of the source PNG is 83 × 56 (≈ 1.48:1) so we
// scale to 0.55" × 0.37" for a subtle but legible mark.
function addKafedraLogo(slide) {
  slide.addImage({
    path: LOGO_HEX, x: M_LEFT, y: H - 0.50, w: 0.55, h: 0.37,
  });
}

function addTitle(slide, text, opts = {}) {
  slide.addText(text, {
    x: M_LEFT, y: TITLE_Y, w: W - M_LEFT - M_RIGHT - 0.6, h: TITLE_H,
    fontFace: F.sans, fontSize: opts.size || 22, bold: true,
    color: opts.color || C.text, align: "left", valign: "middle", margin: 0,
  });
  // Small accent dot to the left of the title (visual motif)
  slide.addShape("ellipse", {
    x: M_LEFT - 0.02, y: TITLE_Y + 0.16, w: 0.20, h: 0.20,
    fill: { color: opts.dotColor || C.gold }, line: { type: "none" },
  });
  // Reposition title to leave room for the dot
  // (we redraw text after dot to keep z-order; not strictly necessary).
}

function addTitleAdv(slide, text, opts = {}) {
  // Variant: title with accent bullet on the left, dark or light background.
  // Optional `opts.section` adds a small uppercase breadcrumb above the title
  // identifying the corresponding template section ("РОЗДІЛ N · НАЗВА").
  const dotColor = opts.dotColor || C.gold;
  const color = opts.color || C.text;
  const onDark = color !== C.text;

  if (opts.section) {
    slide.addText(opts.section, {
      x: M_LEFT, y: 0.10, w: W - M_LEFT - M_RIGHT, h: 0.20,
      fontFace: F.sans, fontSize: 8.5, bold: true, charSpacing: 3,
      color: onDark ? C.textOnDarkMuted : C.textMuted,
      align: "left", valign: "middle", margin: 0,
    });
  }

  slide.addShape("ellipse", {
    x: M_LEFT, y: TITLE_Y + 0.14, w: 0.22, h: 0.22,
    fill: { color: dotColor }, line: { type: "none" },
  });
  slide.addText(text, {
    x: M_LEFT + 0.34, y: TITLE_Y, w: W - M_LEFT - M_RIGHT - 1.6, h: TITLE_H,
    fontFace: F.sans, fontSize: opts.size || 22, bold: true,
    color, align: "left", valign: "middle", margin: 0,
  });
}

// Section-name constants follow the official template structure
// (Шаблон презентації квал роб маг.potm) — 11 content sections + title.
const SEC = {
  S1: "РОЗДІЛ 1 · ДОСЛІДЖЕННЯ",
  S2: "РОЗДІЛ 2 · ОГЛЯД ЛІТЕРАТУРИ (АНАЛОГІВ)",
  S3: "РОЗДІЛ 3 · ПОСТАНОВКА ЗАДАЧІ",
  S4: "РОЗДІЛ 4 · МЕТОДОЛОГІЯ",
  S5: "РОЗДІЛ 5 · АРХІТЕКТУРА СИСТЕМИ",
  S6: "РОЗДІЛ 6 · ПРОГРАМНЕ ЗАБЕЗПЕЧЕННЯ",
  S7: "РОЗДІЛ 7 · ЗМІСТ ЕКСПЕРИМЕНТУ",
  S8: "РОЗДІЛ 8 · РЕЗУЛЬТАТИ ЕКСПЕРИМЕНТУ",
  S9: "РОЗДІЛ 9 · АНАЛІЗ РЕЗУЛЬТАТІВ",
  S10: "РОЗДІЛ 10 · ПУБЛІКАЦІЯ РЕЗУЛЬТАТІВ",
  S11: "РОЗДІЛ 11 · ПІДСУМКИ",
};

// Round white card with subtle shadow + thin top accent bar (drawn as RECT inset).
function addCard(slide, x, y, w, h, opts = {}) {
  const fill = opts.fill || C.bgWhite;
  const radius = opts.radius != null ? opts.radius : 0.06;
  slide.addShape("roundRect", {
    x, y, w, h,
    fill: { color: fill },
    line: { color: opts.borderColor || C.borderSoft, width: opts.borderWidth || 0.75 },
    rectRadius: radius,
    shadow: opts.shadow === false ? undefined :
      { type: "outer", blur: 6, offset: 1, angle: 90, color: "000000", opacity: 0.07 },
  });
}

// ============================================================================
// MAIN BUILD
// ============================================================================

async function build() {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_16x9";
  pres.author = "Нестеренко В.В.";
  pres.title = "Дослідження моделей векторних ембеддінгів для семантичного пошуку";

  const TOTAL = 18;

  // Pre-render all icons we use ----------------------------------------------
  const I = {
    target:   await iconPng(fa.FaBullseye, hash(C.gold)),
    list:     await iconPng(fa.FaListUl, hash(C.primary)),
    search:   await iconPng(fa.FaSearch, hash(C.primary)),
    book:     await iconPng(fa.FaBookOpen, hash(C.primary)),
    eval:     await iconPng(fa.FaCheckCircle, hash(C.emerald)),
    chart:    await iconPng(fa.FaChartBar, hash(C.primary)),
    rank:     await iconPng(fa.FaTrophy, hash(C.gold)),
    speed:    await iconPng(fa.FaTachometerAlt, hash(C.secondary)),
    brain:    await iconPng(fa.FaBrain, hash(C.primary)),
    cog:      await iconPng(fa.FaCogs, hash(C.primary)),
    db:       await iconPng(fa.FaDatabase, hash(C.primary)),
    code:     await iconPng(fa.FaCode, hash(C.secondary)),
    flask:    await iconPng(fa.FaFlask, hash(C.secondary)),
    flag:     await iconPng(fa.FaFlag, hash(C.gold)),
    layers:   await iconPng(fa.FaLayerGroup, hash(C.primary)),
    network:  await iconPng(fa.FaNetworkWired, hash(C.primary)),
    quote:    await iconPng(fa.FaQuoteLeft, hash(C.secondary)),
    laptop:   await iconPng(fa.FaLaptopCode, hash(C.primary)),
    server:   await iconPng(fa.FaServer, hash(C.primary)),
    file:     await iconPng(fa.FaFileAlt, hash(C.primary)),
    award:    await iconPng(fa.FaAward, hash(C.gold)),
    lightbulb:await iconPng(fa.FaLightbulb, hash(C.gold)),
    pie:      await iconPng(fa.FaChartPie, hash(C.primary)),
    line:     await iconPng(fa.FaChartLine, hash(C.primary)),
    formula:  await iconPng(fa.FaSquareRootAlt, hash(C.primary)),
    balance:  await iconPng(fa.FaBalanceScale, hash(C.primary)),
    medical:  await iconPng(fa.FaHeartbeat, hash(C.rose)),
    legal:    await iconPng(fa.FaGavel, hash(C.violet)),
    tech:     await iconPng(fa.FaMicrochip, hash(C.secondary)),
    user:     await iconPng(fa.FaUserGraduate, hash(C.primary)),
    calendar: await iconPng(fa.FaCalendarAlt, hash(C.primary)),
    clock:    await iconPng(fa.FaClock, hash(C.gold)),
    star:     await iconPng(fa.FaStar, hash(C.gold)),
    arrow:    await iconPng(fa.FaArrowRight, hash(C.textMuted)),
    check:    await iconPng(fa.FaCheck, hash(C.emerald)),
    trophy:   await iconPng(fa.FaTrophy, hash(C.gold)),
    medal:    await iconPng(fa.FaMedal, hash(C.gold)),
  };

  // ==========================================================================
  // SLIDE 1 — TITLE  (light background, modern academic feel)
  // ==========================================================================
  {
    const s = pres.addSlide();
    s.background = { color: C.bgWhite };

    // Subtle decorative motif on the right edge: layered translucent circles
    // in brand colours so the page has personality without feeling busy.
    s.addShape("ellipse", {
      x: 7.7, y: -1.4, w: 4.4, h: 4.4,
      fill: { color: C.surfaceCool }, line: { type: "none" },
    });
    s.addShape("ellipse", {
      x: 8.4, y: -0.7, w: 2.9, h: 2.9,
      fill: { color: C.primary, transparency: 88 }, line: { type: "none" },
    });
    s.addShape("ellipse", {
      x: 8.9, y: -0.2, w: 1.6, h: 1.6,
      fill: { color: C.gold, transparency: 78 }, line: { type: "none" },
    });

    // Slim navy bar down the left margin — anchors the composition without
    // a full-width header bar.
    s.addShape("rect", {
      x: 0, y: 0, w: 0.18, h: H,
      fill: { color: C.primary }, line: { type: "none" },
    });

    // Tag chip ("Магістерська кваліфікаційна робота") — soft amber pill
    s.addShape("roundRect", {
      x: M_LEFT, y: 0.50, w: 4.20, h: 0.34,
      fill: { color: C.surfaceGold }, line: { color: C.gold, width: 0.75 },
      rectRadius: 0.16,
    });
    s.addText("МАГІСТЕРСЬКА КВАЛІФІКАЦІЙНА РОБОТА", {
      x: M_LEFT, y: 0.50, w: 4.20, h: 0.34,
      fontFace: F.sans, fontSize: 10, bold: true, charSpacing: 2,
      color: C.goldDark, align: "center", valign: "middle", margin: 0,
    });

    // Main title — deep navy on white
    s.addText("Дослідження моделей векторних ембеддінгів для семантичного пошуку текстових даних у корпоративних базах знань", {
      x: M_LEFT, y: 1.05, w: 8.6, h: 1.95,
      fontFace: F.sans, fontSize: 26, bold: true,
      color: C.text, align: "left", valign: "top", margin: 0,
    });

    // Thin gold separator
    s.addShape("rect", {
      x: M_LEFT, y: 3.10, w: 0.80, h: 0.05,
      fill: { color: C.gold }, line: { type: "none" },
    });

    // Author block
    s.addText([
      { text: "Виконав:  ", options: { color: C.textMuted, fontSize: 13 } },
      { text: "Нестеренко Віталій Вячеславович", options: { color: C.text, fontSize: 14, bold: true, breakLine: true } },
      { text: "Група:  ", options: { color: C.textMuted, fontSize: 13 } },
      { text: "ІПЗм-24-1", options: { color: C.text, fontSize: 14, bold: true } },
    ], {
      x: M_LEFT, y: 3.30, w: 5.0, h: 0.95,
      fontFace: F.sans, valign: "top", margin: 0, paraSpaceAfter: 4,
    });

    s.addText([
      { text: "Науковий керівник:  ", options: { color: C.textMuted, fontSize: 13, breakLine: true } },
      { text: "к.т.н., доцент Русакова Наталія Євгенівна", options: { color: C.text, fontSize: 14, bold: true } },
    ], {
      x: M_LEFT, y: 4.25, w: 6.0, h: 0.85,
      fontFace: F.sans, valign: "top", margin: 0,
    });

    // Date + venue (bottom right) — calendar icon recoloured to brand navy
    const calIconNavy = await iconPng(fa.FaCalendarAlt, hash(C.primary));
    s.addImage({ data: calIconNavy, x: 7.85, y: 4.45, w: 0.20, h: 0.20 });
    s.addText("15 травня 2026   ·   Харків", {
      x: 8.10, y: 4.40, w: 1.85, h: 0.30,
      fontFace: F.sans, fontSize: 11, color: C.textMid,
      align: "left", valign: "middle", margin: 0,
    });

    // Logos in bottom row
    s.addImage({ path: LOGO_XNURE,    x: M_LEFT,        y: H - 0.50, w: 1.80, h: 0.40 });
    s.addImage({ path: LOGO_KAFEDRA,  x: M_LEFT + 2.10, y: H - 0.48, w: 1.80, h: 0.36 });
    s.addImage({ path: LOGO_HEX,      x: M_LEFT + 4.20, y: H - 0.50, w: 0.70, h: 0.40 });
  }

  // ==========================================================================
  // SLIDE 2 — Мета та задачі
  // ==========================================================================
  {
    const s = pres.addSlide();
    s.background = { color: C.bgWhite };
    addTitleAdv(s, "Актуальність, мета та задачі дослідження", { section: SEC.S1 });

    // Goal card — full width gradient-like dark band
    addCard(s, M_LEFT, 1.05, W - M_LEFT - M_RIGHT, 0.85, {
      fill: C.primary, borderColor: C.primary, borderWidth: 0,
    });
    s.addImage({ data: await iconPng(fa.FaBullseye, hash("FFFFFF")), x: 0.65, y: 1.20, w: 0.55, h: 0.55 });
    s.addText([
      { text: "МЕТА:  ", options: { bold: true, color: C.gold, fontSize: 12 } },
      { text: "визначення доцільності застосування сучасних моделей векторних ембеддінгів для підвищення ефективності семантичного пошуку текстових документів у корпоративних базах знань шляхом порівняльного експериментального дослідження.",
        options: { color: C.textOnDark, fontSize: 12 } },
    ], {
      x: 1.40, y: 1.10, w: W - 1.40 - M_RIGHT - 0.10, h: 0.75,
      fontFace: F.sans, valign: "middle", margin: 0,
    });

    // Object / Subject row (two pills)
    const subY = 2.05;
    addCard(s, M_LEFT, subY, 4.55, 0.70, { fill: C.surfaceSoft, borderColor: C.borderSoft });
    s.addText("ОБ'ЄКТ", {
      x: M_LEFT + 0.18, y: subY + 0.06, w: 0.95, h: 0.30,
      fontFace: F.sans, fontSize: 9, bold: true, charSpacing: 2, color: C.gold,
      align: "left", valign: "top", margin: 0,
    });
    s.addText("процеси формування та використання векторних подань текстових даних у системах семантичного пошуку", {
      x: M_LEFT + 0.18, y: subY + 0.30, w: 4.20, h: 0.40,
      fontFace: F.sans, fontSize: 10, color: C.textStrong,
      align: "left", valign: "top", margin: 0,
    });

    addCard(s, M_LEFT + 4.75, subY, 4.55, 0.70, { fill: C.surfaceSoft, borderColor: C.borderSoft });
    s.addText("ПРЕДМЕТ", {
      x: M_LEFT + 4.75 + 0.18, y: subY + 0.06, w: 1.10, h: 0.30,
      fontFace: F.sans, fontSize: 9, bold: true, charSpacing: 2, color: C.secondary,
      align: "left", valign: "top", margin: 0,
    });
    s.addText("моделі векторних ембеддінгів для семантичного пошуку у корпоративних базах знань", {
      x: M_LEFT + 4.75 + 0.18, y: subY + 0.30, w: 4.20, h: 0.40,
      fontFace: F.sans, fontSize: 10, color: C.textStrong,
      align: "left", valign: "top", margin: 0,
    });

    // Tasks header
    s.addText("ЗАДАЧІ ДОСЛІДЖЕННЯ", {
      x: M_LEFT, y: 2.85, w: 4.0, h: 0.30,
      fontFace: F.sans, fontSize: 10, bold: true, charSpacing: 2, color: C.textMuted,
      align: "left", valign: "middle", margin: 0,
    });

    // 6 tasks in 3×2 grid
    const tasks = [
      { n: "1", t: "Аналіз предметної галузі",       icon: I.search },
      { n: "2", t: "Огляд embedding-моделей",        icon: I.book },
      { n: "3", t: "Формування benchmark dataset",   icon: I.db },
      { n: "4", t: "Реалізація системи пошуку",      icon: I.cog },
      { n: "5", t: "Оцінка якості моделей",          icon: I.eval },
      { n: "6", t: "Вибір оптимальної моделі",       icon: I.balance },
    ];
    const cardW = (W - M_LEFT - M_RIGHT - 0.30) / 3;  // 3 columns, 0.15 gap
    const cardH = 0.85;
    const colGap = 0.15;
    const rowGap = 0.15;
    for (let i = 0; i < tasks.length; i++) {
      const col = i % 3, row = Math.floor(i / 3);
      const cx = M_LEFT + col * (cardW + colGap);
      const cy = 3.20 + row * (cardH + rowGap);
      addCard(s, cx, cy, cardW, cardH, { fill: C.bgWhite, borderColor: C.borderSoft });
      // Number circle
      s.addShape("ellipse", {
        x: cx + 0.18, y: cy + 0.20, w: 0.45, h: 0.45,
        fill: { color: C.primary }, line: { type: "none" },
      });
      s.addText(tasks[i].n, {
        x: cx + 0.18, y: cy + 0.20, w: 0.45, h: 0.45,
        fontFace: F.sans, fontSize: 16, bold: true, color: C.textOnDark,
        align: "center", valign: "middle", margin: 0,
      });
      s.addText(tasks[i].t, {
        x: cx + 0.75, y: cy + 0.10, w: cardW - 0.85, h: cardH - 0.20,
        fontFace: F.sans, fontSize: 11, color: C.textStrong, bold: true,
        align: "left", valign: "middle", margin: 0,
      });
    }

    addKafedraLogo(s); addPageNumber(s,2, TOTAL);
  }

  // ==========================================================================
  // SLIDE 3 — Огляд літератури (timeline)
  // ==========================================================================
  {
    const s = pres.addSlide();
    s.background = { color: C.bgWhite };
    addTitleAdv(s, "Огляд літератури: еволюція embedding-моделей", { section: SEC.S2 });

    // Timeline track. y=2.95 leaves a clear ~0.25" gap below the top
    // research-gap callout (which ends near y=2.00) and lets the description
    // boxes underneath the dots breathe down to the bottom info band.
    const trackY = 2.95;
    s.addShape("rect", {
      x: M_LEFT + 0.40, y: trackY, w: W - M_LEFT - M_RIGHT - 0.80, h: 0.025,
      fill: { color: C.border }, line: { type: "none" },
    });

    const events = [
      { year: "2013", model: "Word2Vec",      author: "Mikolov",   note: "Статистичні\nвекторні подання",         color: C.textMuted },
      { year: "2018", model: "BERT",          author: "Devlin",    note: "Трансформери,\nконтекстність",          color: C.secondary },
      { year: "2019", model: "Sentence-BERT", author: "Reimers",   note: "Семантичні\nвекторизатори",             color: C.secondary },
      { year: "2022", model: "E5",            author: "Wang",      note: "Multilingual,\ncontrastive",            color: C.primary },
      { year: "2024", model: "BGE-M3 / nomic",author: "Chen / Nussbaum",note: "Multi-functionality,\nlong context", color: C.primary },
      { year: "2025", model: "Qwen3-Embed.",  author: "Zhang",     note: "Decoder-based,\ninstruction-aware",     color: C.gold },
    ];

    const trackW = W - M_LEFT - M_RIGHT - 0.80;
    const stepW = trackW / (events.length - 1);
    const startX = M_LEFT + 0.40;
    for (let i = 0; i < events.length; i++) {
      const ev = events[i];
      const cx = startX + i * stepW;

      // Year label above
      s.addText(ev.year, {
        x: cx - 0.50, y: trackY - 0.65, w: 1.0, h: 0.30,
        fontFace: F.sans, fontSize: 14, bold: true, color: ev.color,
        align: "center", valign: "middle", margin: 0,
      });

      // Big dot
      s.addShape("ellipse", {
        x: cx - 0.15, y: trackY - 0.135, w: 0.30, h: 0.30,
        fill: { color: ev.color }, line: { color: C.bgWhite, width: 2.5 },
      });

      // Model + author + note below
      s.addText(ev.model, {
        x: cx - 0.85, y: trackY + 0.25, w: 1.70, h: 0.30,
        fontFace: F.sans, fontSize: 12, bold: true, color: C.textStrong,
        align: "center", valign: "middle", margin: 0,
      });
      s.addText(ev.author, {
        x: cx - 0.85, y: trackY + 0.55, w: 1.70, h: 0.22,
        fontFace: F.sans, fontSize: 10, italic: true, color: C.textMuted,
        align: "center", valign: "middle", margin: 0,
      });
      s.addText(ev.note, {
        x: cx - 0.95, y: trackY + 0.78, w: 1.90, h: 0.95,
        fontFace: F.sans, fontSize: 10, color: C.textMid,
        align: "center", valign: "top", margin: 0,
      });
    }

    // Top callout: Дослідницька прогалина
    addCard(s, M_LEFT, 1.05, W - M_LEFT - M_RIGHT, 0.95, {
      fill: C.surfaceCool, borderColor: C.primary, borderWidth: 1,
    });
    s.addImage({ data: I.lightbulb, x: 0.65, y: 1.25, w: 0.55, h: 0.55 });
    s.addText([
      { text: "Дослідницька прогалина:  ", options: { bold: true, color: C.primaryDark, fontSize: 13 } },
      { text: "відсутність порівняльних досліджень embedding-моделей на україномовних доменно-специфічних колекціях. ",
        options: { color: C.textStrong, fontSize: 12 } },
      { text: "Опрацьовано 29 наукових джерел", options: { color: C.gold, fontSize: 12, bold: true } },
      { text: " (Lewis 2020 — RAG, Robertson 2009 — BM25, Thakur 2021 — BEIR, Muennighoff 2023 — MTEB).",
        options: { color: C.textStrong, fontSize: 12 } },
    ], {
      x: 1.40, y: 1.10, w: W - 1.40 - M_RIGHT - 0.20, h: 0.85,
      fontFace: F.sans, valign: "middle", margin: 0,
    });

    // Bottom band: domains badge — anchored just above the footer logo so
    // there is no large empty stripe between timeline notes and the band.
    // Logo sits at y=5.125; we end the band at 5.10 to leave a hairline gap.
    const bandY = 4.70, bandH = 0.40;
    addCard(s, M_LEFT, bandY, W - M_LEFT - M_RIGHT, bandH, {
      fill: C.surfaceSoft, borderColor: C.borderSoft, shadow: false,
    });
    s.addText([
      { text: "Хронологічний охоплюваний період:  ", options: { color: C.textMuted, fontSize: 10.5 } },
      { text: "2013 – 2025  ", options: { color: C.textStrong, fontSize: 11, bold: true } },
      { text: "·   статистичні → контекстні → багатофункціональні   ·  ", options: { color: C.textMuted, fontSize: 10.5 } },
      { text: "12+ років еволюції", options: { color: C.gold, fontSize: 11, bold: true } },
    ], {
      x: M_LEFT, y: bandY, w: W - M_LEFT - M_RIGHT, h: bandH,
      fontFace: F.sans, valign: "middle", align: "center", margin: 0,
    });

    addKafedraLogo(s); addPageNumber(s,3, TOTAL);
  }

  // ==========================================================================
  // SLIDE 4 — Постановка задачі
  // ==========================================================================
  {
    const s = pres.addSlide();
    s.background = { color: C.bgWhite };
    addTitleAdv(s, "Постановка задачі дослідження", { section: SEC.S3 });

    // Left: Problem panel (dark)
    const lx = M_LEFT, ly = CONTENT_TOP, lw = 3.65, lh = CONTENT_H - 0.10;
    s.addShape("roundRect", {
      x: lx, y: ly, w: lw, h: lh,
      fill: { color: C.bgDark }, line: { type: "none" }, rectRadius: 0.10,
    });
    s.addText("ПРОБЛЕМА", {
      x: lx + 0.30, y: ly + 0.20, w: lw - 0.60, h: 0.30,
      fontFace: F.sans, fontSize: 10, bold: true, charSpacing: 3, color: C.gold,
      align: "left", valign: "top", margin: 0,
    });
    s.addText("Недостатньо просто застосувати сучасну embedding-модель — необхідно обґрунтовано визначити модель, що забезпечує оптимальний баланс між:", {
      x: lx + 0.30, y: ly + 0.55, w: lw - 0.60, h: 0.95,
      fontFace: F.sans, fontSize: 11, color: C.textOnDark,
      align: "left", valign: "top", margin: 0, paraSpaceAfter: 4,
    });
    const criteria = [
      { c: "якість семантичного пошуку",         color: C.gold },
      { c: "швидкодія на CPU/GPU",               color: C.secondaryLight },
      { c: "вимоги до пам'яті та ресурсів",      color: C.primaryLight },
      { c: "мовна підтримка",                    color: C.emerald },
      { c: "тип колекції документів",            color: C.rose },
    ];
    // Spread criteria evenly across the remaining panel height so the dark
    // box doesn't have a large empty band at the bottom. Description block
    // ends around ly+1.50; panel inner-bottom is ly+lh-0.20.
    let cy0 = ly + 1.55;
    const critStep = 0.50;
    for (let i = 0; i < criteria.length; i++) {
      const cy = cy0 + i * critStep;
      s.addShape("ellipse", {
        x: lx + 0.30, y: cy + 0.12, w: 0.10, h: 0.10,
        fill: { color: criteria[i].color }, line: { type: "none" },
      });
      s.addText(criteria[i].c, {
        x: lx + 0.50, y: cy, w: lw - 0.80, h: 0.35,
        fontFace: F.sans, fontSize: 11, color: C.textOnDark,
        align: "left", valign: "middle", margin: 0,
      });
    }

    // Right: 5 expected results
    const rx = M_LEFT + lw + 0.25, ry = CONTENT_TOP, rw = W - rx - M_RIGHT;
    s.addText("ОЧІКУВАНІ РЕЗУЛЬТАТИ", {
      x: rx, y: ry, w: rw, h: 0.30,
      fontFace: F.sans, fontSize: 10, bold: true, charSpacing: 3, color: C.textMuted,
      align: "left", valign: "top", margin: 0,
    });

    const results = [
      "Реалізована система семантичного пошуку (PDF / DOCX / TXT / MD)",
      "Власноруч сформований UA benchmark dataset (3 домени)",
      "Кількісне порівняння 4 embedding-моделей + BM25 baseline",
      "Багатокритеріальний вибір моделі (Парето + лінійна згортка)",
      "Обґрунтовані практичні рекомендації для впровадження",
    ];
    const itemH = 0.65;
    const itemGap = 0.10;
    for (let i = 0; i < results.length; i++) {
      const iy = ry + 0.40 + i * (itemH + itemGap);
      addCard(s, rx, iy, rw, itemH, { fill: C.bgWhite, borderColor: C.borderSoft });
      // Number badge
      s.addShape("rect", {
        x: rx, y: iy, w: 0.07, h: itemH,
        fill: { color: i === 0 ? C.gold : i === 1 ? C.secondary : i === 2 ? C.primary : i === 3 ? C.emerald : C.rose },
        line: { type: "none" },
      });
      s.addText(String(i + 1), {
        x: rx + 0.20, y: iy, w: 0.50, h: itemH,
        fontFace: F.sans, fontSize: 22, bold: true, color: C.textStrong,
        align: "center", valign: "middle", margin: 0,
      });
      s.addText(results[i], {
        x: rx + 0.80, y: iy, w: rw - 0.95, h: itemH,
        fontFace: F.sans, fontSize: 12, color: C.textStrong,
        align: "left", valign: "middle", margin: 0,
      });
    }

    addKafedraLogo(s); addPageNumber(s,4, TOTAL);
  }

  // ==========================================================================
  // SLIDE 5 — Сучасні embedding-моделі (4 cards + params chart)
  // ==========================================================================
  {
    const s = pres.addSlide();
    s.background = { color: C.bgWhite };
    addTitleAdv(s, "Сучасні моделі векторних ембеддінгів", { section: SEC.S4 });

    const models = [
      { name: "BGE-M3",           lab: "BAAI",       arch: "Encoder",         params: 568, seq: 8192,  feat: "dense + sparse + multi-vector", train: "self-knowledge distillation", color: C.primary },
      { name: "E5-base",          lab: "intfloat",   arch: "Encoder (ML-E5)", params: 278, seq: 512,   feat: "префікси query: / passage:",   train: "weakly-supervised contrastive", color: C.secondary },
      { name: "nomic-embed v1.5", lab: "Nomic AI",   arch: "Encoder",         params: 137, seq: 8192,  feat: "Matryoshka representations",   train: "long-context contrastive",      color: C.violet },
      { name: "Qwen3-Embed-0.6B", lab: "Alibaba",    arch: "Decoder (LLM)",   params: 596, seq: 32768, feat: "instruction-aware",            train: "foundation-model fine-tuning",  color: C.gold },
    ];

    // 2×2 cards (left), bar chart of params on right
    const cardX = M_LEFT, cardY = CONTENT_TOP, cardW = 2.95, cardH = 1.95;
    const xGap = 0.15, yGap = 0.15;
    for (let i = 0; i < 4; i++) {
      const col = i % 2, row = Math.floor(i / 2);
      const cx = cardX + col * (cardW + xGap);
      const cy = cardY + row * (cardH + yGap);
      addCard(s, cx, cy, cardW, cardH, { fill: C.bgWhite, borderColor: C.borderSoft });
      // Side stripe
      s.addShape("rect", {
        x: cx, y: cy, w: 0.10, h: cardH,
        fill: { color: models[i].color }, line: { type: "none" },
      });
      // Title + lab
      s.addText(models[i].name, {
        x: cx + 0.22, y: cy + 0.10, w: cardW - 0.32, h: 0.30,
        fontFace: F.sans, fontSize: 14, bold: true, color: C.textStrong,
        align: "left", valign: "top", margin: 0,
      });
      s.addText(models[i].lab, {
        x: cx + 0.22, y: cy + 0.40, w: cardW - 0.32, h: 0.20,
        fontFace: F.sans, fontSize: 9.5, italic: true, color: models[i].color,
        align: "left", valign: "top", margin: 0,
      });
      // Stats grid
      const stats = [
        ["параметри",  `${models[i].params}M`],
        ["max токенів", models[i].seq.toLocaleString("uk")],
        ["архітектура", models[i].arch],
      ];
      let sy = cy + 0.65;
      for (const [k, v] of stats) {
        s.addText(k, {
          x: cx + 0.22, y: sy, w: 1.10, h: 0.22,
          fontFace: F.sans, fontSize: 9, color: C.textMuted,
          align: "left", valign: "middle", margin: 0,
        });
        s.addText(v, {
          x: cx + 1.30, y: sy, w: cardW - 1.40, h: 0.22,
          fontFace: F.sans, fontSize: 10, bold: true, color: C.textStrong,
          align: "left", valign: "middle", margin: 0,
        });
        sy += 0.22;
      }
      // Feature
      s.addText(models[i].feat, {
        x: cx + 0.22, y: cy + cardH - 0.42, w: cardW - 0.32, h: 0.36,
        fontFace: F.sans, fontSize: 9.5, italic: true, color: C.textMid,
        align: "left", valign: "middle", margin: 0,
      });
    }

    // Right: bar chart of params + max tokens (mini).
    // Per-bar coloring requires one series per bar — pptxgenjs paints one
    // series uniformly otherwise. We use 4 single-value series, each with the
    // active row labelled and the others as empty strings.
    const chX = cardX + 2 * (cardW + xGap), chY = cardY, chW = W - chX - M_RIGHT;
    addCard(s, chX, chY, chW, 4.05, { fill: C.surfaceSoft, borderColor: C.borderSoft });
    s.addText("ПАРАМЕТРИ МОДЕЛІ", {
      x: chX + 0.20, y: chY + 0.08, w: chW - 0.40, h: 0.25,
      fontFace: F.sans, fontSize: 9, bold: true, charSpacing: 2, color: C.textMuted,
      align: "left", valign: "top", margin: 0,
    });

    const modelLabels = ["BGE-M3", "E5-base", "nomic", "Qwen3"];
    const paramVals  = [568, 278, 137, 596];
    const tokenVals  = [8192, 512, 8192, 32768];
    const modelColors = [C.primary, C.secondary, C.violet, C.gold];

    // For each model: a series with values=[null,…,N,…,null] (one non-null at
    // the model's index) — this paints exactly one bar per series.
    const paramSeries = modelLabels.map((mn, i) => ({
      name: mn, labels: modelLabels,
      values: modelLabels.map((_, j) => j === i ? paramVals[i] : null),
    }));
    s.addChart(pres.charts.BAR, paramSeries, {
      x: chX + 0.10, y: chY + 0.36, w: chW - 0.20, h: 1.55, barDir: "bar",
      barGrouping: "clustered",
      chartColors: modelColors,
      catAxisLabelColor: C.textMid, catAxisLabelFontSize: 9, catAxisLabelFontFace: F.sans,
      valAxisLabelColor: C.textMid, valAxisLabelFontSize: 8, valAxisLabelFontFace: F.sans,
      valGridLine: { color: C.borderSoft, size: 0.5 },
      catGridLine: { style: "none" },
      showValue: true, dataLabelPosition: "outEnd", dataLabelColor: C.textStrong,
      dataLabelFontSize: 9, dataLabelFontFace: F.sans,
      showLegend: false,
      barOverlap: 100,
      chartArea: { fill: { color: C.surfaceSoft } },
      plotArea: { fill: { color: C.surfaceSoft } },
    });

    s.addText("MAX ТОКЕНІВ (ЛОГАРИФМІЧНА ШКАЛА)", {
      x: chX + 0.20, y: chY + 1.95, w: chW - 0.40, h: 0.25,
      fontFace: F.sans, fontSize: 9, bold: true, charSpacing: 2, color: C.textMuted,
      align: "left", valign: "top", margin: 0,
    });
    const tokenSeries = modelLabels.map((mn, i) => ({
      name: mn, labels: modelLabels,
      values: modelLabels.map((_, j) => j === i ? tokenVals[i] : null),
    }));
    s.addChart(pres.charts.BAR, tokenSeries, {
      x: chX + 0.10, y: chY + 2.22, w: chW - 0.20, h: 1.75, barDir: "bar",
      barGrouping: "clustered",
      chartColors: modelColors,
      catAxisLabelColor: C.textMid, catAxisLabelFontSize: 9, catAxisLabelFontFace: F.sans,
      valAxisLabelColor: C.textMid, valAxisLabelFontSize: 8, valAxisLabelFontFace: F.sans,
      valGridLine: { color: C.borderSoft, size: 0.5 },
      catGridLine: { style: "none" },
      showValue: true, dataLabelPosition: "outEnd", dataLabelColor: C.textStrong,
      dataLabelFontSize: 9, dataLabelFontFace: F.sans,
      showLegend: false,
      barOverlap: 100,
      valAxisLogScaleBase: 10,
      chartArea: { fill: { color: C.surfaceSoft } },
      plotArea: { fill: { color: C.surfaceSoft } },
    });

    addKafedraLogo(s); addPageNumber(s,5, TOTAL);
  }

  // ==========================================================================
  // SLIDE 6 — Методологія
  // ==========================================================================
  {
    const s = pres.addSlide();
    s.background = { color: C.bgWhite };
    addTitleAdv(s, "Методологія дослідження", { section: SEC.S4 });

    // Section header — metrics
    s.addText("МЕТРИКИ ОЦІНЮВАННЯ RETRIEVAL-ЯКОСТІ", {
      x: M_LEFT, y: CONTENT_TOP, w: 8.0, h: 0.30,
      fontFace: F.sans, fontSize: 10, bold: true, charSpacing: 3, color: C.textMuted,
      align: "left", valign: "middle", margin: 0,
    });

    // 5 metric cards in a row
    const metrics = [
      { sym: "nDCG", label: "@10", note: "якість ранжування\n(основна)",  icon: I.rank,  color: C.gold },
      { sym: "MRR",  label: "@10", note: "позиція першого\nрелевантного",  icon: I.medal, color: C.primary },
      { sym: "R",    label: "@10", note: "повнота\nпошуку",                 icon: I.search, color: C.secondary },
      { sym: "P",    label: "@10", note: "точність\nвидачі",                icon: I.eval, color: C.emerald },
      { sym: "ms",   label: "/q",  note: "час відповіді\n(швидкодія)",      icon: I.speed, color: C.rose },
    ];
    const mGap = 0.13;
    const mW = (W - M_LEFT - M_RIGHT - 4 * mGap) / 5;
    for (let i = 0; i < metrics.length; i++) {
      const cx = M_LEFT + i * (mW + mGap);
      const cy = CONTENT_TOP + 0.40;
      const m = metrics[i];
      addCard(s, cx, cy, mW, 1.55, { fill: C.bgWhite, borderColor: C.borderSoft });
      // Top color band
      s.addShape("rect", {
        x: cx, y: cy, w: mW, h: 0.10,
        fill: { color: m.color }, line: { type: "none" },
      });
      // Symbol and label
      s.addText([
        { text: m.sym, options: { fontSize: 26, bold: true, color: C.textStrong } },
        { text: m.label, options: { fontSize: 14, color: m.color, bold: true } },
      ], {
        x: cx, y: cy + 0.15, w: mW, h: 0.65,
        fontFace: F.sans, align: "center", valign: "middle", margin: 0,
      });
      s.addText(m.note, {
        x: cx + 0.10, y: cy + 0.85, w: mW - 0.20, h: 0.55,
        fontFace: F.sans, fontSize: 10, color: C.textMid,
        align: "center", valign: "middle", margin: 0,
      });
    }

    // Section header — methods
    s.addText("МЕТОДИ АНАЛІЗУ", {
      x: M_LEFT, y: CONTENT_TOP + 2.10, w: 8.0, h: 0.30,
      fontFace: F.sans, fontSize: 10, bold: true, charSpacing: 3, color: C.textMuted,
      align: "left", valign: "middle", margin: 0,
    });

    // 3 method cards
    const methods = [
      {
        title: "Bootstrap CI",
        desc: "95% довірчий інтервал для nDCG@10",
        sub: "n = 2 000 повторень з повторами",
        color: C.primary, icon: I.flask,
      },
      {
        title: "Парето-домінування",
        desc: "Виявлення Парето-оптимальних альтернатив",
        sub: "у багатокритеріальному просторі",
        color: C.secondary, icon: I.balance,
      },
      {
        title: "Лінійна адитивна згортка",
        desc: "Інтегральна оцінка U(aᵢ)",
        sub: "з ваговими коефіцієнтами критеріїв",
        color: C.gold, icon: I.formula,
      },
    ];
    const mY2 = CONTENT_TOP + 2.45;
    const mW2 = (W - M_LEFT - M_RIGHT - 2 * 0.20) / 3;
    for (let i = 0; i < methods.length; i++) {
      const cx = M_LEFT + i * (mW2 + 0.20);
      addCard(s, cx, mY2, mW2, 1.50, { fill: C.surfaceSoft, borderColor: C.borderSoft });
      s.addShape("ellipse", {
        x: cx + 0.20, y: mY2 + 0.20, w: 0.55, h: 0.55,
        fill: { color: C.bgWhite }, line: { color: methods[i].color, width: 1.5 },
      });
      s.addImage({ data: methods[i].icon, x: cx + 0.30, y: mY2 + 0.30, w: 0.35, h: 0.35 });
      s.addText(methods[i].title, {
        x: cx + 0.85, y: mY2 + 0.18, w: mW2 - 1.0, h: 0.30,
        fontFace: F.sans, fontSize: 13, bold: true, color: C.textStrong,
        align: "left", valign: "middle", margin: 0,
      });
      s.addText(methods[i].desc, {
        x: cx + 0.20, y: mY2 + 0.85, w: mW2 - 0.30, h: 0.30,
        fontFace: F.sans, fontSize: 11, color: C.textStrong,
        align: "left", valign: "middle", margin: 0,
      });
      s.addText(methods[i].sub, {
        x: cx + 0.20, y: mY2 + 1.13, w: mW2 - 0.30, h: 0.30,
        fontFace: F.sans, fontSize: 10, italic: true, color: C.textMuted,
        align: "left", valign: "middle", margin: 0,
      });
    }

    addKafedraLogo(s); addPageNumber(s,6, TOTAL);
  }

  // ==========================================================================
  // SLIDE 7 — Метрики (formulas)
  // ==========================================================================
  {
    const s = pres.addSlide();
    s.background = { color: C.bgWhite };
    addTitleAdv(s, "Метрики оцінювання retrieval-якості", { section: SEC.S4 });

    const formulas = [
      {
        sym: "nDCG@k",
        formula: "DCG@k  /  IDCG@k",
        desc: "Нормалізована якість ранжування",
        explain: "DCG@k = Σᵢ (2^relᵢ − 1) / log₂(i+1) ;\nIDCG@k — DCG ідеального ранжування",
        color: C.gold, range: "0 … 1, ↑",
      },
      {
        sym: "MRR",
        formula: "(1 / |Q|) · Σ (1 / rankq)",
        desc: "Mean Reciprocal Rank",
        explain: "Середнє по запитах: 1 / (позиція першого\nрелевантного документа у видачі)",
        color: C.primary, range: "0 … 1, ↑",
      },
      {
        sym: "Recall@k",
        formula: "|Rel ∩ Retrieved|  /  |Rel|",
        desc: "Повнота пошуку",
        explain: "Частка знайдених релевантних документів\nсеред усіх релевантних у колекції",
        color: C.secondary, range: "0 … 1, ↑",
      },
      {
        sym: "P@k",
        formula: "|Relk|  /  k",
        desc: "Precision@k",
        explain: "Частка релевантних документів\nсеред перших k результатів видачі",
        color: C.emerald, range: "0 … 1, ↑",
      },
    ];

    const fW = (W - M_LEFT - M_RIGHT - 0.20) / 2;
    const fH = 1.92;
    for (let i = 0; i < formulas.length; i++) {
      const col = i % 2, row = Math.floor(i / 2);
      const cx = M_LEFT + col * (fW + 0.20);
      const cy = CONTENT_TOP + row * (fH + 0.18);
      const f = formulas[i];

      addCard(s, cx, cy, fW, fH, { fill: C.bgWhite, borderColor: C.borderSoft });

      // Symbol pill (top-left)
      s.addShape("roundRect", {
        x: cx + 0.20, y: cy + 0.18, w: 1.40, h: 0.42,
        fill: { color: f.color }, line: { type: "none" }, rectRadius: 0.10,
      });
      s.addText(f.sym, {
        x: cx + 0.20, y: cy + 0.18, w: 1.40, h: 0.42,
        fontFace: F.serif, fontSize: 16, bold: true, italic: true, color: C.bgWhite,
        align: "center", valign: "middle", margin: 0,
      });

      // Range tag
      s.addText(f.range, {
        x: cx + fW - 1.10, y: cy + 0.18, w: 0.95, h: 0.42,
        fontFace: F.sans, fontSize: 9.5, color: C.textMuted,
        align: "right", valign: "middle", margin: 0,
      });

      // Description label
      s.addText(f.desc, {
        x: cx + 0.20, y: cy + 0.65, w: fW - 0.40, h: 0.30,
        fontFace: F.sans, fontSize: 12, bold: true, color: C.textStrong,
        align: "left", valign: "middle", margin: 0,
      });

      // Formula display
      s.addShape("rect", {
        x: cx + 0.20, y: cy + 1.00, w: fW - 0.40, h: 0.42,
        fill: { color: C.surfaceCool }, line: { type: "none" },
      });
      s.addText(f.formula, {
        x: cx + 0.20, y: cy + 1.00, w: fW - 0.40, h: 0.42,
        fontFace: F.serif, fontSize: 14, italic: true, bold: true, color: C.primaryDark,
        align: "center", valign: "middle", margin: 0,
      });

      // Explanation
      s.addText(f.explain, {
        x: cx + 0.20, y: cy + 1.45, w: fW - 0.40, h: 0.45,
        fontFace: F.sans, fontSize: 9.5, color: C.textMid,
        align: "left", valign: "top", margin: 0,
      });
    }

    addKafedraLogo(s); addPageNumber(s,7, TOTAL);
  }

  // ==========================================================================
  // SLIDE 8 — МКВ (методологія + ваги)
  // ==========================================================================
  {
    const s = pres.addSlide();
    s.background = { color: C.bgWhite };
    addTitleAdv(s, "Багатокритеріальний вибір моделі (МКВ)", { section: SEC.S4 });

    // 3 method cards (left ~50%)
    const methods = [
      {
        n: "1",
        title: "Парето-домінування",
        formula: "P = { aᵢ ∈ A | ∄ ak ∈ A : ak ≻ aᵢ }",
        desc: "Виявлення альтернатив, які не домінуються жодною іншою за усіма критеріями одночасно",
        color: C.primary,
      },
      {
        n: "2",
        title: "Нормалізація шкал",
        formula: "yᵢⱼ = (xᵢⱼ − min xⱼ) / (max xⱼ − min xⱼ)",
        desc: "Приведення критеріїв до єдиної шкали [0; 1] з врахуванням напряму оптимізації",
        color: C.secondary,
      },
      {
        n: "3",
        title: "Лінійна адитивна згортка",
        formula: "U(aᵢ) = Σⱼ wⱼ · yᵢⱼ ,   де   Σⱼ wⱼ = 1",
        desc: "Інтегральна оцінка корисності з ваговими коефіцієнтами критеріїв",
        color: C.gold,
      },
    ];
    // Slightly narrower method panel so the right-hand weights table can fit
    // its 6 columns without wrapping headers or values.
    const lx = M_LEFT, lw = 4.80;
    let ly = CONTENT_TOP;
    const mh = 1.20;
    const mGap = 0.12;
    for (let i = 0; i < methods.length; i++) {
      const cy = ly + i * (mh + mGap);
      addCard(s, lx, cy, lw, mh, { fill: C.bgWhite, borderColor: C.borderSoft });
      // Big number circle
      s.addShape("ellipse", {
        x: lx + 0.20, y: cy + 0.30, w: 0.60, h: 0.60,
        fill: { color: methods[i].color }, line: { type: "none" },
      });
      s.addText(methods[i].n, {
        x: lx + 0.20, y: cy + 0.30, w: 0.60, h: 0.60,
        fontFace: F.sans, fontSize: 22, bold: true, color: C.textOnDark,
        align: "center", valign: "middle", margin: 0,
      });
      // Title
      s.addText(methods[i].title, {
        x: lx + 0.95, y: cy + 0.12, w: lw - 1.10, h: 0.32,
        fontFace: F.sans, fontSize: 13, bold: true, color: C.textStrong,
        align: "left", valign: "middle", margin: 0,
      });
      // Formula on a soft band
      s.addShape("rect", {
        x: lx + 0.95, y: cy + 0.45, w: lw - 1.10, h: 0.32,
        fill: { color: C.surfaceCool }, line: { type: "none" },
      });
      s.addText(methods[i].formula, {
        x: lx + 0.95, y: cy + 0.45, w: lw - 1.10, h: 0.32,
        fontFace: F.serif, fontSize: 11, italic: true, color: C.primaryDark,
        align: "center", valign: "middle", margin: 0,
      });
      // Description
      s.addText(methods[i].desc, {
        x: lx + 0.95, y: cy + 0.80, w: lw - 1.10, h: 0.35,
        fontFace: F.sans, fontSize: 10, color: C.textMid,
        align: "left", valign: "middle", margin: 0,
      });
    }

    // Right: weight profiles table card
    const rx = lx + lw + 0.20, rw = W - rx - M_RIGHT;
    const ry = CONTENT_TOP, rh = CONTENT_H - 0.10;
    addCard(s, rx, ry, rw, rh, { fill: C.surfaceSoft, borderColor: C.borderSoft });
    s.addText("ПРОФІЛІ ВАГ КРИТЕРІЇВ ЗА ДОМЕНАМИ  (k = 10)", {
      x: rx + 0.15, y: ry + 0.10, w: rw - 0.30, h: 0.30,
      fontFace: F.sans, fontSize: 9, bold: true, charSpacing: 1.5, color: C.textMuted,
      align: "left", valign: "middle", margin: 0,
    });

    // Header @k suffix lifted into the section title so the column labels can
    // stay on one line and the table no longer overflows the parent card.
    const wRows = [
      [{ text: "Домен",   opts: { bold: true, color: C.bgWhite, fontSize: 10, align: "left",   fill: { color: C.bgDark } } },
       { text: "nDCG",    opts: { bold: true, color: C.bgWhite, fontSize: 10, align: "center", fill: { color: C.bgDark } } },
       { text: "MRR",     opts: { bold: true, color: C.bgWhite, fontSize: 10, align: "center", fill: { color: C.bgDark } } },
       { text: "Recall",  opts: { bold: true, color: C.bgWhite, fontSize: 10, align: "center", fill: { color: C.bgDark } } },
       { text: "P",       opts: { bold: true, color: C.bgWhite, fontSize: 10, align: "center", fill: { color: C.bgDark } } },
       { text: "Latency", opts: { bold: true, color: C.bgWhite, fontSize: 10, align: "center", fill: { color: C.bgDark } } }],
      ["Технічний",  "0.30", "0.20", "0.25", "0.05", "0.20"],
      ["Юридичний",  "0.30", "0.30", "0.20", "0.10", "0.10"],
      ["Медичний",   "0.25", "0.15", "0.35", "0.10", "0.15"],
    ];

    // Column widths sum to 3.85" — fits inside the now wider right card
    // (rw - 0.30 ≈ 4.20"). pptxgenjs sizes a table by sum(colW), not by `w`.
    s.addTable(wRows, {
      x: rx + 0.15, y: ry + 0.45, w: rw - 0.30,
      colW: [0.95, 0.55, 0.55, 0.65, 0.50, 0.65],
      rowH: [0.32, 0.32, 0.32, 0.32],
      fontFace: F.sans, fontSize: 10, color: C.textStrong,
      align: "center", valign: "middle",
      border: { type: "solid", pt: 0.5, color: C.borderSoft },
      fill: { color: C.bgWhite },
    });

    // Bottom: legend / chart (visual emphasis)
    const cy = ry + 2.10, cw = rw - 0.30, ch = rh - 2.20;
    s.addText("РОЗПОДІЛ ВАГ ЗА ДОМЕНАМИ", {
      x: rx + 0.15, y: cy, w: cw, h: 0.25,
      fontFace: F.sans, fontSize: 9, bold: true, charSpacing: 1.5, color: C.textMuted,
      align: "left", valign: "middle", margin: 0,
    });

    s.addChart(pres.charts.BAR, [
      { name: "nDCG",    labels: ["Технічний", "Юридичний", "Медичний"], values: [0.30, 0.30, 0.25] },
      { name: "MRR",     labels: ["Технічний", "Юридичний", "Медичний"], values: [0.20, 0.30, 0.15] },
      { name: "Recall",  labels: ["Технічний", "Юридичний", "Медичний"], values: [0.25, 0.20, 0.35] },
      { name: "P@k",     labels: ["Технічний", "Юридичний", "Медичний"], values: [0.05, 0.10, 0.10] },
      { name: "Latency", labels: ["Технічний", "Юридичний", "Медичний"], values: [0.20, 0.10, 0.15] },
    ], {
      x: rx + 0.10, y: cy + 0.30, w: cw + 0.10, h: ch - 0.20, barDir: "bar",
      barGrouping: "percentStacked",
      chartColors: [C.gold, C.primary, C.secondary, C.emerald, C.rose],
      catAxisLabelColor: C.textMid, catAxisLabelFontSize: 9, catAxisLabelFontFace: F.sans,
      valAxisLabelColor: C.textMid, valAxisLabelFontSize: 8, valAxisLabelFontFace: F.sans,
      valGridLine: { color: C.borderSoft, size: 0.5 },
      catGridLine: { style: "none" },
      showLegend: true, legendPos: "b", legendFontSize: 8, legendColor: C.textMid,
      legendFontFace: F.sans,
      chartArea: { fill: { color: C.surfaceSoft } },
      plotArea: { fill: { color: C.surfaceSoft } },
    });

    addKafedraLogo(s); addPageNumber(s,8, TOTAL);
  }

  // ==========================================================================
  // SLIDE 9 — Архітектура системи (flow)
  // ==========================================================================
  {
    const s = pres.addSlide();
    s.background = { color: C.bgWhite };
    addTitleAdv(s, "Архітектура системи семантичного пошуку", { section: SEC.S5 });

    // Flow nodes
    const nodes = [
      { title: "Документи",    sub: "PDF · DOCX\nTXT · MD",        icon: I.file,    color: C.primary },
      { title: "Чанкінг",       sub: "~600 слів\nперекриття",       icon: I.layers,  color: C.secondary },
      { title: "Ембеддінг",     sub: "BGE-M3 · E5\nnomic · Qwen3", icon: I.brain,   color: C.gold },
      { title: "FAISS-індекс",  sub: "IndexFlatIP\nL2-нормалізація",icon: I.db,      color: C.violet },
      { title: "Top-K",         sub: "Cosine\nsimilarity",          icon: I.search,  color: C.emerald },
    ];
    const flowY = CONTENT_TOP + 0.15;
    const flowH = 2.10;
    const totalW = W - M_LEFT - M_RIGHT;
    const arrowW = 0.45;
    const nodeW = (totalW - 4 * arrowW) / 5;

    for (let i = 0; i < nodes.length; i++) {
      const cx = M_LEFT + i * (nodeW + arrowW);
      const cy = flowY;
      const n = nodes[i];

      // Node card
      addCard(s, cx, cy, nodeW, flowH, { fill: C.bgWhite, borderColor: n.color, borderWidth: 1.5 });
      // Top color stripe
      s.addShape("rect", {
        x: cx, y: cy, w: nodeW, h: 0.10,
        fill: { color: n.color }, line: { type: "none" },
      });
      // Icon circle
      s.addShape("ellipse", {
        x: cx + nodeW / 2 - 0.30, y: cy + 0.30, w: 0.60, h: 0.60,
        fill: { color: C.surfaceSoft }, line: { color: n.color, width: 1.5 },
      });
      s.addImage({ data: n.icon, x: cx + nodeW / 2 - 0.18, y: cy + 0.42, w: 0.36, h: 0.36 });
      // Title
      s.addText(n.title, {
        x: cx, y: cy + 0.95, w: nodeW, h: 0.32,
        fontFace: F.sans, fontSize: 13, bold: true, color: C.textStrong,
        align: "center", valign: "middle", margin: 0,
      });
      // Sub
      s.addText(n.sub, {
        x: cx + 0.10, y: cy + 1.30, w: nodeW - 0.20, h: 0.50,
        fontFace: F.sans, fontSize: 9.5, color: C.textMid,
        align: "center", valign: "top", margin: 0,
      });

      // Arrow to next
      if (i < nodes.length - 1) {
        const ax = cx + nodeW + 0.05;
        s.addImage({ data: I.arrow, x: ax, y: cy + flowH / 2 - 0.12, w: 0.36, h: 0.24 });
      }
    }

    // User input + Flask UI overlay
    const uy = flowY + flowH + 0.30;
    const uw = 2.20;
    addCard(s, M_LEFT, uy, uw, 0.55, { fill: C.surfaceCool, borderColor: C.primary });
    s.addImage({ data: I.user, x: M_LEFT + 0.12, y: uy + 0.12, w: 0.30, h: 0.30 });
    s.addText("Запит користувача", {
      x: M_LEFT + 0.50, y: uy, w: uw - 0.60, h: 0.55,
      fontFace: F.sans, fontSize: 11, bold: true, color: C.primaryDark,
      align: "left", valign: "middle", margin: 0,
    });

    addCard(s, W - M_RIGHT - uw, uy, uw, 0.55, { fill: C.surfaceCool, borderColor: C.primary });
    s.addImage({ data: I.laptop, x: W - M_RIGHT - uw + 0.12, y: uy + 0.12, w: 0.30, h: 0.30 });
    s.addText("Flask Web UI", {
      x: W - M_RIGHT - uw + 0.50, y: uy, w: uw - 0.60, h: 0.55,
      fontFace: F.sans, fontSize: 11, bold: true, color: C.primaryDark,
      align: "left", valign: "middle", margin: 0,
    });

    // Components strip — sits just above the footer logo so the slide reads
    // bottom-anchored instead of leaving a 0.4" white stripe.
    const cy = uy + 0.75;
    addCard(s, M_LEFT, cy, W - M_LEFT - M_RIGHT, 0.50, { fill: C.surfaceSoft, borderColor: C.borderSoft, shadow: false });
    s.addText([
      { text: "Основні компоненти:  ", options: { color: C.textMuted, fontSize: 10 } },
      { text: "build_index.py", options: { color: C.primary, fontSize: 10, bold: true, fontFace: F.mono } },
      { text: "  ·  ", options: { color: C.textMuted, fontSize: 10 } },
      { text: "evaluate_benchmark.py", options: { color: C.primary, fontSize: 10, bold: true, fontFace: F.mono } },
      { text: "  ·  ", options: { color: C.textMuted, fontSize: 10 } },
      { text: "embedding_models.py", options: { color: C.primary, fontSize: 10, bold: true, fontFace: F.mono } },
      { text: "  ·  ", options: { color: C.textMuted, fontSize: 10 } },
      { text: "run_all_benchmarks.py", options: { color: C.primary, fontSize: 10, bold: true, fontFace: F.mono } },
      { text: "  ·  ", options: { color: C.textMuted, fontSize: 10 } },
      { text: "app.py", options: { color: C.primary, fontSize: 10, bold: true, fontFace: F.mono } },
    ], {
      x: M_LEFT, y: cy, w: W - M_LEFT - M_RIGHT, h: 0.50,
      fontFace: F.sans, valign: "middle", align: "center", margin: 0,
    });

    addKafedraLogo(s); addPageNumber(s,9, TOTAL);
  }

  // ==========================================================================
  // SLIDE 10 — Технології
  // ==========================================================================
  {
    const s = pres.addSlide();
    s.background = { color: C.bgWhite };
    addTitleAdv(s, "Програмне забезпечення та технології", { section: SEC.S6 });

    const stacks = [
      {
        title: "ML / EMBEDDING",
        icon: I.brain, color: C.primary,
        items: [
          "Python 3.11",
          "sentence-transformers",
          "transformers (HF)",
          "PyTorch (CPU)",
          "trust_remote_code",
          "Cambria Math",
        ],
      },
      {
        title: "RETRIEVAL / DATA",
        icon: I.db, color: C.secondary,
        items: [
          "FAISS (IndexFlatIP)",
          "rank_bm25 (Okapi)",
          "L2-нормалізація",
          "Чанкінг (~600 слів)",
          "PDF / DOCX / TXT / MD",
          "JSONL (chunks, qrels)",
        ],
      },
      {
        title: "WEB / EVAL / ANALYSIS",
        icon: I.chart, color: C.gold,
        items: [
          "Flask (Python)",
          "NumPy, scikit-learn",
          "Bootstrap CI (n=2000)",
          "Pareto + лінійна згортка",
          "Власний benchmark harness",
          "Git + GitHub",
        ],
      },
    ];

    const colW = (W - M_LEFT - M_RIGHT - 2 * 0.20) / 3;
    const colH = CONTENT_H - 0.10;
    for (let i = 0; i < 3; i++) {
      const cx = M_LEFT + i * (colW + 0.20);
      const cy = CONTENT_TOP;
      const st = stacks[i];

      addCard(s, cx, cy, colW, colH, { fill: C.bgWhite, borderColor: C.borderSoft });

      // Top header band
      s.addShape("rect", {
        x: cx, y: cy, w: colW, h: 0.55,
        fill: { color: st.color }, line: { type: "none" },
      });
      s.addImage({ data: await iconPng(
        st.title.includes("ML") ? fa.FaBrain : st.title.includes("RETRIEVAL") ? fa.FaDatabase : fa.FaChartBar,
        hash("FFFFFF"),
      ), x: cx + 0.18, y: cy + 0.13, w: 0.30, h: 0.30 });
      s.addText(st.title, {
        x: cx + 0.55, y: cy, w: colW - 0.60, h: 0.55,
        fontFace: F.sans, fontSize: 11, bold: true, color: C.textOnDark, charSpacing: 2,
        align: "left", valign: "middle", margin: 0,
      });

      // Items
      const itemY0 = cy + 0.75;
      const itemH = (colH - 0.85) / st.items.length;
      for (let j = 0; j < st.items.length; j++) {
        const iy = itemY0 + j * itemH;
        // Tiny icon (square dot)
        s.addShape("rect", {
          x: cx + 0.20, y: iy + itemH / 2 - 0.04, w: 0.08, h: 0.08,
          fill: { color: st.color }, line: { type: "none" },
        });
        s.addText(st.items[j], {
          x: cx + 0.40, y: iy, w: colW - 0.50, h: itemH,
          fontFace: F.sans, fontSize: 11.5, color: C.textStrong,
          align: "left", valign: "middle", margin: 0,
        });
      }
    }

    addKafedraLogo(s); addPageNumber(s,10, TOTAL);
  }

  // ==========================================================================
  // SLIDE 11 — Benchmark dataset
  // ==========================================================================
  {
    const s = pres.addSlide();
    s.background = { color: C.bgWhite };
    addTitleAdv(s, "Benchmark dataset: україномовні доменні колекції", { section: SEC.S7 });

    // Subtitle band
    s.addText([
      { text: "Власноруч сформований ", options: { color: C.textMid, fontSize: 11 } },
      { text: "domain-specific україномовний benchmark", options: { color: C.primaryDark, fontSize: 11, bold: true } },
      { text: ":  3 домени  ·  32 документи  ·  300 запитів  ·  qrels сформовано вручну",
        options: { color: C.textMid, fontSize: 11 } },
    ], {
      x: M_LEFT, y: CONTENT_TOP, w: W - M_LEFT - M_RIGHT, h: 0.30,
      fontFace: F.sans, align: "left", valign: "middle", margin: 0,
    });

    const domains = [
      { name: "Технічний",   docs: 10, queries: 100, color: C.primary,   icon: I.tech,    desc: "програмна інженерія, ML, RAG, аномалії, 3D-друк, лідарні системи" },
      { name: "Юридичний",   docs: 13, queries: 100, color: C.violet,    icon: I.legal,   desc: "кодекси, нормативно-правові акти, законодавство" },
      { name: "Медичний",    docs: 9,  queries: 100, color: C.rose,      icon: I.medical, desc: "клінічні протоколи, медичні рекомендації" },
    ];

    // 3 full-width domain cards. The right-hand "100/100/100" callout was
    // dropped — it duplicated the per-card "100 запитів" and visually drew
    // attention to the symmetry the slide is now communicating in one place.
    const cardX = M_LEFT, cardY = CONTENT_TOP + 0.45;
    const cardW = W - M_LEFT - M_RIGHT;
    const cardH = (CONTENT_H - 0.55 - 2 * 0.12) / 3;
    for (let i = 0; i < domains.length; i++) {
      const cy = cardY + i * (cardH + 0.12);
      const d = domains[i];
      addCard(s, cardX, cy, cardW, cardH, { fill: C.bgWhite, borderColor: C.borderSoft });

      // Side stripe
      s.addShape("rect", {
        x: cardX, y: cy, w: 0.10, h: cardH,
        fill: { color: d.color }, line: { type: "none" },
      });
      // Icon circle (vertically centred)
      const iconCY = cy + cardH / 2;
      s.addShape("ellipse", {
        x: cardX + 0.30, y: iconCY - 0.30, w: 0.60, h: 0.60,
        fill: { color: C.surfaceSoft }, line: { color: d.color, width: 1.5 },
      });
      s.addImage({ data: d.icon, x: cardX + 0.42, y: iconCY - 0.18, w: 0.36, h: 0.36 });

      // Domain name + description block (centre column)
      s.addText(d.name, {
        x: cardX + 1.10, y: cy + 0.20, w: 5.30, h: 0.36,
        fontFace: F.sans, fontSize: 16, bold: true, color: C.textStrong,
        align: "left", valign: "middle", margin: 0,
      });
      s.addText(d.desc, {
        x: cardX + 1.10, y: cy + 0.60, w: 5.30, h: 0.40,
        fontFace: F.sans, fontSize: 11, italic: true, color: C.textMid,
        align: "left", valign: "top", margin: 0,
      });

      // Stats column on the right — documents · queries
      const statsY = cy + 0.20;
      const statsH = cardH - 0.40;
      const statsCols = [
        { v: d.docs,    l: "документи", x: cardX + cardW - 2.85 },
        { v: d.queries, l: "запити",    x: cardX + cardW - 1.40 },
      ];
      // Subtle vertical separator before the stats column
      s.addShape("rect", {
        x: cardX + cardW - 3.00, y: cy + 0.25, w: 0.015, h: cardH - 0.50,
        fill: { color: C.borderSoft }, line: { type: "none" },
      });
      for (const st of statsCols) {
        s.addText(String(st.v), {
          x: st.x, y: statsY, w: 1.30, h: statsH * 0.70,
          fontFace: F.sans, fontSize: 30, bold: true, color: d.color,
          align: "center", valign: "bottom", margin: 0,
        });
        s.addText(st.l, {
          x: st.x, y: statsY + statsH * 0.70 + 0.02, w: 1.30, h: statsH * 0.30,
          fontFace: F.sans, fontSize: 10, color: C.textMuted,
          align: "center", valign: "top", margin: 0,
        });
      }
    }

    addKafedraLogo(s); addPageNumber(s,11, TOTAL);
  }

  // ==========================================================================
  // SLIDE 12 — Приклади benchmark-запитів
  // ==========================================================================
  {
    const s = pres.addSlide();
    s.background = { color: C.bgWhite };
    addTitleAdv(s, "Приклади benchmark-запитів", { section: SEC.S7 });

    const cols = [
      {
        name: "Технічний", color: C.primary, icon: I.tech,
        items: [
          { id: "q1", q: "виявлення аномалій у фінансових транзакціях" },
          { id: "q3", q: "комбінація Autoencoder та Isolation Forest для fraud detection" },
          { id: "q6", q: "прогноз відтоку клієнтів (churn prediction)" },
          { id: "q7", q: "когортний аналіз для сегментації клієнтів" },
          { id: "q8", q: "етичні питання використання персональних даних" },
        ],
      },
      {
        name: "Юридичний", color: C.violet, icon: I.legal,
        items: [
          { id: "L001", q: "що таке цивільна дієздатність фізичної особи" },
          { id: "L002", q: "з якого віку настає повна цивільна дієздатність" },
          { id: "L003", q: "підстави обмеження цивільної дієздатності судом" },
          { id: "L004", q: "поняття юридичної особи та порядок її реєстрації" },
          { id: "L005", q: "що входить до складу спадщини" },
        ],
      },
      {
        name: "Медичний", color: C.rose, icon: I.medical,
        items: [
          { id: "M001", q: "що таке серцево-судинна система" },
          { id: "M002", q: "будова та функції серця людини" },
          { id: "M003", q: "що таке артеріальний тиск і як він регулюється" },
          { id: "M004", q: "велике і мале коло кровообігу — відмінності" },
          { id: "M005", q: "будова дихальної системи людини" },
        ],
      },
    ];

    const colW = (W - M_LEFT - M_RIGHT - 2 * 0.20) / 3;
    const colY = CONTENT_TOP, colH = CONTENT_H - 0.50;

    for (let i = 0; i < cols.length; i++) {
      const cx = M_LEFT + i * (colW + 0.20);
      const c = cols[i];
      addCard(s, cx, colY, colW, colH, { fill: C.bgWhite, borderColor: C.borderSoft });
      // Header band
      s.addShape("rect", {
        x: cx, y: colY, w: colW, h: 0.55,
        fill: { color: c.color }, line: { type: "none" },
      });
      s.addImage({ data: await iconPng(
        c.name === "Технічний" ? fa.FaMicrochip : c.name === "Юридичний" ? fa.FaGavel : fa.FaHeartbeat,
        hash("FFFFFF"),
      ), x: cx + 0.18, y: colY + 0.13, w: 0.30, h: 0.30 });
      s.addText(c.name, {
        x: cx + 0.58, y: colY, w: colW - 0.60, h: 0.55,
        fontFace: F.sans, fontSize: 13, bold: true, color: C.textOnDark,
        align: "left", valign: "middle", margin: 0,
      });

      // Items as quote-style rows
      const itemY0 = colY + 0.70;
      const itemH = (colH - 0.80) / 5;
      for (let j = 0; j < c.items.length; j++) {
        const iy = itemY0 + j * itemH;
        s.addText([
          { text: c.items[j].id + " ", options: { fontSize: 9.5, bold: true, color: c.color, fontFace: F.mono } },
          { text: "« " + c.items[j].q + " »", options: { fontSize: 10, color: C.textStrong, italic: true } },
        ], {
          x: cx + 0.18, y: iy + 0.05, w: colW - 0.30, h: itemH - 0.10,
          fontFace: F.sans, valign: "middle", margin: 0,
        });
        // Separator
        if (j < c.items.length - 1) {
          s.addShape("rect", {
            x: cx + 0.18, y: iy + itemH - 0.02, w: colW - 0.36, h: 0.01,
            fill: { color: C.borderSoft }, line: { type: "none" },
          });
        }
      }
    }

    // Bottom info band
    const ftY = colY + colH + 0.10;
    addCard(s, M_LEFT, ftY, W - M_LEFT - M_RIGHT, 0.40, {
      fill: C.surfaceSoft, borderColor: C.borderSoft, shadow: false,
    });
    s.addText([
      { text: "Типи запитів:  ", options: { color: C.textMuted, fontSize: 10 } },
      { text: "definition · factual · procedural · technical · topic · method · policy · comparison",
        options: { color: C.textStrong, fontSize: 10, bold: true } },
      { text: "    ·    qrels — релевантні чанки сформовано вручну для кожного запиту",
        options: { color: C.textMuted, fontSize: 10 } },
    ], {
      x: M_LEFT, y: ftY, w: W - M_LEFT - M_RIGHT, h: 0.40,
      fontFace: F.sans, valign: "middle", align: "center", margin: 0,
    });

    addKafedraLogo(s); addPageNumber(s,12, TOTAL);
  }

  // ==========================================================================
  // SLIDE 13 — Результати: nDCG@10
  // ==========================================================================
  {
    const s = pres.addSlide();
    s.background = { color: C.bgWhite };
    addTitleAdv(s, "Результати експерименту: nDCG@10", { section: SEC.S8 });

    // Data (verified from results/benchmark_*.json)
    const data = {
      labels: ["Технічний", "Юридичний", "Медичний"],
      models: [
        { name: "BGE-M3",  vals: [0.6722, 0.3065, 0.4339], avg: 0.4708, color: C.gold },
        { name: "Qwen3",   vals: [0.6325, 0.3199, 0.3629], avg: 0.4385, color: C.primary },
        { name: "E5-base", vals: [0.6121, 0.2567, 0.3909], avg: 0.4199, color: C.secondary },
        { name: "BM25",    vals: [0.4861, 0.1875, 0.3222], avg: 0.3319, color: C.textMuted },
        { name: "nomic",   vals: [0.3765, 0.0951, 0.1668], avg: 0.2128, color: C.violet },
      ],
    };

    // Big chart on left (~6.5"), summary cards on right
    const chX = M_LEFT, chY = CONTENT_TOP + 0.05, chW = 6.30, chH = CONTENT_H - 0.20;
    addCard(s, chX, chY, chW, chH, { fill: C.surfaceSoft, borderColor: C.borderSoft });
    s.addText("nDCG@10  ПО ДОМЕНАХ (вище — краще)", {
      x: chX + 0.20, y: chY + 0.08, w: chW - 0.40, h: 0.25,
      fontFace: F.sans, fontSize: 9, bold: true, charSpacing: 2, color: C.textMuted,
      align: "left", valign: "middle", margin: 0,
    });

    s.addChart(pres.charts.BAR, data.models.map(m => ({
      name: m.name, labels: data.labels, values: m.vals,
    })), {
      x: chX + 0.10, y: chY + 0.40, w: chW - 0.20, h: chH - 0.55, barDir: "col",
      chartColors: data.models.map(m => m.color),
      catAxisLabelColor: C.textStrong, catAxisLabelFontSize: 10, catAxisLabelFontFace: F.sans,
      valAxisLabelColor: C.textMid, valAxisLabelFontSize: 9, valAxisLabelFontFace: F.sans,
      valGridLine: { color: C.borderSoft, size: 0.5 },
      catGridLine: { style: "none" },
      showValue: true, dataLabelPosition: "outEnd",
      dataLabelColor: C.textStrong, dataLabelFontSize: 8, dataLabelFontFace: F.sans,
      dataLabelFormatCode: "0.000",
      showLegend: true, legendPos: "b", legendFontSize: 10, legendFontFace: F.sans,
      legendColor: C.textStrong,
      valAxisMinVal: 0, valAxisMaxVal: 0.75,
      chartArea: { fill: { color: C.surfaceSoft } },
      plotArea: { fill: { color: C.surfaceSoft } },
    });

    // Right column: ranking + bootstrap CI callout
    const rx = chX + chW + 0.20, rw = W - rx - M_RIGHT, ry = CONTENT_TOP + 0.05;

    // Ranking card (top — slightly shorter to free space for the CI card below)
    const rankCardH = 2.10;
    addCard(s, rx, ry, rw, rankCardH, { fill: C.bgWhite, borderColor: C.borderSoft });
    s.addText("СЕРЕДНІЙ nDCG@10", {
      x: rx + 0.15, y: ry + 0.10, w: rw - 0.30, h: 0.25,
      fontFace: F.sans, fontSize: 8.5, bold: true, charSpacing: 2, color: C.textMuted,
      align: "left", valign: "middle", margin: 0,
    });

    // Sorted by avg desc
    const sortedAvg = [...data.models].sort((a, b) => b.avg - a.avg);
    const rankH = (rankCardH - 0.40) / 5;
    for (let i = 0; i < sortedAvg.length; i++) {
      const m = sortedAvg[i];
      const ry2 = ry + 0.40 + i * rankH;
      s.addShape("rect", {
        x: rx + 0.15, y: ry2 + 0.04, w: rw - 0.30, h: rankH - 0.06,
        fill: { color: i === 0 ? C.surfaceGold : C.bgWhite },
        line: { type: "none" },
      });
      // Position
      s.addText(`#${i + 1}`, {
        x: rx + 0.15, y: ry2, w: 0.45, h: rankH,
        fontFace: F.sans, fontSize: 11, bold: true, color: i === 0 ? C.gold : C.textMuted,
        align: "center", valign: "middle", margin: 0,
      });
      // Color dot
      s.addShape("ellipse", {
        x: rx + 0.60, y: ry2 + rankH / 2 - 0.07, w: 0.14, h: 0.14,
        fill: { color: m.color }, line: { type: "none" },
      });
      // Name
      s.addText(m.name, {
        x: rx + 0.78, y: ry2, w: rw - 1.45, h: rankH,
        fontFace: F.sans, fontSize: 10.5, bold: i === 0, color: C.textStrong,
        align: "left", valign: "middle", margin: 0,
      });
      // Value
      s.addText(m.avg.toFixed(4), {
        x: rx + rw - 0.65, y: ry2, w: 0.50, h: rankH,
        fontFace: F.mono, fontSize: 10.5, bold: true, color: i === 0 ? C.gold : C.textStrong,
        align: "right", valign: "middle", margin: 0,
      });
    }

    // Bootstrap CI callout (extends to the bottom of the content area)
    const bcY = ry + rankCardH + 0.10;
    const bcH = CONTENT_BOTTOM - bcY - 0.05;
    addCard(s, rx, bcY, rw, bcH, { fill: C.bgDark, borderColor: C.bgDark });
    s.addText("BOOTSTRAP 95% CI", {
      x: rx + 0.15, y: bcY + 0.10, w: rw - 0.30, h: 0.22,
      fontFace: F.sans, fontSize: 8.5, bold: true, charSpacing: 2, color: C.gold,
      align: "left", valign: "middle", margin: 0,
    });
    s.addText("n = 2 000  ·  Технічний домен:", {
      x: rx + 0.15, y: bcY + 0.32, w: rw - 0.30, h: 0.22,
      fontFace: F.sans, fontSize: 9, color: C.textOnDarkMuted,
      align: "left", valign: "middle", margin: 0,
    });
    s.addText([
      { text: "BGE-M3:  ", options: { color: C.gold, fontSize: 10, bold: true } },
      { text: "[0.606 ; 0.737]", options: { color: C.textOnDark, fontSize: 11, bold: true, fontFace: F.mono, breakLine: true } },
      { text: "BM25:    ", options: { color: C.textOnDarkMuted, fontSize: 10, bold: true } },
      { text: "[0.431 ; 0.541]", options: { color: C.textOnDark, fontSize: 11, bold: true, fontFace: F.mono } },
    ], {
      x: rx + 0.15, y: bcY + 0.55, w: rw - 0.30, h: 0.65,
      fontFace: F.sans, valign: "top", margin: 0, paraSpaceAfter: 2,
    });
    s.addText("→ інтервали не перетинаються — статистично значуща перевага семантичного пошуку над лексичним",
    {
      x: rx + 0.15, y: bcY + 1.20, w: rw - 0.30, h: bcH - 1.25,
      fontFace: F.sans, fontSize: 9, color: C.gold, italic: true,
      align: "left", valign: "top", margin: 0,
    });

    addKafedraLogo(s); addPageNumber(s,13, TOTAL);
  }

  // ==========================================================================
  // SLIDE 14 — Деталізовані метрики (4-grid)
  // ==========================================================================
  {
    const s = pres.addSlide();
    s.background = { color: C.bgWhite };
    addTitleAdv(s, "Деталізовані метрики за доменами", { section: SEC.S8 });

    // 4-mini-chart grid: nDCG, MRR, Recall, P@10
    const labels = ["Технічний", "Юридичний", "Медичний"];
    // verified from results/benchmark_*.json
    const stats = {
      ndcg:   [{ n: "BGE-M3", c: C.gold,      v: [0.6722, 0.3065, 0.4339] },
               { n: "Qwen3",  c: C.primary,   v: [0.6325, 0.3199, 0.3629] },
               { n: "E5-base",c: C.secondary, v: [0.6121, 0.2567, 0.3909] },
               { n: "BM25",   c: C.textMuted, v: [0.4861, 0.1875, 0.3222] },
               { n: "nomic",  c: C.violet,    v: [0.3765, 0.0951, 0.1668] }],
      mrr:    [{ n: "BGE-M3", c: C.gold,      v: [0.6993, 0.3831, 0.5054] },
               { n: "Qwen3",  c: C.primary,   v: [0.6660, 0.4022, 0.4256] },
               { n: "E5-base",c: C.secondary, v: [0.6450, 0.3318, 0.4647] },
               { n: "BM25",   c: C.textMuted, v: [0.4808, 0.2533, 0.4003] },
               { n: "nomic",  c: C.violet,    v: [0.4250, 0.1432, 0.2167] }],
      recall: [{ n: "BGE-M3", c: C.gold,      v: [0.8150, 0.3750, 0.5450] },
               { n: "Qwen3",  c: C.primary,   v: [0.7600, 0.3850, 0.4717] },
               { n: "E5-base",c: C.secondary, v: [0.7458, 0.3300, 0.5050] },
               { n: "BM25",   c: C.textMuted, v: [0.7542, 0.2333, 0.3983] },
               { n: "nomic",  c: C.violet,    v: [0.4783, 0.1250, 0.2333] }],
      p10:    [{ n: "BGE-M3", c: C.gold,      v: [0.1430, 0.1060, 0.1400] },
               { n: "Qwen3",  c: C.primary,   v: [0.1380, 0.1100, 0.1290] },
               { n: "E5-base",c: C.secondary, v: [0.1390, 0.0890, 0.1340] },
               { n: "BM25",   c: C.textMuted, v: [0.1340, 0.0640, 0.1000] },
               { n: "nomic",  c: C.violet,    v: [0.0900, 0.0270, 0.0540] }],
    };

    const titles = [
      { key: "ndcg",   title: "nDCG@10",    max: 0.75 },
      { key: "mrr",    title: "MRR@10",     max: 0.80 },
      { key: "recall", title: "Recall@10",  max: 0.90 },
      { key: "p10",    title: "P@10",       max: 0.16 },
    ];

    const gridW = (W - M_LEFT - M_RIGHT - 0.18) / 2;
    const gridH = (CONTENT_H - 0.18) / 2;
    for (let i = 0; i < 4; i++) {
      const col = i % 2, row = Math.floor(i / 2);
      const cx = M_LEFT + col * (gridW + 0.18);
      const cy = CONTENT_TOP + row * (gridH + 0.18);
      addCard(s, cx, cy, gridW, gridH, { fill: C.surfaceSoft, borderColor: C.borderSoft });
      s.addText(titles[i].title, {
        x: cx + 0.20, y: cy + 0.08, w: gridW - 0.40, h: 0.25,
        fontFace: F.sans, fontSize: 11, bold: true, charSpacing: 1.5, color: C.textStrong,
        align: "left", valign: "middle", margin: 0,
      });

      const series = stats[titles[i].key];
      s.addChart(pres.charts.BAR, series.map(m => ({
        name: m.n, labels, values: m.v,
      })), {
        x: cx + 0.10, y: cy + 0.34, w: gridW - 0.20, h: gridH - 0.45, barDir: "col",
        chartColors: series.map(m => m.c),
        catAxisLabelColor: C.textMid, catAxisLabelFontSize: 8, catAxisLabelFontFace: F.sans,
        valAxisLabelColor: C.textMid, valAxisLabelFontSize: 7, valAxisLabelFontFace: F.sans,
        valGridLine: { color: C.borderSoft, size: 0.5 },
        catGridLine: { style: "none" },
        valAxisMinVal: 0, valAxisMaxVal: titles[i].max,
        showLegend: i === 0, legendPos: "b", legendFontSize: 7.5, legendFontFace: F.sans,
        legendColor: C.textMid,
        showValue: false,
        chartArea: { fill: { color: C.surfaceSoft } },
        plotArea: { fill: { color: C.surfaceSoft } },
      });
    }

    addKafedraLogo(s); addPageNumber(s,14, TOTAL);
  }

  // ==========================================================================
  // SLIDE 15 — МКВ результати
  // ==========================================================================
  {
    const s = pres.addSlide();
    s.background = { color: C.bgWhite };
    addTitleAdv(s, "Багатокритеріальний вибір моделі: результати", { section: SEC.S9 });

    // Left: Pareto-optimal callout + insights
    const lx = M_LEFT, lw = 4.30, ly = CONTENT_TOP, lh = CONTENT_H - 0.10;

    // Pareto card
    addCard(s, lx, ly, lw, 1.05, { fill: C.bgDark, borderColor: C.bgDark });
    s.addText("ПАРЕТО-ОПТИМАЛЬНА МНОЖИНА", {
      x: lx + 0.20, y: ly + 0.08, w: lw - 0.40, h: 0.22,
      fontFace: F.sans, fontSize: 9, bold: true, charSpacing: 2, color: C.gold,
      align: "left", valign: "middle", margin: 0,
    });
    s.addText("P = { BGE-M3,  E5-base,  BM25 }", {
      x: lx + 0.20, y: ly + 0.30, w: lw - 0.40, h: 0.36,
      fontFace: F.serif, fontSize: 18, italic: true, bold: true, color: C.textOnDark,
      align: "left", valign: "middle", margin: 0,
    });
    s.addText("3 з 5 альтернатив не домінуються жодною іншою за усіма критеріями одночасно", {
      x: lx + 0.20, y: ly + 0.66, w: lw - 0.40, h: 0.34,
      fontFace: F.sans, fontSize: 9.5, italic: true, color: C.textOnDarkMuted,
      align: "left", valign: "top", margin: 0,
    });

    // 4 insight rows
    const insights = [
      { c: C.gold,      t: "BGE-M3", d: "лідер у 3 доменах за nDCG, MRR, Recall" },
      { c: C.secondary, t: "E5-base", d: "швидка альтернатива, ~70 мс vs ~228 мс у BGE-M3" },
      { c: C.primary,   t: "Qwen3", d: "лідер на юридичному (nDCG=0.320), ~2× повільніший" },
      { c: C.violet,    t: "nomic", d: "найслабші результати; в окремих доменах поступається BM25" },
    ];
    const iY0 = ly + 1.20;
    const iH = (lh - 1.20) / 4;
    for (let i = 0; i < insights.length; i++) {
      const cy = iY0 + i * iH;
      addCard(s, lx, cy + 0.05, lw, iH - 0.10, { fill: C.bgWhite, borderColor: C.borderSoft });
      // Side stripe
      s.addShape("rect", {
        x: lx, y: cy + 0.05, w: 0.10, h: iH - 0.10,
        fill: { color: insights[i].c }, line: { type: "none" },
      });
      s.addText(insights[i].t, {
        x: lx + 0.25, y: cy + 0.10, w: lw - 0.40, h: 0.24,
        fontFace: F.sans, fontSize: 13, bold: true, color: C.textStrong,
        align: "left", valign: "top", margin: 0,
      });
      s.addText(insights[i].d, {
        x: lx + 0.25, y: cy + 0.34, w: lw - 0.40, h: iH - 0.42,
        fontFace: F.sans, fontSize: 10, color: C.textMid,
        align: "left", valign: "top", margin: 0,
      });
    }

    // Right: U-scores chart + table
    const rx = lx + lw + 0.20, rw = W - rx - M_RIGHT, ry = CONTENT_TOP, rh = lh;
    addCard(s, rx, ry, rw, rh, { fill: C.surfaceSoft, borderColor: C.borderSoft });
    s.addText("ІНТЕГРАЛЬНА ОЦІНКА U(aᵢ) ПО ДОМЕНАХ", {
      x: rx + 0.15, y: ry + 0.10, w: rw - 0.30, h: 0.25,
      fontFace: F.sans, fontSize: 9, bold: true, charSpacing: 2, color: C.textMuted,
      align: "left", valign: "middle", margin: 0,
    });

    // U-values (with BM25 in MCDA — matches thesis state).
    // Values mirror those in chapter 5 (after my chapter5 v2 fix):
    //   Tech 0.904, Legal 0.914, Medical 0.912 — averaged across BGE-M3/E5/Qwen3/BM25/nomic.
    // For chart we show top per domain:
    const uVals = [
      { n: "BGE-M3",  c: C.gold,      v: [0.904, 0.872, 0.912] },
      { n: "Qwen3",   c: C.primary,   v: [0.825, 0.914, 0.802] },
      { n: "E5-base", c: C.secondary, v: [0.875, 0.844, 0.901] },
      { n: "BM25",    c: C.textMuted, v: [0.795, 0.741, 0.817] },
      { n: "nomic",   c: C.violet,    v: [0.611, 0.503, 0.568] },
    ];
    s.addChart(pres.charts.BAR, uVals.map(m => ({
      name: m.n, labels: ["Технічний", "Юридичний", "Медичний"], values: m.v,
    })), {
      x: rx + 0.10, y: ry + 0.40, w: rw - 0.20, h: rh - 0.55, barDir: "col",
      chartColors: uVals.map(m => m.c),
      catAxisLabelColor: C.textStrong, catAxisLabelFontSize: 10, catAxisLabelFontFace: F.sans,
      valAxisLabelColor: C.textMid, valAxisLabelFontSize: 9, valAxisLabelFontFace: F.sans,
      valGridLine: { color: C.borderSoft, size: 0.5 },
      catGridLine: { style: "none" },
      showValue: true, dataLabelPosition: "outEnd",
      dataLabelColor: C.textStrong, dataLabelFontSize: 7.5, dataLabelFontFace: F.sans,
      dataLabelFormatCode: "0.00",
      showLegend: true, legendPos: "b", legendFontSize: 9, legendFontFace: F.sans,
      legendColor: C.textStrong,
      valAxisMinVal: 0, valAxisMaxVal: 1.0,
      chartArea: { fill: { color: C.surfaceSoft } },
      plotArea: { fill: { color: C.surfaceSoft } },
    });

    addKafedraLogo(s); addPageNumber(s,15, TOTAL);
  }

  // ==========================================================================
  // SLIDE 16 — Якість vs Швидкодія (scatter)
  // ==========================================================================
  {
    const s = pres.addSlide();
    s.background = { color: C.bgWhite };
    addTitleAdv(s, "Аналіз: якість пошуку vs швидкодія", { section: SEC.S9 });

    // Left: scatter chart
    const chX = M_LEFT, chY = CONTENT_TOP, chW = 5.80, chH = CONTENT_H - 0.10;
    addCard(s, chX, chY, chW, chH, { fill: C.surfaceSoft, borderColor: C.borderSoft });
    s.addText("ТЕХНІЧНИЙ ДОМЕН  ·  nDCG@10  vs  ms / запит", {
      x: chX + 0.20, y: chY + 0.10, w: chW - 0.40, h: 0.25,
      fontFace: F.sans, fontSize: 9, bold: true, charSpacing: 2, color: C.textMuted,
      align: "left", valign: "middle", margin: 0,
    });

    // Scatter: pptxgenjs SCATTER expects [{ name: "X", values: [...] }, ...series]
    // Latency on tech domain (verified): BM25=0.4, E5=70.9, nomic=124.8, BGE-M3=222.5, Qwen3=463.4
    // nDCG on tech: BM25=0.4861, E5=0.6121, nomic=0.3765, BGE-M3=0.6722, Qwen3=0.6325
    const lat = [0.4, 70.9, 124.8, 222.5, 463.4];
    const ndcg = [0.4861, 0.6121, 0.3765, 0.6722, 0.6325];
    s.addChart(pres.charts.SCATTER, [
      { name: "ms",     values: lat },
      { name: "nDCG@10", values: ndcg },
    ], {
      x: chX + 0.10, y: chY + 0.40, w: chW - 0.20, h: chH - 0.55,
      lineSize: 0,
      lineDataSymbol: "circle", lineDataSymbolSize: 14,
      chartColors: [C.primary],
      catAxisTitle: "ms / запит (нижче — краще)",
      catAxisTitleFontSize: 9, catAxisTitleColor: C.textMid, catAxisTitleFontFace: F.sans,
      showCatAxisTitle: true,
      valAxisTitle: "nDCG@10 (вище — краще)",
      valAxisTitleFontSize: 9, valAxisTitleColor: C.textMid, valAxisTitleFontFace: F.sans,
      showValAxisTitle: true,
      catAxisLabelColor: C.textMid, catAxisLabelFontSize: 8, catAxisLabelFontFace: F.sans,
      valAxisLabelColor: C.textMid, valAxisLabelFontSize: 8, valAxisLabelFontFace: F.sans,
      valGridLine: { color: C.borderSoft, size: 0.5 },
      catGridLine: { color: C.borderSoft, size: 0.5 },
      showLegend: false,
      valAxisMinVal: 0, valAxisMaxVal: 0.80,
      catAxisMinVal: 0, catAxisMaxVal: 500,
      chartArea: { fill: { color: C.surfaceSoft } },
      plotArea: { fill: { color: C.surfaceSoft } },
    });

    // Manual point labels (overlay). pptxgenjs scatter axis padding makes
    // exact coordinates unreliable, so labels are tuned by visual inspection
    // rather than a generic linear projection.
    const points = [
      { x: 0.4,   y: 0.4861, n: "BM25",    color: C.textMuted, dx:  0.30, dy: -0.30 },
      { x: 70.9,  y: 0.6121, n: "E5-base", color: C.secondary, dx:  0.00, dy: -0.45 },
      { x: 124.8, y: 0.3765, n: "nomic",   color: C.violet,    dx:  0.00, dy:  0.18 },
      { x: 222.5, y: 0.6722, n: "BGE-M3",  color: C.gold,      dx:  0.00, dy: -0.45 },
      { x: 463.4, y: 0.6325, n: "Qwen3",   color: C.primary,   dx: -0.10, dy: -0.45 },
    ];
    // Plot bounds (inferred): x-axis span 0…500 ms, y-axis span 0…0.80.
    const plotX0 = chX + 0.78, plotX1 = chX + chW - 0.20;
    const plotY0 = chY + chH - 0.95, plotY1 = chY + 0.55;
    const xToPx = (xv) => plotX0 + (xv / 500) * (plotX1 - plotX0);
    const yToPx = (yv) => plotY0 - (yv / 0.80) * (plotY0 - plotY1);
    for (const p of points) {
      const px = xToPx(p.x) + p.dx;
      const py = yToPx(p.y) + p.dy;
      s.addText(p.n, {
        x: px - 0.55, y: py, w: 1.10, h: 0.24,
        fontFace: F.sans, fontSize: 10, bold: true, color: p.color,
        align: "center", valign: "middle", margin: 0,
      });
    }

    // Pareto-frontier annotation (top-left of plot, away from axis labels)
    s.addText("⟵ Парето-фронт", {
      x: chX + 0.85, y: chY + 0.38, w: 1.80, h: 0.25,
      fontFace: F.sans, fontSize: 9.5, italic: true, bold: true, color: C.gold,
      align: "left", valign: "middle", margin: 0,
    });

    // Right: 5 insight cards
    const rx = chX + chW + 0.20, rw = W - rx - M_RIGHT, ry = CONTENT_TOP;
    s.addText("КЛЮЧОВІ ВИСНОВКИ", {
      x: rx, y: ry, w: rw, h: 0.30,
      fontFace: F.sans, fontSize: 10, bold: true, charSpacing: 2, color: C.textMuted,
      align: "left", valign: "middle", margin: 0,
    });
    const cards = [
      { t: "Лідер якості", model: "BGE-M3", note: "найвища nDCG (0.672 на Tech), ~228 мс", color: C.gold,      icon: I.trophy },
      { t: "Найшвидший", model: "BM25",   note: "<3 мс, але якість обмежена",          color: C.textMuted,  icon: I.speed },
      { t: "Альтернатива", model: "E5-base", note: "~3× швидше за BGE-M3, ~70 мс",      color: C.secondary,  icon: I.flag },
      { t: "Лідер на Legal", model: "Qwen3", note: "nDCG=0.320, але ~2× повільніший", color: C.primary,    icon: I.medal },
      { t: "Найслабший", model: "nomic",   note: "поступається навіть BM25",           color: C.violet,     icon: I.network },
    ];
    const cy0 = ry + 0.32, cH = (CONTENT_H - 0.40) / 5;
    for (let i = 0; i < cards.length; i++) {
      const cy = cy0 + i * cH;
      const bodyH = cH - 0.08;
      addCard(s, rx, cy + 0.04, rw, bodyH, { fill: C.bgWhite, borderColor: C.borderSoft });
      s.addShape("rect", {
        x: rx, y: cy + 0.04, w: 0.08, h: bodyH,
        fill: { color: cards[i].color }, line: { type: "none" },
      });
      // Tight three-row layout that fits within bodyH (~0.67"). Earlier we
      // gave the note a 0.10" container which was too small for 9.5 pt text
      // and let the description spill below the card edge.
      s.addText(cards[i].t, {
        x: rx + 0.18, y: cy + 0.07, w: rw - 0.30, h: 0.18,
        fontFace: F.sans, fontSize: 8.5, bold: true, charSpacing: 1.5, color: C.textMuted,
        align: "left", valign: "top", margin: 0,
      });
      s.addText(cards[i].model, {
        x: rx + 0.18, y: cy + 0.24, w: rw - 0.30, h: 0.24,
        fontFace: F.sans, fontSize: 12, bold: true, color: cards[i].color,
        align: "left", valign: "top", margin: 0,
      });
      s.addText(cards[i].note, {
        x: rx + 0.18, y: cy + 0.48, w: rw - 0.30, h: 0.20,
        fontFace: F.sans, fontSize: 9, color: C.textMid,
        align: "left", valign: "top", margin: 0,
      });
    }

    addKafedraLogo(s); addPageNumber(s,16, TOTAL);
  }

  // ==========================================================================
  // SLIDE 17 — Апробація
  // ==========================================================================
  {
    const s = pres.addSlide();
    s.background = { color: C.bgWhite };
    addTitleAdv(s, "Апробація результатів дослідження", { section: SEC.S10 });

    const conferences = [
      {
        n: "1",
        title: "Innovative Research in Science and Economy",
        subtitle: "2-я Міжнародна науково-практична конференція",
        year: "2026",
        type: "Стаття",
        paper: "Порівняльний аналіз ефективності моделей векторних ембеддінгів для задач семантичного пошуку в корпоративних базах знань",
        contrib: "Представлено benchmark-методологію та порівняння 4 моделей у трьох доменних колекціях; обґрунтовано перевагу BGE-M3 над класичним BM25.",
        color: C.primary,
      },
      {
        n: "2",
        title: "Радіоелектроніка та молодь у XXI столітті",
        subtitle: "30-й Міжнародний молодіжний форум, ХНУРЕ",
        year: "2026",
        type: "Тези",
        paper: "Підвищення якості семантичного пошуку в базах знань із використанням векторних подань",
        contrib: "Подано підхід до підвищення якості retrieval у корпоративних базах знань на основі контрастивно-навчених embedding-моделей.",
        color: C.gold,
      },
    ];

    // 2 conference cards stacked
    const cardH = (CONTENT_H - 0.10 - 0.10 - 0.50) / 2;
    let cy = CONTENT_TOP;
    for (let i = 0; i < conferences.length; i++) {
      const c = conferences[i];
      addCard(s, M_LEFT, cy, W - M_LEFT - M_RIGHT, cardH, { fill: C.bgWhite, borderColor: C.borderSoft });
      // Big number on the left
      s.addShape("rect", {
        x: M_LEFT, y: cy, w: 1.20, h: cardH,
        fill: { color: c.color }, line: { type: "none" },
      });
      s.addText(c.n, {
        x: M_LEFT, y: cy + 0.10, w: 1.20, h: cardH * 0.45,
        fontFace: F.serif, fontSize: 42, bold: true, color: C.bgWhite,
        align: "center", valign: "middle", margin: 0,
      });
      s.addText("КОНФЕРЕНЦІЯ", {
        x: M_LEFT, y: cy + cardH * 0.55, w: 1.20, h: 0.20,
        fontFace: F.sans, fontSize: 8, bold: true, charSpacing: 2, color: C.bgWhite,
        align: "center", valign: "middle", margin: 0,
      });
      s.addText(c.year, {
        x: M_LEFT, y: cy + cardH * 0.72, w: 1.20, h: 0.30,
        fontFace: F.sans, fontSize: 14, bold: true, color: C.bgWhite,
        align: "center", valign: "middle", margin: 0,
      });

      // Right content
      const rx = M_LEFT + 1.40, rw = W - rx - M_RIGHT - 0.10;
      // Title
      s.addText(`«${c.title}»`, {
        x: rx, y: cy + 0.10, w: rw, h: 0.30,
        fontFace: F.sans, fontSize: 13, bold: true, color: C.textStrong,
        align: "left", valign: "middle", margin: 0,
      });
      s.addText(c.subtitle, {
        x: rx, y: cy + 0.40, w: rw, h: 0.25,
        fontFace: F.sans, fontSize: 10, italic: true, color: C.textMuted,
        align: "left", valign: "middle", margin: 0,
      });

      // Paper line
      s.addText([
        { text: c.type + ":  ", options: { fontSize: 10, bold: true, color: c.color } },
        { text: `«${c.paper}»`, options: { fontSize: 11, color: C.textStrong, italic: true } },
      ], {
        x: rx, y: cy + 0.70, w: rw, h: 0.50,
        fontFace: F.sans, valign: "top", margin: 0,
      });

      // Contribution
      s.addText([
        { text: "Внесок:  ", options: { fontSize: 9.5, bold: true, color: C.gold } },
        { text: c.contrib, options: { fontSize: 10, color: C.textMid } },
      ], {
        x: rx, y: cy + 1.25, w: rw, h: cardH - 1.30,
        fontFace: F.sans, valign: "top", margin: 0,
      });

      cy += cardH + 0.10;
    }

    // Bottom summary band
    addCard(s, M_LEFT, CONTENT_TOP + 2 * cardH + 0.20, W - M_LEFT - M_RIGHT, 0.50, {
      fill: C.bgDark, borderColor: C.bgDark,
    });
    s.addText([
      { text: "ЗАГАЛОМ:  ", options: { color: C.gold, fontSize: 10, bold: true, charSpacing: 2 } },
      { text: "2 наукові праці   ·   2 виступи на міжнародних конференціях   ·   у співавторстві з науковим керівником   ·   ХНУРЕ, 2026",
        options: { color: C.textOnDark, fontSize: 11, bold: true } },
    ], {
      x: M_LEFT, y: CONTENT_TOP + 2 * cardH + 0.20, w: W - M_LEFT - M_RIGHT, h: 0.50,
      fontFace: F.sans, valign: "middle", align: "center", margin: 0,
    });

    addKafedraLogo(s); addPageNumber(s,17, TOTAL);
  }

  // ==========================================================================
  // SLIDE 18 — Висновки та "Дякую"  (light background to match the deck)
  // ==========================================================================
  {
    const s = pres.addSlide();
    s.background = { color: C.bgWhite };

    // Subtle decorative motif so the slide doesn't read as a blank page
    s.addShape("ellipse", {
      x: -1.0, y: -1.0, w: 4.0, h: 4.0,
      fill: { color: C.surfaceCool }, line: { type: "none" },
    });
    s.addShape("ellipse", {
      x: 8.0, y: 4.0, w: 3.5, h: 3.5,
      fill: { color: C.gold, transparency: 80 }, line: { type: "none" },
    });

    addTitleAdv(s, "Підсумки та практичні рекомендації", { section: SEC.S11 });

    // Goal achieved banner (gold band, dark text)
    addCard(s, M_LEFT, CONTENT_TOP, W - M_LEFT - M_RIGHT, 0.55, {
      fill: C.gold, borderColor: C.gold,
    });
    s.addImage({ data: await iconPng(fa.FaCheck, hash("0F172A")), x: 0.65, y: CONTENT_TOP + 0.13, w: 0.30, h: 0.30 });
    s.addText("ПОСТАВЛЕНУ МЕТУ ДОСЯГНУТО:  обґрунтовано доцільність застосування embedding-моделей для семантичного пошуку у корпоративних базах знань", {
      x: 1.10, y: CONTENT_TOP, w: W - 1.10 - M_RIGHT - 0.10, h: 0.55,
      fontFace: F.sans, fontSize: 11, bold: true, color: C.bgDark,
      align: "left", valign: "middle", margin: 0,
    });

    // 4 recommendation cards in 2×2
    const recs = [
      { t: "BGE-M3", sub: "Основна модель для впровадження", note: "Найкраща якість, стабільність у різних доменах",  color: C.gold,      iconCmp: fa.FaTrophy },
      { t: "E5-base",sub: "Швидка альтернатива",              note: "Прийнятна якість + ~3× швидше за BGE-M3",         color: C.secondary, iconCmp: fa.FaTachometerAlt },
      { t: "Qwen3",  sub: "Лідер на юридичному домені",       note: "Найвища nDCG@10 на Legal (0.320), ~2× повільніший", color: C.primary,   iconCmp: fa.FaMedal },
      { t: "BM25",   sub: "Конкурентний baseline",            note: "Залишається сильним у певних доменах",            color: C.textMuted, iconCmp: fa.FaLayerGroup },
    ];
    const recW = (W - M_LEFT - M_RIGHT - 0.20) / 2;
    const recH = (CONTENT_H - 0.55 - 0.10 - 0.55 - 0.20) / 2;
    for (let i = 0; i < 4; i++) {
      const col = i % 2, row = Math.floor(i / 2);
      const cx = M_LEFT + col * (recW + 0.20);
      const cy = CONTENT_TOP + 0.55 + 0.20 + row * (recH + 0.15);
      addCard(s, cx, cy, recW, recH, {
        fill: C.bgWhite,
        borderColor: recs[i].color, borderWidth: 1.25, shadow: false,
      });
      // Side accent stripe
      s.addShape("rect", {
        x: cx, y: cy, w: 0.10, h: recH,
        fill: { color: recs[i].color }, line: { type: "none" },
      });
      // Icon in tinted circle
      s.addShape("ellipse", {
        x: cx + 0.25, y: cy + 0.18, w: 0.55, h: 0.55,
        fill: { color: recs[i].color }, line: { type: "none" },
      });
      s.addImage({ data: await iconPng(recs[i].iconCmp, hash("FFFFFF")),
        x: cx + 0.36, y: cy + 0.29, w: 0.32, h: 0.32 });

      s.addText(recs[i].t, {
        x: cx + 0.90, y: cy + 0.10, w: recW - 1.05, h: 0.35,
        fontFace: F.sans, fontSize: 16, bold: true, color: recs[i].color,
        align: "left", valign: "middle", margin: 0,
      });
      s.addText(recs[i].sub, {
        x: cx + 0.90, y: cy + 0.46, w: recW - 1.05, h: 0.25,
        fontFace: F.sans, fontSize: 10, bold: true, color: C.textStrong,
        align: "left", valign: "middle", margin: 0,
      });
      s.addText(recs[i].note, {
        x: cx + 0.25, y: cy + 0.85, w: recW - 0.45, h: recH - 0.95,
        fontFace: F.sans, fontSize: 10, color: C.textMid,
        align: "left", valign: "top", margin: 0,
      });
    }

    // Final "Дякую за увагу!" — gold serif italic on white
    s.addText("Дякую за увагу!", {
      x: 0, y: H - 0.85, w: W, h: 0.55,
      fontFace: F.serif, fontSize: 24, bold: true, italic: true, color: C.goldDark,
      align: "center", valign: "middle", margin: 0,
    });

    addKafedraLogo(s);
    addPageNumber(s, 18, TOTAL);
  }

  // ==========================================================================
  // WRITE
  // ==========================================================================
  await pres.writeFile({ fileName: OUTPUT });
  console.log(`OK: wrote ${OUTPUT}`);
}

build().catch(err => {
  console.error(err);
  process.exit(1);
});

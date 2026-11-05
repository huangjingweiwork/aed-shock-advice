// Build report.docx from report.md (a small Markdown subset: % title lines, # / ## headings, paragraphs with
// **bold** / *italic* / `code`, "- " and "1. " lists, | tables |, ![caption](image.png), "Table N:" captions).
// Usage: node build_docx.js   (needs the npm package "docx")
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, Table, TableRow, TableCell, WidthType,
  ShadingType, BorderStyle, ImageRun, Footer, PageNumber, LevelFormat, TableLayoutType,
} = require("docx");

const DIR = __dirname;
const FONT = "Calibri";
const PAGE_W = 11906, MARGIN = 1300, TEXT_W = PAGE_W - 2 * MARGIN; // A4, DXA
const lines = fs.readFileSync(path.join(DIR, "report.md"), "utf8").replace(/\r/g, "").split("\n");

function runs(text, base = {}) {
  // split on **bold**, *italic*, `code`
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), font: FONT, ...base }));
    const t = m[0];
    if (t.startsWith("**")) out.push(new TextRun({ text: t.slice(2, -2), bold: true, font: FONT, ...base }));
    else if (t.startsWith("`")) out.push(new TextRun({ text: t.slice(1, -1), font: "Consolas", size: 19, ...base }));
    else out.push(new TextRun({ text: t.slice(1, -1), italics: true, font: FONT, ...base }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), font: FONT, ...base }));
  return out;
}

function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20), data: b };
}

const border = { style: BorderStyle.SINGLE, size: 4, color: "BFBFBF" };
function table(rows) {
  const cells = rows.map(r => r.replace(/^\||\|$/g, "").split("|").map(c => c.trim()));
  const header = cells[0], body = cells.slice(2);
  const n = header.length;
  // column widths proportional to the longest text in each column (min share)
  // header text may wrap, so it counts by its longest word; body cells count in full
  const word = t => Math.max(...t.split(/\s+/).map(w => w.length));
  // a ✗ is drawn wider than a digit, so it counts as 3; the first (label) column may wrap, so it is capped
  const cellLen = c => (c || "").replace(/\*/g, "").length + ((c || "").includes("✗") ? 2 : 0);
  const len = header.map((h, j) => {
    const l = Math.max(word(h) + 1, ...body.map(r => cellLen(r[j])), 4);
    return j === 0 && n > 6 ? Math.min(l, 18) : l;
  });
  const fs_ = n > 8 ? 16 : 18;
  const tot = len.reduce((a, b) => a + b, 0);
  let widths = len.map(l => Math.max(500, Math.round((TEXT_W * l) / tot)));
  const scale = TEXT_W / widths.reduce((a, b) => a + b, 0);
  widths = widths.map(w => Math.floor(w * scale));
  widths[n - 1] += TEXT_W - widths.reduce((a, b) => a + b, 0);
  const mk = (r, head) => new TableRow({
    tableHeader: head,
    children: r.map((c, j) => new TableCell({
      width: { size: widths[j], type: WidthType.DXA },
      borders: { top: border, bottom: border, left: border, right: border },
      shading: head ? { type: ShadingType.CLEAR, fill: "E8EEF7", color: "auto" } : undefined,
      margins: { top: 40, bottom: 40, left: 80, right: 80 },
      children: [new Paragraph({ children: runs(c, { size: fs_, bold: head || undefined }),
                                 alignment: j > 0 && /^[\d.,%<>≥−\-–+ ✗/]+$/.test(c) ? AlignmentType.RIGHT : AlignmentType.LEFT })],
    })),
  });
  return new Table({ width: { size: TEXT_W, type: WidthType.DXA }, columnWidths: widths, layout: TableLayoutType.FIXED,
                     rows: [mk(header, true), ...body.map(r => mk(r, false))] });
}

const children = [];
const titles = [];
let para = [];
const flush = () => {
  if (!para.length) return;
  const text = para.join(" ");
  para = [];
  if (/^Table \d+:/.test(text)) {
    const [lab, ...rest] = text.split(":");
    children.push(new Paragraph({ keepNext: true, spacing: { before: 160, after: 80 },
      children: [new TextRun({ text: lab + ":", bold: true, font: FONT, size: 20 }), ...runs(rest.join(":"), { size: 20 })] }));
  } else if (inRefs) {
    children.push(new Paragraph({ children: runs(text, { size: 20 }), indent: { left: 400, hanging: 400 },
                                  spacing: { after: 80 } }));
  } else {
    children.push(new Paragraph({ children: runs(text), spacing: { after: 140, line: 288 }, alignment: AlignmentType.JUSTIFIED }));
  }
};
let inRefs = false;
for (let i = 0; i < lines.length; i++) {
  const l = lines[i];
  if (l.startsWith("% ")) { titles.push(l.slice(2)); continue; }
  if (!l.trim()) { flush(); continue; }
  if (l.startsWith("# ") || l.startsWith("## ")) {
    flush();
    const h1 = l.startsWith("# ");
    const text = l.replace(/^#+ /, "");
    inRefs = text === "References";
    children.push(new Paragraph({ heading: h1 ? HeadingLevel.HEADING_1 : HeadingLevel.HEADING_2, children: runs(text) }));
    continue;
  }
  if (l.startsWith("|")) {
    flush();
    const rows = [];
    while (i < lines.length && lines[i].startsWith("|")) rows.push(lines[i++]);
    i--;
    children.push(table(rows));
    children.push(new Paragraph({ children: [], spacing: { after: 120 } }));
    continue;
  }
  const img = l.match(/^!\[(.*)\]\((.*)\)$/);
  if (img) {
    flush();
    const src = process.env.QUICK ? img[2].replace("../results/", "../results_quick/") : img[2];
    const file = path.join(DIR, src);
    if (!fs.existsSync(file)) { console.warn("missing figure", src); continue; }
    const { w, h, data } = pngSize(file);
    const maxW = 600; // px at 96 dpi = 6.25 in
    const width = Math.min(maxW, w / 2), height = Math.round((h * width) / w);
    children.push(new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 120 },
                                  children: [new ImageRun({ type: "png", data, transformation: { width, height } })] }));
    const [lab, ...rest] = img[1].split(":");
    children.push(new Paragraph({ spacing: { after: 200 }, alignment: AlignmentType.LEFT,
      children: [new TextRun({ text: lab + ":", bold: true, font: FONT, size: 19 }), ...runs(rest.join(":"), { size: 19, color: "404040" })] }));
    continue;
  }
  const bullet = l.match(/^- (.*)/), num = l.match(/^(\d+)\. (.*)/);
  if (bullet || num) {
    flush();
    children.push(new Paragraph({ children: runs((bullet || num)[bullet ? 1 : 2]), spacing: { after: 80, line: 276 },
                                  numbering: { reference: bullet ? "bullets" : "numbers", level: 0 } }));
    continue;
  }
  para.push(l.trim());
}
flush();

const head = [
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 },
                  children: [new TextRun({ text: titles[0] || "", bold: true, size: 36, font: FONT })] }),
  ...titles.slice(1).map(t => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 },
                                              children: [new TextRun({ text: t, size: 22, color: "404040", font: FONT })] })),
  new Paragraph({ children: [], spacing: { after: 200 },
                  border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: "2A78D6", space: 4 } } }),
];

const doc = new Document({
  styles: {
    default: { document: { run: { font: FONT, size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 30, bold: true, color: "1C5CAB" },
        paragraph: { spacing: { before: 320, after: 140 }, keepNext: true, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 25, bold: true, color: "0B0B0B" },
        paragraph: { spacing: { before: 220, after: 100 }, keepNext: true, outlineLevel: 1 } },
    ],
  },
  numbering: { config: [
    { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
                                        style: { paragraph: { indent: { left: 500, hanging: 260 } } } }] },
    { reference: "numbers", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
                                        style: { paragraph: { indent: { left: 500, hanging: 300 } } } }] },
  ] },
  sections: [{
    properties: { page: { size: { width: PAGE_W, height: 16838 },
                          margin: { top: MARGIN, bottom: MARGIN, left: MARGIN, right: MARGIN } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ children: [PageNumber.CURRENT], size: 18, color: "808080", font: FONT })] })] }) },
    children: [...head, ...children],
  }],
});
Packer.toBuffer(doc).then(b => {
  fs.writeFileSync(path.join(DIR, "report.docx"), b);
  console.log("wrote report.docx");
});

import fs from "node:fs";
import path from "node:path";

const outDir = path.resolve("figures");
fs.mkdirSync(outDir, { recursive: true });

const C = {
  ink: "#172033",
  muted: "#64748b",
  grid: "#dbe3ee",
  blue: "#2563eb",
  orange: "#ea580c",
  purple: "#7c3aed",
  green: "#059669",
  red: "#dc2626",
  bg: "#ffffff",
};

const esc = (s) =>
  String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");

function svgText(x, y, text, opts = {}) {
  const {
    size = 13,
    weight = 400,
    fill = C.ink,
    anchor = "start",
    rotate = null,
  } = opts;
  const transform = rotate === null ? "" : ` transform="rotate(${rotate} ${x} ${y})"`;
  return `<text x="${x}" y="${y}" font-family="Arial, Helvetica, sans-serif" font-size="${size}" font-weight="${weight}" fill="${fill}" text-anchor="${anchor}"${transform}>${esc(text)}</text>`;
}

function panelFrame(x, y, w, h, title, subtitle) {
  return [
    `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="9" fill="${C.bg}" stroke="${C.grid}"/>`,
    svgText(x + 16, y + 25, title, { size: 15, weight: 700 }),
    svgText(x + 16, y + 44, subtitle, { size: 11, fill: C.muted }),
  ].join("\n");
}

function refusalPanel(x, y, w, h) {
  const names = ["Normal", "Local\nprojection", "Circuit\nablated", "+ local\nrescue", "Global\nprojection"];
  const values = [3.89, 3.59, -8.58, -4.3, -13.62];
  const colors = [C.blue, C.orange, C.red, C.purple, "#991b1b"];
  const left = x + 50;
  const top = y + 66;
  const bottom = y + h - 65;
  const right = x + w - 14;
  const yMin = -15;
  const yMax = 5;
  const sy = (v) => bottom - ((v - yMin) / (yMax - yMin)) * (bottom - top);
  const parts = [
    panelFrame(x, y, w, h, "A. Refusal: broad circuit effect ≠ one-token mediation", "First-token refusal log-odds; n=16"),
  ];

  for (const tick of [-15, -10, -5, 0, 5]) {
    const yy = sy(tick);
    parts.push(`<line x1="${left}" y1="${yy}" x2="${right}" y2="${yy}" stroke="${tick === 0 ? C.muted : C.grid}" stroke-width="${tick === 0 ? 1.3 : 1}"/>`);
    parts.push(svgText(left - 7, yy + 4, tick, { size: 10, fill: C.muted, anchor: "end" }));
  }

  const slot = (right - left) / values.length;
  const barW = slot * 0.58;
  values.forEach((v, i) => {
    const cx = left + slot * (i + 0.5);
    const zero = sy(0);
    const yy = sy(v);
    parts.push(`<rect x="${cx - barW / 2}" y="${Math.min(zero, yy)}" width="${barW}" height="${Math.abs(zero - yy)}" rx="3" fill="${colors[i]}"/>`);
    parts.push(svgText(cx, v >= 0 ? yy - 7 : yy + 15, v.toFixed(2), {
      size: 10,
      weight: 700,
      fill: v >= 0 ? C.ink : "#ffffff",
      anchor: "middle",
    }));
    const [a, b] = names[i].split("\n");
    parts.push(svgText(cx, bottom + 20, a, { size: 10, anchor: "middle" }));
    if (b) parts.push(svgText(cx, bottom + 32, b, { size: 10, anchor: "middle" }));
  });
  return parts.join("\n");
}

function stagePanel(x, y, w, h, letter = "A") {
  const groups = [
    { label: "1×", values: [14.67, 5.95, 26.88] },
    { label: "2×", values: [51.18, 25.9, 92.52] },
  ];
  const series = [
    { label: "Prompt only", color: C.blue },
    { label: "Generation only", color: C.orange },
    { label: "Both", color: C.purple },
  ];
  const left = x + 48;
  const top = y + 90;
  const bottom = y + h - 62;
  const right = x + w - 16;
  const sy = (v) => bottom - (v / 100) * (bottom - top);
  const parts = [
    panelFrame(x, y, w, h, `${letter}. Prompt and generation interventions are complementary`, "Normalized movement toward anti-sycophancy; n=20"),
  ];

  for (const tick of [0, 25, 50, 75, 100]) {
    const yy = sy(tick);
    parts.push(`<line x1="${left}" y1="${yy}" x2="${right}" y2="${yy}" stroke="${C.grid}"/>`);
    parts.push(svgText(left - 7, yy + 4, tick, { size: 10, fill: C.muted, anchor: "end" }));
  }

  const groupW = (right - left) / groups.length;
  const barW = Math.min(36, groupW / 5);
  groups.forEach((g, gi) => {
    const center = left + groupW * (gi + 0.5);
    g.values.forEach((v, si) => {
      const xx = center + (si - 1) * (barW + 7);
      const yy = sy(v);
      parts.push(`<rect x="${xx - barW / 2}" y="${yy}" width="${barW}" height="${bottom - yy}" rx="3" fill="${series[si].color}"/>`);
      parts.push(svgText(xx, yy - 6, `${v.toFixed(1)}%`, { size: 10, weight: 700, anchor: "middle" }));
    });
    parts.push(svgText(center, bottom + 20, g.label, { size: 12, weight: 700, anchor: "middle" }));
  });

  let lx = left + 4;
  series.forEach((s) => {
    parts.push(`<rect x="${lx}" y="${y + h - 26}" width="10" height="10" rx="2" fill="${s.color}"/>`);
    parts.push(svgText(lx + 15, y + h - 17, s.label, { size: 10 }));
    lx += s.label === "Generation only" ? 116 : 94;
  });
  return parts.join("\n");
}

function circuitPanel(x, y, w, h, letter = "B") {
  const prompt = [
    [0.038, 89.8],
    [0.192, 94.9],
    [0.383, 97.5],
    [1.15, 97.5],
  ];
  const e2e = [
    [0.044, -0.3],
    [0.092, 0.2],
    [0.298, 5.1],
    [0.504, 9.7],
    [1.045, 16.9],
    [5.33, 67.9],
    [10.68, 76.8],
  ];
  const left = x + 52;
  const top = y + 68;
  const bottom = y + h - 62;
  const right = x + w - 18;
  const logMin = Math.log10(0.03);
  const logMax = Math.log10(15);
  const sx = (v) => left + ((Math.log10(v) - logMin) / (logMax - logMin)) * (right - left);
  const sy = (v) => bottom - ((Math.max(-5, Math.min(105, v)) + 5) / 110) * (bottom - top);
  const parts = [
    panelFrame(x, y, w, h, `${letter}. Coordinate circuit is much sparser`, "Faithfulness on 64 held-out questions"),
  ];

  for (const tick of [0, 25, 50, 75, 100]) {
    const yy = sy(tick);
    parts.push(`<line x1="${left}" y1="${yy}" x2="${right}" y2="${yy}" stroke="${C.grid}"/>`);
    parts.push(svgText(left - 7, yy + 4, tick, { size: 10, fill: C.muted, anchor: "end" }));
  }
  for (const tick of [0.03, 0.1, 0.3, 1, 3, 10]) {
    const xx = sx(tick);
    parts.push(`<line x1="${xx}" y1="${top}" x2="${xx}" y2="${bottom}" stroke="${C.grid}" stroke-dasharray="2 4"/>`);
    parts.push(svgText(xx, bottom + 18, tick, { size: 10, fill: C.muted, anchor: "middle" }));
  }
  parts.push(svgText((left + right) / 2, bottom + 38, "Retained graph edges (%) — log scale", { size: 10, fill: C.muted, anchor: "middle" }));
  parts.push(svgText(left - 39, (top + bottom) / 2, "Faithfulness (%)", { size: 10, fill: C.muted, anchor: "middle", rotate: -90 }));

  const drawSeries = (data, color) => {
    const points = data.map(([a, b]) => `${sx(a)},${sy(b)}`).join(" ");
    parts.push(`<polyline points="${points}" fill="none" stroke="${color}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>`);
    data.forEach(([a, b]) => {
      parts.push(`<circle cx="${sx(a)}" cy="${sy(b)}" r="4.5" fill="${C.bg}" stroke="${color}" stroke-width="2.5"/>`);
    });
  };
  drawSeries(prompt, C.green);
  drawSeries(e2e, C.red);

  const ly = y + 76;
  parts.push(`<line x1="${right - 252}" y1="${ly - 4}" x2="${right - 230}" y2="${ly - 4}" stroke="${C.green}" stroke-width="3"/>`);
  parts.push(svgText(right - 224, ly, "Prompt coordinate", { size: 10 }));
  parts.push(`<line x1="${right - 112}" y1="${ly - 4}" x2="${right - 90}" y2="${ly - 4}" stroke="${C.red}" stroke-width="3"/>`);
  parts.push(svgText(right - 84, ly, "End-to-end", { size: 10 }));
  return parts.join("\n");
}

function document(width, height, content) {
  return `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">
  <rect width="100%" height="100%" fill="#f8fafc"/>
  ${content}
</svg>
`;
}

const wiml = document(
  700,
  410,
  stagePanel(10, 10, 680, 390, ""),
);

const supervisor = document(
  1450,
  410,
  [
    refusalPanel(10, 10, 460, 390),
    stagePanel(480, 10, 460, 390, "B"),
    circuitPanel(950, 10, 490, 390, "C"),
  ].join("\n"),
);

fs.writeFileSync(path.join(outDir, "wiml_core_results.svg"), wiml);
fs.writeFileSync(path.join(outDir, "supervisor_core_results.svg"), supervisor);
fs.writeFileSync(
  path.join(outDir, "circuit_sparsity.svg"),
  document(720, 410, circuitPanel(10, 10, 700, 390, "")),
);
fs.writeFileSync(
  path.join(outDir, "refusal_mediation.svg"),
  document(660, 410, refusalPanel(10, 10, 640, 390)),
);

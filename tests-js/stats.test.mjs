// SPDX-License-Identifier: AGPL-3.0-or-later
// jsdom tests for the status-bar hardware monitor (renderHwStats / pollHwStats
// in localm/plugins/gui/static/app.js). renderHwStats renders whatever
// /api/stats reports; absent sections must simply not appear, and an all-empty
// payload must hide the readout entirely.

import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { loadApp } from "./harness.mjs";

const GIB = 1024 ** 3;
const STATS = {
  cpu: { percent: 12.4 },
  ram: { used: 4 * GIB, total: 8 * GIB, percent: 50.0 },
  vram: { used: 2 * GIB, total: 16 * GIB, percent: 12.5 },
  gpu: { percent: 30.0 },
};

const settle = () => new Promise((r) => setTimeout(r, 0));

// Well-formed responses for the endpoints app.js's bootstrap touches on load, so
// the unrelated init code (refreshModels -> populateSetupModels, plugins) never
// throws while we exercise the hardware monitor. `calls` records URLs hit.
function goodFetch(calls = []) {
  return async (url) => {
    const u = String(url);
    calls.push(u);
    if (u.endsWith("/api/stats"))
      return { ok: true, status: 200, json: async () => STATS };
    if (u.includes("/api/models"))
      return { ok: true, status: 200, json: async () => ({ models: [], active: "" }) };
    if (u.includes("/api/plugins"))
      return { ok: true, status: 200, json: async () => ({ plugins: [] }) };
    return { ok: true, status: 200, json: async () => ({}) };
  };
}

test("renderHwStats writes a compact CPU/RAM/GPU/VRAM line and unhides", () => {
  const { window } = loadApp({ fetchImpl: goodFetch() });
  assert.equal(typeof window.renderHwStats, "function", "renderHwStats on window");
  const el = window.document.getElementById("hw-stats");
  assert.ok(el, "#hw-stats element exists in index.html");
  window.renderHwStats(STATS);
  assert.equal(el.hidden, false, "shown when there is data");
  assert.match(el.textContent, /CPU 12%/);
  assert.match(el.textContent, /RAM 4\.0\/8\.0 GB/, "RAM reads as used/total, matching VRAM");
  assert.match(el.textContent, /GPU 30%/);
  assert.match(el.textContent, /VRAM 2\.0\/16\.0 GB/);
});

test("renderHwStats orders the metrics CPU, RAM, GPU, VRAM", () => {
  const { window } = loadApp({ fetchImpl: goodFetch() });
  const el = window.document.getElementById("hw-stats");
  window.renderHwStats(STATS);
  // The row wraps after two metrics at the sidebar's width, so this order is
  // what puts the system on line one and the graphics card on line two. The
  // old order (CPU, RAM, VRAM, GPU) split the card's pair across the wrap.
  assert.deepEqual(
    [...el.querySelectorAll("span")].map((s) => s.textContent.split(" ")[0]),
    ["CPU", "RAM", "GPU", "VRAM"]);
});

test("renderHwStats falls back to RAM percent when used/total are absent", () => {
  const { window } = loadApp({ fetchImpl: goodFetch() });
  const el = window.document.getElementById("hw-stats");
  // An older server, or a proxy returning a partial payload: the readout must
  // still show RAM rather than dropping the metric silently.
  window.renderHwStats({ ram: { percent: 61.4 } });
  assert.match(el.textContent, /RAM 61%/);
  assert.equal(el.hidden, false);
});

test("renderHwStats omits RAM entirely when the payload carries no memory data", () => {
  const { window } = loadApp({ fetchImpl: goodFetch() });
  const el = window.document.getElementById("hw-stats");
  window.renderHwStats({ ram: {}, cpu: { percent: 7 } });
  assert.doesNotMatch(el.textContent, /RAM/, "no RAM span when there is nothing to report");
  assert.match(el.textContent, /CPU 7%/, "the other metrics still render");
});

test("renderHwStats hides the readout when nothing is measurable", () => {
  const { window } = loadApp({ fetchImpl: goodFetch() });
  const el = window.document.getElementById("hw-stats");
  window.renderHwStats(STATS);
  assert.equal(el.hidden, false);
  window.renderHwStats({});              // negative: empty payload
  assert.equal(el.hidden, true, "empty stats hide the readout");
});

test("renderHwStats shows VRAM total-only when free is unknown", () => {
  const { window } = loadApp({ fetchImpl: goodFetch() });
  const el = window.document.getElementById("hw-stats");
  window.renderHwStats({ vram: { total: 8 * GIB } });
  assert.match(el.textContent, /VRAM 8\.0 GB/);
  assert.doesNotMatch(el.textContent, /\//, "no used/total slash when free unknown");
});

test("renderHwStats tints the VRAM figure by fullness when used is known", () => {
  const { window } = loadApp({ fetchImpl: goodFetch() });
  const el = window.document.getElementById("hw-stats");
  // 2/16 GiB = 12.5% -> low band. The tint rides on its own span so ONLY the VRAM
  // figure is coloured, not the whole CPU/RAM/GPU row.
  window.renderHwStats({ vram: { used: 2 * GIB, total: 16 * GIB } });
  const span = el.querySelector(".vram-usage");
  assert.ok(span, "the VRAM figure is a .vram-usage span when used is known");
  assert.match(span.textContent, /VRAM 2\.0\/16\.0 GB/);
  assert.ok(span.classList.contains("vram-ok"), "low usage -> vram-ok band");
});

test("renderHwStats uses the high band when VRAM is nearly full", () => {
  const { window } = loadApp({ fetchImpl: goodFetch() });
  const el = window.document.getElementById("hw-stats");
  window.renderHwStats({ vram: { used: 15 * GIB, total: 16 * GIB } });   // 93.75%
  assert.ok(el.querySelector(".vram-usage.vram-full"), "over 90% full -> vram-full band");
});

test("renderHwStats gives the VRAM figure NO colour when only total is known", () => {
  // A stale or process-blind reading arrives as total-only (no used): there is
  // nothing to be "full" of, so no tint - and never a wrong number shown as live.
  const { window } = loadApp({ fetchImpl: goodFetch() });
  const el = window.document.getElementById("hw-stats");
  window.renderHwStats({ vram: { total: 16 * GIB } });
  assert.equal(el.querySelector(".vram-usage"), null, "no .vram-usage span when free is unknown");
  assert.match(el.textContent, /VRAM 16\.0 GB/);
});

test("pollHwStats fetches /api/stats and renders the result", async () => {
  const calls = [];
  const { window } = loadApp({ fetchImpl: goodFetch(calls) });
  await window.pollHwStats();
  await settle();
  assert.ok(calls.some((u) => u.endsWith("/api/stats")), "polls GET /api/stats");
  assert.match(window.document.getElementById("hw-stats").textContent, /CPU 12%/);
});

// --------------------------------------------------------------------------- //
//  The readout WRAPS, it never truncates                                       //
//                                                                              //
//  It used to be `white-space: nowrap` + `overflow: hidden` + `text-overflow:  //
//  ellipsis`, so at the sidebar's width the line was cut - and because VRAM is //
//  rendered last of CPU/RAM/VRAM/GPU, the figure that got cut was the one the  //
//  readout exists for ("VRAM 3.6/1..."). jsdom does no layout, so the wrap     //
//  itself is unobservable here; what IS checkable is that the truncating       //
//  declarations are gone from the REAL stylesheet and the wrapping ones are    //
//  present, plus the DOM shape the wrap depends on.                            //
// --------------------------------------------------------------------------- //

// COMMENTS ARE STRIPPED FIRST, and that is not tidiness. The rule carries a
// comment naming the very declarations that must be absent (it explains what
// was removed and why), so a raw match reports "text-overflow: ellipsis is
// still here" against prose while the declaration is long gone. Caught by this
// test failing on a correct stylesheet - the check has to read declarations,
// not the words near them.
const hwStatsCss = () => {
  const css = readFileSync(
    fileURLToPath(new URL("../localm/plugins/gui/static/style.css", import.meta.url)), "utf8");
  const m = css.match(/\n\.hw-stats \{([\s\S]*?)\n\}/);
  assert.ok(m, ".hw-stats rule found in style.css");
  return m[1].replace(/\/\*[\s\S]*?\*\//g, "");
};

// The column pinning lives in its own rules, OUTSIDE `.hw-stats { ... }`, so the
// rule-scoped reader above cannot see it. Same comment-stripping, same reason.
const styleCss = () => readFileSync(
  fileURLToPath(new URL("../localm/plugins/gui/static/style.css", import.meta.url)), "utf8")
  .replace(/\/\*[\s\S]*?\*\//g, "");

test("hw-stats: the rule does not truncate", () => {
  const rule = hwStatsCss();
  assert.doesNotMatch(rule, /text-overflow:\s*ellipsis/,
    "ellipsis would silently cut the VRAM figure off again");
  assert.doesNotMatch(rule, /white-space:\s*nowrap/,
    "nowrap on the container is what forced the truncation");
  assert.doesNotMatch(rule, /overflow:\s*hidden/, "hidden overflow clips the wrapped second line");
});

test("hw-stats: the rule lays the metrics out as a grid instead", () => {
  const css = hwStatsCss();
  assert.match(css, /display:\s*grid/, "metrics must be allowed onto a second line");
  assert.match(css, /grid-template-columns:\s*auto auto/,
    "two columns: load on the left, memory on the right");
});

// The 2x2 must NOT depend on where the row happens to wrap. It is pinned, so an
// absent metric leaves an empty cell rather than pulling the next one across -
// and absence is routine here (no psutil means no CPU/RAM, an AMD card reports
// no GPU%), which is exactly the case a wrap-based layout got wrong.
test("hw-stats: load and memory are pinned to their own columns", () => {
  const css = styleCss();
  assert.match(css, /\[data-metric="cpu"\][\s\S]{0,80}\[data-metric="gpu"\]\s*{\s*grid-column:\s*1/,
    "CPU and GPU share column 1");
  assert.match(css, /\[data-metric="ram"\][\s\S]{0,80}\[data-metric="vram"\]\s*{\s*grid-column:\s*2/,
    "RAM and VRAM share column 2");
});

// Column-major reading is the whole point of the layout, so it is asserted on the
// RENDERED spans rather than only in CSS: a future change to the emit order in
// renderHwStats would otherwise pass the CSS checks above while scrambling the grid.
test("hw-stats: each metric carries the column tag the grid needs", () => {
  const { window } = loadApp({ fetchImpl: goodFetch() });
  window.renderHwStats(STATS);
  const el = window.document.getElementById("hw-stats");
  const tags = [...el.querySelectorAll("span")].map((s) => s.dataset.metric);
  assert.deepEqual(tags, ["cpu", "ram", "gpu", "vram"],
    "row 1 is the system (CPU, RAM), row 2 is the card (GPU, VRAM)");
});

test("hw-stats: each metric is one unbreakable span, with no stray separator text", () => {
  const { window } = loadApp({ fetchImpl: goodFetch() });
  const el = window.document.getElementById("hw-stats");
  window.renderHwStats(STATS);
  const spans = [...el.querySelectorAll("span")];
  assert.equal(spans.length, 4, "one span per metric (CPU/RAM/VRAM/GPU)");
  // Every child is an element: a " · " TEXT node between spans is precisely what
  // gets stranded at the end of a wrapped line, so the separator moved to CSS.
  assert.equal([...el.childNodes].every((n) => n.nodeType === 1), true,
    "no separator text nodes between the metric spans");
  // The whole VRAM figure travels together - a wrap must not split "2.0/16.0 GB".
  const vram = spans.find((s) => /VRAM/.test(s.textContent));
  assert.equal(vram.textContent, "VRAM 2.0/16.0 GB", "the VRAM span carries the complete figure");
});

// MULTI-GPU. The aggregate is not merely less detailed on a 2-card board, it is
// actively misleading: 22/48 GB reads as comfortable while card 0 sits at 20/24.
// These pin the per-card rows so that cannot silently regress back to one figure.
const MULTI = {
  cpu: { percent: 12.4 },
  ram: { used: 4 * GIB, total: 8 * GIB, percent: 50.0 },
  gpu: { percent: 30.0 },
  vram: {
    used: 22 * GIB, total: 48 * GIB, percent: 45.8,
    devices: [
      { index: 0, name: "RTX 4090", total: 24 * GIB, used: 20 * GIB, percent: 83.3 },
      { index: 1, name: "RTX 3090", total: 24 * GIB, used: 2 * GIB, percent: 8.3 },
    ],
  },
};

test("hw-stats: a multi-GPU board gets one row per card, not the combined figure", () => {
  const { window } = loadApp({ fetchImpl: goodFetch() });
  window.renderHwStats(MULTI);
  const el = window.document.getElementById("hw-stats");
  const spans = [...el.querySelectorAll("span")];
  const text = spans.map((s) => s.textContent);

  assert.deepEqual(spans.map((s) => s.dataset.metric),
    ["cpu", "ram", "gpu", "vram", "gpu", "vram"],
    "CPU|RAM, then one GPU|VRAM pair per card");
  assert.deepEqual(text.slice(2), ["GPU0", "20.0/24.0 GB", "GPU1", "2.0/24.0 GB"],
    "each card names itself and reports ITS OWN used/total");

  // The combined 22/48 must not appear anywhere: it is the number that hides the
  // full card, which is the entire reason the breakdown exists.
  assert.ok(!text.some((t) => t.includes("22.0/48.0")),
    "the combined figure must not be shown alongside the per-card rows");

  // The near-full card must carry the warm band while the empty one does not -
  // a per-card figure with a board-wide colour would defeat the point.
  assert.match(spans[3].className, /vram-busy|vram-full/, "card 0 at 83% reads as busy");
  assert.match(spans[5].className, /vram-ok/, "card 1 at 8% reads as fine");
});

test("hw-stats: the unattributable aggregate GPU% is dropped on a multi-GPU board", () => {
  const { window } = loadApp({ fetchImpl: goodFetch() });
  window.renderHwStats(MULTI);
  const el = window.document.getElementById("hw-stats");
  const text = [...el.querySelectorAll("span")].map((s) => s.textContent);
  // /api/stats sends ONE system-wide percent with no card attribution, so beside a
  // named card it would be a borrowed number. Dropped rather than mis-attributed.
  assert.ok(!text.some((t) => /^GPU \d+%$/.test(t)),
    "a board-wide GPU% must not sit beside a specific card");
  // ...but it is still shown when there IS only one card to attribute it to.
  window.renderHwStats(STATS);
  const single = [...el.querySelectorAll("span")].map((s) => s.textContent);
  assert.ok(single.some((t) => /^GPU \d+%$/.test(t)),
    "single-card boards keep the aggregate GPU%");
});

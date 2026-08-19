// SPDX-License-Identifier: AGPL-3.0-or-later
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

// WHAT THIS FILE GUARDS, and why it is able to guard it at all.
//
// The bug: re-rendering a page (sorting the model list, saving a settings
// section) made the whole page jump to the top. MEASURED in Chrome, both cases:
// our JavaScript never called a scroll API and nothing reloaded. The rebuild
// left the scroll container with NO VISIBLE CONTENT for one layout, so
// scrollHeight fell to exactly clientHeight, the browser clamped scrollTop to 0
// because there was nothing left to scroll, and the content came back 20-50 ms
// later with the position already gone.
//
//     models    scrollHeight 1232 -> 945 (== clientHeight) -> 1429, scrollTop 287 -> 0
//     settings  scrollHeight 3157 -> 889 (== clientHeight) -> 3157, scrollTop 966 -> 0
//
// jsdom HAS NO LAYOUT ENGINE: scrollTop, scrollHeight and clientHeight are always
// 0 there, so a test asserting on scroll position here could not fail and would
// be theatre. This file therefore does NOT test scroll.
//
// It tests the PRECONDITION, which is pure DOM and fully observable in jsdom:
// at every point where the render suspends, the container still has something
// visible in it. A suspension point is exactly where a real browser gets to lay
// out and paint, so "visible content at every suspension" is the property that
// makes the clamp unreachable.
//
// The sampling is tied to FETCH, deliberately, not to a timer. Every suspension
// in both render paths is an awaited fetch, and in a real browser that await
// lasts a network round-trip - which is when the browser paints. A macrotask
// sampler would prove nothing: with an instant fetch stub the whole rebuild
// collapses into a single task, so the intermediate state a real browser sees
// for tens of milliseconds is never observable from a setTimeout.
//
// Both tests assert a SAMPLE COUNT before asserting on content. A run whose
// render died early takes zero samples and would otherwise pass while measuring
// nothing - the vacuous-green shape that already cost this investigation
// several trials.

const SCHEMA = {
  fields: [
    { key: "n_ctx", widget: "number", label: "Context window", help: "",
      group: "Engine", owner: "core", min: 512, step: 512, default: 4096 },
    { key: "mode", widget: "select", label: "Session persistence", help: "",
      group: "Privacy", owner: "core", options: ["privacy", "log", "full"],
      default: "log" },
    { key: "require_auth", widget: "toggle", label: "Require an API key",
      help: "", group: "Security", owner: "core", default: false },
  ],
};

// The three awaited builders in refreshSettingsPage. Each one's FIRST statement
// is an unconditional `await fetch(...)`, so all three are guaranteed suspension
// points rather than ones that merely happen to fire for this fixture.
const BUILDER_URLS = ["/v1/media/config", "/v1/tts/config", "/v1/plugins/settings"];

const MODELS = {
  models: [
    { name: "alpha.gguf", model_type: "llm", source: "local", size_bytes: 10, mtime: 1 },
    { name: "beta.gguf", model_type: "llm", source: "local", size_bytes: 20, mtime: 2 },
  ],
  active: "alpha.gguf",
};

const drain = async (n = 12) => {
  for (let i = 0; i < n; i++) await new Promise((r) => setTimeout(r, 0));
};

test("settings: a re-render never shrinks the page while it rebuilds", async () => {
  const samples = [];
  let sampling = false;
  let win = null;

  const res = loadAppWithPages({
    fetchImpl: async (url) => {
      const u = String(url);
      if (sampling && win && BUILDER_URLS.includes(u)) {
        const doc = win.document;
        samples.push({
          url: u,
          formChildren: doc.getElementById("config-form").childElementCount,
          // .settings-section is `display: none` in style.css and only
          // .settings-section.active is `display: block`, so this count IS the
          // page's visible height in the only sense that matters here.
          visibleSections:
            doc.querySelectorAll("#settings-content .settings-section.active").length,
          visibleInForm:
            doc.querySelectorAll("#config-form .settings-section.active").length,
          formTotal: doc.querySelectorAll("#config-form .settings-section").length,
        });
      }
      if (u === "/v1/config/schema") {
        return { ok: true, status: 200, json: async () => SCHEMA, text: async () => "" };
      }
      return {
        ok: true, status: 200, text: async () => "",
        json: async () => ({ models: [], active: "", conversations: [], plugins: [] }),
      };
    },
  });
  win = res.window;

  // FIRST render: the state the user is looking at, scrolled down.
  runScript(win, "refreshSettingsPage();");
  await drain();
  const visibleNow = () =>
    win.document.querySelectorAll("#settings-content .settings-section.active").length;
  const baseline = visibleNow();
  // The BASELINE is what makes this test able to fail. Counting "is anything at
  // all visible" is NOT enough: index.html ships static sections outside
  // #config-form (e.g. #sec-performance, data-group="model"), and one of those
  // keeps its .active right through a form rebuild. So the naive count never
  // reaches zero and the bug hides behind it - measured, this exact test passed
  // on the unfixed code until the assertion became a comparison against the
  // baseline. What the user experiences is the page SHRINKING, not emptying.
  assert.ok(baseline > 0,
    "precondition: the first render must leave visible sections, or the second "
    + "render has no visible state to destroy and this test measures nothing");

  // SECOND render - what clicking "Save <section>" actually does, since
  // saveSettingsSection calls refreshSettingsPage() on success.
  sampling = true;
  runScript(win, "refreshSettingsPage();");
  await drain();

  assert.equal(samples.length, BUILDER_URLS.length,
    "expected one sample per awaited builder, got " + samples.length
    + " (" + samples.map((s) => s.url).join(", ") + ") - a short count means the "
    + "render stopped early and the assertions below would be vacuous");

  for (const s of samples) {
    assert.ok(s.formChildren > 0,
      "the settings form was EMPTY while suspended on " + s.url);
    // ASSERT ON THE VISIBLE CONTENT, not on a class name or a call order: this is
    // the property the user experiences. A page that loses most of its height is
    // shorter than the viewport, so there is nothing left to scroll and the
    // browser clamps scrollTop to 0.
    assert.ok(s.visibleSections >= baseline,
      "the settings page LOST visible content while it suspended on " + s.url
      + ": " + baseline + " sections were on screen before the re-render, only "
      + s.visibleSections + " during it (" + s.formTotal + " rebuilt in the form, "
      + s.visibleInForm + " of them shown). The browser lays out during that "
      + "await, finds a page shorter than the viewport, and clamps scrollTop "
      + "to 0 - the settings scroll jump.");
  }
});

test("models: a re-render never empties the table it is rebuilding", async () => {
  const samples = [];
  let sampling = false;
  let win = null;

  const res = loadAppWithPages({
    fetchImpl: async (url) => {
      const u = String(url);
      if (sampling && win && u.startsWith("/api/models")) {
        samples.push({
          children: win.document.getElementById("models-table").childElementCount,
        });
      }
      if (u.startsWith("/api/models")) {
        return { ok: true, status: 200, json: async () => MODELS, text: async () => "" };
      }
      return {
        ok: true, status: 200, text: async () => "",
        json: async () => ({ models: [], active: "", conversations: [], plugins: [] }),
      };
    },
  });
  win = res.window;

  runScript(win, "refreshModelsPage();");
  await drain();
  assert.ok(win.document.getElementById("models-table").childElementCount > 0,
    "precondition: the first render must put something in the table, or there is "
    + "nothing for the second render to blank and this test measures nothing");

  // A sort, a type-tab click and every row action all re-enter here.
  sampling = true;
  runScript(win, "refreshModelsPage();");
  await drain();

  assert.ok(samples.length > 0,
    "expected at least one /api/models fetch during the re-render; none means the "
    + "render stopped early and the assertion below would be vacuous");

  for (const s of samples) {
    assert.ok(s.children > 0,
      "the models table was EMPTY while the re-render waited on /api/models. An "
      + "empty scroll container has nothing to scroll, so the browser clamps "
      + "scrollTop to 0 and the rows arrive too late to put it back.");
  }
});

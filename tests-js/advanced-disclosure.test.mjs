// SPDX-License-Identifier: AGPL-3.0-or-later
// The "Advanced" disclosure (.adv-fold) on chat's #params drawer and the three
// Studio generation forms.

import test from "node:test";
import assert from "node:assert/strict";
import { loadApp, loadAppWithPages, runScript } from "./harness.mjs";

// Every field, by surface, as the spec assigns it.
const SURFACES = {
  "chat #params": {
    fold: "p-advanced",
    advanced: ["p-top-p", "p-top-k", "p-repeat-penalty", "p-max-tokens", "p-seed", "p-grammar"],
    primary: ["p-persona", "p-system", "p-temperature", "p-web", "p-memory", "p-speak", "p-kb"],
  },
  "studio images": {
    fold: "img-advanced",
    advanced: ["img-seed", "img-guidance", "img-denoise", "img-cfg",
               "img-lora-strength-model", "img-lora-strength-clip"],
    primary: ["img-prompt", "img-negative", "img-lora", "img-input", "img-reload-llm"],
  },
  "studio music": {
    fold: "music-advanced",
    advanced: ["music-seed", "music-steps", "music-cfg"],
    primary: ["music-tags", "music-lyrics", "music-duration"],
  },
  "studio video": {
    fold: "video-advanced",
    advanced: ["video-width", "video-height", "video-seed", "video-steps", "video-cfg"],
    primary: ["video-prompt", "video-negative", "video-seconds", "video-fps", "video-image"],
  },
};

for (const [label, spec] of Object.entries(SURFACES)) {
  test(`${label}: advanced fields sit inside the fold, primary fields do not`, () => {
    const { window: win } = loadApp();
    const doc = win.document;
    const fold = doc.getElementById(spec.fold);
    assert.ok(fold, `${spec.fold} should exist`);
    assert.equal(fold.tagName, "DETAILS");
    assert.ok(fold.classList.contains("adv-fold"));

    for (const id of spec.advanced) {
      const f = doc.getElementById(id);
      assert.ok(f, `${id} should exist`);
      assert.equal(f.closest("details.adv-fold")?.id, spec.fold,
        `${id} is specified as advanced, so it must sit inside #${spec.fold}`);
    }
    for (const id of spec.primary) {
      const f = doc.getElementById(id);
      assert.ok(f, `${id} should exist`);
      assert.equal(f.closest("details.adv-fold"), null,
        `${id} is specified as always-visible, so it must NOT be inside any fold`);
    }
  });

  test(`${label}: the fold starts closed and its summary opens it`, () => {
    const { window: win } = loadApp();
    const fold = win.document.getElementById(spec.fold);
    assert.equal(fold.open, false, "a fold must start collapsed");
    fold.querySelector("summary").click();
    assert.equal(fold.open, true, "clicking the summary opens it");
    fold.querySelector("summary").click();
    assert.equal(fold.open, false, "clicking again closes it");
  });

  test(`${label}: the summary names what is inside, so it can be judged unopened`, () => {
    const { window: win } = loadApp();
    const summary = win.document.getElementById(spec.fold).querySelector("summary");
    assert.match(summary.textContent, /Advanced/);
    const hint = summary.querySelector(".adv-fold-hint");
    assert.ok(hint && hint.textContent.trim().length > 0,
      "a bare 'Advanced' tells the user nothing about whether to open it");
  });
}

test("no fold opens on a bare page load", () => {
  const { window: win } = loadApp();
  runScript(win, "window.__opened = revealFilledAdvanced(document);");
  assert.equal(win.__opened, 0, "a freshly loaded page has nothing set to reveal");
  for (const fold of win.document.querySelectorAll("details.adv-fold")) {
    assert.equal(fold.open, false, `#${fold.id} must be shut on a fresh load`);
  }
});

test("revealFilledAdvanced opens a fold holding a value, and only that fold", () => {
  const { window: win } = loadApp();
  win.document.getElementById("video-seed").value = "1234";
  runScript(win, "window.__opened = revealFilledAdvanced(document);");
  assert.equal(win.__opened, 1);
  assert.equal(win.document.getElementById("video-advanced").open, true);
  assert.equal(win.document.getElementById("p-advanced").open, false,
    "an unrelated surface's fold must stay shut");
});

test("a select resting on its blank option does not count as set", () => {
  // img-lora's first option is value="" ("None").
  const { window: win } = loadApp();
  const lora = win.document.getElementById("img-lora");
  assert.equal(lora.value, "", "precondition: the LoRA select starts blank");
  runScript(win, "window.__opened = revealFilledAdvanced(document);");
  assert.equal(win.__opened, 0);
});

test("the 'n set' badge counts values while the fold is closed", () => {
  const { window: win } = loadApp();
  const fold = win.document.getElementById("video-advanced");
  const badge = fold.querySelector(".adv-fold-count");
  assert.equal(badge.hidden, true, "nothing set: no badge");

  win.document.getElementById("video-seed").value = "7";
  win.document.getElementById("video-steps").value = "12";
  runScript(win, "updateAdvancedCount(document.getElementById('video-advanced'));");
  assert.equal(badge.hidden, false);
  assert.equal(badge.textContent, "2 set",
    "a collapsed fold holding live overrides must say so, not read as empty");

  win.document.getElementById("video-seed").value = "";
  win.document.getElementById("video-steps").value = "";
  runScript(win, "updateAdvancedCount(document.getElementById('video-advanced'));");
  assert.equal(badge.hidden, true, "clearing the values clears the badge");
});

test("typing into a folded field updates its badge without any per-fold wiring", () => {
  const { window: win } = loadApp();
  const seed = win.document.getElementById("music-seed");
  seed.value = "99";
  seed.dispatchEvent(new win.Event("input", { bubbles: true }));
  const badge = win.document.getElementById("music-advanced").querySelector(".adv-fold-count");
  assert.equal(badge.textContent, "1 set");
});

/* ---- the writers: applyPersona and "reuse settings" ---- */

test("applyPersona reveals the sampling values it just restored", () => {
  const { window: win } = loadApp();
  // personaCache is a top-level `export let`, not on window: seed it from a
  // script in the same realm.
  runScript(win, `personaCache = [{
    name: "tight", system: "Be terse.",
    params: { temperature: 0.2, top_k: 20, max_tokens: 512 },
  }];`);
  const fold = win.document.getElementById("p-advanced");
  assert.equal(fold.open, false, "precondition: the fold starts shut");

  runScript(win, 'window.__applied = applyPersona("tight");');
  assert.equal(win.__applied, true, "the persona should apply");
  assert.equal(win.document.getElementById("p-top-k").value, "20",
    "precondition: the persona really did write a folded field");
  assert.equal(fold.open, true,
    "applyPersona toasts success, so the values it restored must be visible");
});

test("applyPersona leaves the fold shut when it restored nothing advanced", () => {
  // applyPersona writes "" for unset params.
  const { window: win } = loadApp();
  runScript(win, `personaCache = [{
    name: "warm", system: "Be friendly.", params: { temperature: 0.9 },
  }];`);
  runScript(win, 'applyPersona("warm");');
  assert.equal(win.document.getElementById("p-temperature").value, "0.9");
  assert.equal(win.document.getElementById("p-advanced").open, false,
    "a persona that set nothing advanced must not expand the fold");
});

test("'reuse settings' reveals the seed and guidance it just restored", () => {
  // showImageDetail calls fetchImageURL(...).then(...) with no .catch, so the
  // fetch stub must resolve.
  const { window: win } = loadAppWithPages({
    fetchImpl: async () => ({
      ok: true, status: 200, json: async () => ({}), text: async () => "",
      blob: async () => ({ type: "image/png" }),
    }),
  });
  runScript(win, `showImageDetail({
    name: "fox.png",
    meta: { prompt: "a fox", seed: 4242, guidance: 3.5, denoise: null },
  });`);
  const reuse = [...win.document.querySelectorAll("button")]
    .find((b) => b.textContent === "reuse settings");
  assert.ok(reuse, "the reuse button renders only when item.meta.prompt is set");

  const fold = win.document.getElementById("img-advanced");
  assert.equal(fold.open, false, "precondition: the fold starts shut");
  reuse.click();
  assert.equal(win.document.getElementById("img-seed").value, "4242",
    "precondition: reuse really did write a folded field");
  assert.equal(fold.open, true,
    "reuse toasts 'Settings restored', so the restored seed must be visible");
});

test("'reuse settings' reveals the cfg it just restored (S4: CLI/GUI parity)", () => {
  const { window: win } = loadAppWithPages({
    fetchImpl: async () => ({
      ok: true, status: 200, json: async () => ({}), text: async () => "",
      blob: async () => ({ type: "image/png" }),
    }),
  });
  runScript(win, `showImageDetail({
    name: "fox.png",
    meta: { prompt: "a fox", negative_prompt: "blurry", cfg: 4.2 },
  });`);
  const reuse = [...win.document.querySelectorAll("button")]
    .find((b) => b.textContent === "reuse settings");

  const fold = win.document.getElementById("img-advanced");
  assert.equal(fold.open, false, "precondition: the fold starts shut");
  reuse.click();
  assert.equal(win.document.getElementById("img-cfg").value, "4.2",
    "precondition: reuse really did write cfg into its folded field");
  assert.equal(fold.open, true,
    "reuse toasts 'Settings restored', so the restored cfg must be visible");
});

test("'reuse settings' leaves the fold shut for an entry with no advanced values", () => {
  const { window: win } = loadAppWithPages({
    fetchImpl: async () => ({
      ok: true, status: 200, json: async () => ({}), text: async () => "",
      blob: async () => ({ type: "image/png" }),
    }),
  });
  runScript(win, `showImageDetail({
    name: "plain.png",
    meta: { prompt: "a fox", seed: null, guidance: null, denoise: null, cfg: null,
            lora_strength_model: null, lora_strength_clip: null },
  });`);
  const reuse = [...win.document.querySelectorAll("button")]
    .find((b) => b.textContent === "reuse settings");
  reuse.click();
  assert.equal(win.document.getElementById("img-prompt").value, "a fox",
    "precondition: the primary fields still got restored");
  assert.equal(win.document.getElementById("img-advanced").open, false,
    "nothing advanced was restored, so nothing should expand");
});

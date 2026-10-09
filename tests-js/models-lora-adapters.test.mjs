// SPDX-License-Identifier: AGPL-3.0-or-later
// The Models page shows GGUF LoRA adapters: which base each is attached to, which
// adapters the loaded model runs with, and an attach/detach flow whose refusals
// (a mismatched architecture) are shown where the user is looking.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

const settle = () => new Promise((r) => setTimeout(r, 0));

function makeFetch(models, calls, { attach = null, detach = null, info = null } = {}) {
  return async (url, opts = {}) => {
    const u = String(url);
    if (u.startsWith("/api/models/adapters/attach")) {
      calls.push({ route: "attach", body: JSON.parse(opts.body) });
      const r = attach || { ok: true, body: { status: "attached", needs_reload: false } };
      return { ok: r.ok, status: r.ok ? 200 : 400, json: async () => (r.ok ? r.body : { detail: r.detail }) };
    }
    if (u.startsWith("/api/models/adapters/detach")) {
      calls.push({ route: "detach", body: JSON.parse(opts.body) });
      const r = detach || { ok: true, body: { status: "detached", needs_reload: false } };
      return { ok: r.ok, status: r.ok ? 200 : 404, json: async () => (r.ok ? r.body : { detail: r.detail }) };
    }
    if (u.startsWith("/v1/models/") && info) {
      return { ok: true, status: 200, json: async () => info };
    }
    if (u === "/api/models" || u.startsWith("/api/models?")) {
      return { ok: true, status: 200, json: async () => ({ models, active: null }) };
    }
    return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
  };
}

const BASE = { name: "qwen3-0.6b", active: false, loaded: false, model_type: "llm", size_bytes: 100 };
const OTHER = { name: "tiny-chat", active: false, loaded: false, model_type: "llm", size_bytes: 50 };
const FREE_ADAPTER = {
  name: "style-lora", active: false, loaded: false, model_type: "lora", size_bytes: 10,
  adapter: true, adapter_file: "style-lora.gguf",
};
const ATTACHED_ADAPTER = {
  name: "tone-lora", active: false, loaded: false, model_type: "lora", size_bytes: 10,
  adapter: true, adapter_file: "tone-lora.gguf", base: "qwen3-0.6b", base_registered: true, scale: 0.8,
};

async function boot(models, calls = [], opts = {}) {
  const toasts = [];
  const { window } = loadAppWithPages({ fetchImpl: makeFetch(models, calls, opts) });
  window.toast = (msg, isError) => toasts.push({ msg: String(msg), isError: !!isError });
  await window.refreshModelsPage();
  await settle();
  return { window, toasts, calls };
}

function row(window, name) {
  const tr = [...window.document.querySelectorAll("#models-table tbody tr")]
    .find((r) => r.querySelector(".name")?.textContent === name);
  assert.ok(tr, `no row rendered for ${name}`);
  return tr;
}
const badges = (window, name) => [...row(window, name).querySelectorAll(".adapter-badge")];
const buttonNamed = (window, name, text) =>
  [...row(window, name).querySelectorAll("button")].find((b) => b.textContent === text);

test("an attached adapter row names its base and scale and offers change and detach", async () => {
  const { window } = await boot([BASE, ATTACHED_ADAPTER]);
  const b = badges(window, "tone-lora");
  assert.equal(b.length, 1);
  assert.equal(b[0].textContent, "attached to qwen3-0.6b ×0.8");
  assert.ok(buttonNamed(window, "tone-lora", "change"));
  assert.ok(buttonNamed(window, "tone-lora", "detach"));
  assert.equal(buttonNamed(window, "tone-lora", "attach"), undefined);
});

test("an unattached adapter row says so and offers attach but not detach", async () => {
  const { window } = await boot([BASE, FREE_ADAPTER]);
  assert.equal(badges(window, "style-lora")[0].textContent, "not attached");
  assert.ok(buttonNamed(window, "style-lora", "attach"));
  assert.equal(buttonNamed(window, "style-lora", "detach"), undefined);
});

test("an adapter whose base is no longer registered is flagged", async () => {
  const gone = { ...ATTACHED_ADAPTER, base_registered: false };
  const { window } = await boot([gone]);
  const b = badges(window, "tone-lora")[0];
  assert.ok(b.classList.contains("adapter-warn"));
  assert.match(b.title, /no longer registered/);
});

test("a plain model carries no adapter badge and no adapter buttons", async () => {
  const { window } = await boot([BASE]);
  assert.equal(badges(window, "qwen3-0.6b").length, 0);
  assert.equal(buttonNamed(window, "qwen3-0.6b", "attach"), undefined);
});

test("a base that is not loaded counts its attached adapters", async () => {
  const base = { ...BASE, adapters: [{ name: "tone-lora", file: "tone-lora.gguf", scale: 0.8 }] };
  const { window } = await boot([base]);
  const b = badges(window, "qwen3-0.6b");
  assert.equal(b.length, 1);
  assert.equal(b[0].textContent, "1 LoRA attached");
  assert.match(b[0].title, /tone-lora ×0\.8/);
});

test("a loaded base running exactly its attached adapters shows each as active", async () => {
  const base = {
    ...BASE, loaded: true,
    adapters: [{ name: "tone-lora", file: "tone-lora.gguf", scale: 0.8 }],
    applied_adapters: [{ name: "tone-lora.gguf", scale: 0.8 }],
  };
  const { window } = await boot([base]);
  const b = badges(window, "qwen3-0.6b");
  assert.equal(b.length, 1);
  assert.ok(b[0].classList.contains("adapter-active"));
  assert.equal(b[0].textContent, "LoRA tone-lora.gguf ×0.8");
});

test("a loaded base whose attached set differs from what it runs asks for a reload", async () => {
  const base = {
    ...BASE, loaded: true,
    adapters: [{ name: "tone-lora", file: "tone-lora.gguf", scale: 0.8 }],
    applied_adapters: [],
  };
  const { window } = await boot([base]);
  const b = badges(window, "qwen3-0.6b");
  assert.equal(b.length, 1);
  assert.ok(b[0].classList.contains("adapter-warn"));
  assert.match(b[0].textContent, /reload/);
});

test("a base running under another of its names counts as live and shows its adapters", async () => {
  const base = {
    ...BASE, loaded: false, adapter_resident: true,
    adapters: [{ name: "tone-lora", file: "tone-lora.gguf", scale: 0.8 }],
    applied_adapters: [{ name: "tone-lora.gguf", scale: 0.8 }],
  };
  const { window } = await boot([base]);
  const b = badges(window, "qwen3-0.6b");
  assert.equal(b.length, 1);
  assert.ok(b[0].classList.contains("adapter-active"));
});

test("a base running under another of its names with different adapters asks for a reload", async () => {
  const base = {
    ...BASE, loaded: false, adapter_resident: true,
    adapters: [{ name: "tone-lora", file: "tone-lora.gguf", scale: 0.8 }],
    applied_adapters: [],
  };
  const { window } = await boot([base]);
  assert.match(badges(window, "qwen3-0.6b")[0].textContent, /reload/);
});

test("a loaded base still running an adapter that was since detached asks for a reload", async () => {
  const base = { ...BASE, loaded: true, applied_adapters: [{ name: "tone-lora.gguf", scale: 1 }] };
  const { window } = await boot([base]);
  const b = badges(window, "qwen3-0.6b");
  assert.equal(b.length, 1);
  assert.match(b[0].textContent, /reload/);
});

test("the attach dialog offers only chat models as bases and posts adapter, base and scale", async () => {
  const embedding = { name: "embedder", active: false, loaded: false, model_type: "embedding" };
  const calls = [];
  const { window, toasts } = await boot([BASE, OTHER, embedding, FREE_ADAPTER], calls);
  buttonNamed(window, "style-lora", "attach").onclick();
  const sel = window.document.getElementById("adapter-base");
  assert.deepEqual([...sel.options].map((o) => o.value).sort(), ["qwen3-0.6b", "tiny-chat"]);
  sel.value = "tiny-chat";
  window.document.getElementById("adapter-scale").value = "0.5";
  window.document.querySelector(".adapter-attach-confirm").onclick();
  await settle(); await settle();
  assert.deepEqual(calls, [{ route: "attach", body: { adapter: "style-lora", base: "tiny-chat", scale: 0.5 } }]);
  assert.equal(window.document.getElementById("modal").style.display, "none");
  assert.ok(toasts.some((t) => !t.isError && t.msg === "'style-lora' attached to 'tiny-chat'"),
    JSON.stringify(toasts));
});

test("changing an attached adapter preselects its current base and scale", async () => {
  const { window } = await boot([BASE, OTHER, ATTACHED_ADAPTER]);
  buttonNamed(window, "tone-lora", "change").onclick();
  assert.equal(window.document.getElementById("adapter-base").value, "qwen3-0.6b");
  assert.equal(window.document.getElementById("adapter-scale").value, "0.8");
});

test("a refusal from the server is shown inside the dialog and the dialog stays open", async () => {
  const detail = "cannot attach 'style-lora' to 'qwen3-0.6b': adapter architecture 'llama' does not match 'qwen3'";
  const { window, toasts } = await boot([BASE, FREE_ADAPTER], [], { attach: { ok: false, detail } });
  buttonNamed(window, "style-lora", "attach").onclick();
  window.document.querySelector(".adapter-attach-confirm").onclick();
  await settle(); await settle();
  const err = window.document.querySelector(".adapter-error");
  assert.equal(err.hidden, false);
  assert.equal(err.textContent, detail);
  assert.equal(window.document.getElementById("modal").style.display, "flex");
  assert.equal(toasts.length, 0, "a refusal is not also reported as a toast");
});

test("a scale of zero is refused in the dialog without a request", async () => {
  const calls = [];
  const { window } = await boot([BASE, FREE_ADAPTER], calls);
  buttonNamed(window, "style-lora", "attach").onclick();
  window.document.getElementById("adapter-scale").value = "0";
  window.document.querySelector(".adapter-attach-confirm").onclick();
  await settle();
  assert.deepEqual(calls, []);
  assert.match(window.document.querySelector(".adapter-error").textContent, /other than 0/);
});

test("attaching to a loaded base tells the user to reload it", async () => {
  const { window, toasts } = await boot([BASE, FREE_ADAPTER], [],
    { attach: { ok: true, body: { status: "attached", needs_reload: true } } });
  buttonNamed(window, "style-lora", "attach").onclick();
  window.document.querySelector(".adapter-attach-confirm").onclick();
  await settle(); await settle();
  assert.ok(toasts.some((t) => /Unload and load 'qwen3-0\.6b' again/.test(t.msg)), JSON.stringify(toasts));
});

test("with no chat model registered the dialog says there is nothing to attach to", async () => {
  const { window } = await boot([FREE_ADAPTER]);
  buttonNamed(window, "style-lora", "attach").onclick();
  assert.equal(window.document.getElementById("adapter-base"), null);
  assert.match(window.document.getElementById("modal-body").textContent, /No chat models are registered/);
});

test("detach posts the adapter name and reports it", async () => {
  const calls = [];
  const { window, toasts } = await boot([BASE, ATTACHED_ADAPTER], calls);
  await buttonNamed(window, "tone-lora", "detach").onclick();
  await settle();
  assert.deepEqual(calls, [{ route: "detach", body: { adapter: "tone-lora" } }]);
  assert.ok(toasts.some((t) => !t.isError && t.msg === "'tone-lora' detached"), JSON.stringify(toasts));
});

test("a failed detach is an error toast, not a success", async () => {
  const { window, toasts } = await boot([BASE, ATTACHED_ADAPTER], [],
    { detach: { ok: false, detail: "'tone-lora' is not an attached adapter" } });
  await buttonNamed(window, "tone-lora", "detach").onclick();
  await settle();
  assert.deepEqual(toasts, [{ msg: "'tone-lora' is not an attached adapter", isError: true }]);
});

test("model details list the adapters the loaded model runs with", async () => {
  const info = {
    path: "x.gguf", model_type: "llm", source: "", size_bytes: 1, sha256: "", aliases: [],
    active: true, loaded: true, adapters: [{ name: "tone-lora.gguf", scale: 0.8 }],
  };
  const { window } = await boot([BASE], [], { info });
  await window.showModelDetail("qwen3-0.6b");
  const text = window.document.getElementById("modal-body").textContent;
  assert.match(text, /LoRA adapters/);
  assert.match(text, /tone-lora\.gguf ×0\.8/);
});

test("model details have no adapter row when none is applied", async () => {
  const info = {
    path: "x.gguf", model_type: "llm", source: "", size_bytes: 1, sha256: "", aliases: [],
    active: true, loaded: true,
  };
  const { window } = await boot([BASE], [], { info });
  await window.showModelDetail("qwen3-0.6b");
  assert.doesNotMatch(window.document.getElementById("modal-body").textContent, /LoRA adapters/);
});

// SPDX-License-Identifier: AGPL-3.0-or-later
// The native media model pickers (app/native-models.js) on the Images, Video
// and Music pages, and the registered-model dropdowns Settings puts in front of
// the native model fields (pages/settings.js enhanceNativeModelFields).
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

const IMAGE_OPTIONS = {
  models: [
    { name: "fluxunchained", file: "fluxunchained-dev-q8-0-v2.gguf", kind: "diffusion",
      family: "flux", family_label: "FLUX.1" },
    { name: "sd-turbo", file: "sd_turbo-f16-q8_0.gguf", kind: "checkpoint", family: "sd2",
      family_label: "Stable Diffusion 2.x" },
  ],
  slots: {
    clip_l: [{ name: "clip_l", file: "clip_l.safetensors", kind: "clip_l" },
             { name: "ViT-L-14-TEXT", file: "ViT-L-14-TEXT.safetensors", kind: "clip_l" }],
    clip_g: [], llm: [],
    t5xxl: [{ name: "t5xxl_fp8", file: "t5xxl_fp8.safetensors", kind: "t5xxl" }],
    vae: [{ name: "ae", file: "ae.safetensors", kind: "vae", family: "flux", family_label: "FLUX.1" }],
  },
  loras: [{ name: "aidma-flux", file: "aidma.safetensors", kind: "lora", family: "flux" }],
  current: { model: "", clip_l: "", clip_g: "", t5xxl: "", llm: "", vae: "" },
};

function preview(model, extra = {}) {
  if (!model) return null;
  if (model === "sd-turbo") {
    return { model, kind: "checkpoint", family: "sd2", family_label: "Stable Diffusion 2.x",
             needs: [], components: {}, missing: [], notes: [], error: null };
  }
  return { model, kind: "diffusion", family: "flux", family_label: "FLUX.1",
           needs: ["clip_l", "t5xxl", "vae"],
           components: {
             clip_l: { name: extra.clip_l || "clip_l", auto: !extra.clip_l },
             t5xxl: { name: "t5xxl_fp8", auto: true }, vae: { name: "ae", auto: true } },
           missing: [], notes: [], error: null };
}

function makeFetch(calls, { status = 200 } = {}) {
  return async (url) => {
    const u = String(url);
    calls.push(u);
    if (u.includes("/native-models")) {
      const q = new URL(u, "http://x").searchParams;
      if (status === 400 && q.get("model") === "gone") {
        return { ok: false, status: 400, json: async () => ({ detail: "'gone' is not a registered model." }) };
      }
      const body = { ...IMAGE_OPTIONS, preview: preview(q.get("model"), { clip_l: q.get("clip_l") }) };
      if (u.includes("/api/music/")) {
        return { ok: true, status: 200, json: async () => ({
          components: { dit: [{ name: "acestep-dit", file: "dit.gguf" }], text_encoder: [],
                        vae: [{ name: "ace-vae", file: "vae.gguf" }], lm: [] },
          current: { dit: "", text_encoder: "", vae: "", lm: "" } }) };
      }
      return { ok: true, status: 200, json: async () => body };
    }
    if (u.includes("/api/imagine/backend")) {
      return { ok: true, status: 200, json: async () => ({ choice: "auto", active: "native",
        native: { model: null, runtime: "vulkan", recommended: { name: "sd-turbo" } } }) };
    }
    return { ok: true, status: 200, text: async () => "",
             json: async () => ({ models: [], active: "", conversations: [], plugins: [] }) };
  };
}

async function settle(win, code) {
  runScript(win, `globalThis.__p = (async () => { ${code} })();`);
  await win.__p;
}

function app(opts = {}) {
  const calls = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(calls, opts), ...opts });
  return { win, calls, $: (id) => win.document.getElementById(id) };
}

test("native image backend shows the picker; a Flux model lists its companion slots", async () => {
  const { win, $ } = app();
  await settle(win, "await refreshImageBackend(); await imageModelPicker.refresh();");
  const mount = $("img-native-models");
  assert.equal(mount.hidden, false);
  const selects = mount.querySelectorAll("select");
  assert.equal(selects.length, 1);
  const model = selects[0];
  assert.deepEqual([...model.querySelectorAll("optgroup")].map((g) => g.label),
                   ["FLUX.1", "Stable Diffusion 2.x"]);
  model.value = "fluxunchained";
  model.dispatchEvent(new win.Event("change"));
  await settle(win, "await imageModelPicker.refresh();");
  const slots = [...mount.querySelectorAll("select[data-slot]")].map((s) => s.dataset.slot);
  assert.deepEqual(slots, ["clip_l", "t5xxl", "vae"]);
  const clip = mount.querySelector('select[data-slot="clip_l"]');
  assert.equal(clip.options[0].textContent, "Automatic (clip_l)");
  assert.match(mount.textContent, /FLUX\.1 model: uses CLIP-L text encoder, T5 text encoder, VAE\./);
});

test("the choice sends only what was picked, and persists per page", async () => {
  const { win, $ } = app();
  await settle(win, "await imageModelPicker.setModels({ model: 'fluxunchained', clip_l: 'ViT-L-14-TEXT' });");
  await settle(win, "imageModelPicker.setLoras([{ name: 'aidma-flux', strength: 0.7 }, { name: '' }]);");
  runScript(win, "globalThis.__c = JSON.stringify(imageModelPicker.choice());");
  assert.deepEqual(JSON.parse(win.__c), {
    native_models: { model: "fluxunchained", clip_l: "ViT-L-14-TEXT" },
    loras: [{ name: "aidma-flux", strength: 0.7 }],
  });
  const saved = JSON.parse(win.localStorage.getItem("localm.nativeModels.image"));
  assert.equal(saved.models.clip_l, "ViT-L-14-TEXT");
  const clip = $("img-native-models").querySelector('select[data-slot="clip_l"]');
  assert.equal(clip.value, "ViT-L-14-TEXT");
});

test("choosing Automatic clears a slot and Settings default clears the model", async () => {
  const { win, $ } = app();
  await settle(win, "await imageModelPicker.setModels({ model: 'fluxunchained', clip_l: 'ViT-L-14-TEXT' });");
  const clip = $("img-native-models").querySelector('select[data-slot="clip_l"]');
  clip.value = "";
  clip.dispatchEvent(new win.Event("change"));
  runScript(win, "globalThis.__c = JSON.stringify(imageModelPicker.choice());");
  assert.deepEqual(JSON.parse(win.__c), { native_models: { model: "fluxunchained" } });
  const model = $("img-native-models").querySelector("select");
  model.value = "";
  model.dispatchEvent(new win.Event("change"));
  runScript(win, "globalThis.__c = JSON.stringify(imageModelPicker.choice());");
  assert.deepEqual(JSON.parse(win.__c), {});
});

test("a saved choice the server refuses is dropped instead of breaking the page", async () => {
  const { win, $ } = app({ status: 400,
    seedLocalStorage: { "localm.nativeModels.image": JSON.stringify({ models: { model: "gone" }, loras: [] }) } });
  await settle(win, "await imageModelPicker.refresh();");
  runScript(win, "globalThis.__c = JSON.stringify(imageModelPicker.choice());");
  assert.deepEqual(JSON.parse(win.__c), {});
  assert.ok($("img-native-models").querySelector("select"));
});

test("unreadable stored state falls back to an empty choice", async () => {
  const { win } = app({ seedLocalStorage: { "localm.nativeModels.image": "{not json" } });
  runScript(win, "globalThis.__c = JSON.stringify(imageModelPicker.choice());");
  assert.deepEqual(JSON.parse(win.__c), {});
});

test("music picker offers each ACE-Step component and sends the picks", async () => {
  const { win, $ } = app();
  await settle(win, "await musicModelPicker.refresh();");
  const dit = $("music-native-models").querySelector('select[data-slot="dit"]');
  assert.deepEqual([...dit.options].map((o) => o.value), ["", "acestep-dit"]);
  dit.value = "acestep-dit";
  dit.dispatchEvent(new win.Event("change"));
  runScript(win, "globalThis.__c = JSON.stringify(musicModelPicker.choice());");
  assert.deepEqual(JSON.parse(win.__c), { native_models: { dit: "acestep-dit" } });
});

test("Settings puts a registered-model dropdown in front of a native model field", async () => {
  const { win } = app();
  runScript(win, `
    globalThis.__ctrl = buildSettingControl({ key: "native_model", widget: "text", label: "Native model",
                                              default: "" });
    globalThis.__path = buildSettingControl({ key: "native_vae", widget: "text", label: "VAE",
                                              default: "D:/m/my_vae.safetensors" });
    document.body.append(__ctrl.node, __path.node);`);
  await settle(win, "await enhanceNativeModelFields('image', [__ctrl, __path]);");
  const sel = win.__ctrl.node.querySelector("select.native-model-select");
  const input = win.__ctrl.node.querySelector("input");
  assert.ok(sel && input.hidden);
  sel.value = "fluxunchained";
  sel.dispatchEvent(new win.Event("change"));
  assert.equal(input.value, "fluxunchained");
  assert.equal(win.__ctrl.read(), "fluxunchained");
  const psel = win.__path.node.querySelector("select.native-model-select");
  const pinput = win.__path.node.querySelector("input");
  assert.equal(psel.value, "__custom_path__");
  assert.equal(pinput.hidden, false);
  assert.equal(win.__path.read(), "D:/m/my_vae.safetensors");
});

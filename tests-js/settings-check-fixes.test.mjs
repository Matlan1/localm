// SPDX-License-Identifier: AGPL-3.0-or-later
// Settings and privacy controls that must report what really happened: deleting
// every conversation, saving the logo style, the new-key QR, the speech-model
// button, the media launcher hint and the web "ask" mode.
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

const settle = async (n = 12) => {
  for (let i = 0; i < n; i++) await new Promise((r) => setTimeout(r, 0));
};
const res = (body, status = 200) => ({
  ok: status < 400, status, json: async () => body, text: async () => "<svg></svg>",
});

/* ---- delete all conversations -------------------------------------------- */

/** A server store: GET pages it like the real route, DELETE removes a row unless
 *  the id is refused. */
function conversationStore(count, { refuse = [], unreadable = [], listStatus = 200, noOpDelete = [] } = {}) {
  const rows = [];
  for (let i = 0; i < count; i++) {
    rows.push({ id: `c${String(i).padStart(4, "0")}`, title: `chat ${i}`, updated_at: 10000 - i });
  }
  const store = { rows, unreadable: [...unreadable], deletes: [], lists: 0 };
  store.fetchImpl = async (url, opts = {}) => {
    const u = String(url);
    const method = (opts.method || "GET").toUpperCase();
    if (u.startsWith("/api/conversations?")) {
      store.lists += 1;
      if (store.lists > 1 && listStatus !== 200) return res({}, listStatus);
      const q = new URL(u, "http://x").searchParams;
      const offset = Number(q.get("offset") || 0);
      const limit = Number(q.get("limit") || 200);
      return res({ enabled: true, conversations: store.rows.slice(offset, offset + limit),
                   total: store.rows.length, unreadable: store.unreadable });
    }
    const m = u.match(/^\/api\/conversations\/([^?]+)$/);
    if (m && method === "DELETE") {
      const id = decodeURIComponent(m[1]);
      store.deletes.push(id);
      if (refuse.includes(id)) return res({ detail: "refused" }, 403);
      if (!noOpDelete.includes(id)) {
        store.rows = store.rows.filter((c) => c.id !== id);
        store.unreadable = store.unreadable.filter((x) => x !== id);
      }
      return res({ status: "deleted", id });
    }
    return res({ models: [], active: "", plugins: [], keys: [] });
  };
  return store;
}

async function clickDeleteAll(win) {
  runScript(win, `
    chat.persist = true;
    chat.privacy = false;
    chat.modeKnown = true;
    window.__toasts = [];
    const __t = toast;
    toast = (m, e) => { window.__toasts.push({ m: String(m), e: !!e }); __t(m, e); };
    confirmDanger = (a, b, c, fn) => { window.__done = fn(); };
  `);
  win.document.getElementById("gui-clear-convs").onclick();
  await win.__done;
}

test("delete all removes every stored conversation, past the first 200", async () => {
  const store = conversationStore(500);
  const { window: win } = loadAppWithPages({ fetchImpl: store.fetchImpl });
  await settle();
  win.localStorage.setItem("localm.conversations", JSON.stringify([{ id: "c0000", messages: [] }]));
  await clickDeleteAll(win);

  assert.equal(store.rows.length, 0, `${store.rows.length} conversations stayed on the server`);
  assert.equal(new Set(store.deletes).size, 500, "each conversation is deleted once");
  assert.equal(win.localStorage.getItem("localm.conversations"), null, "the browser copy is gone too");
  assert.ok(!win.__toasts.some((x) => x.e), JSON.stringify(win.__toasts));
});

test("delete all also removes files the list reports as unreadable", async () => {
  const store = conversationStore(3, { unreadable: ["broken1", "broken2"] });
  const { window: win } = loadAppWithPages({ fetchImpl: store.fetchImpl });
  await settle();
  await clickDeleteAll(win);
  assert.deepEqual(store.unreadable, [], "an unreadable chat file stayed on disk");
  assert.equal(store.rows.length, 0);
});

test("a refused delete still removes the rest, says how many remain and keeps the page", async () => {
  const refuse = ["c0000", "c0250"];
  const store = conversationStore(500, { refuse });
  const { window: win } = loadAppWithPages({ fetchImpl: store.fetchImpl });
  await settle();
  win.localStorage.setItem("localm.conversations", JSON.stringify([{ id: "c0000", messages: [] }]));
  runScript(win, `chat.conversations = [{ id: "c0000", messages: [] }, { id: "c0001", messages: [] }];
                  chat.activeId = "c0001";`);
  await clickDeleteAll(win);

  assert.deepEqual(store.rows.map((c) => c.id).sort(), refuse,
    "only the refused conversations may remain on the server");
  const errs = win.__toasts.filter((x) => x.e);
  assert.equal(errs.length, 1, JSON.stringify(win.__toasts));
  assert.match(errs[0].m, /^2 conversations could not be deleted/);
  assert.ok(win.localStorage.getItem("localm.conversations") !== null,
    "the failure path stays on the page instead of the success path's cache wipe");
  const left = win.eval("chat.conversations.map((c) => c.id)");
  assert.deepEqual([...left], ["c0000"], "the sidebar lists what is still stored");
  assert.equal(win.eval("chat.activeId"), "c0000");
});

test("a row still listed after an OK delete counts as a failure and the loop ends", async () => {
  const store = conversationStore(2, { noOpDelete: ["c0001"] });
  const { window: win } = loadAppWithPages({ fetchImpl: store.fetchImpl });
  await settle();
  await clickDeleteAll(win);
  assert.deepEqual(store.rows.map((c) => c.id), ["c0001"]);
  assert.match(win.__toasts.find((x) => x.e).m, /^1 conversation could not be deleted/);
});

test("a list that cannot be read is reported, never treated as an empty store", async () => {
  const store = conversationStore(300, { listStatus: 500 });
  const { window: win } = loadAppWithPages({ fetchImpl: store.fetchImpl });
  await settle();
  // The boot sequence may have used the first list call; this makes the next one fail.
  store.lists = 1;
  await clickDeleteAll(win);
  assert.equal(store.rows.length, 300, "nothing could be deleted");
  assert.match(win.__toasts.find((x) => x.e).m, /Could not read the server's list/);
});

/* ---- logo style ----------------------------------------------------------- */

test("a server that refuses the logo style says so instead of reporting success", async () => {
  const fetchImpl = async (url, opts = {}) => {
    if (String(url).includes("/v1/config") && opts.method === "PATCH") return res({}, 403);
    return res({ models: [], active: "", plugins: [] });
  };
  const { window: win } = loadAppWithPages({ fetchImpl });
  await settle();
  const saved = await win.eval('setLogoStyle("localm")');
  assert.equal(saved, false);
  const toastEl = win.document.getElementById("toast");
  assert.match(toastEl.textContent, /logo style could not be saved/);
  assert.equal(toastEl.className, "error");
});

test("a saved logo style reports true and raises no error", async () => {
  const fetchImpl = async () => res({ models: [], active: "", plugins: [] });
  const { window: win } = loadAppWithPages({ fetchImpl });
  await settle();
  assert.equal(await win.eval('setLogoStyle("localm")'), true);
  assert.notEqual(win.document.getElementById("toast").className, "error");
});

/* ---- new-key QR ----------------------------------------------------------- */

test("a QR the server will not render says so; the key stays usable", async () => {
  const fetchImpl = async (url, opts = {}) => {
    if (String(url) === "/api/pairing/qr" && opts.method === "POST") return res({}, 403);
    return res({ models: [], active: "", plugins: [] });
  };
  const { window: win } = loadAppWithPages({ fetchImpl });
  await settle();
  const box = win.document.createElement("div");
  await win.renderKeyQR(box, "k");
  assert.match(box.textContent, /QR code could not be shown/);
  assert.equal(box.querySelector(".key-qr"), null);
});

test("a rendered QR shows no unavailable note", async () => {
  const fetchImpl = async () => res({ models: [], active: "", plugins: [] });
  const { window: win } = loadAppWithPages({ fetchImpl });
  await settle();
  const box = win.document.createElement("div");
  await win.renderKeyQR(box, "k");
  assert.ok(box.querySelector(".key-qr"), "the QR renders");
  assert.doesNotMatch(box.textContent, /could not be shown/);
});

/* ---- speech model button -------------------------------------------------- */

const VOICE_SCHEMA = { fields: [
  { key: "voice_stt_language", widget: "text", label: "Language", help: "", group: "Voice",
    owner: "voice", value: "", default: "" },
] };

async function voiceBox(statusBody) {
  const fetchImpl = async (url) => {
    const u = String(url);
    if (u === "/v1/config/schema") return res(VOICE_SCHEMA);
    if (u === "/api/voice/status") return res(statusBody);
    return res({ models: [], active: "", plugins: [], conversations: [] });
  };
  const { window: win } = loadAppWithPages({ fetchImpl });
  await settle();
  runScript(win, 'caps.fsAccess = "host"; refreshSettingsPage();');
  await settle(30);
  const box = win.document.querySelector(".voice-model-setup-box");
  assert.ok(box, "the speech model box renders");
  return { status: box.querySelector(".voice-setup-status"),
           btn: box.querySelector(".voice-download-model-btn") };
}

test("a cached speech model offers no download button", async () => {
  const { status, btn } = await voiceBox({ available: true, model_cached: true, model: "base",
                                           can_download: false, reason: "", download_blocked: "" });
  assert.match(status.textContent, /downloaded and ready/);
  assert.equal(btn.style.display, "none");
});

test("a download the server refuses hides the button and shows its reason", async () => {
  const { status, btn } = await voiceBox({
    available: false, model_cached: false, model: "base", can_download: false,
    reason: "generic reason", download_blocked: "'huggingface.co' is on the deny list (net_deny).",
  });
  assert.equal(btn.style.display, "none");
  assert.match(status.textContent, /deny list/);
  assert.doesNotMatch(status.textContent, /generic reason/);
});

test("a missing package hides the button and shows the install reason", async () => {
  const { status, btn } = await voiceBox({
    available: false, model_cached: false, model: "", can_download: false,
    reason: "needs the faster-whisper package", download_blocked: "",
  });
  assert.equal(btn.style.display, "none");
  assert.match(status.textContent, /faster-whisper/);
});

test("a download the server allows shows the button", async () => {
  const { btn } = await voiceBox({ available: false, model_cached: false, model: "base",
                                   can_download: true, reason: "", download_blocked: "" });
  assert.notEqual(btn.style.display, "none");
  assert.match(btn.textContent, /Download speech model now/);
});

/* ---- media launcher hint -------------------------------------------------- */

const MEDIA_SCHEMA = { fields: [
  { key: "n_ctx", widget: "number", label: "Context window", help: "", group: "Engine", value: 0, default: 0 },
] };

async function mediaSubsection(launchField, routes = {}) {
  const calls = [];
  const fetchImpl = async (url, opts = {}) => {
    const u = String(url);
    const method = (opts.method || "GET").toUpperCase();
    calls.push(`${method} ${u}`);
    if (routes[`${method} ${u}`]) return routes[`${method} ${u}`]();
    if (u === "/v1/config/schema") return res(MEDIA_SCHEMA);
    if (u === "/v1/media/config" && method === "GET") {
      return res({ plugins: [{ plugin: "image", label: "Image", fields: [launchField] }] });
    }
    return res({ models: [], active: "", plugins: [], conversations: [] });
  };
  const { window: win } = loadAppWithPages({ fetchImpl });
  await settle();
  runScript(win, 'caps.fsAccess = "host"; refreshSettingsPage();');
  await settle(30);
  const sub = [...win.document.querySelectorAll(".media-subsection")]
    .find((s) => s.dataset.plugin === "image");
  assert.ok(sub, "the image subsection renders");
  return { win, sub, calls };
}

test("a blank launch command shows the launcher localm detected", async () => {
  const { sub } = await mediaSubsection({
    key: "launch_cmd", widget: "text", label: "Launch", help: "", value: "",
    is_override: false, auto: '"C:/comfy/run.bat"',
  });
  const input = sub.querySelector('[data-key="launch_cmd"]');
  assert.equal(input.value, '"C:/comfy/run.bat"');
  assert.ok(input.classList.contains("auto-detected"));
});

test("Create launch script appears translated and posts its endpoint", async () => {
  const endpoint = "/api/comfyui/create-launcher?workdir=C%3A%2Fcomfy";
  const { win, sub, calls } = await mediaSubsection({
    key: "launch_cmd", widget: "text", label: "Launch", help: "", value: "", is_override: false,
    auto: "",
    action: { label: "server text", endpoint, success_msg: "server done" },
  });
  const btn = [...sub.querySelectorAll("button")].find((b) => /launch script/i.test(b.textContent));
  assert.ok(btn, "the action button renders");
  assert.equal(btn.textContent, "Create launch script", "the catalog text wins over the server's");
  await btn.onclick();
  await settle();
  assert.ok(calls.includes(`POST ${endpoint}`), calls.join("\n"));
  assert.match(win.document.getElementById("toast").textContent, /Launcher script created/);
});

/* ---- web ask mode --------------------------------------------------------- */

for (const [name, reply, expected] of [
  ["a key without config:read", () => res({}, 403), true],
  ["an unreachable server", () => { throw new Error("down"); }, true],
  ["a reply with no net_mode", () => res({}), true],
  ["ask", () => res({ net_mode: "ask" }), true],
  ["allow", () => res({ net_mode: "allow" }), false],
]) {
  test(`webModeIsAsk: ${name} -> ${expected}`, async () => {
    let armed = false;
    const fetchImpl = async (url) => {
      if (armed && String(url) === "/v1/config") return reply();
      return res({ models: [], active: "", plugins: [] });
    };
    const { window: win } = loadAppWithPages({ fetchImpl });
    await settle();
    armed = true;
    assert.equal(await win.webModeIsAsk(), expected);
  });
}

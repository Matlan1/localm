// SPDX-License-Identifier: AGPL-3.0-or-later
// Page load reads a plugin's routes only when /api/capabilities lists the plugin
// as active; the server-down overlay says when the server was shut down on
// request; "Reset and re-enter key" signs the browser out on the server and says
// when it could not; the peer route dialog never loads a local copy unless asked.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync, readdirSync, existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

import { loadApp, runScript } from "./harness.mjs";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const APP = join(ROOT, "localm", "plugins", "gui", "static", "app");
const BUILTIN = join(ROOT, "localm", "plugins", "builtin");

const json = (obj, status = 200) => ({
  ok: status >= 200 && status < 300, status, statusText: String(status),
  headers: { get: () => null },
  json: async () => obj, text: async () => JSON.stringify(obj),
});

async function tick(n = 10) {
  for (let i = 0; i < n; i++) await new Promise((r) => setTimeout(r, 20));
}

const PLUGIN_URLS = ["/api/memory", "/api/rag/collections", "/api/voice/status", "/api/coder/sessions"];

function bootWith(plugins) {
  const urls = [];
  const fetchImpl = async (url) => {
    const u = String(url);
    urls.push(u);
    if (u === "/api/capabilities") return json({ plugins, core: {} });
    if (u.startsWith("/api/models")) return json({ models: [], active: "" });
    if (u === "/api/voice/status") return json({ available: true });
    if (u === "/api/rag/collections") return json({ collections: [] });
    if (u === "/api/coder/sessions") return json({ sessions: [] });
    return json({});
  };
  const { window } = loadApp({ fetchImpl });
  return { window, urls };
}

test("page load reads no plugin route when the capabilities list no active plugin", async () => {
  const { window, urls } = bootWith([
    { name: "memory", active: false, installed: true },
    { name: "voice", active: false, installed: false },
  ]);
  await tick();
  assert.ok(urls.includes("/api/capabilities"), "the capabilities were read");
  for (const u of PLUGIN_URLS) {
    assert.equal(urls.filter((x) => x === u).length, 0, `${u} was read for an inactive plugin`);
  }
  const mic = window.document.getElementById("chat-mic");
  assert.ok(mic.classList.contains("unavailable"), "the mic is greyed for a voice plugin that is not active");
  assert.match(mic.title, /voice/i);
});

test("page load reads each active plugin's route once", async () => {
  const { urls } = bootWith(["memory", "rag", "voice", "coder"].map((name) => ({ name, active: true })));
  await tick();
  for (const u of PLUGIN_URLS) {
    assert.equal(urls.filter((x) => x === u).length, 1, `${u} read once for an active plugin`);
  }
});

test("a capabilities read that gets no answer runs every plugin read instead of guessing absent", async () => {
  const urls = [];
  const fetchImpl = async (url) => {
    const u = String(url);
    urls.push(u);
    if (u === "/api/capabilities") return json({ detail: "boom" }, 500);
    if (u.startsWith("/api/models")) return json({ models: [], active: "" });
    if (u === "/api/voice/status") return json({ available: true });
    if (u === "/api/rag/collections") return json({ collections: [] });
    if (u === "/api/coder/sessions") return json({ sessions: [] });
    return json({});
  };
  const { window } = loadApp({ fetchImpl });
  await tick();
  for (const u of PLUGIN_URLS) assert.equal(urls.filter((x) => x === u).length, 1, `${u} read once`);
  const mic = window.document.getElementById("chat-mic");
  assert.ok(!mic.classList.contains("unavailable"), "the mic is not greyed as not installed");
});

// Function bodies of *file*, keyed by the declared name.
function functionBodies(file) {
  const src = readFileSync(join(APP, file), "utf-8").replace(/\r/g, "");
  const out = {};
  const re = /^(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(/gm;
  let m;
  while ((m = re.exec(src))) {
    const end = src.indexOf("\n}\n", m.index);
    out[m[1]] = src.slice(m.index, end === -1 ? undefined : end);
  }
  return out;
}

test("every page-load read of a plugin route in init.js goes through the plugin gate", () => {
  const init = readFileSync(join(APP, "init.js"), "utf-8").replace(/\r/g, "");
  const plugins = readdirSync(BUILTIN).filter((d) => {
    const toml = join(BUILTIN, d, "plugin.toml");
    return existsSync(toml) && !/^protected\s*=\s*true/m.test(readFileSync(toml, "utf-8"));
  });
  assert.ok(plugins.includes("memory") && plugins.includes("voice"), "builtin plugins found");
  const imports = [...init.matchAll(/^import \{([^}]*)\} from "\.\/([\w-]+\.js)";$/gm)];
  const bodies = {};
  for (const [, names, file] of imports) {
    const fb = functionBodies(file);
    for (const n of names.split(",").map((s) => s.trim())) if (fb[n]) bodies[n] = fb[n];
  }
  const probeBlock = init.slice(init.indexOf("export const PLUGIN_BOOT_PROBES"),
    init.indexOf("]);", init.indexOf("export const PLUGIN_BOOT_PROBES")));
  const gated = [];
  for (const [name, body] of Object.entries(bodies)) {
    const routes = [...body.matchAll(/["`]\/api\/([a-z_-]+)/g)].map((m) => m[1]);
    const plugin = routes.find((seg) => plugins.includes(seg));
    if (!plugin) continue;
    gated.push(name);
    const direct = new RegExp(`(^|[^\\w.])${name}\\(`, "m");
    assert.ok(!direct.test(init), `${name} (reads /api/${plugin}) is called directly in init.js`);
    assert.match(probeBlock, new RegExp(`plugin: "${plugin}", run: ${name}\\b`),
      `${name} is not listed under plugin "${plugin}" in PLUGIN_BOOT_PROBES`);
  }
  assert.deepEqual(gated.sort(), ["reattachSessions", "refreshKbSelect", "refreshMemory", "refreshVoiceStatus"]);
});

test("a requested shut down shows its own overlay with no key reset and no reconnect poll", async () => {
  const { window } = loadApp({ fetchImpl: async () => { throw new TypeError("down"); } });
  const doc = window.document;
  window.onServerUnreachable({ reason: "shutdown" });
  const ov = doc.getElementById("reconnect-overlay");
  assert.match(ov.textContent, /was shut down/);
  assert.doesNotMatch(ov.textContent, /Reconnecting|re-enter key/i);
  window.onServerUnreachable({ sawDown: true });
  assert.match(ov.textContent, /was shut down/, "a later unreachable call keeps the shut down overlay");
  runScript(window, "window.__timer = _reconnectTimer;");
  assert.equal(window.__timer, null, "no reconnect poll runs after a shut down");
  ov.querySelector("button").click();
  await tick(3);
  assert.match(ov.querySelector(".reconnect-note").textContent, /still not running/);
});

function resetSetup(logout) {
  const calls = [];
  const fetchImpl = async (url, opts = {}) => {
    const u = String(url);
    calls.push({ url: u, method: opts.method || "GET", headers: opts.headers || {} });
    if (u === "/api/session") return json({ authed: true, csrf: "tok-1" });
    if (u === "/api/session/logout") return logout();
    if (u.startsWith("/api/models")) return json({ models: [], active: "" });
    return json({});
  };
  const { window } = loadApp({ fetchImpl });
  return { window, calls };
}

test("reset signs the browser out on the server, keeps localStorage and clears the page session", async () => {
  const { window, calls } = resetSetup(() => json({ authed: false, warnings: [] }));
  await tick();
  runScript(window, `localStorage.setItem("localm.theme", "dark"); sessionStorage.setItem("localm.loginOk", "1");`);
  try { await window.resetClientState(); } catch { /* jsdom location.reload */ }
  const out = calls.filter((c) => c.url === "/api/session/logout");
  assert.equal(out.length, 1, "one sign-out request");
  assert.equal(out[0].method, "POST");
  assert.equal(out[0].headers["X-CSRF-Token"], "tok-1", "the session's CSRF token is sent");
  assert.equal(window.localStorage.getItem("localm.theme"), "dark", "localStorage is kept");
  assert.equal(window.sessionStorage.length, 0, "the page session is cleared");
});

test("reset says so and clears nothing when the server cannot be reached", async () => {
  const { window } = resetSetup(() => { throw new TypeError("down"); });
  await tick();
  runScript(window, `sessionStorage.setItem("localm.loginOk", "1");`);
  const note = window.document.createElement("div");
  const reloaded = await window.resetClientState(note);
  assert.equal(reloaded, false);
  assert.match(note.textContent, /could not be signed out/);
  assert.equal(window.sessionStorage.getItem("localm.loginOk"), "1", "nothing cleared");
});

test("reset shows the server's refusal", async () => {
  const { window } = resetSetup(() => json({ detail: "Missing or invalid CSRF token." }, 403));
  await tick();
  const note = window.document.createElement("div");
  assert.equal(await window.resetClientState(note), false);
  assert.match(note.textContent, /could not be signed out: Missing or invalid CSRF token/);
});

test("reset says when the server kept the session and offers a reload", async () => {
  const { window } = resetSetup(() => json({ authed: false, warnings: ["session store not written"] }));
  await tick();
  const note = window.document.createElement("div");
  const button = window.document.createElement("button");
  assert.equal(await window.resetClientState(note, button), false);
  assert.match(note.textContent, /could not end the session on its side: session store not written/);
  assert.equal(button.textContent, "Reload");
});

const PEER = { instance_id: "peer-1", host: "127.0.0.1", port: 9555, scheme: "http", model: "shared" };

function peerSetup({ routeStatus = 200, models = { models: [] } } = {}) {
  const calls = [];
  const state = { routeStatus, models };
  const impl = async (url, opts = {}) => {
    const u = String(url);
    let body;
    try { body = opts.body ? JSON.parse(opts.body) : null; } catch { body = opts.body; }
    calls.push({ url: u, method: opts.method || "GET", body });
    if (u === "/v1/models/shared/peer-offer") {
      return json({ available: true, peer: { ...PEER, requires_key: true } });
    }
    if (u === "/v1/models/shared/peer-route") {
      return state.routeStatus === 200
        ? json({ status: "routed", model: "shared", peer: PEER })
        : json({ detail: "the key was refused" }, state.routeStatus);
    }
    if (u === "/api/models/load") return json({ status: "loaded", model: "shared" });
    if (u.startsWith("/api/models")) return json(state.models);
    return json({});
  };
  const { window } = loadApp({ fetchImpl: impl });
  return { window, calls, state };
}

function modalButton(doc, label) {
  return [...doc.querySelectorAll("#modal button")].find((b) => b.textContent === label);
}

test("Route with an empty key keeps the dialog open and loads nothing", async () => {
  const { window, calls } = peerSetup();
  const doc = window.document;
  window.switchModel("shared");
  await tick();
  modalButton(doc, "Route").click();
  await tick(3);
  assert.equal(doc.getElementById("modal").style.display, "flex", "the dialog stays open");
  assert.match(doc.querySelector("#modal .peer-route-error").textContent, /API key/);
  assert.equal(calls.filter((c) => c.url === "/v1/models/shared/peer-route").length, 0);
  assert.equal(calls.filter((c) => c.url === "/api/models/load").length, 0);
});

test("the x cancels the switch and loads nothing", async () => {
  const { window, calls } = peerSetup();
  const doc = window.document;
  const pending = window.switchModel("shared");
  await tick();
  doc.getElementById("modal-close").click();
  const res = await pending;
  assert.equal(res.status, "cancelled");
  assert.equal(calls.filter((c) => c.url === "/api/models/load").length, 0, "no local load on cancel");
});

test("a refused key asks again with the reason and never loads locally on its own", async () => {
  const { window, calls } = peerSetup({ routeStatus: 403 });
  const doc = window.document;
  const pending = window.switchModel("shared");
  await tick();
  doc.querySelector("#modal input[type=password]").value = "wrong";
  modalButton(doc, "Route").click();
  await tick();
  assert.equal(doc.getElementById("modal").style.display, "flex", "the dialog is shown again");
  assert.match(doc.querySelector("#modal .peer-route-error").textContent, /the key was refused/);
  assert.equal(calls.filter((c) => c.url === "/api/models/load").length, 0, "no local load after a refused key");
  modalButton(doc, "Load here instead").click();
  await pending;
  assert.equal(calls.filter((c) => c.url === "/api/models/load").length, 1, "the local load runs when chosen");
});

test("an accepted route is kept for a reload of the tab and restored while the server holds it", async () => {
  const withRoute = {
    models: [{ name: "local-one", active: true, loaded: true }, { name: "shared" }],
    active: "local-one",
    peer_routes: { shared: { host: "127.0.0.1", port: 9555, instance_id: "peer-1" } },
  };
  const { window } = peerSetup({ models: withRoute });
  const doc = window.document;
  await tick();
  runScript(window, "chat.modeKnown = true; chat.serverPrivacy = false;");
  const pending = window.switchModel("shared");
  await tick();
  doc.querySelector("#modal input[type=password]").value = "peer-key";
  modalButton(doc, "Route").click();
  await pending;
  assert.equal(window.sessionStorage.getItem("localm.peerSelected"), "shared");

  runScript(window, "_peerSelected = null; _peerRestore = sessionStorage.getItem('localm.peerSelected');");
  await window.refreshModels();
  runScript(window, "window.__active = modelCache.active; window.__peer = isPeerSelected();");
  assert.equal(window.__active, "shared", "the restored route is the selection again");
  assert.equal(window.__peer, true);

  runScript(window, "chat.serverPrivacy = true;");
  await window.refreshModels();
  assert.equal(window.sessionStorage.getItem("localm.peerSelected"), null, "not kept in privacy mode");
});

// SPDX-License-Identifier: AGPL-3.0-or-later
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages } from "./harness.mjs";

// The Settings "Companion app" card must show a phone-REACHABLE address (the
// host's LAN / Tailscale IP), never the loopback origin - 127.0.0.1 on a phone
// is the phone itself. companionView() is the pure decision point; the URLs are
// built from THIS page's own scheme + port (one port across every interface).

const tick = () => new Promise((r) => setTimeout(r, 50));

function fetchCompanion(resp) {
  return (url) => {
    if (String(url).includes("/api/companion")) {
      return Promise.resolve({ ok: true, status: 200,
        json: async () => resp, text: async () => "" });
    }
    return Promise.resolve({ ok: true, status: 200,
      json: async () => ({ models: [], active: "" }), text: async () => "" });
  };
}

test("companionView: network bind lists LAN then Tailscale, built from scheme+port", () => {
  const { window } = loadAppWithPages({});
  const view = window.companionView(
    { network_bind: true, lan: "192.168.1.50", tailscale: "100.101.102.103" },
    { protocol: "https:", port: "8642" });
  assert.equal(view.urls.length, 2);
  assert.equal(view.urls[0].kind, "Wi-Fi / LAN");
  assert.equal(view.urls[0].url, "https://192.168.1.50:8642/");
  assert.equal(view.urls[1].kind, "Tailscale");
  assert.equal(view.urls[1].url, "https://100.101.102.103:8642/");
  assert.equal(view.hint, "");
  assert.ok(!view.urls.some((u) => /127\.0\.0\.1|localhost/.test(u.url)),
    "never offers the loopback address");
});

test("companionView: no Tailscale -> only the LAN address is listed", () => {
  const { window } = loadAppWithPages({});
  const view = window.companionView(
    { network_bind: true, lan: "10.0.0.7", tailscale: "" },
    { protocol: "https:", port: "8642" });
  assert.equal(view.urls.length, 1);
  assert.equal(view.urls[0].url, "https://10.0.0.7:8642/");
  assert.equal(view.hint, "");
});

test("companionView: loopback bind shows the -H 0.0.0.0 hint and no address", () => {
  const { window } = loadAppWithPages({});
  const view = window.companionView(
    { network_bind: false, lan: "", tailscale: "" },
    { protocol: "http:", port: "8642" });
  assert.equal(view.urls.length, 0);
  assert.match(view.hint, /-H 0\.0\.0\.0/);
});

test("companionView: loopback hint names the GUI path (Bind address + Restart)", () => {
  // F1: the card must no longer concede that only a terminal can cause the
  // bind - Settings > Server > Bind address is the first-class route now.
  const { window } = loadAppWithPages({});
  const view = window.companionView(
    { network_bind: false, lan: "", tailscale: "" },
    { protocol: "http:", port: "8642" });
  assert.match(view.hint, /Bind address/);
  assert.match(view.hint, /Restart server/);
});

test("companionView: a bind_fallback reason outranks every generic hint", () => {
  // The server REFUSED a configured network bind (e.g. no strong API key) and
  // stayed on loopback: the card must say WHY, or the Settings change looks
  // like it silently did nothing.
  const { window } = loadAppWithPages({});
  const reason = "The configured bind address (0.0.0.0) was not applied: no API key is set.";
  const view = window.companionView(
    { network_bind: false, lan: "", tailscale: "", bind_fallback: reason },
    { protocol: "http:", port: "8642" });
  assert.equal(view.urls.length, 0);
  assert.equal(view.hint, reason);
});

test("companionView: network bind but no detectable IP -> generic hint, never loopback", () => {
  const { window } = loadAppWithPages({});
  const view = window.companionView(
    { network_bind: true, lan: "", tailscale: "" },
    { protocol: "https:", port: "8642" });
  assert.equal(view.urls.length, 0);
  assert.ok(view.hint.length > 0);
  assert.ok(!/127\.0\.0\.1/.test(view.hint));
});

test("refreshCompanion: renders the reachable addresses and hides the hint", async () => {
  const { window } = loadAppWithPages({ fetchImpl: fetchCompanion(
    { network_bind: true, lan: "192.168.1.50", tailscale: "100.101.102.103" }) });
  await tick();
  await window.refreshCompanion();
  const list = window.document.getElementById("companion-addrs");
  const hint = window.document.getElementById("companion-hint");
  assert.equal(list.style.display, "block");
  const links = list.querySelectorAll("a.companion-addr-url");
  assert.equal(links.length, 2);
  // The harness page is served on :8642, so the URLs carry that port.
  assert.equal(links[0].getAttribute("href"), "http://192.168.1.50:8642/");
  assert.equal(hint.style.display, "none");
});

test("refreshCompanion: loopback bind hides addresses, shows the bind hint", async () => {
  const { window } = loadAppWithPages({ fetchImpl: fetchCompanion(
    { network_bind: false, lan: "", tailscale: "" }) });
  await tick();
  await window.refreshCompanion();
  const list = window.document.getElementById("companion-addrs");
  const hint = window.document.getElementById("companion-hint");
  assert.equal(list.style.display, "none");
  assert.equal(hint.style.display, "block");
  assert.match(hint.textContent, /-H 0\.0\.0\.0/);
});

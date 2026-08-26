// SPDX-License-Identifier: AGPL-3.0-or-later
// The remote-image proxy, driven in a REAL browser against the real enforcing CSP.
//
// The jsdom unit suite covers the client rewrite and tests/test_image_proxy.py
// covers the route, but neither can answer the question the feature exists for:
// does the BROWSER contact the remote origin. jsdom enforces no CSP and issues no
// image loads, so "the browser never reached that host" is unfalsifiable there.
// Here a local origin server records every request it receives, so the answer is
// read off that log rather than inferred.
//
// Runs on whichever engine --browser selects. The CSP arm at the end is why that
// matters: img-src 'self' data: blob: is the barrier the OFF state rests on, and
// it is enforced by the browser rather than by localm.

import { test, expect } from "@playwright/test";
import http from "node:http";
import zlib from "node:zlib";
import { BASE_URL } from "./_env.mjs";

function crc32(buf) {
  let c;
  let crc = 0xFFFFFFFF;
  for (let n = 0; n < buf.length; n++) {
    c = (crc ^ buf[n]) & 0xFF;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xEDB88320 ^ (c >>> 1) : c >>> 1;
    crc = c ^ (crc >>> 8);
  }
  return (crc ^ 0xFFFFFFFF) >>> 0;
}

/** A real 64x64 PNG, built here so the test needs no fixture file and no network. */
function png(w = 64, h = 64) {
  const row = Buffer.concat([Buffer.from([0]),
                             Buffer.from(Array.from({ length: w * 3 }, (_, i) => (i % 3 === 0 ? 220 : 40)))]);
  const raw = Buffer.concat(Array.from({ length: h }, () => row));
  const chunk = (tag, data) => {
    const body = Buffer.concat([Buffer.from(tag, "latin1"), data]);
    const len = Buffer.alloc(4);
    len.writeUInt32BE(data.length);
    const crc = Buffer.alloc(4);
    crc.writeUInt32BE(crc32(body));
    return Buffer.concat([len, body, crc]);
  };
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(w, 0);
  ihdr.writeUInt32BE(h, 4);
  ihdr[8] = 8;
  ihdr[9] = 2;
  return Buffer.concat([Buffer.from([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]),
                        chunk("IHDR", ihdr),
                        chunk("IDAT", zlib.deflateSync(raw)),
                        chunk("IEND", Buffer.alloc(0))]);
}

/** A stand-in remote image host that RECORDS every request it receives, so "the
 *  browser never reached this origin" is a reading rather than an inference. */
async function startOrigin() {
  const body = png();
  const seen = [];
  const server = http.createServer((req, res) => {
    seen.push({ path: req.url, headers: req.headers });
    res.writeHead(200, { "Content-Type": "image/png", "Content-Length": body.length });
    res.end(body);
  });
  await new Promise((r) => server.listen(0, "127.0.0.1", r));
  const { port } = server.address();
  return { url: (p) => `http://127.0.0.1:${port}${p}`, seen, close: () => server.close() };
}

const patchConfig = (page, patch) => page.evaluate(async (p) => {
  const r = await fetch("/v1/config", {
    method: "PATCH", headers: window.authHeaders(), body: JSON.stringify(p),
  });
  return r.status;
}, patch);

/** Render `![](url)` through the SHIPPED renderMarkdown into a fresh node. */
const renderRemoteImage = (page, url, id) => page.evaluate(([u, i]) => {
  const box = document.createElement("div");
  box.id = i;
  document.body.appendChild(box);
  window.renderMarkdown(box, "reply text\n\n![alt](" + u + ")\n");
}, [url, id]);

/** A refused image is REPLACED by a `.img-blocked` note carrying the route's
 *  reason, so read whichever of the two is present. */
const imgState = (page, id) => page.evaluate((i) => {
  const box = document.getElementById(i);
  const note = box && box.querySelector(".img-blocked");
  if (note) {
    return {
      blocked: true,
      hasSrc: false,
      naturalWidth: 0,
      proxySrc: note.dataset.lmProxySrc || null,
      failed: note.dataset.lmProxyFailed || null,
      reason: (note.textContent || "").trim(),
      visible: note.getBoundingClientRect().height > 0,
    };
  }
  const img = box && box.querySelector("img");
  if (!img) return { missing: true };
  return {
    blocked: false,
    srcScheme: (img.getAttribute("src") || "").split(":")[0] || "(none)",
    hasSrc: img.hasAttribute("src"),
    proxySrc: img.dataset.lmProxySrc || null,
    failed: img.dataset.lmProxyFailed || null,
    naturalWidth: img.naturalWidth,
  };
}, id);

async function boot(page) {
  await page.goto(BASE_URL);
  await page.waitForFunction(() => typeof window.renderMarkdown === "function"
                                && typeof window.authHeaders === "function");
  // The route fetches through netpolicy, whose SSRF guard refuses private
  // addresses, and the stand-in origin above is one.
  expect(await patchConfig(page, { net_allow_private: true })).toBe(200);
}

test("OFF (the default): the browser never contacts the remote origin, and the reader is told",
  async ({ page }) => {
    const origin = await startOrigin();
    try {
      await boot(page);
      expect(await patchConfig(page, { gui_proxy_remote_images: false })).toBe(200);
      await renderRemoteImage(page, origin.url("/off.png"), "off");

      await expect.poll(async () => (await imgState(page, "off")).failed).toBe("1");
      const st = await imgState(page, "off");
      expect(st.hasSrc, "no src attribute at all, so nothing can be fetched").toBe(false);
      expect(st.naturalWidth, "nothing rendered").toBe(0);
      expect(st.proxySrc, "what the model asked for is kept for diagnostics")
        .toBe(origin.url("/off.png"));
      expect(origin.seen, "the REMOTE ORIGIN saw no request from anyone").toEqual([]);
      // And the reader is TOLD. Before this the reply just had a hole in it: a
      // src-less <img> paints the model's alt text, or nothing at all when the
      // model wrote `![](...)`, so an image could vanish with no way to know.
      expect(st.visible, "the note is actually on screen, not a 0x0 element").toBe(true);
      expect(st.reason).toContain("Image not shown");
      expect(st.reason, "and carries the route's own reason, which names the setting")
        .toContain("Show remote images in replies");
    } finally { origin.close(); }
  });

test("ON: the image renders, and the request to the remote origin came from the SERVER",
  async ({ page }) => {
    // The control for the test above. Without it, "the origin saw nothing" is
    // equally consistent with a harness that never issued the render at all.
    const origin = await startOrigin();
    try {
      await boot(page);
      expect(await patchConfig(page, { gui_proxy_remote_images: true })).toBe(200);
      await renderRemoteImage(page, origin.url("/on.png"), "on");

      await expect.poll(async () => (await imgState(page, "on")).naturalWidth).toBe(64);
      expect((await imgState(page, "on")).srcScheme, "served from a blob, not the remote URL")
        .toBe("blob");

      expect(origin.seen.length).toBe(1);
      const h = origin.seen[0].headers;
      expect(h["user-agent"] || "", "localm's own outbound agent, not the browser's")
        .toContain("localm");
      expect(h.referer, "the remote host learns no referrer").toBeUndefined();
      expect(h.origin, "and no origin").toBeUndefined();
      expect(h["sec-fetch-dest"], "and none of the browser's fetch metadata").toBeUndefined();
    } finally { origin.close(); }
  });

test("turning it OFF stops an image already on screen, once the page cache is cleared",
  async ({ page }) => {
    const origin = await startOrigin();
    try {
      await boot(page);
      expect(await patchConfig(page, { gui_proxy_remote_images: true })).toBe(200);
      const url = origin.url("/toggle.png");
      await renderRemoteImage(page, url, "t1");
      await expect.poll(async () => (await imgState(page, "t1")).naturalWidth).toBe(64);

      // The route starts refusing immediately, but the client keeps a
      // page-lifetime blob cache keyed on the URL, so a re-render is served from
      // it. That is this assertion's control.
      expect(await patchConfig(page, { gui_proxy_remote_images: false })).toBe(200);
      await renderRemoteImage(page, url, "t2");
      await expect.poll(async () => (await imgState(page, "t2")).naturalWidth).toBe(64);

      // saveSettings() calls this on every successful save.
      await page.evaluate(() => window.clearImageProxyCache());
      await renderRemoteImage(page, url, "t3");
      await expect.poll(async () => (await imgState(page, "t3")).failed).toBe("1");
      expect((await imgState(page, "t3")).hasSrc, "the cleared cache cannot re-serve it")
        .toBe(false);
    } finally { origin.close(); }
  });

test("a reply linking a remote image with the proxy OFF does not make the app reload itself",
  async ({ page }) => {
    // The proxy answers 403 while the feature is off. That 403 rides on the
    // open-mode shell token, and the fetch wrapper's stale-credential recovery
    // used to read it as a refused credential: unregister the service worker,
    // drop the caches, reload, mid-reply.
    const origin = await startOrigin();
    try {
      await boot(page);
      expect(await patchConfig(page, { gui_proxy_remote_images: false })).toBe(200);
      await page.evaluate(() => { window.__reloadCanary = "ALIVE"; });
      await renderRemoteImage(page, origin.url("/reload.png"), "r1");
      await expect.poll(async () => (await imgState(page, "r1")).failed).toBe("1");

      await page.waitForTimeout(4000);
      expect(await page.evaluate(() => window.__reloadCanary || null),
             "the page kept its identity, so it never reloaded").toBe("ALIVE");
      expect(await page.evaluate(() => sessionStorage.getItem("localm.shellReset")),
             "and the one-shot credential recovery was never armed").toBe(null);
    } finally { origin.close(); }
  });

test("CSP: this engine refuses a remote <img> outright, which is what the OFF state rests on",
  async ({ page }) => {
    // R41 verified the shell's CSP on Chrome only. img-src is the barrier that
    // keeps "the browser never contacts the remote origin" true even if the
    // client rewrite were removed, so it is worth measuring per engine.
    const origin = await startOrigin();
    try {
      await boot(page);
      // Bypass renderMarkdown entirely: a raw remote <img>, which is exactly what
      // the rewrite exists to prevent, straight into the document.
      await page.evaluate((u) => {
        const img = document.createElement("img");
        img.id = "csp-probe";
        img.src = u;
        document.body.appendChild(img);
      }, origin.url("/csp-remote.png"));
      // Fires-control: a SAME-ORIGIN image must still load, so a zero below can
      // never be "the probe never ran".
      await page.evaluate(() => {
        const img = document.createElement("img");
        img.id = "csp-control";
        img.src = "/icon-192.png";
        document.body.appendChild(img);
      });
      await expect.poll(async () => page.evaluate(
        () => document.getElementById("csp-control").naturalWidth)).toBeGreaterThan(0);

      await page.waitForTimeout(1500);
      expect(await page.evaluate(() => document.getElementById("csp-probe").naturalWidth),
             "the remote image did not load").toBe(0);
      expect(origin.seen, "and the engine never issued the request at all").toEqual([]);
    } finally { origin.close(); }
  });

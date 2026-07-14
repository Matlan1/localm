// SPDX-License-Identifier: AGPL-3.0-or-later
/* localm PWA service worker.
 *
 * Makes the GUI installable and lets the app shell open instantly (and offline),
 * while NEVER caching API/model traffic - chat, model loads, and plugin calls
 * must always hit the live server. The data lives on the server this app was
 * served from; the worker only caches the static front-end.
 */
// Bump this whenever the cached shell assets (style.css, app/*.js, pages/*.js,
// index.html, icons) change, so an installed PWA drops the old cache on activate
// and re-precaches the new files instead of serving stale cache-first assets.
// v49: pages/settings.js changed (the ComfyUI setup log no longer erases itself
// on failure, #621) - a stale-cache miss on THIS exact bump is what made that
// fix invisible to an already-installed PWA in the field.
// v50: pages/settings.js changed again (the managed_comfy_enabled checkbox was
// removed - a stale-cached client would keep showing it AND get "unknown config
// key" on save, since the server no longer accepts that field).
// v51: index.html and pages/models.js changed (the ComfyUI re-scan button moved
// out of its hidden tab-gated row into the "Add a model" card) - a stale-cached
// client would keep the button invisible on the All/LLMs tabs.
// v52: app/helpers.js and pages/images.js/music.js/video.js changed (the
// ComfyUI missing-model download offer) - a stale-cached client would keep
// missing the pre-generate model check and the download confirm modal.
const CACHE = "localm-shell-v52";
const SHELL = [
  "/", "/index.html", "/style.css",
  // GUI ES-module entry + every app/* and pages/* module (the import graph).
  "/app/main.js",
  "/app/client-log.js", "/app/helpers.js", "/app/icons.js", "/app/picker.js", "/app/theme.js",
  "/app/logo.js", "/app/tabs.js", "/app/models-sidebar.js", "/app/chat.js",
  "/app/cmdk.js", "/app/settings-perf.js", "/app/coder.js", "/app/slash.js",
  "/app/init.js",
  // pages.js was split per page (same load order); precache each part.
  "/pages/dispatch.js", "/pages/models.js", "/pages/images.js",
  "/pages/plugins.js", "/pages/settings.js", "/pages/workflow.js",
  "/pages/music.js", "/pages/video.js", "/pages/knowledge.js",
  "/icon.svg", "/icon-192.png", "/icon-512.png", "/icon-maskable-512.png",
  "/manifest.webmanifest",
  "/vendor/marked.min.js", "/vendor/purify.min.js", "/vendor/highlight.min.js",
  "/vendor/katex.min.js", "/vendor/auto-render.min.js",
  "/vendor/github-dark.min.css", "/vendor/katex.min.css",
  // Inter (UI typeface) - precache so the shell renders with it offline / on first
  // paint instead of flashing the fallback then reflowing.
  "/vendor/inter/inter-latin-400-normal.woff2", "/vendor/inter/inter-latin-500-normal.woff2",
  "/vendor/inter/inter-latin-600-normal.woff2", "/vendor/inter/inter-latin-700-normal.woff2",
];

self.addEventListener("install", (e) => {
  // Pre-cache the shell; best-effort so one missing asset is not fatal. Force a
  // network fetch (cache: "reload") so a NEW worker never re-caches a stale copy
  // from the browser's HTTP cache on update - it must precache the new assets.
  e.waitUntil(
    caches.open(CACHE)
      .then((c) => Promise.allSettled(
        SHELL.map((u) => c.add(new Request(u, { cache: "reload" })))))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener("activate", (e) => {
  // Drop old shell versions (ONLY our own localm-shell-* caches) so an updated
  // app never serves stale assets. The transformers.js model cache
  // ("transformers-cache", the Kokoro TTS weights) must SURVIVE a shell bump; a
  // blanket k !== CACHE filter re-downloaded it every update (REC-KOKORO-RELOAD).
  e.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(
        keys.filter((k) => k.startsWith("localm-shell-") && k !== CACHE)
            .map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (e) => {
  const req = e.request;
  if (req.method !== "GET") return;                       // never touch writes
  const url = new URL(req.url);
  if (url.origin !== self.location.origin) return;        // only our own origin
  // API / model / plugin traffic and the CA cert are ALWAYS live, never cached.
  // /localm-ca.crt must hit the network: if the SW handled it, a navigate-mode
  // download could fall back to cached index.html instead of the cert, saving an
  // .html file from the "Install certificate" link (J2).
  if (/^\/(api|v1|plugins|localm-ca\.crt)(\/|$)/.test(url.pathname)) return;

  // Navigations: network-first so the app updates; fall back to the cached
  // shell when offline so the installed app still opens.
  if (req.mode === "navigate") {
    e.respondWith(fetch(req).catch(() => caches.match("/index.html")));
    return;
  }

  // Static assets: cache-first, then fill the cache on a miss.
  e.respondWith(
    caches.match(req).then((hit) => {
      if (hit) return hit;
      return fetch(req).then((res) => {
        if (res && res.ok) {
          const copy = res.clone();
          caches.open(CACHE).then((c) => c.put(req, copy));
        }
        return res;
      });
    })
  );
});

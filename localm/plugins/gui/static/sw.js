// SPDX-License-Identifier: AGPL-3.0-or-later
/* localm PWA service worker.
 *
 * Caches the static front-end shell only. API, model and plugin traffic always
 * hits the live server.
 */
// Placeholder. The GUI server substitutes the real value on every request to
// /sw.js: a content digest of the static assets it serves. Never edit by hand.
const CACHE = "localm-shell-dev";
const SHELL = [
  "/", "/index.html", "/style.css",
  // GUI ES-module entry plus every app/* and pages/* module.
  "/app/main.js",
  "/app/client-log.js", "/app/helpers.js", "/app/icons.js", "/app/picker.js", "/app/theme.js",
  "/app/logo.js", "/app/tabs.js", "/app/models-sidebar.js", "/app/chat.js",
  "/app/cmdk.js", "/app/settings-perf.js", "/app/coder.js", "/app/slash.js",
  "/app/init.js", "/app/media-gallery.js",
  // Per-page modules.
  "/pages/dispatch.js", "/pages/models.js", "/pages/images.js",
  "/pages/plugins.js", "/pages/settings.js", "/pages/workflow.js",
  "/pages/music.js", "/pages/video.js", "/pages/knowledge.js",
  "/icon.svg", "/icon-192.png", "/icon-512.png", "/icon-maskable-512.png",
  "/manifest.webmanifest",
  "/vendor/marked.min.js", "/vendor/purify.min.js", "/vendor/highlight.min.js",
  "/vendor/katex.min.js", "/vendor/auto-render.min.js",
  "/vendor/github-dark.min.css", "/vendor/katex.min.css",
  // Inter (UI typeface).
  "/vendor/inter/inter-latin-400-normal.woff2", "/vendor/inter/inter-latin-500-normal.woff2",
  "/vendor/inter/inter-latin-600-normal.woff2", "/vendor/inter/inter-latin-700-normal.woff2",
];

self.addEventListener("install", (e) => {
  // Pre-cache the shell, best-effort: one missing asset is not fatal.
  // cache: "reload" bypasses the browser's own HTTP cache.
  e.waitUntil(
    caches.open(CACHE)
      .then((c) => Promise.allSettled(
        SHELL.map((u) => c.add(new Request(u, { cache: "reload" })))))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener("activate", (e) => {
  // Drop old shell versions. Only localm-shell-* caches are deleted; every
  // other cache, including the transformers.js model cache, must survive.
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
  // API, model and plugin traffic and the CA cert are always live, never
  // cached, and must reach the network.
  if (/^\/(api|v1|plugins|localm-ca\.crt)(\/|$)/.test(url.pathname)) return;

  // Navigations: network-first, falling back to the cached shell when offline.
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

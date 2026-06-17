/* localm PWA service worker.
 *
 * Makes the GUI installable and lets the app shell open instantly (and offline),
 * while NEVER caching API/model traffic - chat, model loads, and plugin calls
 * must always hit the live server. The data lives on the server this app was
 * served from; the worker only caches the static front-end.
 */
const CACHE = "localm-shell-v1";
const SHELL = [
  "/", "/index.html", "/style.css", "/app.js", "/pages.js",
  "/icon.svg", "/manifest.webmanifest",
  "/vendor/marked.min.js", "/vendor/purify.min.js", "/vendor/highlight.min.js",
  "/vendor/katex.min.js", "/vendor/auto-render.min.js",
  "/vendor/github-dark.min.css", "/vendor/katex.min.css",
];

self.addEventListener("install", (e) => {
  // Pre-cache the shell; addAll is best-effort so one missing asset is not fatal.
  e.waitUntil(
    caches.open(CACHE)
      .then((c) => Promise.allSettled(SHELL.map((u) => c.add(u))))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener("activate", (e) => {
  // Drop old shell versions so an updated app does not serve stale assets.
  e.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (e) => {
  const req = e.request;
  if (req.method !== "GET") return;                       // never touch writes
  const url = new URL(req.url);
  if (url.origin !== self.location.origin) return;        // only our own origin
  // API / model / plugin traffic is ALWAYS live - never served from cache.
  if (/^\/(api|v1|plugins)(\/|$)/.test(url.pathname)) return;

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

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
// v53: index.html and pages/models.js changed (a new "Embedding" Model
// Manager tab + type option) - a stale-cached client would keep lacking the
// tab and the per-row type <select> would be missing the 'embedding' choice.
// v54: pages/models.js changed again (the "Unload all" toast now also counts
// the shared embedding model's release, and a row for a registered model that
// IS the resident embedder now correctly shows loaded/Unload) - a stale-cached
// client would keep under-reporting freed VRAM and miss the per-row control.
// v55: pages/models.js's "Import from ComfyUI..." Browse button was fixed (it
// silently did nothing - openModal() has no stack, so the folder picker it
// opened destroyed the Import dialog's own DOM before the picked path could be
// written back); pages/settings.js and style.css dropped the rotating
// per-plugin accent color on the Media section's three boxes in favor of one
// consistent accent. A stale-cached client would keep the broken Browse
// button and the mismatched accent colors.
// v56: pages/settings.js's managed-ComfyUI panel gained a "corrupt install /
// needs repair" state (Repair button, POST /api/comfy/repair) distinguishing
// an abandoned setup attempt from a clean "not set up" - a stale-cached
// client would keep showing the old dead-end Set-up-button-that-just-409s.
// v57: pages/plugins.js now drives the External plugins card off the engine API
// (/api/plugins + /api/plugins/install-external + /api/plugins/{name}/uninstall)
// instead of the deleted /v1/plugins, which 404'd the whole card (REG-585), and
// app/models-sidebar.js publishes the active model as soon as a load lands so
// chat is not falsely refused for up to 30s afterwards (REG-471), and
// pages/models.js names the alias the server actually stored (REG-562). A
// stale-cached client would keep calling routes that no longer exist, keep the
// false "No model loaded" refusal, and keep naming an alias that does not exist.
// v58: chat.js's refreshCtxLimit() privacy-mode branch now wipes the live
// in-memory conversation only once per page load, not on every 30s poll tick
// (GUI-LIVE-WIPE) - a stale-cached client would keep erasing a user's own
// first message every 30 seconds in privacy mode.
// v59: pages/knowledge.js changed (reindex paths are batched in groups of 50 to
// respect the server's request size cap) - a stale-cached client would keep
// failing to re-index collections with more than 50 documents.
// v60: pages/settings.js changed (the experimental "Split media across GPUs"
// toggle in the Media section) - a stale-cached client would keep lacking the
// control.
// v61: app/models-sidebar.js's renderHwStats() now renders the status-bar VRAM
// figure as a span with a subtle fullness colour, shown ONLY when the reading is
// a fresh, device-global measurement (the backend withholds used/free for a stale
// or process-blind one) - a stale-cached client would keep presenting an
// untrustworthy VRAM number as live and lack the fullness colour.
// v62: the v58 privacyWiped latch is now reset when the poll sees the server
// leave privacy mode, so a LATER restart back into privacy mode wipes newly-
// accumulated non-privacy leftovers again instead of leaving them painted in
// the sidebar (AUD-PRIV-2) - a stale-cached client would suppress that wipe.
// v63: app/settings-perf.js no longer hides the Main GPU / Split rows on an
// INCONCLUSIVE /api/gpus probe (probe_status timeout/busy) - a stale-cached
// client would keep concluding "single GPU" from a wedged driver and hide
// controls a multi-GPU box owns.
// v68: knowledge.js changed (reindex-needed badge fix + gpu_fallback_reason
// surfacing) - bump so an already-installed PWA picks up the new bytes.
// v73: knowledge.js now surfaces documents a folder re-sync found missing from
// disk (flagged, not deleted) - a stale-cached client would show the collection
// as fully healthy while part of its index no longer matches the filesystem.
// v74: index.html, pages/settings.js and style.css changed (the Settings page
// gained a search box that filters every group at once) - a stale-cached client
// would keep the old markup, so the box would be missing entirely.
// v75: coder.js renders the new episodes_recalled event (which past lessons a
// run pulled in, with the id to forget one by) - a stale-cached client would
// silently ignore the event and show nothing.
// v76: coder.js changed (set_todos tool cards show plan progress on the head
// line) - bump so an already-installed PWA picks up the new bytes.
// v77: the tts plugin's server-side settings got a real write surface - a new
// Text-to-speech settings section (pages/settings.js), the per-browser voice
// override is now named and clearable (app/settings-perf.js), and the chat
// picker label says which store it writes (index.html). A stale-cached client
// would keep the old picker that silently diverged from the server default.
// v78: a coder approval card now names the sub-agent that is asking
// (app/coder.js renders the confirm_request event's new `agent` field, styled by
// style.css) - a stale-cached client would drop the label and show the anonymous
// prompt that made two concurrent children indistinguishable.
// v79: pages/knowledge.js changed - a failed collection delete or document
// removal now shows the server's reason instead of a bare "Delete failed". Those
// endpoints answer 409 while another localm process is writing the collection,
// and that detail names the holder and says it is worth retrying; a stale-cached
// client would keep swallowing it and leave the user with a dead end.
// v80: a coder task whose verification could not run is now labelled "not
// verified" on the final line (app/coder.js reads the final event's new
// `verify_state`) - a stale-cached client would render the old unqualified
// "Task finished" and report an unverified task as a clean one.
// v81: app/settings-perf.js changed - a chat send that failed before any token
// streamed no longer has its rendered error wiped a moment later, and the
// request now falls back to the loaded model when the dropdown is empty or
// desynced - a stale-cached client would keep both bugs.
// v82: app/main.js now installs window.X as a live getter into each module's
// namespace object instead of a one-time value copy, so reassigned `export let`
// bindings (modelCache and friends) stay live on window instead of freezing at
// their value from first load - a stale-cached client would keep the old
// snapshot-copy loop and window.modelCache would never reflect model switches.
// v83: pages/knowledge.js handles the new "unknown" embedding status, which the
// server now returns instead of a file-existence answer when the model is an
// owner-chosen path and the caller is not the owner. A stale-cached client would
// fall through to the else-branch and tell that user the model is "not installed"
// and to pick another one - both false, and the second one unactionable.
// v84: app/models-sidebar.js's refreshModels() now refreshes the Settings "Live
// tuning" VRAM estimate when its 30s poll finds the active model changed from
// somewhere else (another tab/device, the CLI, an MCP client) - a stale-cached
// client would keep the old poll and the estimate would stay stuck on whatever
// model was active when Settings was opened.
// v85: app/models-sidebar.js and pages/models.js now warn (instead of a plain
// success toast) when a model load's own response reports a partial/zero GPU
// offload (AGENTS.md rule 5 - a silent CPU fallback must not read as success).
// A stale-cached client would keep toasting an unqualified "switched" even
// when the server told it the load degraded to slow CPU layers.
// v87: app/coder.js handles a new "assistant_text" event (a post-parse fix-up
// the agent loop now sends so a tool call written in a shape the live token
// stream cannot hide, e.g. a ```json fence, does not linger as raw JSON in
// the chat bubble once it has actually been executed). A stale-cached client
// would silently drop this event and keep showing the leaked JSON forever.
// v89: index.html gained a LoRA picker + strength controls on the Image page,
// and pages/images.js now populates/reads them. A stale-cached client would
// keep the old form with no way to select a LoRA at all.
// v90: pages/models.js and style.css changed (HF search results now show a
// model's architecture family, a Mixture-of-Experts badge, and an approximate
// parameter count) - a stale-cached client would keep the old rows with none
// of that information.
// v91: the "Report a bug" form (index.html) and its handler (models.js) split
// into what-I-did / what-I-expected / what-happened fields (#958) - a
// stale-cached client would keep the old single-textarea form and the new
// fields would never reach the server.
// v92: ADR-0004 Unit B - a "Warm up now" action next to Settings' embedding_model
// field (settings.js) plus its CSS (style.css). A stale-cached client would keep
// serving the old settings page with no button at all.
// v93: app/settings-perf.js changed - a failed chat send now parses the JSON
// error body instead of slicing the raw text, so the user sees the server's
// actual reason (and, for a VRAM-overflow 503, its full "Options:" list)
// instead of truncated raw JSON markup. A stale-cached client would keep
// showing the old sliced-and-mangled error.
const CACHE = "localm-shell-v93";
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

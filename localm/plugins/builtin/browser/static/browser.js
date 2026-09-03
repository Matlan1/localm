// SPDX-License-Identifier: AGPL-3.0-or-later
// Browser plugin client entry (ES module).
//
// loadClientPlugins() imports this for the active browser plugin and calls
// register(ctx) with ctx = { registerTTS, toast, authHeaders, voicesChanged }.
//
// A plugin tab's section is not created for us, so this builds its own
// #view-browser into #main (idempotently) and chains window.onViewShown.
//
// The live view is the job's own SSE stream: the worker pushes one "frame"
// event per rendered frame and this sets it as the image source. Frames are the
// only event type a job does not keep in its replay history, so re-opening the
// tab shows the current picture rather than every picture since it started.
//
// SECURITY: every server-originating string reaches the DOM via textContent,
// never innerHTML. A frame's payload is used only as a data: image source, and
// is rejected unless it is base64.

const API = "/api/browser";

// A frame is base64 and nothing else. Anything failing this is not rendered,
// so a payload cannot become another URL scheme.
const B64 = /^[A-Za-z0-9+/]*={0,2}$/;

/** The data: URL for one frame payload, or null when it is not base64. Exported
 *  so the guard itself is what gets tested, not a copy of its pattern. */
export function frameSrc(data) {
  if (typeof data !== "string" || !data || !B64.test(data)) return null;
  return "data:image/jpeg;base64," + data;
}

function el(tag, cls, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined && text !== null) n.textContent = String(text);
  return n;
}

/** Load this plugin's own stylesheet, once.
 *
 *  The tab builds itself into the SPA's #main, and the host stylesheet has no
 *  rules for any browser-* class, so without this every control is unstyled and
 *  the frame overflows its container. Resolved from the module's own URL so it
 *  works wherever the plugin's assets are mounted. */
function ensureStyles() {
  const id = "browser-plugin-styles";
  if (document.getElementById(id)) return;
  const link = document.createElement("link");
  link.id = id;
  link.rel = "stylesheet";
  link.href = new URL("./browser.css", import.meta.url).href;
  (document.head || document.documentElement).appendChild(link);
}

export function register(ctx) {
  ctx = ctx || {};
  ensureStyles();
  const toast = typeof ctx.toast === "function" ? ctx.toast : () => {};
  const authHeaders =
    typeof ctx.authHeaders === "function" ? ctx.authHeaders : () => ({});

  const main = document.getElementById("main") || document.body;
  let view = document.getElementById("view-browser");
  if (view) return;                       // already built by an earlier register
  view = el("section", "view");
  view.id = "view-browser";
  main.appendChild(view);

  const bar = el("div", "browser-bar");
  const url = el("input", "browser-url");
  url.type = "text";
  url.placeholder = "https://example.com";
  const go = el("button", "primary", "Open");
  const stop = el("button", "", "Stop");
  stop.disabled = true;
  const watch = el("button", "", "Watch the agent");
  watch.hidden = true;
  watch.title = "Show the browser the coding agent is driving";
  bar.append(url, go, stop, watch);

  const shot = el("img", "browser-frame");
  shot.alt = "Live view of the automated browser";
  const status = el("div", "browser-status", "No browser open.");
  const refused = el("ul", "browser-refused");

  view.append(el("h2", "", "Browser"), bar, status, shot, refused);

  let jobId = null;
  let abort = null;
  // "idle" nothing open, "own" this tab's own browser, "agent" watching the
  // one the coding agent drives. Only "own" is drivable: /api/browser/navigate
  // resolves the caller's OWN gui- session, never the agent's, so an address
  // bar in "agent" mode would drive a browser other than the one on screen.
  let mode = "idle";
  // True while a request is in flight, so a second click cannot open a second
  // browser or navigate one that has not finished opening.
  let busy = false;

  function applyControls() {
    const idle = mode === "idle";
    const agent = mode === "agent";
    go.textContent = idle ? "Open" : "Go";
    go.disabled = agent || busy;
    go.title = agent
      ? "The coding agent drives this browser. The tab is watching it."
      : (idle ? "Open a browser at this address"
              : "Send this browser to another address");
    stop.disabled = idle;
    url.disabled = agent;
  }

  function setMode(next) { mode = next; applyControls(); }
  function setBusy(on) { busy = on; applyControls(); }

  function showFrame(data) {
    const src = frameSrc(data);
    if (src) shot.src = src;
  }

  function addRefusal(text) {
    const item = el("li", "", text);
    refused.appendChild(item);
    while (refused.childElementCount > 50) refused.removeChild(refused.firstChild);
  }

  /** Read the job's SSE stream and paint each frame. */
  async function stream(id) {
    abort = new AbortController();
    let res;
    try {
      res = await fetch("/api/jobs/" + encodeURIComponent(id) + "/events",
                        { headers: authHeaders(), signal: abort.signal });
    } catch (e) {
      status.textContent = "Live view disconnected.";
      return;
    }
    if (!res.ok || !res.body) {
      status.textContent = "Live view unavailable (" + res.status + ").";
      return;
    }
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    let lastLine = "";
    for (;;) {
      let chunk;
      try {
        chunk = await reader.read();
      } catch (e) {
        break;                            // aborted, or the connection dropped
      }
      if (chunk.done) break;
      buf += dec.decode(chunk.value, { stream: true });
      const parts = buf.split("\n\n");
      buf = parts.pop();
      for (const part of parts) {
        const line = part.split("\n").find((l) => l.startsWith("data:"));
        if (!line) continue;
        let ev;
        try {
          ev = JSON.parse(line.slice(5).trim());
        } catch (e) {
          continue;
        }
        if (ev.type === "frame") showFrame(ev.data);
        else if (ev.type === "line" && (ev.line || ev.text)) {
          lastLine = ev.line || ev.text;
          status.textContent = lastLine;
        } else if (ev.type === "end") {
          setMode("idle");
          const failed = ev.status && ev.status !== "done" && ev.status !== "cancelled";
          if (failed) {
            status.textContent = lastLine || ("Browser stopped: " + ev.status);
            toast(status.textContent, true);
          } else {
            status.textContent = "Browser closed.";
          }
        }
      }
    }
  }

  async function open() {
    setMode("own");
    setBusy(true);
    status.textContent = "Starting the browser...";
    try {
      const r = await fetch(API + "/session", {
        method: "POST", headers: authHeaders(),
        body: JSON.stringify({ url: url.value.trim() || null }),
      });
      const data = await r.json().catch(() => ({}));
      if (!r.ok) {
        setMode("idle");
        status.textContent = data.detail || "Could not open a browser.";
        toast(status.textContent, true);
        return;
      }
      jobId = data.job_id;
      stream(jobId);
      poll();
    } catch (e) {
      setMode("idle");
      status.textContent = "Could not open a browser.";
    } finally {
      setBusy(false);
    }
  }

  async function close() {
    if (abort) { try { abort.abort(); } catch (e) { /* already gone */ } }
    abort = null;
    const id = jobId;
    const own = mode === "own";
    setMode("idle");
    jobId = null;
    // The worker loops until it is asked to stop, and its exit is what closes
    // the browser (own view) or hands the agent's back with its screencast off
    // (agent view). Without this the job outlives the viewer in either mode.
    if (id) {
      try {
        await fetch("/api/jobs/" + encodeURIComponent(id) + "/cancel",
                    { method: "POST", headers: authHeaders() });
      } catch (e) { /* it may already have ended */ }
    }
    if (own) {
      try {
        await fetch(API + "/stop", { method: "POST", headers: authHeaders() });
      } catch (e) { /* the server may already have closed it */ }
    }
    status.textContent = own ? "Browser closed."
                             : "Stopped watching the agent browser.";
  }

  /** Drive the already-open browser to another address.
   *
   *  Without this the tab could reach only ONE url per session: the address
   *  field was disabled while a browser was open, and Open would POST /session
   *  again and take that route's "already open for this key" refusal. */
  async function navigate() {
    const target = url.value.trim();
    if (!target) return;
    setBusy(true);
    status.textContent = "Loading " + target + "...";
    try {
      const r = await fetch(API + "/navigate", {
        method: "POST", headers: authHeaders(),
        body: JSON.stringify({ url: target }),
      });
      const data = await r.json().catch(() => ({}));
      if (!r.ok) {
        status.textContent = data.detail || "Could not go there.";
        toast(status.textContent, true);
        return;
      }
      if (data.ok) {
        status.textContent = "Opened " + (data.url || target);
      } else {
        status.textContent = "Refused: " + (data.refused || data.error || target);
        toast(status.textContent, true);
      }
    } catch (e) {
      status.textContent = "Could not go there.";
    } finally {
      setBusy(false);
    }
  }

  /** The address bar's action: open the first browser, then navigate it. */
  function submit() {
    if (mode === "own") navigate();
    else if (mode === "idle") open();
  }

  /** Refresh the refusal list, so a blocked destination is visible rather than
   *  showing only as a page that did not load. */
  async function poll() {
    if (!jobId) return;
    try {
      const r = await fetch(API + "/state", { headers: authHeaders() });
      if (r.ok) {
        const st = await r.json();
        refused.replaceChildren();
        for (const b of st.blocked || []) addRefusal(b.url + "  -  " + b.reason);
      }
    } catch (e) { /* transient; the next tick retries */ }
    if (jobId) setTimeout(poll, 2000);
  }

  async function refreshAgentOffer() {
    // The agent's browser is a different session from this tab's own, so it is
    // only watchable while the agent actually has one open.
    try {
      const r = await fetch(API + "/agent", { headers: authHeaders() });
      if (!r.ok) { watch.hidden = true; return; }
      const st = await r.json();
      watch.hidden = !st.available;
    } catch (e) {
      watch.hidden = true;
    }
  }

  async function watchAgent() {
    setMode("agent");
    status.textContent = "Attaching to the agent browser...";
    try {
      const r = await fetch(API + "/agent", {
        method: "POST", headers: authHeaders(),
      });
      const data = await r.json().catch(() => ({}));
      if (!r.ok) {
        setMode("idle");
        status.textContent = data.detail || "No agent browser to watch.";
        toast(status.textContent, true);
        return;
      }
      jobId = data.job_id;
      stream(jobId);
      poll();
    } catch (e) {
      setMode("idle");
      status.textContent = "Could not attach to the agent browser.";
    }
  }

  go.onclick = submit;
  stop.onclick = close;
  watch.onclick = watchAgent;
  url.onkeydown = (e) => { if (e.key === "Enter" && !go.disabled) submit(); };
  setMode("idle");

  const prev = window.onViewShown;
  window.onViewShown = (name) => {
    if (prev) prev(name);
    if (name === "browser") {
      refreshAgentOffer();
      if (jobId) poll();
    }
  };
}

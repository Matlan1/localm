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

export function register(ctx) {
  ctx = ctx || {};
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

  function setRunning(on) {
    go.disabled = on;
    stop.disabled = !on;
    url.disabled = on;
  }

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
          setRunning(false);
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
    setRunning(true);
    status.textContent = "Starting the browser...";
    try {
      const r = await fetch(API + "/session", {
        method: "POST", headers: authHeaders(),
        body: JSON.stringify({ url: url.value.trim() || null }),
      });
      const data = await r.json().catch(() => ({}));
      if (!r.ok) {
        setRunning(false);
        status.textContent = data.detail || "Could not open a browser.";
        toast(status.textContent, true);
        return;
      }
      jobId = data.job_id;
      stream(jobId);
      poll();
    } catch (e) {
      setRunning(false);
      status.textContent = "Could not open a browser.";
    }
  }

  async function close() {
    if (abort) { try { abort.abort(); } catch (e) { /* already gone */ } }
    abort = null;
    try {
      await fetch(API + "/stop", { method: "POST", headers: authHeaders() });
    } catch (e) { /* the server may already have closed it */ }
    setRunning(false);
    jobId = null;
    status.textContent = "Browser closed.";
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
    setRunning(true);
    status.textContent = "Attaching to the agent browser...";
    try {
      const r = await fetch(API + "/agent", {
        method: "POST", headers: authHeaders(),
      });
      const data = await r.json().catch(() => ({}));
      if (!r.ok) {
        setRunning(false);
        status.textContent = data.detail || "No agent browser to watch.";
        toast(status.textContent, true);
        return;
      }
      jobId = data.job_id;
      stream(jobId);
      poll();
    } catch (e) {
      setRunning(false);
      status.textContent = "Could not attach to the agent browser.";
    }
  }

  go.onclick = open;
  stop.onclick = close;
  watch.onclick = watchAgent;
  url.onkeydown = (e) => { if (e.key === "Enter" && !go.disabled) open(); };

  const prev = window.onViewShown;
  window.onViewShown = (name) => {
    if (prev) prev(name);
    if (name === "browser") {
      refreshAgentOffer();
      if (jobId) poll();
    }
  };
}

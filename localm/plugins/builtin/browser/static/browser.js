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

/** Map client click coordinates to page coordinates on an image with
 *  object-fit: contain, or null when outside the rendered content. */
export function frameCoords(img, clientX, clientY) {
  const nw = img.naturalWidth;
  const nh = img.naturalHeight;
  if (!nw || !nh) return null;
  const rect = img.getBoundingClientRect ? img.getBoundingClientRect() : null;
  if (!rect || !rect.width || !rect.height) return null;

  const imgRatio = nw / nh;
  const elemRatio = rect.width / rect.height;

  let rw, rh, ox, oy;
  if (elemRatio > imgRatio) {
    rh = rect.height;
    rw = rh * imgRatio;
    ox = (rect.width - rw) / 2;
    oy = 0;
  } else {
    rw = rect.width;
    rh = rw / imgRatio;
    ox = 0;
    oy = (rect.height - rh) / 2;
  }

  const cx = clientX - rect.left - ox;
  const cy = clientY - rect.top - oy;
  if (cx < 0 || cx > rw || cy < 0 || cy > rh) return null;

  return {
    x: Math.round(cx * (nw / rw)),
    y: Math.round(cy * (nh / rh)),
  };
}

/** Read one job's SSE stream and dispatch each event. Exported so a second
 *  caller (the coder session's opt-in inline mirror) reuses the exact same
 *  parsing rather than a second copy of it - a job's own replay history never
 *  keeps more than the latest frame (see jobs.FRAME_EVENT), so a caller with
 *  its own re-derived parser is the only way this could quietly diverge.
 *
 *  Resolves once the stream ends (aborted, disconnected, or an "end" event).
 *  onFetchFailed and onUnavailable are two different failures a caller may
 *  want to tell apart: the first is the network request itself failing
 *  (offline, aborted before a response arrived), the second is a response
 *  that arrived but refused the request (job not found, not owned). */
export async function watchFrames(jobId, {
  authHeaders, signal, onFrame, onLine, onEnd, onFetchFailed, onUnavailable,
} = {}) {
  let res;
  try {
    res = await fetch("/api/jobs/" + encodeURIComponent(jobId) + "/events",
                      { headers: authHeaders ? authHeaders() : {}, signal });
  } catch (e) {
    if (onFetchFailed) onFetchFailed(e);
    return;
  }
  if (!res.ok || !res.body) {
    if (onUnavailable) onUnavailable(res.status);
    return;
  }
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
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
      if (ev.type === "frame") { if (onFrame) onFrame(ev.data); }
      else if (ev.type === "line" && (ev.line || ev.text)) {
        if (onLine) onLine(ev.line || ev.text);
      } else if (ev.type === "end") {
        if (onEnd) onEnd(ev);
      }
    }
  }
}

function el(tag, cls, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined && text !== null) n.textContent = String(text);
  return n;
}

/** The GUI's translation of key through window.tOr (app/i18n.js, bridged onto
 *  window by app/main.js), or the English fallback with its {name} params
 *  filled in when this module runs without the GUI shell. */
function tr(key, fallback, params) {
  if (typeof window !== "undefined" && typeof window.tOr === "function") {
    return window.tOr(key, fallback, params);
  }
  return fallback.replace(/\{(\w+)\}/g, (whole, name) =>
    (params && Object.prototype.hasOwnProperty.call(params, name)
      ? String(params[name]) : whole));
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
  const go = el("button", "btn-primary", "Open");
  const stop = el("button", "btn-secondary", "Stop");
  stop.disabled = true;
  const watch = el("button", "btn-secondary", "Watch the agent");
  watch.hidden = true;
  watch.title = "Show the browser the coding agent is driving";
  bar.append(url, go, stop, watch);

  const shot = el("img", "browser-frame");
  shot.alt = "Live view of the automated browser";
  shot.hidden = true;
  shot.tabIndex = 0;
  // Names the keys that leave the frame. Shown while the frame holds focus
  // and its keys go to the page.
  const keysHint = el("div", "browser-keys-hint");
  keysHint.id = "browser-keys-hint";
  keysHint.hidden = true;
  shot.setAttribute("aria-describedby", keysHint.id);
  function paintKeysHint() {
    keysHint.textContent = tr("browser.liveView.keysHint",
      "Keys go to the page · Esc releases the keyboard · Shift+Tab moves back");
  }
  paintKeysHint();
  document.addEventListener("localm:language", paintKeysHint);
  const status = el("div", "browser-status", "No browser open.");
  const refused = el("ul", "browser-refused");

  view.append(el("h2", "", "Browser"), bar, status, shot, keysHint, refused);

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

  let wheelTimer = null;
  let pendingDx = 0;
  let pendingDy = 0;

  // Clicks, keys, typed text and wheel deltas for the page, oldest first.
  // pumpInput() sends the head and waits for its answer before it sends the
  // next, so one input request is in flight at a time and requests reach the
  // browser in the order they were made. Typed text and wheel deltas merge into
  // an entry of the same kind waiting at the tail.
  const inputQueue = [];
  let inputBusy = false;
  // Incremented, and inputQueue emptied, whenever the view leaves "own" mode.
  // A request answered after that changes nothing on screen.
  let inputEpoch = 0;
  // True from a failed input request until one succeeds again.
  let inputFailing = false;

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

  function setMode(next) {
    mode = next;
    // Clearing here rather than at each call site: every path back to idle
    // runs through this, and a frame left behind shows the last screenshot of
    // a browser that is no longer open.
    if (next === "idle") {
      shot.hidden = true;
      shot.removeAttribute("src");
    }
    if (next !== "own") {
      if (wheelTimer) { clearTimeout(wheelTimer); wheelTimer = null; }
      pendingDx = 0;
      pendingDy = 0;
      inputQueue.length = 0;
      inputEpoch++;
      inputBusy = false;
      inputFailing = false;
      keysHint.hidden = true;
    }
    applyControls();
  }
  function setBusy(on) { busy = on; applyControls(); }

  function showFrame(data) {
    const src = frameSrc(data);
    if (src) { shot.src = src; shot.hidden = false; }
  }

  function addRefusal(text) {
    const item = el("li", "", text);
    refused.appendChild(item);
    while (refused.childElementCount > 50) refused.removeChild(refused.firstChild);
  }

  /** Read the job's SSE stream and paint each frame. */
  async function stream(id) {
    abort = new AbortController();
    let lastLine = "";
    await watchFrames(id, {
      authHeaders,
      signal: abort.signal,
      onFrame: showFrame,
      onLine: (text) => { lastLine = text; status.textContent = text; },
      onEnd: (ev) => {
        setMode("idle");
        const failed = ev.status && ev.status !== "done" && ev.status !== "cancelled";
        if (failed) {
          status.textContent = lastLine || ("Browser stopped: " + ev.status);
          toast(status.textContent, true);
        } else {
          status.textContent = "Browser closed.";
        }
      },
      onFetchFailed: () => { status.textContent = "Live view disconnected."; },
      onUnavailable: (code) => {
        status.textContent = "Live view unavailable (" + code + ").";
      },
    });
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

  /** Put one input for the page on inputQueue, after any wheel deltas gathered
   *  before it, and start sending if nothing is in flight. Input made outside
   *  "own" mode is dropped. */
  function sendInput(path, body) {
    if (mode !== "own") return;
    if (path !== "/scroll") flushWheel();
    const tail = inputQueue[inputQueue.length - 1];
    if (tail && tail.path === path && path === "/type") {
      tail.body.text += body.text;
    } else if (tail && tail.path === path && path === "/scroll") {
      tail.body.delta_x += body.delta_x;
      tail.body.delta_y += body.delta_y;
    } else {
      inputQueue.push({ path, body });
    }
    pumpInput();
  }

  /** Send the head of inputQueue unless an input request is already in flight,
   *  then the next one once it is answered. */
  async function pumpInput() {
    if (inputBusy || !inputQueue.length) return;
    inputBusy = true;
    const epoch = inputEpoch;
    const { path, body } = inputQueue.shift();
    try {
      await deliverInput(path, body, epoch);
    } finally {
      if (epoch === inputEpoch) {
        inputBusy = false;
        pumpInput();
      }
    }
  }

  /** POST one input to the page. A failure is shown on the status line, and
   *  toasted when it follows a success, and never thrown: the inputs queued
   *  behind it are still sent. */
  async function deliverInput(path, body, epoch) {
    let problem = null;
    try {
      const r = await fetch(API + path, {
        method: "POST", headers: authHeaders(), body: JSON.stringify(body),
      });
      const data = await r.json().catch(() => ({}));
      if (!r.ok) {
        problem = typeof data.detail === "string" && data.detail
          ? data.detail : "HTTP " + r.status;
      } else if (data && data.ok === false) {
        problem = data.error || "HTTP " + r.status;
      }
    } catch (err) {
      problem = (err && err.message) || String(err);
    }
    if (epoch !== inputEpoch) return;
    if (problem) {
      status.textContent = tr("browser.liveView.inputFailed",
        "Input did not reach the browser: {detail}", { detail: problem });
      if (!inputFailing) toast(status.textContent, true);
      inputFailing = true;
    } else if (inputFailing) {
      inputFailing = false;
      status.textContent = tr("browser.liveView.inputRestored",
        "Input is reaching the browser again.");
    }
  }

  function flushWheel() {
    if (wheelTimer) { clearTimeout(wheelTimer); wheelTimer = null; }
    if (!pendingDx && !pendingDy) return;
    const dx = pendingDx;
    const dy = pendingDy;
    pendingDx = 0;
    pendingDy = 0;
    sendInput("/scroll", { delta_x: dx, delta_y: dy });
  }

  shot.onclick = (e) => {
    if (mode !== "own") return;
    if (typeof shot.focus === "function") shot.focus();
    const coords = frameCoords(shot, e.clientX, e.clientY);
    if (!coords) return;
    sendInput("/click", { x: coords.x, y: coords.y, button: "left" });
  };

  shot.addEventListener("wheel", (e) => {
    if (mode !== "own") return;
    e.preventDefault();
    pendingDx += e.deltaX;
    pendingDy += e.deltaY;
    if (!wheelTimer) {
      wheelTimer = setTimeout(() => {
        wheelTimer = null;
        flushWheel();
      }, 40);
    }
  }, { passive: false });

  shot.addEventListener("focus", () => { keysHint.hidden = mode !== "own"; });
  shot.addEventListener("blur", () => { keysHint.hidden = true; });

  shot.onkeydown = (e) => {
    if (mode !== "own") return;
    // Esc takes keyboard focus off the frame and is not sent to the page.
    // Shift+Tab keeps its default, moving focus to the control before it.
    if (e.key === "Escape") {
      e.preventDefault();
      shot.blur();
      return;
    }
    if (e.key === "Tab" && e.shiftKey) return;
    const navKeys = [
      "Backspace", "Enter", "Tab",
      "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight",
      "PageUp", "PageDown", "Home", "End", "Delete",
    ];
    if (navKeys.includes(e.key)) {
      e.preventDefault();
      sendInput("/key", { key: e.key });
      return;
    }
    if (!e.ctrlKey && !e.altKey && !e.metaKey && e.key && e.key.length === 1) {
      e.preventDefault();
      sendInput("/type", { text: e.key });
    }
  };

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

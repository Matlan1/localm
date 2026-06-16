# localm GUI: web-first vs Tauri (historical ADR)

Historical architecture-decision record for why the web-first GUI shipped instead
of the originally planned Tauri shell. Kept for the rationale; for current GUI
behaviour see `docs/gui.md` and `docs/architecture.md`.

---

**Implemented:** web-first GUI served by the FastAPI server (`localm/plugins/gui/`). `localm gui` starts the server and opens the browser. See `docs/gui.md`.

**Original plan:** Tauri 2 + Svelte 5 native shell. Deliberately deferred, not abandoned.

## Why web-first shipped instead of Tauri

- Zero runtime dependencies and no npm dependency tree. The frontend is vanilla JS with three pinned, vendored single-file libraries (marked, DOMPurify, highlight.js). No install scripts, no supply chain surface.
- Fully offline once installed, matching the project's offline-first policy.
- The Tauri plan's own architecture had the frontend talking to the FastAPI server over HTTP. That layer is now built and tested; a Tauri shell can wrap the same frontend later without rework (native window + sidecar management is all it adds).
- Iteration speed: refresh the page instead of recompiling Rust.

## What exists today

- `localm gui [MODEL]` command: engine startup, port auto-pick, browser launch
- Chat: streaming, markdown + highlighted code, params drawer (temperature, top-p, max tokens, seed, system prompt), conversations in localStorage, stop, copy buttons, usage stats (TTFT, tok/s)
- Model selector with live engine switching under the inference semaphore
- Coder sessions: per-session Agent in a worker thread, SSE event stream (tokens, tool cards, results), browser approval flow with unified diff previews, stop / end session, busy-state handling
- Agent event hooks (`on_event`, `confirm_handler`, `request_stop`) usable by any future frontend, including a Tauri shell
- Bearer auth honoured when `LOCALM_API_KEY` is set; CORS stays localhost-locked

## What's next

Open GUI work is tracked in one place: the **Web GUI** section of [TODO.md](../TODO.md) (missing chat features, coder features, and pages, including the eventual Tauri 2 shell). This file only records the architecture decision and its rationale.

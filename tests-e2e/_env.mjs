// SPDX-License-Identifier: AGPL-3.0-or-later
// Shared environment for the GUI end-to-end (real-browser) test. Everything here
// is resolved at runtime relative to the repo - no hardcoded absolute or
// machine-specific paths (AGENTS.md rule 1), and it depends only on this repo's
// own venv + a throwaway data dir (rule 4).

import path from "node:path";
import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
export const REPO_ROOT = path.resolve(HERE, "..");

// The Python that runs localm: this project's own venv (self-contained, rule 4).
// Walk UP from the repo to find it, so this works both in a normal clone (.venv at
// the repo root) AND in a git worktree, whose .venv lives in the main working tree
// a few levels up. Fall back to `python` on PATH. LOCALM_E2E_PYTHON forces a
// specific interpreter but should almost never be needed.
function resolvePython() {
  if (process.env.LOCALM_E2E_PYTHON) return process.env.LOCALM_E2E_PYTHON;
  const rel = process.platform === "win32"
    ? path.join(".venv", "Scripts", "python.exe")
    : path.join(".venv", "bin", "python");
  let dir = REPO_ROOT;
  for (let i = 0; i < 8; i++) {
    const cand = path.join(dir, rel);
    if (existsSync(cand)) return cand;
    const parent = path.dirname(dir);
    if (parent === dir) break;         // filesystem root
    dir = parent;
  }
  return "python";
}
export const PYTHON = resolvePython();

// A throwaway, isolated data dir so the test never touches real models/config/
// chat history. Per-run (pid-suffixed) so a stale, still-locked dir from a crashed
// prior run is never reused; recreated fresh in global-setup.
//
// Under the repo's own gitignored `.claude/`, NOT the OS temp dir: a run writes a
// whole LOCALM_HOME plus every plugin it installs, and on Windows os.tmpdir() puts
// that on the system drive. LOCALM_E2E_HOME still overrides it.
export const LOCALM_HOME =
  process.env.LOCALM_E2E_HOME
  || path.join(REPO_ROOT, ".claude", "e2e-home", `run-${process.pid}`);

// A distinctive fixed port. reuseExistingServer is OFF, so if this port is busy
// (e.g. another session), the test FAILS LOUDLY starting its own server rather
// than attaching to a stranger's - it never adopts or kills a foreign server.
export const PORT = Number(process.env.LOCALM_E2E_PORT || 8795);
export const BASE_URL = `http://127.0.0.1:${PORT}`;

// The user-facing plugins whose nav tabs must switch. Installing these is what
// reproduces the #357 condition (plugin tabs blank): a kernel-only boot would not
// have caught it. chat is the baseline (always-on) plugin and jobs `requires` it,
// so install it FIRST or jobs will not fully load in a fresh home.
export const PLUGINS = ["chat", "coder", "image", "music", "video", "jobs", "rag"];

// Every tab that must build, activate, and render after a real boot.
export const EXPECTED_TABS = [
  "chat", "coder", "images", "music", "video",
  "knowledge", "jobs", "models", "plugins", "settings",
];

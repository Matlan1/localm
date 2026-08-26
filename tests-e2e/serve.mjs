// SPDX-License-Identifier: AGPL-3.0-or-later
// Ordered launcher for the e2e server: build a fresh throwaway LOCALM_HOME,
// INSTALL the user-facing plugins, THEN start the GUI - all in one process, so the
// server never binds before its plugin routes/assets exist. (A Playwright
// globalSetup does NOT reliably finish before webServer starts, which raced the
// server ahead of the install: plugin routes 404'd and the jobs client_entry -
// /plugins/jobs/jobs.js - failed to load, so #view-jobs never built.)
//
// Playwright runs this as the webServer command and tree-kills it (and the localm
// child) on teardown, so nothing is left bound.

import { execFileSync, spawn } from "node:child_process";
import { rmSync, mkdirSync } from "node:fs";
import { PYTHON, REPO_ROOT, LOCALM_HOME, PORT, PLUGINS } from "./_env.mjs";

try { rmSync(LOCALM_HOME, { recursive: true, force: true }); }
catch (e) { console.warn(`[e2e] could not clear ${LOCALM_HOME} (reusing it):`, e.message); }
mkdirSync(LOCALM_HOME, { recursive: true });

const env = { ...process.env, LOCALM_HOME };

// Every localm call this harness makes goes through here. A `plugin install`
// without --no-deps is REFUSED rather than run: PYTHON is the developer venv that
// every other session on the machine shares, and the install would resolve the
// plugins' pip extras into it.
function runLocalm(args) {
  if (args[0] === "plugin" && args[1] === "install" && !args.includes("--no-deps")) {
    throw new Error("[e2e] refusing `localm plugin install` without --no-deps: it would "
                    + "install pip extras into the shared developer venv this harness runs.");
  }
  execFileSync(PYTHON, ["-m", "localm", ...args], { cwd: REPO_ROOT, env, stdio: "inherit" });
}

// Install (which also enables) each first-party plugin, ONE target per call
// (`plugin install` takes a single TARGET). --no-deps: a boot/switch smoke needs
// only the nav tab + page module, not the generation backends, and pip extras
// would make this slow + online. chat is the baseline that jobs `requires`.
for (const name of PLUGINS) {
  runLocalm(["plugin", "install", name, "--no-deps"]);
}

// Now serve. Inherit stdio so [WebServer] output is visible; forward termination.
const srv = spawn(PYTHON, ["-m", "localm", "gui", "--no-browser", "-p", String(PORT)],
  { cwd: REPO_ROOT, env, stdio: "inherit" });
srv.on("exit", (code) => process.exit(code ?? 0));
for (const sig of ["SIGINT", "SIGTERM"]) process.on(sig, () => srv.kill());

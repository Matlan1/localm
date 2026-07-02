// SPDX-License-Identifier: AGPL-3.0-or-later
// Playwright config for the GUI end-to-end (real-browser) smoke. This is the ONE
// automated check that loads the real shipped ES-module graph in a real browser,
// fires DOMContentLoaded, and drives each page - the class of "the page does not
// load" bug the jsdom unit harness structurally cannot catch (it strips
// import/export and never boots). It is NOT part of `npm test` (which stays fast
// and needs no browser); run it explicitly with `npm run test:e2e`.
//
// Prereqs (once): `npm install` then `npx playwright install chromium`, and a
// localm importable from this repo (its .venv, or `python` on PATH with localm
// installed).

import { defineConfig } from "@playwright/test";
import { REPO_ROOT, BASE_URL } from "./tests-e2e/_env.mjs";

export default defineConfig({
  testDir: "./tests-e2e",
  fullyParallel: false,
  workers: 1,                       // one shared server instance
  forbidOnly: true,
  retries: 0,
  reporter: "list",
  timeout: 60_000,
  use: {
    baseURL: BASE_URL,
    headless: true,
    launchOptions: { args: ["--no-sandbox"] },   // harmless on Windows; portable
  },
  // tests-e2e/serve.mjs installs the plugins THEN starts the GUI in one ordered
  // process (no globalSetup race). Playwright waits for it to answer and tree-kills
  // ONLY this child (and its localm subprocess) on exit. reuseExistingServer:false
  // -> if the port is busy it fails loudly instead of adopting (or later killing) a
  // server it did not start.
  webServer: {
    command: "node tests-e2e/serve.mjs",
    url: BASE_URL,
    cwd: REPO_ROOT,
    reuseExistingServer: false,
    timeout: 120_000,
    stdout: "pipe",
    stderr: "pipe",
  },
});

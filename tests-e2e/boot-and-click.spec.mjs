// SPDX-License-Identifier: AGPL-3.0-or-later
// GUI boot-and-click smoke in a REAL browser (headless chromium). This is the one
// check that loads the actual shipped ES-module graph, fires DOMContentLoaded, and
// drives each page like a user - the exact thing the jsdom unit harness cannot do
// (it strips import/export into one shared realm and never boots). It would have
// caught BOTH shipped "blank page" bugs directly:
//   * #357 - reassigning the imported VIEWS threw "Assignment to constant variable"
//     in renderNav, so plugin tabs never activated (caught by the tab-switch loop);
//   * #358 - a top-level JSON.parse of corrupt localStorage aborted the module
//     graph and blanked the shell (caught by the corrupt-storage boot test).

import { test, expect } from "@playwright/test";
import { EXPECTED_TABS } from "./_env.mjs";

// Attach uncaught-error listeners BEFORE navigating. A read-only-import
// reassignment, a missing/misnamed export, and a top-level throw all surface as a
// page error (uncaught exception); resource 404s do NOT (those are console
// messages), so an empty pageerror list is a precise "the graph evaluated" signal.
function trackErrors(page) {
  const pageErrors = [];
  page.on("pageerror", (e) => pageErrors.push(String(e.message || e)));
  // Surface (don't swallow) browser-side failures so a diagnosis is in the test
  // output, not lost: a failed dynamic import (e.g. a plugin client_entry) logs a
  // console error + a failed request rather than a pageerror.
  page.on("console", (m) => {
    if (m.type() === "error") console.log(`[browser console.error] ${m.text()}`);
  });
  page.on("requestfailed", (r) => {
    console.log(`[request failed] ${r.url()} -> ${r.failure()?.errorText || "?"}`);
  });
  page.on("response", (r) => {
    if (r.status() >= 400) console.log(`[response ${r.status()}] ${r.url()}`);
  });
  return pageErrors;
}

test("every nav tab boots, activates, and renders in a real browser", async ({ page }) => {
  const pageErrors = trackErrors(page);

  await page.goto("/");
  // The plugin nav is built from /api/plugins during the DOMContentLoaded boot; if
  // the module graph failed to evaluate, this button never appears and we fail
  // here instead of hanging - proof the real ESM graph ran end to end.
  await expect(page.locator("#nav-coder")).toBeVisible({ timeout: 30_000 });

  for (const name of EXPECTED_TABS) {
    const nav = page.locator(`#nav-${name}`);
    await expect(nav, `nav button #nav-${name} should exist`).toHaveCount(1);
    await nav.click();
    const view = page.locator(`#view-${name}`);
    await expect(view, `#view-${name} should activate on click`).toHaveClass(/\bactive\b/);
    await expect(view, `#view-${name} should be visible`).toBeVisible();
    const children = await page.locator(`#view-${name} *`).count();
    expect(children, `#view-${name} should render content`).toBeGreaterThan(3);
    // exactly one view active at a time (a real switch, not an overlay)
    const active = await page.locator('[id^="view-"].active').count();
    expect(active, `exactly one active view after clicking ${name}`).toBe(1);
  }

  expect(pageErrors, "no uncaught JS errors during boot + navigation").toEqual([]);
});

test("boot survives CORRUPT localStorage (does not blank the shell)", async ({ page }) => {
  const pageErrors = trackErrors(page);
  const warnings = [];
  page.on("console", (m) => { if (m.type() === "warning") warnings.push(m.text()); });

  // Poison both JSON-backed keys BEFORE any page script runs. Pre-#358 this threw
  // SyntaxError at module-eval time in chat.js and aborted the whole graph.
  await page.addInitScript(() => {
    localStorage.setItem("localm.conversations", "{oops not json");
    localStorage.setItem("localm.convCollapsed", "&&&broken");
  });

  await page.goto("/");
  // The app must still boot: nav built, chat view rendered.
  await expect(page.locator("#nav-coder")).toBeVisible({ timeout: 30_000 });
  await expect(page.locator("#nav-chat")).toBeVisible();
  const chatChildren = await page.locator("#view-chat *").count();
  expect(chatChildren, "chat view still renders").toBeGreaterThan(3);

  expect(pageErrors, "corrupt storage must not throw an uncaught error").toEqual([]);
  // ...and the corruption is SURFACED, not silently swallowed (AGENTS.md rule 5).
  expect(warnings.some((w) => /corrupt localStorage/i.test(w)),
    "a warning should surface the corrupt localStorage entry").toBe(true);
});

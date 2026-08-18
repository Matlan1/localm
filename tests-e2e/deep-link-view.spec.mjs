// SPDX-License-Identifier: AGPL-3.0-or-later
// Deep links to a PLUGIN tab must land on that tab, in a real browser.
//
// `/?view=images` (or music/video/knowledge/coder/jobs) used to land on the MODELS
// page instead: at the moment init.js reads `?view=`, VIEWS still held only
// CORE_VIEWS, because the plugin tabs are appended by rebuildViews() once
// /api/capabilities resolves. A valid plugin tab therefore failed the
// `VIEWS.includes()` test and was silently replaced by "models" - and the line
// above it had already stripped the query, so nothing could retry it.
//
// Only a real browser can catch this: the jsdom unit harness strips import/export
// into one shared realm and never boots the module graph, so it has no boot
// ordering to get wrong. `localm gui` itself prints `/?view=models` (a CORE view),
// which is why this path was never exercised in normal use.
//
// Each test gets a fresh browser context, so every navigation below is a genuine
// COLD boot with empty localStorage - the exact condition that reproduces it.

import { test, expect } from "@playwright/test";
import { EXPECTED_TABS } from "./_env.mjs";

// Kernel views (tabs.js CORE_VIEWS) are in VIEWS from module-eval, so they never
// exercised this path. Everything else is a plugin tab and is what regressed.
const CORE = ["chat", "models", "plugins", "settings"];
const PLUGIN_TABS = EXPECTED_TABS.filter((t) => !CORE.includes(t));

function trackErrors(page) {
  const pageErrors = [];
  page.on("pageerror", (e) => pageErrors.push(String(e.message || e)));
  page.on("console", (m) => {
    if (m.type() === "error") console.log(`[browser console.error] ${m.text()}`);
  });
  // Name the URL behind any 4xx/5xx. A bare "failed to load resource" console
  // line cannot be told apart from a real missing plugin asset, and rule 5 only
  // allows ignoring a warning once it is PROVEN harmless.
  page.on("response", (r) => {
    if (r.status() >= 400) console.log(`[response ${r.status()}] ${r.url()}`);
  });
  return pageErrors;
}

// Which view the shell actually settled on, sampled through expect.poll so it is
// read at failure time rather than at call time.
async function activeViewId(page) {
  return await page.locator('[id^="view-"].active').first()
    .getAttribute("id").catch(() => null);
}

for (const tab of PLUGIN_TABS) {
  test(`deep link /?view=${tab} lands on ${tab}, not models`, async ({ page }) => {
    const pageErrors = trackErrors(page);

    await page.goto(`/?view=${tab}`);

    // Wait for the boot to SETTLE before judging. Both of these prove the async
    // work the deep link races has finished: the nav button only exists once
    // renderNav() ran off /api/capabilities (so VIEWS is populated), and
    // #view-jobs only exists once loadClientPlugins() imported the jobs client
    // entry. Gating on them is what makes a still-inactive view below mean "the
    // deep link was consumed and lost" rather than "we did not wait long enough".
    await expect(page.locator(`#nav-${tab}`)).toBeVisible({ timeout: 30_000 });
    await expect(page.locator(`#view-${tab}`)).toBeAttached({ timeout: 30_000 });

    // Assert on the DESTINATION first: it is the property under test, and a
    // failure names the page actually landed on rather than reading as a
    // tweakable class-matcher. Polled, so the value reported on failure is the
    // state AT failure time: an id interpolated eagerly into a message string
    // instead reports the static boot state (view-chat) and sends the next
    // reader after the wrong page.
    await expect.poll(() => activeViewId(page), {
      timeout: 15_000,
      message: `deep link /?view=${tab} should land on #view-${tab}`,
    }).toBe(`view-${tab}`);
    await expect(page.locator(`#view-${tab}`), `#view-${tab} should be visible`)
      .toBeVisible();

    // Exactly one view active. This is the guard against "fixing" the bug by
    // dropping the VIEWS.includes() test: _applyActiveClasses only ever touches
    // views listed in VIEWS, so activating a not-yet-registered tab strips
    // .active from chat and gives it to nothing - a blank shell, which is worse
    // than landing on the wrong page.
    const active = await page.locator('[id^="view-"].active').count();
    expect(active, `exactly one active view after deep-linking ${tab}`).toBe(1);

    const children = await page.locator(`#view-${tab} *`).count();
    expect(children, `#view-${tab} should render content`).toBeGreaterThan(3);

    expect(pageErrors, "no uncaught JS errors during a deep-linked boot").toEqual([]);
  });
}

test("deep link to a CORE view still works", async ({ page }) => {
  const pageErrors = trackErrors(page);
  await page.goto("/?view=settings");
  await expect(page.locator("#view-settings"))
    .toHaveClass(/\bactive\b/, { timeout: 30_000 });
  expect(await page.locator('[id^="view-"].active').count()).toBe(1);
  expect(pageErrors, "no uncaught JS errors").toEqual([]);
});

test("deep link to an UNKNOWN view still falls back to models", async ({ page }) => {
  const pageErrors = trackErrors(page);
  // A name that is not a kernel view and not any plugin's tab. Falling back to
  // models here is deliberate and must survive the fix: the point is to preserve
  // a tab that IS real once the plugin set is known, not to trust any string.
  await page.goto("/?view=not-a-real-view");
  await expect(page.locator("#view-models"))
    .toHaveClass(/\bactive\b/, { timeout: 30_000 });
  await expect(page.locator("#view-not-a-real-view")).toHaveCount(0);
  expect(await page.locator('[id^="view-"].active').count()).toBe(1);
  expect(pageErrors, "no uncaught JS errors").toEqual([]);
});

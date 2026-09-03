// SPDX-License-Identifier: AGPL-3.0-or-later
// Every button in the coder session toolbar has to be inside the viewport and
// hit-testable at a normal laptop width. The bar needs about 1380px of its own,
// and the coder column is far narrower than that once the sidebar and the
// sessions rail take their share, so a nowrap bar puts Stop and End off-screen
// with no way to scroll to them. jsdom cannot see this: it has no layout engine,
// so every one of these buttons "exists" there whatever the CSS does.

import { test, expect } from "@playwright/test";

const BUTTONS = [
  "coder-undo", "coder-files", "coder-controls", "coder-memory",
  "coder-bg", "coder-compact", "coder-export", "coder-log",
  "coder-history", "coder-stop", "coder-end",
];

test("every coder toolbar button is reachable at 1280px", async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 900 });
  await page.goto("/?view=coder");
  await expect(page.locator("#setup-cwd")).toBeVisible({ timeout: 30_000 });

  // The bar only renders with a session open; show it directly rather than
  // starting a real agent session, which needs a loaded model.
  await page.evaluate(() => document.getElementById("coder-bar").classList.add("open"));
  await page.waitForTimeout(200);

  const unreachable = await page.evaluate((ids) => {
    const bad = [];
    const vw = document.documentElement.clientWidth;
    for (const id of ids) {
      const el = document.getElementById(id);
      if (!el) { bad.push({ id, why: "missing" }); continue; }
      const r = el.getBoundingClientRect();
      if (r.width === 0 && r.height === 0) continue;   // hidden by state, not layout
      if (r.right > vw || r.left < 0) {
        bad.push({ id, why: "outside the viewport", right: Math.round(r.right), vw });
      }
    }
    return bad;
  }, BUTTONS);

  expect(unreachable, "a toolbar button off-screen cannot be clicked or scrolled to")
    .toEqual([]);
});

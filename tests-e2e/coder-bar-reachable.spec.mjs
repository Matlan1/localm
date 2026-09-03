// SPDX-License-Identifier: AGPL-3.0-or-later
// Every button in the coder session toolbar has to stay inside the bar's own box
// and remain hit-testable at a normal laptop width. The bar needs about 1380px
// and the coder column is far narrower once the sidebar and the sessions rail
// take their share, so a nowrap bar pushes Stop and End underneath the rail with
// no way to scroll to them. jsdom cannot see this: it has no layout engine, so
// every one of these buttons "exists" there whatever the CSS does.

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

  // Put the bar in the state a live session gives it, without needing a loaded
  // model: the bar shown, one session in the switcher, and a project path of the
  // length a real one has. Those three are what make the row overflow.
  await page.evaluate(() => {
    const bar = document.getElementById("coder-bar");
    bar.classList.add("open");
    const sel = document.getElementById("session-select");
    const opt = document.createElement("option");
    opt.textContent = "invoice-demo (24fd0d)";
    sel.appendChild(opt);
    document.getElementById("coder-cwd").textContent = "D:\\projects\\invoice-demo";
    document.getElementById("coder-state").textContent = "idle";
  });
  await page.waitForTimeout(300);

  const report = await page.evaluate((ids) => {
    const bar = document.getElementById("coder-bar");
    const barBox = bar.getBoundingClientRect();
    const overflowing = [];
    let measured = 0;
    for (const id of ids) {
      const el = document.getElementById(id);
      if (!el) { overflowing.push({ id, why: "missing" }); continue; }
      const r = el.getBoundingClientRect();
      if (r.width === 0 && r.height === 0) continue;
      measured += 1;
      const hit = document.elementFromPoint(
        Math.round(r.left + r.width / 2), Math.round(r.top + r.height / 2));
      const covered = !hit || !(hit === el || el.contains(hit));
      if (r.right > barBox.right + 1 || covered) {
        overflowing.push({
          id,
          right: Math.round(r.right),
          barRight: Math.round(barBox.right),
          covered,
        });
      }
    }
    return { measured, overflowing };
  }, BUTTONS);

  // Guard the fixture itself: if the buttons stopped rendering, "nothing
  // overflows" would pass while measuring nothing at all.
  expect(report.measured, "the fixture must actually lay out the toolbar buttons")
    .toBeGreaterThanOrEqual(BUTTONS.length - 1);
  expect(report.overflowing, "a toolbar button past the bar's edge is covered by the sessions rail and cannot be clicked")
    .toEqual([]);
});

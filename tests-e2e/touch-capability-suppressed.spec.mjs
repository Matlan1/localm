// SPDX-License-Identifier: AGPL-3.0-or-later
// Every context built from ./_fixtures.mjs reports no touch capability.

import { test, expect } from "./_fixtures.mjs";

test("navigator reports no touch capability", async ({ page }) => {
  const maxTouchPoints = await page.evaluate(() => navigator.maxTouchPoints);
  expect(maxTouchPoints).toBe(0);
});

test("pointer media query reports fine, not coarse", async ({ page }) => {
  const pointerCoarse = await page.evaluate(
    () => window.matchMedia && window.matchMedia("(pointer: coarse)").matches,
  );
  expect(pointerCoarse).toBe(false);
});

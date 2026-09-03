// SPDX-License-Identifier: AGPL-3.0-or-later
// A <select> sitting in a `.card .row` next to a flex:1 <input>. Only a real
// browser can catch this: jsdom has no CSS layout engine, so the select's
// `width: 100%` flex-basis starving its siblings still passes every jsdom
// assertion that only checks the DOM structure exists.

import { test, expect } from "@playwright/test";

async function widthOf(page, id) {
  return await page.locator(`#${id}`).evaluate((el) => el.getBoundingClientRect().width);
}

test("the model search input keeps its width beside the source select", async ({ page }) => {
  await page.goto("/?view=models");
  await expect(page.locator("#disc-query")).toBeVisible({ timeout: 30_000 });

  const source = await widthOf(page, "disc-source");
  const query = await widthOf(page, "disc-query");

  expect(source, "the source select must be wide enough to read its options")
    .toBeGreaterThan(80);
  expect(source, "the source select must not claim the whole row")
    .toBeLessThan(300);
  expect(query, "the search input must keep the rest of the row")
    .toBeGreaterThan(source * 2);
});

test("no visible input in a card row is collapsed by a sibling", async ({ page }) => {
  await page.goto("/?view=models");
  await expect(page.locator("#disc-query")).toBeVisible({ timeout: 30_000 });

  const collapsed = await page.evaluate(() => {
    const bad = [];
    for (const input of document.querySelectorAll(".card .row input[type='text']")) {
      const box = input.getBoundingClientRect();
      if (box.width === 0 && box.height === 0) continue;
      if (box.width < 60) bad.push({ id: input.id, width: Math.round(box.width) });
    }
    return bad;
  });

  expect(collapsed, "an input squeezed under 60px is unusable, whatever else shares the row")
    .toEqual([]);
});

test("every select in a card row sizes to its content", async ({ page }) => {
  await page.goto("/?view=models");
  await expect(page.locator("#disc-query")).toBeVisible({ timeout: 30_000 });

  const offenders = await page.evaluate(() => {
    const bad = [];
    for (const sel of document.querySelectorAll(".card .row select")) {
      const style = getComputedStyle(sel);
      if (style.flexGrow !== "0" || style.flexShrink !== "0") {
        bad.push({ id: sel.id, grow: style.flexGrow, shrink: style.flexShrink });
      }
    }
    return bad;
  });

  expect(offenders, "a card-row select must not grow or shrink against its siblings")
    .toEqual([]);
});

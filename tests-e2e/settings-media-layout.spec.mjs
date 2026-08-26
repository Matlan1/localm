// SPDX-License-Identifier: AGPL-3.0-or-later
// Settings > Media renders three side-by-side subsections (Image, Music, Video).
// Only a real browser can catch a layout regression here: jsdom has no CSS layout
// engine, so a broken grid-template-columns still passes every jsdom assertion
// that merely checks the DOM structure exists.

import { test, expect } from "@playwright/test";

function rectsOverlap(a, b) {
  return a.x < b.x + b.width && b.x < a.x + a.width
    && a.y < b.y + b.height && b.y < a.y + a.height;
}

// The container's OWN boundingBox() does not grow to cover an overflowing
// child (overflow: visible content is drawn outside the box but does not
// enlarge it), so a container-only check misses overflow-based overlap. This
// takes the envelope of every descendant instead, matching what a user
// actually sees on screen.
async function visualFootprint(locator) {
  return await locator.evaluate((el) => {
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const n of [el, ...el.querySelectorAll("*")]) {
      const r = n.getBoundingClientRect();
      if (r.width === 0 && r.height === 0) continue;
      minX = Math.min(minX, r.left);
      minY = Math.min(minY, r.top);
      maxX = Math.max(maxX, r.right);
      maxY = Math.max(maxY, r.bottom);
    }
    return { x: minX, y: minY, width: maxX - minX, height: maxY - minY };
  });
}

test("Media settings subsections do not overlap", async ({ page }) => {
  const pageErrors = [];
  page.on("pageerror", (e) => pageErrors.push(String(e.message || e)));

  await page.goto("/?view=settings");
  await page.locator('.settings-nav-link[data-target="media"]').click();

  const grid = page.locator("#settings-sec-media .media-settings-grid");
  await expect(grid).toBeAttached({ timeout: 30_000 });
  const subs = grid.locator(".media-subsection");
  await expect(subs).toHaveCount(3, { timeout: 30_000 });

  const boxes = [];
  for (const sub of await subs.all()) {
    const box = await visualFootprint(sub);
    expect(box, "each media subsection must have a real layout box").not.toBeNull();
    boxes.push(box);
  }

  for (const box of boxes) {
    expect(box.width, "a media subsection must not be squeezed below a usable width")
      .toBeGreaterThan(150);
  }
  for (let i = 0; i < boxes.length; i++) {
    for (let j = i + 1; j < boxes.length; j++) {
      expect(rectsOverlap(boxes[i], boxes[j]),
        `media subsection ${i} and ${j} must not overlap: ${JSON.stringify(boxes[i])} vs ${JSON.stringify(boxes[j])}`
      ).toBe(false);
    }
  }

  expect(pageErrors, "no uncaught JS errors").toEqual([]);
});

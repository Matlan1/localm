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
    expect(Number.isFinite(box.width), "each media subsection must have a real layout box")
      .toBe(true);
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

// The page's own mobile breakpoint (@media (max-width: 760px)) stacks every
// .settings-fields grid to one column. .media-subsection .settings-fields has
// higher CSS specificity than that bare selector (two classes vs one), so a
// media query alone does not make it lose the cascade - it must be repeated
// inside the breakpoint. The real page's own .media-settings-grid sizing
// happens to keep every subsection narrow enough that this cascade defect
// stays invisible on the actual Settings page at any width up to the
// breakpoint - so this asserts the cascade rule directly, on an isolated
// element wide enough to expose it, rather than relying on the real page's
// current width math to reproduce it.
test("nested .settings-fields still stacks to one column below the mobile breakpoint", async ({ page }) => {
  await page.setViewportSize({ width: 700, height: 812 });
  await page.goto("/?view=settings");

  const columns = await page.evaluate(() => {
    const probe = document.createElement("div");
    probe.className = "media-subsection";
    probe.style.width = "400px";   // comfortably above the 2-column threshold
    const fields = document.createElement("div");
    fields.className = "settings-fields";
    probe.appendChild(fields);
    document.body.appendChild(probe);
    const n = getComputedStyle(fields).gridTemplateColumns.trim().split(/\s+/).length;
    probe.remove();
    return n;
  });
  expect(columns, "a media subsection's fields must stack to one column below the "
    + "page's own mobile breakpoint, matching every other settings section, even "
    + "when the subsection itself is wide enough for two").toBe(1);
});

// SPDX-License-Identifier: AGPL-3.0-or-later
// Shared `test`/`expect` for every spec in this directory. Extends the base
// `context` fixture so every page this suite opens reports no touch capability.

import { test as base, expect } from "@playwright/test";

export const test = base.extend({
  context: async ({ context }, use) => {
    // Overrides the value Chromium reports for the host's own hardware.
    // See touch-capability-suppressed.spec.mjs.
    await context.addInitScript(() => {
      Object.defineProperty(Navigator.prototype, "maxTouchPoints", {
        get: () => 0,
        configurable: true,
      });
    });
    await use(context);
  },
});

export { expect };

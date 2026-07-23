// SPDX-License-Identifier: AGPL-3.0-or-later
// Minimal ES module fixture: an `export let` binding that gets REASSIGNED
// (not mutated), mirroring app/models-sidebar.js's `modelCache` and the ~20
// siblings across app/*+pages/* that app/main.js's window-export loop must
// reflect live, not freeze at their first-load value.
export let counter = { n: 0 };
export function bump() {
  counter = { n: counter.n + 1 };
}

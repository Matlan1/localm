// SPDX-License-Identifier: AGPL-3.0-or-later
// ES module fixture: an `export let` binding that is reassigned, not mutated.
export let counter = { n: 0 };
export function bump() {
  counter = { n: counter.n + 1 };
}

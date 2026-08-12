// SPDX-License-Identifier: AGPL-3.0-or-later
// LM-DA-047: privacy mode's "no traces, not even localStorage" guarantee used
// to rest on two independently hand-maintained mechanisms - a per-call-site
// `if (!chat.privacy) localStorage.setItem(...)` guard at 13 sites across 7
// files, and helpers.js's INSTANCE_SCOPED_KEYS wipe list (11 entries) - with
// nothing checking they agreed. They had already drifted: two write-gated
// keys ("localm.studioOpen", "localm.backendHintDismissed") were missing from
// the wipe list, so a user who used those features before enabling privacy
// mode kept the trace indefinitely while the sidebar reported "privacy mode -
// this session only".
//
// The fix routes every instance-scoped write through chat.js's
// lsSetScoped(key, value) instead of a hand-written guard. This test is the
// mechanical backstop: it SOURCE-SCANS the real shipped files for every
// lsSetScoped call site (not a hand-typed duplicate list - a duplicate list
// here would just be a THIRD place to keep in sync) and asserts each key it
// finds is registered in INSTANCE_SCOPED_KEYS. A future call site that adds a
// write-gate without a matching wipe entry - LM-DA-047's exact failure mode -
// fails this test before it ever ships, regardless of which page/feature it
// belongs to or whether any test happens to exercise that code path.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { loadApp, runScript } from "./harness.mjs";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const STATIC = join(ROOT, "localm", "plugins", "gui", "static");
const read = (p) => readFileSync(join(STATIC, p), "utf-8");

// Every file that can legitimately call lsSetScoped. Deliberately exhaustive
// over app/ + pages/ (not just the 7 files LM-DA-047 named) so a NEW file
// introducing a fresh call site is covered automatically, not just the ones
// known to be broken today.
const SCANNED_FILES = [
  "app/chat.js", "app/coder.js", "app/models-sidebar.js", "app/settings-perf.js",
  "app/tabs.js", "pages/images.js", "pages/video.js", "pages/knowledge.js",
];

// Resolve a JS source string to a list of {file, key} for every
// lsSetScoped(<arg>, ...) call, where <arg> is either a quoted literal or an
// identifier that resolves to a `const IDENT = "literal";` declared anywhere
// in the SAME file (the pattern settings-perf.js's BACKEND_HINT_DISMISSED_KEY
// already used before this fix, kept working on purpose - reusing an existing
// constant is better style than a second literal in the same file).
function extractScopedWrites(file) {
  const src = read(file);
  const constMap = new Map();
  for (const m of src.matchAll(/const\s+(\w+)\s*=\s*"([^"]+)"\s*;/g)) {
    constMap.set(m[1], m[2]);
  }
  const found = [];
  // (?<!function ) excludes chat.js's own `export function lsSetScoped(key, value) {`
  // declaration - only actual CALL sites (where the first arg is a real key) matter here.
  for (const m of src.matchAll(/(?<!function )lsSetScoped\(\s*(?:"([^"]+)"|'([^']+)'|(\w+))\s*,/g)) {
    let key = m[1] || m[2];
    if (!key && m[3]) {
      key = constMap.get(m[3]);
      assert.ok(key, `${file}: lsSetScoped(${m[3]}, ...) - cannot statically ` +
        `resolve "${m[3]}" to a same-file string constant. Use a literal or a ` +
        `\`const ${m[3]} = "localm...";\` in this file so this test can verify it.`);
    }
    assert.ok(key, `${file}: found an lsSetScoped(...) call this scanner could not parse`);
    found.push({ file, key });
  }
  return found;
}

test("LM-DA-047: every lsSetScoped(key, ...) call site's key is registered in INSTANCE_SCOPED_KEYS", () => {
  const { window } = loadApp();
  runScript(window, "window.__keys = INSTANCE_SCOPED_KEYS;");
  const wipeList = new Set(window.__keys);
  assert.ok(wipeList.size > 0, "sanity check: INSTANCE_SCOPED_KEYS is not empty");

  const writes = SCANNED_FILES.flatMap(extractScopedWrites);
  assert.ok(writes.length > 0, "sanity check: the scanner found at least one lsSetScoped call site");

  const missing = writes.filter((w) => !wipeList.has(w.key));
  assert.deepEqual(missing, [],
    "every privacy-gated write must have a matching INSTANCE_SCOPED_KEYS entry " +
    "(helpers.js) - found write-gated key(s) with no wipe entry: " +
    JSON.stringify(missing));
});

test("LM-DA-047: the old hand-written `if (!chat.privacy) localStorage.setItem` " +
     "guard must not reappear - new instance-scoped writes go through lsSetScoped", () => {
  const offenders = [];
  for (const file of [...SCANNED_FILES, "app/helpers.js"]) {
    // Strip `//` line comments first - this file's own doc comments quote the
    // retired pattern verbatim as history, which is not a reappearance of it.
    const src = read(file).replace(/\/\/[^\n]*/g, "");
    if (/if\s*\(\s*!chat\.privacy\s*\)\s*\{?\s*(?:try\s*\{\s*)?localStorage\.setItem/.test(src)) {
      offenders.push(file);
    }
  }
  assert.deepEqual(offenders, [],
    "found the pre-LM-DA-047 inline guard pattern reintroduced in: " + offenders.join(", "));
});

// SPDX-License-Identifier: AGPL-3.0-or-later
// Guards the HAND-VENDORED jsQR at localm/plugins/gui/static/vendor/jsQR.js.
//
// WHY THIS FILE EXISTS. Same reason as vendor-dompurify.test.mjs and
// vendor-katex.test.mjs: this file has no entry in package.json, so no
// dependency tooling can see it and a published advisory against it produces no
// PR, no alert and no red check. The vendor README named jsQR, marked and
// highlight.js as the three with no guard at all; this closes jsQR.
//
// WHAT CHANGED, and it is the reason a guard is now POSSIBLE. The shipped file
// used to be `jsQR.min.js`, and the README recorded why it was the worst of the
// three to pin: "no version string and no upstream minified artefact to hash
// against". That was exactly right - npm publishes only an UNMINIFIED
// dist/jsQR.js, so the vendored minified bytes came from a CDN's own minifier,
// which is not reproducible. Measured 2026-08-13 while trying: the shipped file
// matched NO published artefact, and jsdelivr's current 1.3.0 / 1.3.1 / 1.4.0
// minified builds all differ from it and from each other even after normalising
// line endings and stripping the sourceMappingURL comment. Its provenance was
// genuinely unrecoverable.
//
// So the file was replaced with npm's own dist/jsQR.js, whose tarball was
// verified against the registry's published shasum
// (8efb8d0a7cc6863cb6d95116b9069123ce9eb2d1) before use. It is unminified and
// therefore named honestly - jsQR.js, not jsQR.min.js. That costs 126 KB on a
// LAZILY loaded asset: models-sidebar.js only injects it when a scan starts on a
// browser without a native BarcodeDetector, so most users never fetch it at all.
// Being able to say which upstream build ships is worth more than those bytes.
//
// THE HASH IS THE VERSION PIN, not a belt-and-braces extra. jsQR's dist carries
// no version string of any kind (checked: the only "version" token in the file
// refers to QR symbol versions, not the library's). There is nothing to read at
// runtime, so content identity is the only thing that can say which build this
// is - the same treatment katex.min.css and auto-render.min.js get.
//
// THE HASH IS TAKEN OVER CRLF-NORMALISED BYTES, and that is load-bearing rather
// than tidiness. This repo has core.autocrlf=true and no .gitattributes rule for
// vendor/, so a file containing newlines is checked out with different bytes
// than the blob git stores - the README measured the old jsQR at 130469 B in git
// and 130470 B in a Windows working tree. A raw-byte hash would pass on one
// platform and fail on the other, and the failure would read as a corrupted
// vendor drop rather than a line-ending artefact.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import crypto from "node:crypto";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";

const VENDOR = new URL("../localm/plugins/gui/static/vendor/", import.meta.url);

// Provenance pin. sha256, base64, over CRLF-normalised bytes. Recorded from the
// npm registry artefact, tarball shasum-verified:
//   npm pack jsqr@1.4.0  ->  package/dist/jsQR.js
// Bumping jsQR means updating BOTH constants together.
const VENDORED_VERSION = "1.4.0";
const PINNED_JSQR = "vEDIoVGWI2sjFNsIVvcsoLSZgM1UE7jIUqc0n1/uCFk=";

function normalisedHash(buf) {
  const lf = Buffer.from(buf.toString("latin1").replace(/\r\n/g, "\n"), "latin1");
  return crypto.createHash("sha256").update(lf).digest("base64");
}

/** Load the REAL vendored bytes. It is a UMD bundle, so require() takes the
 *  `module.exports` branch - the same file the browser loads as a classic
 *  script. Evaluating the actual artefact is the point: an npm import would be
 *  structurally incapable of saying anything about the file that ships. */
function loadJsQR() {
  const require = createRequire(import.meta.url);
  const path = fileURLToPath(new URL("jsQR.js", VENDOR));
  delete require.cache[path];
  const mod = require(path);
  return mod && mod.default ? mod.default : mod;
}

// A real QR symbol, generated once and embedded so the test needs no fixture
// file and no network. 25x25 modules, version 2, error correction M.
const QR_PAYLOAD = "localm-vendor-guard-7Q4M";
const QR_MODULES = [
  "1111111000000010101111111",
  "1000001001001101101000001",
  "1011101010111001001011101",
  "1011101010010111001011101",
  "1011101011111000101011101",
  "1000001011010011101000001",
  "1111111010101010101111111",
  "0000000011101001100000000",
  "1011111000100101101111100",
  "1011010110110100100101000",
  "0000101110011011110100111",
  "0110110010111010100100010",
  "1000111100100101011011110",
  "1110010111101010100101000",
  "1010111011100111111010011",
  "1001000011101011101100001",
  "1001011011001010111111110",
  "0000000011100011100010100",
  "1111111000011000101011011",
  "1000001011010011100010001",
  "1011101011011101111111100",
  "1011101011001011111110111",
  "1011101011100011000100001",
  "1000001000100000110101001",
  "1111111011111010010111111",
];

/** Render the module grid to RGBA at `scale` px per module, with a quiet zone. */
function renderQR(scale = 4, quiet = 4) {
  const n = QR_MODULES.length;
  const side = (n + quiet * 2) * scale;
  const data = new Uint8ClampedArray(side * side * 4).fill(255);
  for (let my = 0; my < n; my++) {
    for (let mx = 0; mx < n; mx++) {
      if (QR_MODULES[my][mx] !== "1") continue;
      for (let dy = 0; dy < scale; dy++) {
        for (let dx = 0; dx < scale; dx++) {
          const x = (mx + quiet) * scale + dx;
          const y = (my + quiet) * scale + dy;
          const i = (y * side + x) * 4;
          data[i] = data[i + 1] = data[i + 2] = 0;
        }
      }
    }
  }
  return { data, width: side, height: side };
}

test("the vendored jsQR is the pinned upstream build (content identity)", () => {
  const got = normalisedHash(fs.readFileSync(new URL("jsQR.js", VENDOR)));
  assert.equal(got, PINNED_JSQR,
    `jsQR.js does not match the pinned jsqr@${VENDORED_VERSION} npm artefact. ` +
    "If this was a deliberate bump, update VENDORED_VERSION and PINNED_JSQR " +
    "together; if not, the vendored file has been modified or corrupted.");
});

test("the old un-pinnable minified drop is gone", () => {
  // The whole point of the replacement: jsQR.min.js could not be traced to any
  // upstream artefact. If it reappears, provenance has been lost again.
  assert.equal(fs.existsSync(new URL("jsQR.min.js", VENDOR)), false,
    "jsQR.min.js is back - it has no reproducible upstream source, use the npm dist");
});

test("the Apache-2.0 license text ships beside it", () => {
  const txt = fs.readFileSync(new URL("LICENSE.jsqr", VENDOR), "utf8");
  assert.match(txt, /Apache License/i);
  assert.match(txt, /Version 2\.0/);
});

test("the vendored bytes actually decode a real QR symbol", () => {
  // Not "the module loads": a guard that only checks the file parses would pass
  // on a truncated or half-replaced drop that cannot decode anything.
  const jsQR = loadJsQR();
  assert.equal(typeof jsQR, "function", "jsQR did not expose its UMD export");
  const { data, width, height } = renderQR();
  const res = jsQR(data, width, height);
  assert.ok(res, "jsQR returned null for a valid QR symbol");
  assert.equal(res.data, QR_PAYLOAD);
});

test("jsQR returns null for an image with no QR in it", () => {
  // The negative half. Without it, a stub that returned a fixed truthy object
  // would satisfy the decode test above.
  const jsQR = loadJsQR();
  const side = 120;
  const blank = new Uint8ClampedArray(side * side * 4).fill(255);
  assert.equal(jsQR(blank, side, side), null,
    "jsQR claimed to decode a blank image");
});

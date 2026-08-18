// SPDX-License-Identifier: AGPL-3.0-or-later
// Guards the HAND-VENDORED onnxruntime-web runtime at
// localm/plugins/builtin/tts/static/vendor/onnxruntime/. Two files, and they
// are one build: ort-wasm-simd-threaded.jsep.mjs is emscripten glue that loads
// ort-wasm-simd-threaded.jsep.wasm sitting beside it.
//
// WHY THIS FILE EXISTS. Same reason as vendor-dompurify / vendor-katex /
// vendor-jsqr / vendor-highlightjs: nothing in package.json names any of this,
// so no dependency scanner can ever see a vulnerable version sitting here, and
// "nothing happened" is indistinguishable from "we are up to date". A vendored
// WebAssembly runtime is the worst member of that set to leave unguarded,
// because it is 21 MB of opaque binary that no reviewer reads.
//
// AND ONE THING THE OTHER GUARDS DO NOT HAVE TO CHECK: THE PAIRING. The glue
// and the JS that drives it ship as one release. kokoro.min.js has
// @huggingface/transformers compiled INTO it and builds its runtime URL from
// its own env.version, so vendoring a runtime from a different transformers
// release pairs mismatched halves. That failure surfaces at model-load time as
// "no available backend found", which reads exactly like the CDN outage this
// vendoring replaced, so the test asserts the version agreement directly
// rather than leaving it to be rediscovered.
//
// HASHING: THE TWO FILES NEED DIFFERENT TREATMENT, and it is not tidiness.
// This repo has core.autocrlf=true and no .gitattributes rule covering these
// paths. The .mjs is text (measured: 125 LF, zero CR upstream), so git converts
// it on checkout and a raw-byte hash would pass on Linux and fail on Windows.
// The .wasm opens with a NUL byte, so git classifies it binary and never
// converts it - and there a CRLF-normalised hash would be actively WRONG,
// because a real 0x0D 0x0A pair inside the binary would be rewritten by the
// normaliser and the pin would stop describing the bytes that ship.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import crypto from "node:crypto";

const VENDOR = new URL(
  "../localm/plugins/builtin/tts/static/vendor/", import.meta.url);
const RUNTIME = new URL("onnxruntime/", VENDOR);

// Provenance pin. Recorded from the npm registry artefact, whose integrity hash
// npm verifies during `npm pack`:
//   npm pack @huggingface/transformers@3.8.1  ->  package/dist/
const VENDORED_TRANSFORMERS = "3.8.1";
const MJS = "ort-wasm-simd-threaded.jsep.mjs";
const WASM = "ort-wasm-simd-threaded.jsep.wasm";
const PINNED_MJS_HASH = "CPuG7EM8eL+wMsXYSmi46OWo2BJo+jniQxQXmldnpbk=";   // CRLF-normalised
const PINNED_WASM_HASH = "xGZV6KlK/EUzjUyyuEBHX4jlAS1SRQmRblBQecAL+jk="; // raw bytes
const EXPECTED_WASM_BYTES = 21596019;

function readRuntime(name) {
  return fs.readFileSync(new URL(name, RUNTIME));
}
/** sha256 (base64) over CRLF-normalised bytes. Text artefacts only. */
function normalisedHash(buf) {
  const lf = Buffer.from(buf.toString("latin1").replace(/\r\n/g, "\n"), "latin1");
  return crypto.createHash("sha256").update(lf).digest("base64");
}
/** sha256 (base64) over the bytes exactly as they sit on disk. */
function rawHash(buf) {
  return crypto.createHash("sha256").update(buf).digest("base64");
}

test("the vendored runtime glue is the recorded upstream artefact", () => {
  const got = normalisedHash(readRuntime(MJS));
  assert.equal(got, PINNED_MJS_HASH,
    `${MJS} does not match the pinned @huggingface/transformers@`
    + `${VENDORED_TRANSFORMERS} dist artefact. If this was a deliberate bump, `
    + "move BOTH runtime files together and update VENDORED_TRANSFORMERS, "
    + "PINNED_MJS_HASH, PINNED_WASM_HASH and vendor/NOTICE.md; if not, the "
    + "vendored file has been modified or corrupted.");
});

test("the vendored runtime binary is the recorded upstream artefact", () => {
  const buf = readRuntime(WASM);
  assert.equal(buf.length, EXPECTED_WASM_BYTES,
    `${WASM} is ${buf.length} bytes, expected ${EXPECTED_WASM_BYTES}`);
  assert.equal(rawHash(buf), PINNED_WASM_HASH,
    `${WASM} does not match the pinned upstream binary.`);
});

test("the vendored binary is a real WebAssembly module, not a placeholder", () => {
  // A hash pin alone would happily pin a truncated or stub file forever. This
  // reads the format's own header, and then hands the bytes to the engine that
  // will actually run them: WebAssembly.validate is the only check here that
  // the 21 MB is a loadable module rather than 21 MB of the right hash.
  const buf = readRuntime(WASM);
  assert.deepEqual([...buf.subarray(0, 4)], [0x00, 0x61, 0x73, 0x6d],
    "no \\0asm magic: this is not a WebAssembly binary");
  assert.deepEqual([...buf.subarray(4, 8)], [1, 0, 0, 0],
    "unexpected WebAssembly version word");
  assert.ok(WebAssembly.validate(buf),
    "WebAssembly.validate rejected the vendored runtime, so no browser could "
    + "instantiate it either");
});

test("the glue loads its wasm from beside itself, by the vendored name", () => {
  // The two files are only "one build" if the glue actually asks for the
  // binary we shipped. It resolves the name against wasmPaths, which tts.js
  // points at this directory, so a rename on either side is a 404 at first
  // speak and nothing earlier would notice.
  const src = readRuntime(MJS).toString("utf8");
  assert.ok(src.includes(`"${WASM}"`),
    `${MJS} does not name ${WASM}; the pair has drifted apart`);
  assert.ok(src.includes("import.meta.url"),
    `${MJS} no longer resolves against import.meta.url, so its pthread workers `
    + "would not come from this directory");
});

test("the vendored runtime matches the transformers.js compiled into the bundle", () => {
  // THE PAIRING CHECK. kokoro.min.js carries transformers.js inside it and
  // derives its runtime URL from env.version; the vendored files come from that
  // same npm release. Reading the version out of the shipped bundle means this
  // fails when someone rebuilds the bundle and forgets the runtime, which is
  // the realistic way these two drift.
  const bundle = fs.readFileSync(new URL("kokoro.min.js", VENDOR)).toString("latin1");
  assert.ok(bundle.includes(`"${VENDORED_TRANSFORMERS}"`),
    `kokoro.min.js does not contain the version string `
    + `"${VENDORED_TRANSFORMERS}". The bundle was rebuilt against a different `
    + "@huggingface/transformers, so the vendored onnxruntime runtime under "
    + "vendor/onnxruntime/ is now from the wrong release. Re-vendor both "
    + "together (see vendor/NOTICE.md) - a mismatched pair fails at model-load "
    + "time with \"no available backend found\", which looks identical to the "
    + "CDN outage this vendoring removed.");
});

test("nothing in the tts plugin still points at a CDN for the runtime", () => {
  // The whole point of vendoring. tts.js must default to the local directory,
  // and the shipped template must agree with it - two files, one value, and
  // only the pair decides what an offline browser loads.
  const PLUGIN = new URL("../localm/plugins/builtin/tts/", import.meta.url);
  const tts = fs.readFileSync(new URL("static/tts.js", PLUGIN)).toString("utf8");
  const tpl = JSON.parse(
    fs.readFileSync(new URL("tts.example.json", PLUGIN)).toString("utf8"));

  assert.equal(tpl.wasm_paths, "vendor/onnxruntime/",
    "tts.example.json no longer ships the vendored runtime path. Blank is not "
    + "neutral: it falls through to the bundle's cdn.jsdelivr.net default, "
    + "which the GUI's CSP refuses.");
  assert.ok(tts.includes(`cfg.wasm_paths || "${tpl.wasm_paths}"`),
    "tts.js does not fall back to the shipped default, so a browser that "
    + "cannot reach /api/tts/config would silently try the CDN");

  // CODE lines only. The first version of this asserted the origin was absent
  // from the whole file and went red on the comment that EXPLAINS why the
  // origin was removed - so the cheapest way to satisfy it would have been to
  // delete the explanation. A test is a specification, and that one specified
  // the wrong thing.
  const codeLines = tts.split("\n").filter((l) => {
    const s = l.trim();
    return s && !s.startsWith("//") && !s.startsWith("*") && !s.startsWith("/*");
  });
  const offending = codeLines.filter((l) => /cdn\.jsdelivr\.net/.test(l));
  assert.deepEqual(offending, [],
    "tts.js has executable code naming a CDN origin again");
});

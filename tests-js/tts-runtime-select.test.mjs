// SPDX-License-Identifier: AGPL-3.0-or-later
// Which onnxruntime-web build the TTS plugin loads in which browser, and that
// every build it can pick is vendored.

import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";

const STATIC = new URL("../localm/plugins/builtin/tts/static/", import.meta.url);
const { isSafariBelow26, runtimePaths } = await import(new URL("tts-util.js", STATIC).href);

const SAFARI = (v) => ({
  vendor: "Apple Computer, Inc.",
  userAgent: `Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 `
    + `(KHTML, like Gecko) Version/${v} Safari/605.1.15`,
});
const CHROME = {
  vendor: "Google Inc.",
  userAgent: "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    + "(KHTML, like Gecko) Chrome/151.0.0.0 Safari/537.36",
};
const FIREFOX = { vendor: "", userAgent: "Mozilla/5.0 (X11; Linux x86_64; rv:140.0) Gecko/20100101 Firefox/140.0" };
const IOS_CHROME = {
  vendor: "Apple Computer, Inc.",
  userAgent: "Mozilla/5.0 (iPhone; CPU iPhone OS 17_5 like Mac OS X) AppleWebKit/605.1.15 "
    + "(KHTML, like Gecko) CriOS/126.0.0.0 Mobile/15E148 Safari/604.1",
};

test("Safari below 26 is recognised, Safari 26 and other browsers are not", () => {
  assert.equal(isSafariBelow26(SAFARI(18)), true);
  assert.equal(isSafariBelow26(SAFARI(17)), true);
  assert.equal(isSafariBelow26(SAFARI(26)), false);
  assert.equal(isSafariBelow26(CHROME), false);
  assert.equal(isSafariBelow26(FIREFOX), false);
  assert.equal(isSafariBelow26(IOS_CHROME), false);
  assert.equal(isSafariBelow26(undefined), false);
});

test("Safari below 26 without WebGPU gets the plain build, everything else the asyncify build", () => {
  const dir = "http://x/plugins/tts/vendor/onnxruntime/";
  assert.deepEqual(runtimePaths(dir, { safariBelow26: true, webgpu: false }), {
    mjs: dir + "ort-wasm-simd-threaded.mjs",
    wasm: dir + "ort-wasm-simd-threaded.wasm",
  });
  for (const opts of [{ safariBelow26: true, webgpu: true },
                      { safariBelow26: false, webgpu: false },
                      { safariBelow26: false, webgpu: true }, undefined]) {
    assert.deepEqual(runtimePaths(dir, opts), {
      mjs: dir + "ort-wasm-simd-threaded.asyncify.mjs",
      wasm: dir + "ort-wasm-simd-threaded.asyncify.wasm",
    }, JSON.stringify(opts));
  }
});

test("every build runtimePaths can pick is vendored", () => {
  const dir = new URL("vendor/onnxruntime/", STATIC);
  for (const opts of [{ safariBelow26: true, webgpu: false }, { safariBelow26: false }]) {
    const p = runtimePaths("", opts);
    for (const name of [p.mjs, p.wasm]) {
      assert.ok(fs.existsSync(new URL(name, dir)), `${name} is not vendored`);
    }
  }
});

test("tts.js hands onnxruntime the browser's runtime pair", () => {
  const src = fs.readFileSync(new URL("tts.js", STATIC), "utf8");
  assert.match(src, /onnx\.wasm\.wasmPaths = runtimePaths\(/);
  assert.match(src, /isSafariBelow26\(navigator\)/);
});

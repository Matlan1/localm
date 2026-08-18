# Vendored third-party code (TTS plugin)

`kokoro.min.js` is a self-contained browser ESM bundle produced from upstream
npm packages so the plugin runs offline without a CDN. All bundled code is under
permissive licenses (no copyleft, no non-commercial terms):

| Package | Version | License | Text |
|---|---|---|---|
| `kokoro-js` | 1.2.1 | Apache-2.0 | `LICENSE.kokoro-js` |
| `@huggingface/transformers` (transformers.js) | 3.8.1 | Apache-2.0 | `LICENSE.transformers-js` |
| `onnxruntime-web` | 1.22.0-dev.20250409-89f8206ba4 | MIT (c) Microsoft Corporation | `LICENSE.onnxruntime-web` |
| `phonemizer` | ^1.2 | MIT | (bundled, see the kokoro-js notice) |

`voices.json` is voice metadata (names, language, gender, quality grade) extracted
from kokoro-js for the voice picker. The 28 shipped voices are English only
(en-us / en-gb); no CC-BY non-English voices are referenced, so no extra
attribution is required.

## `onnxruntime/` - the ONNX runtime, vendored so TTS works OFFLINE

    onnxruntime/ort-wasm-simd-threaded.jsep.mjs      44484 B
    onnxruntime/ort-wasm-simd-threaded.jsep.wasm  21596019 B

Byte-exact copies of `dist/` from `@huggingface/transformers@3.8.1`, which is the
transformers.js version compiled into `kokoro.min.js` (its `env.version` string
reads `3.8.1`, and the bundle builds its runtime URL from that value). Those two
filenames are the ONLY runtime artefacts the bundle names, confirmed by reading
it rather than assumed. The `.mjs` is emscripten glue that loads the `.wasm`
beside it and spawns its own pthread workers from its own module URL, so the two
must stay in one directory together.

Until 2026-08-18 these were NOT vendored: the bundle defaulted `wasmPaths` to
`https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1/dist/`, so neural
TTS needed the public internet on every cold load and never worked offline or
behind a filtering proxy. That also forced `cdn.jsdelivr.net` into the GUI's
Content-Security-Policy `script-src` (a dynamic `import()` is a module script).
Vendoring removes the network dependency and the CSP grant together.

`tts.example.json` ships `"wasm_paths": "vendor/onnxruntime/"` and `tts.js` falls
back to the same value, so an out-of-the-box install never reaches a CDN.

### Re-vendoring

npm verifies the registry integrity hash during `npm pack`, so take the artefact
from there and copy it unmodified. Do not hand-edit or re-minify.

```
npm pack @huggingface/transformers@<version>
tar -xzf huggingface-transformers-<version>.tgz
cp package/dist/ort-wasm-simd-threaded.jsep.mjs \
   localm/plugins/builtin/tts/static/vendor/onnxruntime/
cp package/dist/ort-wasm-simd-threaded.jsep.wasm \
   localm/plugins/builtin/tts/static/vendor/onnxruntime/
cp package/LICENSE localm/plugins/builtin/tts/static/vendor/LICENSE.transformers-js
npm test    # then update the pins in tests-js/vendor-onnxruntime.test.mjs
```

**The version MUST match the transformers.js compiled into `kokoro.min.js`.** The
emscripten glue and the JS that drives it are one build; pairing a newer runtime
with an older bundle is the same half-bump hazard the GUI's KaTeX note describes,
except it fails at model-load time rather than visibly. If you rebuild
`kokoro.min.js`, re-read its `env.version` and re-vendor to match.

`onnxruntime-web` publishes no `LICENSE` file in its npm package (checked: the
tarball contains none), so `LICENSE.onnxruntime-web` is the text from the
microsoft/onnxruntime repository at commit `89f8206ba4f1c22c39e0297fb55272e8ce8cd7d0`,
which is the exact commit the pinned dev version names.

### These are invisible to dependency tooling

Same hazard the GUI's `vendor/README.md` spells out: nothing in `package.json`
names any of this, so no scanner can ever flag a vulnerable version sitting here.
`tests-js/vendor-onnxruntime.test.mjs` is the replacement signal - it pins the
version and the content hashes, and asserts the runtime is a real, loadable
emscripten module rather than only that the files exist.

## What is NOT vendored (fetched once at runtime, then cached in-browser)

- The Kokoro model weights (~86 MB ONNX) from the Apache-2.0 Hugging Face repo
  named in `tts.example.json` (`model`), fetched on first use and cached by the
  browser (Cache API / IndexedDB), exactly like the chat model. No user text goes
  with the request; it is a public model asset.

For a fully air-gapped FIRST run, host the weights locally and point `model` at
them; the runtime itself no longer needs anything from the network.

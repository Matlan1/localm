# Vendored third-party code (TTS plugin)

`kokoro.min.js` is a self-contained browser ESM bundle produced from upstream
npm packages so the plugin runs offline without a CDN:

| Package | Version | License | Text |
|---|---|---|---|
| `kokoro-js` | 1.2.1 | Apache-2.0 | `LICENSE.kokoro-js` |
| `@huggingface/transformers` (transformers.js) | 3.8.1 | Apache-2.0 | `LICENSE.transformers-js` |
| `onnxruntime-web` | 1.22.0-dev.20250409-89f8206ba4 | MIT (c) Microsoft Corporation | `LICENSE.onnxruntime-web` |
| `phonemizer` | 1.2.1 | Apache-2.0 | (bundled, see the kokoro-js notice) |
| eSpeak NG (compiled into `phonemizer`) | see below | GPL-3.0-or-later | `LICENSE.espeak-ng` |

`phonemizer` is a thin wrapper around a WebAssembly build of eSpeak NG, and that
build plus its `espeak-ng-data` payload are compiled into `kokoro.min.js`. eSpeak
NG is GPL-3.0-or-later, so this bundle is not wholly permissive. localm is
AGPL-3.0-or-later, and section 13 of the AGPL grants permission to combine a
covered work with a GPL-3.0 work, so the combination is licensed; the resulting
distribution carries eSpeak NG under the GPL.

Until this was checked the table above listed `phonemizer` as MIT and described
the bundle as carrying no copyleft. Both were wrong: npm and the package's own
`LICENSE` give Apache-2.0, and eSpeak NG has been inside the bundle all along.

`voices.json` is voice metadata (names, language, gender, quality grade) extracted
from kokoro-js for the voice picker. It lists 41 voices: 28 English (en-us /
en-gb) and 13 for es, fr, hi, it and pt-br. The grades are upstream's own and the
key is absent for the voices upstream does not grade, rather than filled with an
invented value. All of them come from the Apache-2.0 model repo named in
`tts.example.json`; no CC-BY voices are referenced, so no extra attribution is
required.

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

### `kokoro.min.js` carries one appended line

The bundle keeps no export for its eSpeak NG engine, so one line is APPENDED to
the end of the upstream artefact, changing no existing byte:

    export{Y8 as espeakWorker,te as espeakFS};

`g2p.js` needs those two to select a language and to install a dictionary. The
names are minifier output and will differ in any rebuilt bundle: re-locate them
by content (the `new ne.eSpeakNGWorker` promise, and the object carrying
`createDataFile`/`unlink`) rather than by name. A re-vendor that drops the line
fails `tests-js/tts-g2p.test.mjs` rather than breaking speech at runtime.

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

## `espeak-ng-data/` - the non-English pronunciation dictionaries

eSpeak NG inside `kokoro.min.js` already carries every language's letter-to-sound
rules and voice definitions, but only the English dictionary, so selecting any
other language produced English phonemes. These are the missing dictionaries for
the five languages the shipped Kokoro voices cover:

    es_dict      49285 B  sha256 e7c6347e407d5c14f283eeada18b86c6d560b03b62f53659473109bbff096757
    fr_dict      63727 B  sha256 e399ab924c4d10beef1fc310b30ea56e4ddfd8b4b64b8ed978e9c65394d49b2d
    hi_dict      92143 B  sha256 5a68c9532624e57ac845b26ce1e2e5034c4f6353bede46ecbe57e583ec8effd6
    it_dict     154408 B  sha256 7ce5b6b4e2ee251516708584267a413a3c02b2fa07cb527a2eb421fbbb3b12cf
    pt_dict      76389 B  sha256 94c689153e12e9c5e0ecbf60518a93ebb2ea0fbe805c63d97fa84c43331424e9

They are read by `g2p.js`, which fetches one on first use of that language and
writes it into the running WebAssembly filesystem. An English-only user fetches
none of them.

Taken from the `espeakng-loader` 0.2.4 wheel on PyPI
(`espeakng_loader-0.2.4-py3-none-win_amd64.whl`), which redistributes an upstream
eSpeak NG build. They are data files of eSpeak NG and carry its GPL-3.0-or-later
licence; `LICENSE.espeak-ng` is the upstream `COPYING`.

That build is NOT the one compiled into `kokoro.min.js`: its `phondata`,
`phontab` and `phonindex` differ in size from the bundled copies. A dictionary
encodes indices into those phoneme tables, so a mismatched pair would produce
wrong phonemes rather than an error. Only the `_dict` files are taken, and the
pairing is checked rather than assumed: swapping the wheel's `en_dict` over the
bundled one leaves English phonemization byte-identical, while a deliberately
corrupted dictionary changes it. `tests-js/tts-g2p.test.mjs` pins the resulting
phonemes for all five languages against Kokoro's own alphabet.

## What is NOT vendored (fetched once at runtime, then cached in-browser)

- The Kokoro model weights (~86 MB ONNX) from the Apache-2.0 Hugging Face repo
  named in `tts.example.json` (`model`), fetched on first use and cached by the
  browser (Cache API / IndexedDB), exactly like the chat model. No user text goes
  with the request; it is a public model asset.

For a fully air-gapped FIRST run, host the weights locally and point `model` at
them; the runtime itself no longer needs anything from the network.

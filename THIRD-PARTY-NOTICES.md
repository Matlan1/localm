# Third-Party Notices

localm is licensed under AGPL-3.0-or-later (see [LICENSE](LICENSE)). It includes,
or at release time bundles, third-party components that remain under their own
licenses. The AGPL covers localm's own code, not these components. Their licenses
and copyrights are reproduced or referenced below.

## Vendored in this repository (browser assets)

Shipped under `localm/plugins/*/static/vendor/` so the GUI runs offline without a
CDN. Each keeps its own license:

| Component | Used for | License | Copyright |
|---|---|---|---|
| marked | Markdown rendering | MIT | (c) 2018-2026 MarkedJS, (c) 2011-2018 Christopher Jeffrey; its license file also carries the Markdown notice, (c) 2004 John Gruber |
| DOMPurify (`purify`) | HTML sanitizing | Apache-2.0 OR MPL-2.0 | (c) Cure53 and contributors |
| highlight.js (`highlight` + `github-dark` theme) | Code highlighting | BSD-3-Clause | (c) 2006 Ivan Sagalaev and contributors |
| KaTeX (`katex`, `auto-render`, `KaTeX_*` fonts) | Math rendering | MIT | (c) 2013-2020 Khan Academy and other contributors |
| Inter (`vendor/inter`, Latin woff2 + `OFL.txt`) | UI typeface | SIL OFL 1.1 | (c) 2016 The Inter Project Authors |
| jsQR | QR-code camera scanning (phone pairing) | Apache-2.0 | (c) Cosmo Wolfe and contributors |

Each license text sits beside the vendored files in
`localm/plugins/gui/static/vendor/`, copied unchanged from the upstream release
that is vendored: `LICENSE.marked`, `LICENSE.dompurify` (Apache-2.0) and
`LICENSE.dompurify-mpl` (MPL-2.0), `LICENSE.highlightjs`, `LICENSE.katex`,
`LICENSE.jsqr` and `inter/OFL.txt`.

The TTS plugin vendors a separate browser bundle (`kokoro.min.js`) under
`localm/plugins/builtin/tts/static/vendor/`. Its components and licenses are
documented in
[`localm/plugins/builtin/tts/static/vendor/NOTICE.md`](localm/plugins/builtin/tts/static/vendor/NOTICE.md),
with each license text beside the bundle (`LICENSE.kokoro-js`,
`LICENSE.transformers-js`, `LICENSE.onnxruntime-web`, `LICENSE.espeak-ng`):

| Component | Used for | License | Copyright |
|---|---|---|---|
| kokoro-js | Kokoro text-to-speech model runner | Apache-2.0 | per upstream; text in `LICENSE.kokoro-js` |
| transformers.js (`@huggingface/transformers`) | Model loading and inference in the browser | Apache-2.0 | per upstream; text in `LICENSE.transformers-js` |
| onnxruntime-web | ONNX runtime (WebAssembly) | MIT | (c) Microsoft Corporation |
| phonemizer | Text-to-phoneme wrapper | Apache-2.0 | per upstream; covered by the kokoro-js notice |
| eSpeak NG (compiled into `phonemizer`, plus the `espeak-ng-data/` dictionaries) | Phoneme engine and non-English pronunciation dictionaries | GPL-3.0-or-later | (c) the eSpeak NG authors; text in `LICENSE.espeak-ng` |

The TTS bundle is not wholly permissive. eSpeak NG is GPL-3.0-or-later and is
compiled into `kokoro.min.js` (its WebAssembly build and `espeak-ng-data` payload),
and its five non-English dictionaries are committed beside the bundle. Section 13
of the AGPL permits combining an AGPL-covered work with a GPL-3.0 work, so the
combination is licensed, and the combined bundle carries eSpeak NG under the GPL.

The test suite carries four Gemma 4 chat templates from Google's `google/gemma-4-E2B-it` and
`google/gemma-4-31B-it` model repositories (Apache-2.0, (c) Google LLC) as test fixtures in
[`tests/fixtures/gemma4_chat_templates/`](tests/fixtures/gemma4_chat_templates/README.md). They are
not shipped in the package.

## Bundled at release time (native inference binaries)

The `localm-llama-runtime` package carries native inference binaries that are NOT
committed to this repository; they are provisioned locally by `localm setup-llama`.
`setup-llama` copies the upstream `LICENSE` into the runtime `lib/` alongside the
binaries at provision time (as `LICENSE.llama-cpp`), falling back to a bundled
MIT notice if the archive omits one, so any installer or release that ships these
binaries also ships their license text:

| Component | License |
|---|---|
| llama.cpp | MIT (c) 2023-2026 The ggml authors |
| ggml | MIT (c) 2023-2026 The ggml authors |
| AMD ROCm runtime libraries (GPU builds) | Permissive (MIT / BSD-style); see AMD's distribution |
| libgomp (GCC's OpenMP runtime, Linux; from Debian's `libgomp1` package) | GPL-3.0-or-later WITH GCC Runtime Library Exception 3.1; `setup-llama` writes the notice beside it as `LICENSE.libgomp` |

Two more native runtimes are downloaded from their upstream releases at the
user's request and are not redistributed by this repository:
stable-diffusion.cpp (MIT; `localm setup-sdcpp`, loaded into a localm worker
process) and KoboldCpp (AGPL-3.0; `localm setup-music`, run as a separate
program, see the arm's-length section below). localm unpacks each as upstream
ships it and adds or removes no license file: the stable-diffusion.cpp archive
carries `stable-diffusion.cpp.txt` and `ggml.txt` (both MIT texts), and the
KoboldCpp runtime carries `LICENSE.md` (AGPL-3.0) and a `vendor_licenses/`
folder; both stay in the runtime folder beside the binaries.

The NVIDIA CUDA runtime libraries (cudart and cuBLAS) are proprietary, not open
source. They are downloaded only when the user picks a CUDA build
(`localm setup-llama --backend cuda`, `localm setup-sdcpp --backend cuda`): from
NVIDIA's own PyPI packages (`nvidia-cuda-runtime`, `nvidia-cublas` and their
`-cu12` variants, which declare `LicenseRef-NVIDIA-Proprietary`) on Linux, and
from the `cudart` bundle of the upstream llama.cpp or stable-diffusion.cpp
release on Windows. They are not part of this repository, and localm copies the
library files without a license file; NVIDIA's CUDA license terms apply to them.

## Python dependencies (installed from PyPI, not redistributed here)

localm declares its Python dependencies in `pyproject.toml`. They are installed
from PyPI by pip or uv and are NOT redistributed inside this repository, so no
license text is reproduced here. The great majority of declared dependencies are
permissive (MIT / BSD / Apache-2.0 / HPND / ISC / PSF / OLDAP-2.8), with these exceptions:

- **MPL-2.0**, in `certifi` (an unmodified CA-certificate bundle), `hypothesis`
  (the `dev` extra, tests only) and, transitively, `tqdm` (dual-licensed
  MPL-2.0 / MIT).
- **LGPL**, in `zeroconf` (LGPL-2.1-or-later; the mDNS / DNS-SD service-discovery
  library behind `localm.local` network naming) and, under the optional `gpu`
  and `hf` extras, `num2words` (LGPL) and `easydict` (LGPL-3.0).
- **GPL-3.0-only**, in `PyQt6` and `PyQt6-WebEngine`, pulled in transitively by
  `pywebview[qt]` under the optional `desktop` extra on Linux only (see
  `pyproject.toml`'s `desktop` extra and `localm/appface.py`'s Linux
  `gui="qt"` selection). Both are Riverbank Computing packages dual-licensed
  under GPL-3.0-only or a paid commercial license; a `pip install
  "localm[desktop]"` on Linux takes the GPL-3.0-only terms by default. This
  extra is optional, best-effort (localm falls back to a browser tab if it is
  absent or fails to start), and not installed by either `setup.sh` or
  `setup.bat` by default.

MPL-2.0 and the LGPL are both weak copyleft: each library above is
imported unmodified as a separate package installed from PyPI, at arm's length,
and imposes no obligation on localm beyond preserving the upstream license
notices. GPL-3.0-only is strong copyleft and works differently: whether it is
compatible with distributing it alongside localm's own AGPL-3.0-or-later code is
a legal question this notice does not resolve. Anyone distributing a build that
includes the `desktop[qt]` extra on Linux should get their own confirmation
rather than relying on this file.

One dependency's license is not confirmed here: `rocm`, `rocm-sdk-core`, and
`rocm-sdk-libraries-gfx103x-all` (pulled in by the optional `[gpu]` extra, for the
AMD ROCm PyTorch build on Windows) publish no license field or classifier on
PyPI. They wrap AMD's own ROCm SDK components, which AMD distributes under
permissive terms upstream, but that is not independently confirmable from the
PyPI package metadata itself; anyone relying on the `[gpu]` extra on AMD hardware
should check AMD's ROCm distribution terms directly rather than this notice.

## Arm's-length tools (not bundled, linked, or redistributed)

localm interoperates with some external programs by invoking them over HTTP or as
separate subprocesses, for example ComfyUI (GPL-3.0; image / music / video generation), the
ComfyUI-GGUF node (Apache-2.0) and KoboldCpp (AGPL-3.0; native music).
These independent programs, which the user installs or localm downloads
unmodified from upstream on request, are not bundled, statically or
dynamically linked, or redistributed by localm, so their licenses (including
GPL / AGPL) do not extend to localm or its users.

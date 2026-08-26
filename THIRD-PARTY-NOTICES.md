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
| marked | Markdown rendering | MIT | (c) 2011-2024 Christopher Jeffrey and contributors |
| DOMPurify (`purify`) | HTML sanitizing | Apache-2.0 OR MPL-2.0 | (c) Cure53 and contributors |
| highlight.js (`highlight` + `github-dark` theme) | Code highlighting | BSD-3-Clause | (c) 2006 Ivan Sagalaev and contributors |
| KaTeX (`katex`, `auto-render`, `KaTeX_*` fonts) | Math rendering | MIT | (c) 2013-2020 Khan Academy and other contributors |
| Inter (`vendor/inter`, Latin woff2 + `OFL.txt`) | UI typeface | SIL OFL 1.1 | (c) 2016 The Inter Project Authors |
| jsQR | QR-code camera scanning (phone pairing) | Apache-2.0 | (c) Cosmo Wolfe and contributors |

The TTS plugin vendors a separate browser bundle. Its components and licenses are
documented in
[`localm/plugins/builtin/tts/static/vendor/NOTICE.md`](localm/plugins/builtin/tts/static/vendor/NOTICE.md)
(kokoro-js and transformers.js under Apache-2.0; onnxruntime-web and phonemizer
under MIT), with the kokoro-js license text in the adjacent `LICENSE.kokoro-js`.
All are permissive.

## Bundled at release time (native inference binaries)

The `localm-llama-runtime` package carries native inference binaries that are NOT
committed to this repository; they are provisioned locally by `localm setup-llama`.
`setup-llama` copies the upstream `LICENSE` into the runtime `lib/` alongside the
binaries at provision time (as `LICENSE.llama-cpp`), falling back to a bundled
MIT notice if the archive omits one, so any installer or release that ships these
binaries also ships their license text:

| Component | License |
|---|---|
| llama.cpp | MIT (c) 2023-2024 The ggml authors |
| ggml | MIT (c) 2023-2024 The ggml authors |
| AMD ROCm runtime libraries (GPU builds) | Permissive (MIT / BSD-style); see AMD's distribution |

## Python dependencies (installed from PyPI, not redistributed here)

localm declares its Python dependencies in `pyproject.toml`. They are installed
from PyPI by pip or uv and are NOT redistributed inside this repository, so no
license text is reproduced here. The great majority of declared dependencies are
permissive (MIT / BSD / Apache-2.0 / HPND / ISC / PSF), with three exceptions:

- **MPL-2.0**, in `certifi` (an unmodified CA-certificate bundle) and `tqdm`
  (dual-licensed MPL-2.0 / MIT).
- **LGPL-2.1-or-later**, in `zeroconf` (the mDNS / DNS-SD service-discovery
  library behind `localm.local` network naming).
- **GPL-3.0-only**, in `PyQt6` and `PyQt6-WebEngine`, pulled in transitively by
  `pywebview[qt]` under the optional `desktop` extra on Linux only (see
  `pyproject.toml`'s `desktop` extra and `localm/appface.py`'s Linux
  `gui="qt"` selection). Both are Riverbank Computing packages dual-licensed
  under GPL-3.0-only or a paid commercial license; a `pip install
  "localm[desktop]"` on Linux takes the GPL-3.0-only terms by default. This
  extra is optional, best-effort (localm falls back to a browser tab if it is
  absent or fails to start), and not installed by either `setup.sh` or
  `setup.bat` by default.

MPL-2.0 and LGPL-2.1-or-later are both weak copyleft: each library above is
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
separate subprocesses, for example ComfyUI (image / music / video generation).
These independent programs the user installs are not bundled, statically or
dynamically linked, or redistributed by localm, so their licenses (including
GPL / AGPL) do not extend to localm or its users.

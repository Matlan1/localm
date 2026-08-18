# Vendored browser libraries (hand-maintained)

Every file in this directory is a byte-exact copy of an upstream `dist/` build,
committed straight into the repo. The GUI has no build step: `index.html` loads
these as classic scripts and the app code references them as window globals, so
they are served from disk exactly as they arrive from upstream.

Licenses and copyright for each component are recorded in the repo-root
[`THIRD-PARTY-NOTICES.md`](../../../../../THIRD-PARTY-NOTICES.md).

## These are INVISIBLE to dependency tooling. That is the thing to remember.

None of these libraries has an entry in `package.json` or `package-lock.json`,
and there is no lockfile anywhere that names them. `package.json` covers only
the jsdom/Playwright test harness, never the shipped frontend.

The consequence is not subtle: **no dependency scanner, Dependabot included, can
ever see a vulnerable version sitting here.** A published CVE against one of
these libraries produces no PR, no alert, and no red check. Nothing at all
happens, and "nothing happened" is indistinguishable from "we are up to date".

That is not hypothetical. The vendored DOMPurify sat at 3.2.6 across the whole
affected range of CVE-2026-0540 / GHSA-v2wj-7wpq-c8vv (a rawtext `SAFE_FOR_XML`
sanitization bypass affecting 3.1.3 through 3.3.1, fixed in 3.3.2) and nothing in
the toolchain noticed, because there was nothing that could.

## What replaces the missing signal

`tests-js/vendor-dompurify.test.mjs` and `tests-js/vendor-katex.test.mjs` load
the real vendored bytes and assert both a version floor and the actual security
behaviour. They are the only things in the repo that read a vendored file, and
they run under `npm test`.

DOMPurify matters most because it is the FIRST barrier on the main shell, and
for a long time it was the only enforcing one. That is no longer true: the
shell's CSP now ENFORCES with a per-request script-src nonce (PR #1291), so a
sanitizer bypass has something behind it. Both were measured separately in a
real browser on 2026-08-18: with the sanitizer in place an injected handler is
stripped before the browser ever sees it (no CSP violation is raised at all),
and with the sanitizer removed the same payload reaches the DOM and the CSP
refuses to run it. Two independent barriers, so do not read either one as
optional. KaTeX has its own guard for the same reason DOMPurify does: it sat on 0.16.11 for the whole affected range of
CVE-2025-23207 / GHSA-cg87-wmx4-v546 and nothing noticed.

`tests-js/vendor-jsqr.test.mjs` does the same for jsQR, and additionally decodes
a real QR symbol with the vendored bytes - a hash alone would pass on a file that
parses but cannot decode.

`tests-js/vendor-highlightjs.test.mjs` does the same for highlight.js. It pins
the version and content hash, actually highlights real code, and asserts a
GROWTH-SHAPE property rather than a version floor alone: highlight.js 11.12.0
fixed two ReDoS bugs (a c/cpp type-token regex, issue #4362, and a recursive
xml sublanguage reference) with no CVE and no GHSA advisory for either, so no
scanner anywhere could ever have flagged the vulnerable 11.9.0 build that
shipped here. `app/helpers.js` calls `hljs.highlightElement` synchronously on
the main thread for every `pre code` block, and the language comes straight
off a model-emitted fenced code block, so this was a real, reachable
main-thread freeze from ordinary model output, not a theoretical one.

`tests-js/vendor-marked.test.mjs` closes the last gap: marked used to be the one
library with no guard at all. It is pinned differently from the others out of
necessity - MEASURED, `window.marked.version` is undefined on this build, so
there is no runtime version to assert and no banner-vs-runtime cross-check to
make. It pins the banner plus a content hash instead (the same treatment
auto-render.min.js, katex.min.css and jsQR.js get), asserts marked still parses
real markdown, and asserts the option set helpers.js applies at load. One test
asserts the version field is still ABSENT, so if upstream ever adds one the
guard fails and gets upgraded to a real floor rather than quietly pinning bytes
forever.

It also pins the pipeline's actual contract: marked does NOT sanitize. Its own
`sanitize` option was removed upstream in v7 and it passes raw HTML through by
design, which is precisely why DOMPurify has to run on its OUTPUT. Advisory
status for 12.0.2 was established from affected-VERSION RANGES rather than by
reading upstream code for a quoted function, against two independent sources
each with a control query that returned a known-positive: OSV reports 0 vulns,
and all 18 marked advisories in the GitHub advisory database were range-tested
against 12.0.2 with none matching (every one is bounded above by 4.0.10 or
lower, except GHSA-6v9c-7cg6-27q7 which is >= 18.0.0).

### Two things the KaTeX guard had to do differently

**Not every library carries a version.** `katex.min.js` exposes a runtime
`katex.version`, but `auto-render.min.js` and `katex.min.css` carry no version
string at all, and none of the three has a banner comment. Those two are pinned
by recorded content hash instead, which is the only thing that can say which
upstream build they came from. `jsQR.js` has no version string either and is
pinned the same way.

**jsQR used to be UNPINNABLE, and the fix was to change which artefact ships.**
It was `jsQR.min.js`, and npm publishes only an UNMINIFIED `dist/jsQR.js` - so
the vendored bytes had come from a CDN's own minifier, which is not
reproducible. Measured 2026-08-13: the shipped file matched NO published
artefact, and jsdelivr's current 1.3.0 / 1.3.1 / 1.4.0 minified builds all
differed from it and from each other even after normalising line endings and
stripping the `sourceMappingURL` comment. Its provenance was genuinely
unrecoverable. It was replaced with npm's own `dist/jsQR.js` (tarball verified
against the registry shasum), which is unminified and therefore named honestly.
That costs 126 KB on an asset `models-sidebar.js` loads LAZILY - only when a
scan starts on a browser without a native `BarcodeDetector` - so most users
never fetch it. **If you are tempted to re-minify it for size, you would be
trading a verifiable artefact for an unverifiable one, which is the exact
problem this replaced.**

**A content hash must be taken over CRLF-normalised bytes.** This repo has
`core.autocrlf=true` and no `.gitattributes` rule covering this directory, so
any vendored file containing a newline is checked out with different bytes than
the blob git stores. Measured: `katex.min.css` is 23335 B in git and 23336 B in
a Windows working tree; the old `jsQR.min.js` 130469 vs 130470;
`highlight.min.js` 121727 vs 122939. Only `katex.min.js` and `auto-render.min.js` contain no
newline at all, so only those two hash identically either way. A raw-byte hash
would pass on one platform and fail on the other, and the failure would read as
a corrupted vendor drop rather than a line-ending artefact.

## Re-vendoring

Take the upstream `dist/` artefact unmodified. Do not hand-edit these files: the
DOMPurify test asserts that the banner comment agrees with the library's own
runtime `version`, precisely so a half-finished or hand-patched drop fails loudly
instead of looking fine.

```
npm pack dompurify@<version>          # npm verifies the registry integrity hash
tar -xzf dompurify-<version>.tgz
cp package/dist/purify.min.js localm/plugins/gui/static/vendor/purify.min.js
npm test                              # the vendor guard runs here
```

```
npm pack jsqr@<version>
tar -xzf jsqr-<version>.tgz
cp package/dist/jsQR.js  localm/plugins/gui/static/vendor/jsQR.js
cp package/LICENSE       localm/plugins/gui/static/vendor/LICENSE.jsqr
npm test    # then update VENDORED_VERSION + PINNED_JSQR in tests-js/vendor-jsqr.test.mjs
```

KaTeX is three files that MUST move together, from one release. A half-bump
renders visibly broken: 0.18.0's one breaking change prefixed the structural CSS
classes (`base` became `katex-base`, and likewise `strut` and `sizing`), so a new
`katex.min.js` beside an old `katex.min.css` produces unstyled math while every
assertion about the file you did replace still passes. `vendor-katex.test.mjs`
checks that pairing directly.

```
npm pack katex@<version>
tar -xzf katex-<version>.tgz
cp package/dist/katex.min.js            localm/plugins/gui/static/vendor/katex.min.js
cp package/dist/katex.min.css           localm/plugins/gui/static/vendor/katex.min.css
cp package/dist/contrib/auto-render.min.js \
                        localm/plugins/gui/static/vendor/auto-render.min.js
npm test    # then update VENDORED_VERSION + PINNED in tests-js/vendor-katex.test.mjs
```

Check `package/dist/fonts/` against `vendor/fonts/` when you bump KaTeX. All 20
woff2 files were byte-identical between 0.16.11 and 0.18.4, so that directory has
not needed to move so far, but the CSS references them by name and a font
revision would.

```
npm pack @highlightjs/cdn-assets@<version>   # npm verifies the registry integrity hash
tar -xzf highlightjs-cdn-assets-<version>.tgz
cp package/highlight.min.js localm/plugins/gui/static/vendor/highlight.min.js
npm test    # then update VENDORED_VERSION + PINNED_HASH in tests-js/vendor-highlightjs.test.mjs
```

```
npm pack marked@<version>             # npm verifies the registry integrity hash
tar -xzf marked-<version>.tgz
cp package/marked.min.js localm/plugins/gui/static/vendor/marked.min.js
npm test    # then update VENDORED_VERSION + PINNED_HASH in tests-js/vendor-marked.test.mjs
```

`marked.min.js` ships from the PACKAGE ROOT of `marked`, NOT from `lib/`.
`lib/marked.umd.js` is a different artefact of the same version and is not what
`index.html` loads. Re-check the advisory ranges when you bump: the guard pins
bytes, which cannot tell you a newer version is safe.

`highlight.min.js` ships from the PACKAGE ROOT of `@highlightjs/cdn-assets`, not
from `es/`, which is a separate ES-module build of the same version and is NOT
what `index.html` loads as a classic script. Check `package/styles/github-dark.min.css`
against the vendored copy too (CRLF-normalised) before assuming it needs
re-vendoring: 11.9.0 to 11.12.0 changed nothing in that file, so it was left
alone for that bump, but a future release may not be so quiet.

You do not need to touch the service worker. `sw.js`'s `CACHE` constant is a
content digest computed per request over every cacheable asset under
`static/` (see `localm/plugins/gui/web.py`'s `_compute_sw_cache_value`), and this
directory is inside that set, so replacing a file here changes the served cache
name on its own and returning browsers re-fetch it.

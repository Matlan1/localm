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

DOMPurify matters most because it is the sole enforcing XSS barrier on the main
shell (the shell's CSP is still report-only). KaTeX has its own guard for the
same reason DOMPurify does: it sat on 0.16.11 for the whole affected range of
CVE-2025-23207 / GHSA-cg87-wmx4-v546 and nothing noticed.

`marked`, `highlight.js` and `jsQR` still have no guard. Until they do, check
them by hand when you touch this directory.

### Two things the KaTeX guard had to do differently

**Not every library carries a version.** `katex.min.js` exposes a runtime
`katex.version`, but `auto-render.min.js` and `katex.min.css` carry no version
string at all, and none of the three has a banner comment. Those two are pinned
by recorded content hash instead, which is the only thing that can say which
upstream build they came from. `jsQR.min.js` has the same problem in a worse
form (no version string and no upstream minified artefact to hash against).

**A content hash must be taken over CRLF-normalised bytes.** This repo has
`core.autocrlf=true` and no `.gitattributes` rule covering this directory, so
any vendored file containing a newline is checked out with different bytes than
the blob git stores. Measured: `katex.min.css` is 23335 B in git and 23336 B in
a Windows working tree; `jsQR.min.js` 130469 vs 130470; `highlight.min.js`
121727 vs 122939. Only `katex.min.js` and `auto-render.min.js` contain no
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

You do not need to touch the service worker. `sw.js`'s `CACHE` constant is a
content digest computed per request over every cacheable asset under
`static/` (see `localm/plugins/gui/web.py`'s `_compute_sw_cache_value`), and this
directory is inside that set, so replacing a file here changes the served cache
name on its own and returning browsers re-fetch it.

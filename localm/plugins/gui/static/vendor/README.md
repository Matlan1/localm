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

`tests-js/vendor-dompurify.test.mjs` loads the real `purify.min.js` bytes and
asserts both a version floor and the actual security behaviour. It is the only
thing in the repo that reads a vendored file, and it runs under `npm test`.

DOMPurify matters most because it is the sole enforcing XSS barrier on the main
shell (the shell's CSP is still report-only), so it gets the version floor. The
other libraries here have no such guard yet. Until they do, check them by hand
when you touch this directory.

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

You do not need to touch the service worker. `sw.js`'s `CACHE` constant is a
content digest computed per request over every cacheable asset under
`static/` (see `localm/plugins/gui/web.py`'s `_compute_sw_cache_value`), and this
directory is inside that set, so replacing a file here changes the served cache
name on its own and returning browsers re-fetch it.

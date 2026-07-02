# GUI end-to-end tests (real browser)

`npm test` (the jsdom unit suite under `tests-js/`) runs each GUI module in ONE
shared classic-script realm with `import`/`export` stripped, and never fires
`DOMContentLoaded`. That is fast and good for unit logic, but it structurally
**cannot** catch "the page does not load" bugs: real ES-module binding semantics
(a reassigned import throws), a missing/misnamed export (module-graph load
failure), a circular-import TDZ, or a top-level throw during module evaluation. Two
such blank-page bugs shipped GREEN before this suite existed (a reassigned `VIEWS`
that broke every plugin tab; a `JSON.parse` of corrupt `localStorage` that aborted
boot).

This suite loads the **real shipped ES-module graph in a real browser** (headless
Chromium via Playwright), fires the actual boot, and drives every page like a user.
It is the automated version of the "run `localm gui` and click around" smoke that
used to be manual and therefore skipped.

## Run it

```
npm install                        # once (adds @playwright/test)
npx playwright install chromium    # once (downloads the browser to a machine cache)
npm run test:e2e
```

`tests-e2e/serve.mjs` builds a throwaway `LOCALM_HOME`, installs the user-facing
plugins into it, then starts `localm gui --no-model` on a fixed port - in that
order, so the server never binds before its plugin routes exist. Playwright starts
that launcher, waits for it, and tree-kills only that child on exit (it never
adopts or kills a server it did not start). The Python that runs localm is this
repo's own `.venv`, discovered by walking up from the repo (so it works in a normal
clone and in a git worktree); set `LOCALM_E2E_PYTHON` to force a specific one.

## What it asserts

- **Boot + switch:** every nav tab (chat, coder, images, music, video, knowledge,
  jobs, models, plugins, settings) builds, activates on click, becomes the single
  active view, and renders content - with no uncaught page error. This catches the
  reassigned-import / missing-export / broken-handler class directly.
- **Corrupt storage:** with both JSON-backed `localStorage` keys poisoned before
  any script runs, the app still boots and surfaces a warning (never a blank shell).

Both assertions are proven to fail when their bug is reintroduced, so a green run
means something.

## Run it when

Before merging any change to `localm/plugins/gui/static/` (JS, `index.html`, the
service worker, or a plugin's client assets). It is not part of `npm test` so the
unit suite stays fast and browser-free.

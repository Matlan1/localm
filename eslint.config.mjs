// SPDX-License-Identifier: AGPL-3.0-or-later
import js from "@eslint/js";
import globals from "globals";

export default [
  {
    ignores: ["localm/plugins/gui/static/vendor/**"],
  },
  js.configs.recommended,
  {
    // First-party GUI app code, shipped as native ES modules (index.html
    // loads app/main.js with type="module"; pages/* are dynamically
    // imported). vendor/*, loaded as classic <script> tags, put marked,
    // DOMPurify, hljs and renderMathInElement on the global scope.
    files: [
      "localm/plugins/gui/static/app/**/*.js",
      "localm/plugins/gui/static/pages/**/*.js",
    ],
    languageOptions: {
      ecmaVersion: "latest",
      sourceType: "module",
      globals: {
        ...globals.browser,
        // Vendor libraries loaded as classic <script> tags (index.html), put on
        // the global scope rather than imported.
        marked: "readonly",
        DOMPurify: "readonly",
        hljs: "readonly",
        renderMathInElement: "readonly",
        // First-party helpers icons.js also publishes as `window.iconEl = iconEl`
        // (app/icons.js) for callers that reference it without importing it.
        iconEl: "readonly",
      },
    },
  },
  {
    // The PWA service worker: hand-maintained, first-party, but a classic
    // (non-module) script running in the ServiceWorkerGlobalScope.
    files: ["localm/plugins/gui/static/sw.js"],
    languageOptions: {
      ecmaVersion: "latest",
      sourceType: "script",
      globals: {
        ...globals.serviceworker,
      },
    },
  },
  {
    // jsdom test harness (Node, native ES modules).
    files: ["tests-js/**/*.mjs"],
    languageOptions: {
      ecmaVersion: "latest",
      sourceType: "module",
      globals: {
        ...globals.node,
      },
    },
  },
];

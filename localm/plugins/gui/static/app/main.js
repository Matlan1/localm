// SPDX-License-Identifier: AGPL-3.0-or-later
/* localm GUI ES-module entry (loaded as <script type="module"> from index.html).
 * Imports every app/* and pages/* module in load order, so their top-level side
 * effects run in order, then attaches every public binding to window, where the
 * index.html install-UI module and runtime-injected client plugins read it as
 * window.X.
 *
 * Each window.X is a getter into the module namespace object, so it always reads
 * the current value of an `export let` binding that was reassigned at runtime.
 * There is no setter, so `window.X = ...` throws in strict mode; reassign the
 * binding through the module itself. */
import * as m0 from "./client-log.js";
import * as m1 from "./helpers.js";
import * as mIcons from "./icons.js";
import * as mPk from "./picker.js";
import * as m2 from "./theme.js";
import * as m3 from "./logo.js";
import * as m4 from "./tabs.js";
import * as m5 from "./models-sidebar.js";
import * as m6 from "./chat.js";
import * as m7 from "./cmdk.js";
import * as m8 from "./settings-perf.js";
import * as m9 from "./coder.js";
import * as m10 from "./slash.js";
import * as m11 from "./init.js";
import * as m12 from "../pages/dispatch.js";
import * as m13 from "../pages/models.js";
import * as m14 from "../pages/images.js";
import * as m15 from "../pages/plugins.js";
import * as m16 from "../pages/settings.js";
import * as m17 from "../pages/workflow.js";
import * as m18 from "../pages/music.js";
import * as m19 from "../pages/video.js";
import * as m20 from "../pages/knowledge.js";

for (const mod of [m0, m1, mIcons, mPk, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15, m16, m17, m18, m19, m20]) {
  for (const name of Object.keys(mod)) {
    try {
      Object.defineProperty(window, name, { get: () => mod[name], configurable: true, enumerable: true });
    } catch (_) { /* read-only global; skip */ }
  }
}

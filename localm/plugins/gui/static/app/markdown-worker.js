// SPDX-License-Identifier: AGPL-3.0-or-later
/* localm GUI - Markdown worker, started by helpers.js as a module worker. Parses
   each request with the same vendored marked and options as the page and posts
   the unsanitised HTML back; the page sanitises it. */
import "../vendor/marked.min.js";
import { markdownWorkerReply } from "./markdown-parse.js";

self.onmessage = (e) => {
  self.postMessage(markdownWorkerReply(self.marked, e.data));
};

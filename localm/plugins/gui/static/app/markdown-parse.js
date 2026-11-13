// SPDX-License-Identifier: AGPL-3.0-or-later
/* localm GUI - the Markdown parse step, shared by the page (helpers.js) and the
   Markdown worker (markdown-worker.js). Pure: no DOM, no globals; the caller
   passes the marked library in. */
"use strict";

/** The marked options of every reply and document render, on either thread. */
export const MARKED_OPTIONS = Object.freeze({ breaks: true });

const _VOID_TAGS = new Set([
  "area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta",
  "param", "source", "track", "wbr",
]);
const _TAG_NAME = /^\/?([A-Za-z][A-Za-z0-9-]*)/;

/** The deepest element nesting in *html*, counting every start tag that is not
 *  void or self-closed as one level until its end tag. An unclosed start tag
 *  counts until the end of the string. Linear in the length of *html*. */
export function htmlNestingDepth(html) {
  const s = String(html || "");
  let depth = 0;
  let max = 0;
  let at = s.indexOf("<");
  while (at >= 0) {
    const end = s.indexOf(">", at + 1);
    if (end < 0) break;
    const m = _TAG_NAME.exec(s.slice(at + 1, Math.min(end, at + 64)));
    if (m) {
      const closing = s.charCodeAt(at + 1) === 47;
      const name = m[1].toLowerCase();
      if (closing) {
        if (depth > 0) depth -= 1;
      } else if (!_VOID_TAGS.has(name) && s.charCodeAt(end - 1) !== 47) {
        depth += 1;
        if (depth > max) max = depth;
      }
    }
    at = s.indexOf("<", end + 1);
  }
  return max;
}

/** Parse each Markdown source of *parts* with *lib* (the vendored marked) and
 *  MARKED_OPTIONS.
 *
 *  Returns `{ html, depth }`: `html[i]` is the HTML of `parts[i]`, or null where
 *  `parts[i]` is null, and `depth` is the deepest element nesting across them
 *  (htmlNestingDepth). The HTML is NOT sanitised. Throws whatever marked throws. */
export function parseMarkdownParts(lib, parts) {
  const html = parts.map((p) => (p === null ? null : lib.parse(p, MARKED_OPTIONS)));
  let depth = 0;
  for (const h of html) if (h !== null) depth = Math.max(depth, htmlNestingDepth(h));
  return { html, depth };
}

/** The Markdown worker's reply to one request `{ id, parts }`: `{ id, html, depth }`
 *  (parseMarkdownParts) on success, `{ id, error }` with marked's message when
 *  it threw. Never throws. */
export function markdownWorkerReply(lib, data) {
  const id = data && data.id;
  try {
    return { id, ...parseMarkdownParts(lib, (data && data.parts) || []) };
  } catch (e) {
    return { id, error: String((e && e.message) || e) };
  }
}

// SPDX-License-Identifier: AGPL-3.0-or-later
// App release tag order shared by the update routes: PEP 440 prerelease ranks.

// A prerelease suffix: dev, a, alpha, b, beta or rc, then a number with no
// separator (dev1, rc1, b2, alpha3), in any case; a dotted number (rc.1) is not one.
export const PRERELEASE_SUFFIX_RE = /^(dev|rc|a|b|alpha|beta)(\d+)$/i;

// Rank of each prerelease kind, in PEP 440 order: dev < a = alpha < b = beta < rc.
// A final release ranks above every kind.
export const PRERELEASE_RANK = { dev: 0, a: 1, alpha: 1, b: 2, beta: 2, rc: 3 };
export const FINAL_RANK = 4;

// [major, minor, patch, kind rank, prerelease number] for an app tag, or null when
// the tag does not start with vX.Y.Z or its suffix is neither empty nor a prerelease
// suffix (v0.3.2.1, v0.3.2-hotfix, v0.3.2+b, v0.3.2a, v0.3.2-rc.1). A final release
// has rank FINAL_RANK and number 0.
export function tagKey(tag) {
  const m = /^v(\d+)\.(\d+)\.(\d+)(.*)$/.exec(String(tag || ""));
  if (!m) return null;
  const suffix = m[4].replace(/^[-.]/, "");
  if (!suffix) return [Number(m[1]), Number(m[2]), Number(m[3]), FINAL_RANK, 0];
  const p = PRERELEASE_SUFFIX_RE.exec(suffix);
  if (!p) return null;
  return [Number(m[1]), Number(m[2]), Number(m[3]),
    PRERELEASE_RANK[p[1].toLowerCase()], Number(p[2])];
}

// True when app tag *a* orders at or below app tag *b*.
export function tagAtMost(a, b) {
  const ka = tagKey(a), kb = tagKey(b);
  if (!ka || !kb) return false;
  for (let i = 0; i < 5; i++) {
    if (ka[i] !== kb[i]) return ka[i] < kb[i];
  }
  return true;
}

// True when *tag* is an app tag with a prerelease suffix (v0.3.4-dev1, v0.3.4rc1).
export function isPrereleaseTag(tag) {
  const k = tagKey(tag);
  return k !== null && k[3] !== FINAL_RANK;
}

// True when *client*, a version with or without the leading "v", orders below app tag
// *tag*. A value that is not a version is not older.
export function clientOlderThan(client, tag) {
  const c = "v" + String(client).replace(/^v/, "");
  return tagAtMost(c, tag) && !tagAtMost(tag, c);
}

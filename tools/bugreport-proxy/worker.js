// SPDX-License-Identifier: AGPL-3.0-or-later
//
// localm bug-report proxy - a tiny Cloudflare Worker.
//
// Why this exists: localm is shipped to 1-2 testers while the repo is private.
// We want their in-app "Send to maintainer" button to file a GitHub issue WITHOUT
// giving each tester a GitHub account and WITHOUT baking a GitHub token into the
// distributed app (a token in the app leaks the moment someone opens the build,
// and even a scoped Issues:write token can read every issue and spam the repo).
//
// Instead the app POSTs the (user-reviewed) report to THIS Worker, which holds a
// fine-grained GitHub token as an encrypted secret and creates the issue. The
// token never leaves Cloudflare, it can only file issues through this rate-limited
// endpoint, and it rotates here without re-shipping the app.
//
// Deploy + secrets: see README.md in this folder.
//
// Request:  POST <worker-url>  { "title": "...", "body": "..." }
//           optional header  X-Localm-Token: <SHARED_SECRET>
// Response: 201 { "ok": true, "url": "<issue url>", "number": 123 }  on success
//           4xx/5xx { "error": "...", ... }                          otherwise

export default {
  async fetch(request, env) {
    if (request.method === "OPTIONS") return cors(new Response(null, { status: 204 }));
    if (request.method !== "POST") return json({ error: "POST only" }, 405);

    // Optional shared-secret gate. Low-value (it only permits filing an issue via
    // this proxy), but set SHARED_SECRET to keep random scanners from spamming.
    if (env.SHARED_SECRET) {
      const got = request.headers.get("X-Localm-Token") || "";
      if (!timingSafeEqual(got, env.SHARED_SECRET)) return json({ error: "unauthorized" }, 401);
    }
    if (!env.GITHUB_TOKEN || !env.TARGET_REPO) {
      return json({ error: "proxy misconfigured: set GITHUB_TOKEN and TARGET_REPO" }, 500);
    }

    let payload;
    try { payload = await request.json(); } catch { return json({ error: "invalid JSON" }, 400); }
    const title = (String(payload.title || "").slice(0, 200).trim()) || "localm bug report";
    const body = String(payload.body || "").slice(0, 60000);
    if (!body.trim()) return json({ error: "empty report body" }, 400);

    const issue = { title, body };
    // Labels are optional and must ALREADY EXIST on the repo (GitHub rejects an
    // issue that names a non-existent label), so they are opt-in via env.
    if (env.ISSUE_LABELS) {
      const labels = env.ISSUE_LABELS.split(",").map((s) => s.trim()).filter(Boolean);
      if (labels.length) issue.labels = labels;
    }

    const gh = await fetch(`https://api.github.com/repos/${env.TARGET_REPO}/issues`, {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${env.GITHUB_TOKEN}`,
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "localm-bugreport-proxy",
        "Content-Type": "application/json",
      },
      body: JSON.stringify(issue),
    });
    if (!gh.ok) {
      const detail = (await gh.text().catch(() => "")).slice(0, 500);
      return json({ error: "GitHub rejected the issue", status: gh.status, detail }, 502);
    }
    const data = await gh.json();
    return json({ ok: true, url: data.html_url, number: data.number }, 201);
  },
};

function json(obj, status = 200) {
  return cors(new Response(JSON.stringify(obj), {
    status, headers: { "Content-Type": "application/json" },
  }));
}

// The localm app posts server-side (no browser), so CORS is not required, but
// allowing it is harmless and lets a browser-based caller work too.
function cors(resp) {
  resp.headers.set("Access-Control-Allow-Origin", "*");
  resp.headers.set("Access-Control-Allow-Methods", "POST, OPTIONS");
  resp.headers.set("Access-Control-Allow-Headers", "Content-Type, X-Localm-Token");
  return resp;
}

// Constant-time-ish compare so the secret check does not leak its contents via
// per-character timing. (Length is not hidden; acceptable for a low-value secret.)
function timingSafeEqual(a, b) {
  if (typeof a !== "string" || typeof b !== "string" || a.length !== b.length) return false;
  let out = 0;
  for (let i = 0; i < a.length; i++) out |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return out === 0;
}

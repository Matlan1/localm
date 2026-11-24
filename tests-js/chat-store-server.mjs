// SPDX-License-Identifier: AGPL-3.0-or-later
// An in-memory stand-in for the conversation store of
// localm/plugins/builtin/chat/plug.py and /v1/config. The rules of its PUT, GET
// and DELETE are those of fixtures/chat-store-contract.json, which
// tests/test_chat_store_contract.py runs against the real routes.

const plain = (x) => JSON.parse(JSON.stringify(x));

function textOf(m) {
  if (typeof m.content === "string") return m.content;
  return (m.content || []).filter((p) => p.type === "text").map((p) => p.text).join("");
}

function mediaOf(m) {
  const parts = (Array.isArray(m.content) ? m.content : []).filter((p) => p.type !== "text")
    .map(({ refusedBy: _r, notSent: _n, name: _a, seconds: _s, ...p }) => p);
  return parts.length ? JSON.stringify(parts) : "";
}

function countTurns(messages, out) {
  for (const m of messages || []) {
    const key = JSON.stringify(typeof m.text === "string" ? ["user", m.text, ""]
      : m.kind ? [m.kind, m.query || m.url || "", ""] : [m.role, textOf(m), mediaOf(m)]);
    out.set(key, (out.get(key) || 0) + 1);
    countTurns(m.compacted, out);
  }
  return out;
}

function turns(data) {
  const out = countTurns(data.messages, new Map());
  for (const rec of data.branches || []) {
    (rec.tails || []).forEach((tail, i) => { if (i !== rec.current) countTurns(tail, out); });
  }
  return out;
}

function wouldLose(stored, body) {
  const have = turns(body);
  return [...turns(stored)].some(([key, n]) => (have.get(key) || 0) < n);
}

export function makeServer({ instance = "backend-x", principal = "owner" } = {}) {
  const s = { files: new Map(), deleted: new Set(), puts: [], mode: "full", calls: [], failPut: 0,
              gui: [], chatReply: null, instance, principal, pageSize: 0, hold: null, loseAnswer: 0,
              chatPosts: [], active: "", refuseKey: false, failGet: 0 };
  let n = 0;
  const res = (status, body) => ({ ok: status < 400, status, statusText: String(status),
    headers: { get: () => null }, text: async () => JSON.stringify(body), json: async () => plain(body) });
  const row = (r) => ({ id: r.id, title: r.title, updated_at: r.updated_at, rev: r.rev, save_id: r.save_id ?? null,
    pinned: !!r.pinned, folder: r.folder || null, n_messages: r.messages.length });
  const stale = (r, deleted = false) => res(409, { detail: {
    error: "stale", rev: r ? r.rev : null, row: r ? row(r) : null, deleted } });
  /** Stores a conversation as another browser would. Returns its revision. */
  s.put = (id, data) => {
    const r = { id, title: "t", updated_at: Date.now(), branches: [], params: {}, messages: [], ...data, rev: "r" + (++n) };
    s.files.set(id, r);
    return r.rev;
  };
  /** Deletes a conversation as another browser would. */
  s.remove = (id) => { s.files.delete(id); s.deleted.add(id); };
  /** The answer of the store to one request. */
  s.store = (method, id, body, headers = {}) => {
    const stored = s.files.get(id);
    if (s.mode === "privacy") return res(403, { detail: "privacy" });
    const meant = headers["X-LocaLM-Chat-Store"];
    if (method === "DELETE" && meant != null && meant !== s.instance + " " + s.principal) {
      return res(403, { detail: { error: "owner" } });
    }
    if (method === "GET") {
      if (stored) return res(200, stored);
      return s.deleted.has(id) ? res(410, { detail: "deleted" }) : res(404, { detail: "none" });
    }
    if (method === "DELETE") {
      const had = s.files.has(id);
      s.remove(id);
      return res(200, { status: had ? "deleted" : "absent", id });
    }
    if ((body.instance && body.instance !== s.instance) || (body.principal && body.principal !== s.principal)) {
      return res(403, { detail: { error: "owner" } });
    }
    if (!stored && s.deleted.has(id)) return stale(null, true);
    if ("base_rev" in body) {
      if ((stored ? stored.rev : null) !== body.base_rev) return stale(stored);
    } else if (stored && wouldLose(stored, body)) return stale(stored);
    const params = "params" in body ? body.params : (stored && stored.params) || {};
    return res(200, { status: "saved", id, rev: s.put(id, { ...body, params }) });
  };
  s.fetch = async (url, opts = {}) => {
    const u = String(url);
    const method = opts.method || "GET";
    s.calls.push(method + " " + u);
    if (s.hold) await s.hold(method, u);
    if (s.refuseKey && !u.startsWith("/api/session")) return res(401, { detail: "key" });
    if (u === "/v1/config") {
      return res(200, { effective_mode: s.mode, n_ctx_max: 16384, instance_id: s.instance });
    }
    if (u.startsWith("/api/conversations?")) {
      const who = { instance: s.instance, principal: s.principal };
      if (s.mode === "privacy") return res(200, { enabled: false, conversations: [], stored: s.files.size, ...who });
      const all = [...s.files.values()].map(row);
      const offset = Number(/offset=(\d+)/.exec(u)?.[1] || 0);
      const rows = s.pageSize ? all.slice(offset, offset + s.pageSize) : all;
      return res(200, { enabled: true, conversations: rows, total: all.length, unreadable: [], ...who });
    }
    const m = /^\/api\/conversations\/([^?]+)$/.exec(u);
    if (m) {
      const body = opts.body ? JSON.parse(opts.body) : null;
      const id = decodeURIComponent(m[1]);
      if (method === "PUT") s.puts.push({ id, body });
      if (method === "GET" && s.failGet > 0) { s.failGet--; return res(500, { detail: "unreadable" }); }
      if (method === "PUT" && s.failPut) return res(s.failPut, { detail: "no" });
      const answer = s.store(method, id, body, opts.headers);
      if (method === "PUT" && s.loseAnswer > 0) {
        s.loseAnswer--;
        throw new TypeError("Failed to fetch");
      }
      return answer;
    }
    if (u === "/api/gui-error") s.gui.push(String(opts.body));
    if (u === "/v1/chat/completions") s.chatPosts.push(JSON.parse(opts.body));
    if (u === "/v1/chat/completions" && s.chatReply) return s.chatReply(JSON.parse(opts.body), opts);
    return res(200, { models: s.active ? [{ name: s.active }] : [], active: s.active, plugins: [], items: [] });
  };
  return s;
}

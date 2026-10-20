// SPDX-License-Identifier: AGPL-3.0-or-later
// The setup panel's "hooks" button: the web form of the coder REPL's /hooks,
// /hooks trust and /hooks forget.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages } from "./harness.mjs";

const HOOKS = {
  configured: true, cwd: "/tmp/project",
  config_path: "/tmp/project/.localcoder/config.toml",
  digest: "abc123", warnings: [],
  hooks: [{ event: "pre_tool_use", index: 1, label: "pre_tool_use hook #1", matcher: "write_file",
            command: "python hooks/guard.py", timeout: 30, error: null }],
};

function makeFetch(calls, hooksData) {
  return async (url, opts = {}) => {
    const u = String(url);
    const method = opts.method || "GET";
    calls.push({ url: u, method, body: opts.body ? JSON.parse(opts.body) : null });
    if (u.startsWith("/api/coder/hooks?")) {
      return { ok: true, status: 200, json: async () => hooksData, text: async () => "" };
    }
    if (u.startsWith("/api/coder/hooks/")) {
      return { ok: true, status: 200, json: async () => ({ trusted: true }), text: async () => "" };
    }
    return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
  };
}

function buttonByText(win, text) {
  return [...win.document.querySelectorAll("#modal-body button")]
    .find((b) => b.textContent.trim() === text);
}

test("hooks: lists the project's hooks and trusts the digest it was shown", async () => {
  const calls = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(calls, { ...HOOKS, trusted: false }) });
  assert.ok(win.document.getElementById("setup-hooks"), "the setup panel has a hooks button");
  win.document.getElementById("setup-cwd").value = "/tmp/project";
  await win.openHooksModal();
  const get = calls.find((c) => c.url.startsWith("/api/coder/hooks?"));
  assert.match(get.url, /cwd=%2Ftmp%2Fproject/);
  const body = win.document.getElementById("modal-body").textContent;
  assert.match(body, /pre_tool_use hook #1: python hooks\/guard\.py/);
  assert.match(body, /Not trusted/);
  const trust = buttonByText(win, "trust these hooks");
  assert.ok(trust, "an untrusted project offers a trust button");
  assert.equal(buttonByText(win, "stop trusting"), undefined);
  await trust.onclick();
  const post = calls.find((c) => c.url === "/api/coder/hooks/trust");
  assert.equal(post.method, "POST");
  assert.deepEqual(post.body, { cwd: "/tmp/project", digest: "abc123" });
});

test("hooks: trusted hooks offer stop trusting, which posts forget", async () => {
  const calls = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(calls, { ...HOOKS, trusted: true }) });
  win.document.getElementById("setup-cwd").value = "/tmp/project";
  await win.openHooksModal();
  assert.equal(buttonByText(win, "trust these hooks"), undefined);
  const forget = buttonByText(win, "stop trusting");
  assert.ok(forget);
  await forget.onclick();
  const post = calls.find((c) => c.url === "/api/coder/hooks/forget");
  assert.equal(post.method, "POST");
  assert.deepEqual(post.body, { cwd: "/tmp/project" });
});

test("hooks: a project with none says so, and no cwd never reaches the server", async () => {
  const calls = [];
  const { window: win } = loadAppWithPages({
    fetchImpl: makeFetch(calls, { configured: false, cwd: "/tmp/project", warnings: [] }) });
  win.document.getElementById("setup-cwd").value = "";
  await win.openHooksModal();
  assert.equal(calls.filter((c) => c.url.startsWith("/api/coder/hooks")).length, 0);
  win.document.getElementById("setup-cwd").value = "/tmp/project";
  await win.openHooksModal();
  assert.match(win.document.getElementById("modal-body").textContent,
    /No tool hooks or MCP servers in this project/);
});

test("hooks: MCP servers are listed with the hooks and trusted with the same digest", async () => {
  const calls = [];
  const data = {
    ...HOOKS, trusted: false, hooks: [],
    mcp_servers: [{ name: "db", argv: ["npx", "-y", "server-sqlite"], env: {}, trusted_calls: true,
                    line: 'MCP server "db": ["npx", "-y", "server-sqlite"] (trusted = true)' }],
  };
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(calls, data) });
  win.document.getElementById("setup-cwd").value = "/tmp/project";
  await win.openHooksModal();
  const body = win.document.getElementById("modal-body").textContent;
  assert.match(body, /MCP server "db": \["npx", "-y", "server-sqlite"\]/);
  assert.match(body, /its tool calls are not confirmed/);
  assert.match(body, /no MCP server is started/);
  assert.equal(buttonByText(win, "trust these hooks"), undefined);
  const trust = buttonByText(win, "trust these commands");
  assert.ok(trust, "an untrusted MCP project offers a trust button");
  await trust.onclick();
  const post = calls.find((c) => c.url === "/api/coder/hooks/trust");
  assert.deepEqual(post.body, { cwd: "/tmp/project", digest: "abc123" });
});

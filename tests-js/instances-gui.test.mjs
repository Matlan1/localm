// SPDX-License-Identifier: AGPL-3.0-or-later
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

function makeFetch(instances, calls, stopResponse) {
  return async (url, opts = {}) => {
    const u = String(url);
    if (u.startsWith("/api/instances/") && u.endsWith("/stop")) {
      calls.push({ url: u, method: opts.method });
      if (stopResponse && stopResponse.ok === false) {
        return {
          ok: false, status: stopResponse.status || 502,
          json: async () => ({ detail: stopResponse.detail }),
        };
      }
      return { ok: true, status: 200, json: async () => (stopResponse ? stopResponse.body : {}) };
    }
    if (u === "/api/instances") {
      return { ok: true, status: 200, json: async () => ({ instances }) };
    }
    return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
  };
}

const SELF_ROW = {
  instance_id: "selfinstance0001", self: true, alive: true,
  root_dir: "/proj/self", mode: "full", scheme: "http", host: "127.0.0.1",
  port: 8642, address: "http://127.0.0.1:8642", pid: 111, started: "t0",
};
const OTHER_ROW = {
  instance_id: "otherinstance0002", self: false, alive: true,
  root_dir: "/proj/other", mode: "api", scheme: "http", host: "127.0.0.1",
  port: 8643, address: "http://127.0.0.1:8643", pid: 222, started: "t0",
};
const UNREACHABLE_ROW = {
  instance_id: "deadinstance00003", self: false, alive: false,
  root_dir: "/proj/dead", mode: "api", scheme: "http", host: "0.0.0.0",
  port: 9000, address: "http://0.0.0.0:9000", pid: 333, started: "t0",
};

function instancesTable(window) {
  return window.document.querySelector("#instances-list table.data-table");
}
function instancesRows(window) {
  const table = instancesTable(window);
  return table ? [...table.querySelectorAll("tbody tr")] : [];
}

test("instances-gui: lists other instances and filters out self", async () => {
  const { window } = loadAppWithPages({
    fetchImpl: makeFetch([SELF_ROW, OTHER_ROW], []),
  });
  await window.refreshInstancesCard();
  await new Promise((r) => setTimeout(r, 0));

  const rows = instancesRows(window);
  assert.equal(rows.length, 1, "only the non-self instance is rendered");
  assert.ok(rows[0].textContent.includes("/proj/other"),
    "the row shows the other instance's directory");
  assert.ok(!rows[0].textContent.includes("/proj/self"),
    "the server's own row must not appear in this card");
});

test("instances-gui: shows the bind address and liveness status per row", async () => {
  const { window } = loadAppWithPages({
    fetchImpl: makeFetch([OTHER_ROW, UNREACHABLE_ROW], []),
  });
  await window.refreshInstancesCard();
  await new Promise((r) => setTimeout(r, 0));

  const rows = instancesRows(window);
  assert.equal(rows.length, 2);
  const live = rows.find((tr) => tr.textContent.includes("/proj/other"));
  const dead = rows.find((tr) => tr.textContent.includes("/proj/dead"));
  assert.ok(live.textContent.includes("http://127.0.0.1:8643"), "live row shows its address");
  assert.ok(live.textContent.toLowerCase().includes("live"));
  assert.ok(dead.textContent.includes("http://0.0.0.0:9000"), "unreachable row still shows its address");
  assert.ok(dead.textContent.toLowerCase().includes("no answer"));
});

test("instances-gui: only self running shows the empty state, not an empty table", async () => {
  const { window } = loadAppWithPages({
    fetchImpl: makeFetch([SELF_ROW], []),
  });
  await window.refreshInstancesCard();
  await new Promise((r) => setTimeout(r, 0));

  assert.equal(instancesRows(window).length, 0);
  const box = window.document.querySelector("#instances-list");
  assert.ok(box.querySelector(".empty-state"), "an empty result renders the designed empty state");
});

test("instances-gui: a genuinely empty registry also shows the empty state", async () => {
  const { window } = loadAppWithPages({
    fetchImpl: makeFetch([], []),
  });
  await window.refreshInstancesCard();
  await new Promise((r) => setTimeout(r, 0));

  const box = window.document.querySelector("#instances-list");
  assert.ok(box.querySelector(".empty-state"));
});

test("instances-gui: Stop asks confirmDanger naming the instance, and declining makes no request", async () => {
  const calls = [];
  const { window } = loadAppWithPages({
    fetchImpl: makeFetch([OTHER_ROW], calls),
  });
  await window.refreshInstancesCard();
  await new Promise((r) => setTimeout(r, 0));

  let seenTitle = null, seenMessage = null;
  runScript(window, `
    confirmDanger = (title, message, label, onConfirm) => {
      window.__seenTitle = title;
      window.__seenMessage = message;
      window.__seenLabel = label;
      // decline: never call onConfirm()
    };
  `);
  const rows = instancesRows(window);
  const stopBtn = [...rows[0].querySelectorAll("button")]
    .find((b) => b.textContent === "Stop");
  assert.ok(stopBtn, "a Stop button is rendered for the other instance");
  stopBtn.onclick();
  await new Promise((r) => setTimeout(r, 0));

  seenTitle = window.__seenTitle;
  seenMessage = window.__seenMessage;
  assert.match(seenMessage, /\/proj\/other/, "the confirmation names the target directory");
  assert.deepEqual(calls, [], "declining the confirmation must not call the stop route");
});

test("instances-gui: confirming Stop POSTs the instance id, toasts, and refreshes the list", async () => {
  const calls = [];
  const fetchCalls = [];
  const baseFetch = makeFetch([OTHER_ROW], calls, { body: { status: "stopped" } });
  const { window } = loadAppWithPages({
    fetchImpl: async (url, opts) => { fetchCalls.push(String(url)); return baseFetch(url, opts); },
  });
  const toasts = [];
  window.toast = (msg, isError) => toasts.push({ msg: String(msg), isError: !!isError });
  await window.refreshInstancesCard();
  await new Promise((r) => setTimeout(r, 0));

  runScript(window, `confirmDanger = (title, message, label, onConfirm) => onConfirm();`);
  const stopBtn = [...instancesRows(window)[0].querySelectorAll("button")]
    .find((b) => b.textContent === "Stop");
  const listFetchesBefore = fetchCalls.filter((u) => u === "/api/instances").length;
  stopBtn.onclick();
  await new Promise((r) => setTimeout(r, 0));
  await new Promise((r) => setTimeout(r, 0));

  assert.equal(calls.length, 1);
  assert.equal(calls[0].method, "POST");
  assert.equal(calls[0].url, "/api/instances/otherinstance0002/stop");
  assert.ok(toasts.some((t) => !t.isError && t.msg.includes("/proj/other")),
    `expected a success toast naming the stopped instance, got: ${JSON.stringify(toasts)}`);
  const listFetchesAfter = fetchCalls.filter((u) => u === "/api/instances").length;
  assert.ok(listFetchesAfter > listFetchesBefore, "a successful stop refreshes the card");
});

test("instances-gui: a server-side failure is reported as an error, not swallowed as success", async () => {
  const calls = [];
  const { window } = loadAppWithPages({
    fetchImpl: makeFetch([OTHER_ROW], calls,
      { ok: false, status: 502, detail: "Could not confirm instance stopped (pid 222)." }),
  });
  const toasts = [];
  window.toast = (msg, isError) => toasts.push({ msg: String(msg), isError: !!isError });
  await window.refreshInstancesCard();
  await new Promise((r) => setTimeout(r, 0));

  runScript(window, `confirmDanger = (title, message, label, onConfirm) => onConfirm();`);
  const stopBtn = [...instancesRows(window)[0].querySelectorAll("button")]
    .find((b) => b.textContent === "Stop");
  stopBtn.onclick();
  await new Promise((r) => setTimeout(r, 0));

  assert.ok(toasts.some((t) => t.isError && t.msg.includes("Could not confirm instance stopped")),
    `expected the server's own reason surfaced as an error toast, got: ${JSON.stringify(toasts)}`);
  assert.ok(!toasts.some((t) => !t.isError), "must not ALSO report success");
  assert.equal(stopBtn.disabled, false, "the button must re-enable after a failed attempt");
});

test("instances-gui: a read-only key (403 on the list) hides the card instead of crashing", async () => {
  const { window } = loadAppWithPages({
    fetchImpl: async (url) => {
      if (String(url) === "/api/instances") {
        return { ok: false, status: 403, json: async () => ({ detail: "forbidden" }) };
      }
      return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
    },
  });
  await window.refreshInstancesCard();
  await new Promise((r) => setTimeout(r, 0));

  assert.equal(instancesRows(window).length, 0);
  const box = window.document.querySelector("#instances-list");
  assert.equal(box.children.length, 0, "hidden, not an empty-state or an error dump");
});

// A row from a DIFFERENT install (same_install: false) is listed but carries no
// Stop button: this server holds no shutdown credential for another install, and
// the route refuses such an id with 409. Rows with same_install absent or true
// keep the button, so an older payload renders exactly as before.

const FOREIGN_ROW = {
  instance_id: "foreigninstance04", self: false, alive: true,
  same_install: false, root_dir: "/proj/elsewhere", mode: "full",
  scheme: "http", host: "127.0.0.1", port: 8644,
  address: "http://127.0.0.1:8644", pid: 444, started: null,
};

test("instances-gui: an instance of another install is listed", async () => {
  const { window } = loadAppWithPages({
    fetchImpl: makeFetch([SELF_ROW, FOREIGN_ROW], []),
  });
  await window.refreshInstancesCard();
  await new Promise((r) => setTimeout(r, 0));

  const rows = instancesRows(window);
  assert.equal(rows.length, 1, "the other install's instance is rendered");
  assert.ok(rows[0].textContent.includes("/proj/elsewhere"),
    "its directory is shown, so the card really does span installs");
  assert.ok(rows[0].textContent.includes("http://127.0.0.1:8644"),
    "its address is shown");
});

test("instances-gui: an instance of another install gets a label, not a Stop button",
  async () => {
    const calls = [];
    const { window } = loadAppWithPages({
      fetchImpl: makeFetch([SELF_ROW, FOREIGN_ROW], calls),
    });
    window.confirmDanger = () => {
      throw new Error("confirmDanger must not be reachable for another install");
    };
    await window.refreshInstancesCard();
    await new Promise((r) => setTimeout(r, 0));

    const row = instancesRows(window)[0];
    const stopBtn = [...row.querySelectorAll("button")]
      .find((b) => b.textContent === "Stop");
    assert.equal(stopBtn, undefined,
      "no Stop button: the route refuses a cross-install id with 409");

    const label = row.querySelector(".instances-foreign");
    assert.ok(label, "the action cell carries the other-install label instead");
    assert.equal(label.textContent, "other install");
    assert.ok(/different LocaLM install/.test(label.title || ""),
      "the label explains why it cannot be stopped from here");
    assert.equal(calls.length, 0, "nothing was requested for that row");
  });

test("instances-gui: a same-install row keeps its Stop button", async () => {
  const { window } = loadAppWithPages({
    fetchImpl: makeFetch(
      [SELF_ROW, { ...OTHER_ROW, same_install: true }, FOREIGN_ROW], []),
  });
  await window.refreshInstancesCard();
  await new Promise((r) => setTimeout(r, 0));

  const rows = instancesRows(window);
  assert.equal(rows.length, 2);
  const own = rows.find((r) => r.textContent.includes("/proj/other"));
  const foreign = rows.find((r) => r.textContent.includes("/proj/elsewhere"));
  assert.ok([...own.querySelectorAll("button")].some((b) => b.textContent === "Stop"),
    "this install's own instance is still stoppable");
  assert.equal(own.querySelector(".instances-foreign"), null);
  assert.equal(
    [...foreign.querySelectorAll("button")].find((b) => b.textContent === "Stop"),
    undefined);
});

test("instances-gui: a row with no directory reported renders a placeholder", async () => {
  const { window } = loadAppWithPages({
    fetchImpl: makeFetch([{ ...FOREIGN_ROW, root_dir: null }], []),
  });
  await window.refreshInstancesCard();
  await new Promise((r) => setTimeout(r, 0));

  const cells = instancesRows(window)[0].querySelectorAll("td");
  assert.equal(cells[0].textContent, "(not reported)",
    "a network-bound peer omits root_dir from /whoami; the cell says so");
});

// SPDX-License-Identifier: AGPL-3.0-or-later
// The parameters drawer's per-chat fields (system prompt, sampling, seed,
// GBNF grammar, persona) belong to the conversation on screen: a new
// conversation starts blank and an older one gets its own values back.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp, runScript } from "./harness.mjs";

const FIELDS = [
  ["p-grammar", 'root ::= "a" "'],
  ["p-system", "You are terse."],
  ["p-temperature", "0.3"],
  ["p-top-p", "0.8"],
  ["p-top-k", "12"],
  ["p-repeat-penalty", "1.2"],
  ["p-max-tokens", "256"],
  ["p-seed", "42"],
];

function boot() {
  const { window: win } = loadApp();
  const doc = win.document;
  const val = (id) => doc.getElementById(id).value;
  const type = (id, v) => {
    const f = doc.getElementById(id);
    f.value = v;
    f.dispatchEvent(new win.Event("input", { bubbles: true }));
  };
  // newConversation() ids come from Date.now(): wait so two clicks never share one.
  const plus = async () => {
    await new Promise((r) => setTimeout(r, 3));
    doc.getElementById("new-conv").click();
  };
  const open = (convId) => {
    const i = win.chat.conversations.findIndex((c) => c.id === convId);
    doc.querySelectorAll("#conv-list .conv-item")[i].click();
  };
  runScript(win, "window.chat = chat;");
  return { win, doc, val, type, plus, open };
}

async function tick() {
  for (let i = 0; i < 6; i++) await new Promise((r) => setTimeout(r, 0));
}

test("a grammar set in chat A is absent in a new chat B and back in A", async () => {
  const { win, doc, val, type, plus, open } = boot();
  await plus();
  const a = win.chat.activeId;
  type("p-grammar", 'root ::= "a" "');
  assert.equal(val("p-grammar"), 'root ::= "a" "');

  await plus();
  const b = win.chat.activeId;
  assert.notEqual(a, b);
  assert.equal(val("p-grammar"), "", "the new chat's panel must not show A's grammar");
  runScript(win, "window.__p = chatParams();");
  assert.equal(win.__p.grammar, null, "the new chat must not send A's grammar");
  assert.equal(doc.querySelector("#p-advanced .adv-fold-count").hidden, true);

  open(a);
  await tick();
  assert.equal(win.chat.activeId, a);
  assert.equal(val("p-grammar"), 'root ::= "a" "', "chat A gets its own grammar back");
  assert.equal(doc.getElementById("p-advanced").open, true,
    "the Advanced fold opens when a restored value sits inside it");
  runScript(win, "window.__p = chatParams();");
  assert.equal(win.__p.grammar, 'root ::= "a" "');

  open(b);
  await tick();
  assert.equal(val("p-grammar"), "", "B is still clean after the round trip");
});

for (const [id, value] of FIELDS) {
  test(`${id} does not leak into a new chat and returns with its own chat`, async () => {
    const { win, val, type, plus, open } = boot();
    await plus();
    const a = win.chat.activeId;
    type(id, value);
    await plus();
    assert.equal(val(id), "", `${id} must be blank in the new chat`);
    open(a);
    await tick();
    assert.equal(val(id), value, `${id} must come back in chat A`);
  });
}

test("two chats keep different values for the same field", async () => {
  const { win, val, type, plus, open } = boot();
  await plus();
  const a = win.chat.activeId;
  type("p-temperature", "0.1");
  await plus();
  const b = win.chat.activeId;
  type("p-temperature", "1.5");
  open(a);
  await tick();
  assert.equal(val("p-temperature"), "0.1");
  open(b);
  await tick();
  assert.equal(val("p-temperature"), "1.5");
});

test("a persona applied in chat A stays with chat A", async () => {
  const { win, doc, val, plus, open } = boot();
  await plus();
  const a = win.chat.activeId;
  runScript(win, `personaCache = [{ name: "pirate", system: "Arr.", params: { temperature: 0.7, top_k: 9 } }];
    const o = document.createElement("option"); o.value = "pirate"; o.textContent = "pirate";
    document.getElementById("p-persona").appendChild(o);
    applyPersona("pirate");`);
  assert.equal(val("p-system"), "Arr.");
  await plus();
  assert.equal(val("p-system"), "");
  assert.equal(val("p-temperature"), "");
  assert.equal(val("p-top-k"), "");
  assert.equal(val("p-persona"), "");
  open(a);
  await tick();
  assert.equal(val("p-system"), "Arr.");
  assert.equal(val("p-temperature"), "0.7");
  assert.equal(val("p-top-k"), "9");
  assert.equal(val("p-persona"), "pirate");
  assert.equal(doc.getElementById("p-advanced").open, true);
});

test("deleting the chat on screen does not hand its values to the next chat", async () => {
  const { win, doc, val, type, plus } = boot();
  await plus();
  await plus();
  const second = win.chat.activeId;
  type("p-grammar", 'root ::= "x"');
  const rows = [...doc.querySelectorAll("#conv-list .conv-item")];
  const idx = win.chat.conversations.findIndex((c) => c.id === second);
  const dels = rows[idx].querySelectorAll("button.del");
  dels[dels.length - 1].click();
  await tick();
  assert.notEqual(win.chat.activeId, second);
  assert.equal(val("p-grammar"), "");
});

test("values typed with no conversation open go to the conversation created next", () => {
  const { win, val, type } = boot();
  runScript(win, "chat.conversations = []; chat.activeId = null; renderChat();");
  type("p-grammar", 'root ::= "y"');
  runScript(win, "newConversation();");
  assert.equal(val("p-grammar"), 'root ::= "y"');
  assert.equal(win.chat.conversations[0].params.grammar, 'root ::= "y"');
});

test("conv.params holds only non-blank per-chat fields and is dropped when all are blank", async () => {
  const { win, type, plus } = boot();
  await plus();
  const conv = () => win.chat.conversations.find((c) => c.id === win.chat.activeId);
  assert.equal(conv().params, undefined);
  type("p-seed", "7");
  assert.deepEqual(JSON.parse(JSON.stringify(conv().params)), { seed: "7" });
  type("p-seed", "");
  assert.equal(conv().params, undefined);
});

test("global fields (web, memory, speak) are not per-chat", async () => {
  const { win, doc, plus, open } = boot();
  await plus();
  const a = win.chat.activeId;
  const web = doc.getElementById("p-web");
  web.checked = true;
  web.dispatchEvent(new win.Event("change", { bubbles: true }));
  await plus();
  assert.equal(web.checked, true, "web access stays as the user set it");
  open(a);
  await tick();
  assert.equal(web.checked, true);
});

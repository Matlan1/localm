// SPDX-License-Identifier: AGPL-3.0-or-later
// Inline event handler attributes in an artifact are rewritten into nonce-stamped
// scripts that attach the same code with addEventListener.
import { test } from "node:test";
import assert from "node:assert/strict";
import vm from "node:vm";
import { JSDOM, VirtualConsole } from "jsdom";
import { loadApp } from "./harness.mjs";

const NONCE = "test-nonce-h4ndl3r";

function srcdoc(code, lang = "html") {
  const { window: win } = loadApp();
  win.__LOCALM_CSP_NONCE__ = NONCE;
  return win.artifactSrcdoc(code, lang);
}

// Runs the srcdoc as the iframe would, with scripts enabled. jsdom enforces no
// CSP, so this checks the bindings work, not that the policy admits them.
const frames = [];
function run(out) {
  const dom = new JSDOM(out, { runScripts: "dangerously" });
  frames.push(dom.window);
  return dom.window;
}
test.after(() => { for (const w of frames) w.close(); });

function parse(out) {
  return new JSDOM(out).window.document;
}

function handlerAttrs(doc) {
  const found = [];
  for (const node of doc.querySelectorAll("*")) {
    for (const a of node.attributes) {
      if (/^on[a-z]+$/i.test(a.name)) found.push(`${node.localName}[${a.name}]`);
    }
  }
  return found;
}

function generatedScripts(doc) {
  return [...doc.querySelectorAll("script")].filter((s) => /addEventListener/.test(s.textContent));
}

test("a static onclick becomes a nonce'd script binding it, with no on* attribute left", () => {
  const out = srcdoc('<button id="b" onclick="this.textContent = \'clicked\'">go</button>');
  const doc = parse(out);
  assert.deepEqual(handlerAttrs(doc), [], out);
  const scripts = generatedScripts(doc);
  assert.equal(scripts.length, 1, out);
  assert.equal(scripts[0].getAttribute("nonce"), NONCE, out);
  assert.match(scripts[0].textContent, /addEventListener\("click"/);
  assert.ok(doc.getElementById("b").hasAttribute("data-localm-on"));

  const win = run(out);
  const b = win.document.getElementById("b");
  b.click();
  assert.equal(b.textContent, "clicked");
});

test("several handlers on one element and handlers on many elements are all bound", () => {
  const out = srcdoc(
    '<input id="i" oninput="this.dataset.i = value" onchange="this.dataset.c = \'1\'">'
    + '<button id="a" onclick="window.hits = (window.hits || 0) + 1">a</button>'
    + '<button id="b" onclick="window.hits = (window.hits || 0) + 10">b</button>');
  const doc = parse(out);
  assert.deepEqual(handlerAttrs(doc), [], out);
  assert.equal(generatedScripts(doc).length, 4, out);
  const ids = [...doc.querySelectorAll("[data-localm-on]")].map((e) => e.getAttribute("data-localm-on"));
  assert.equal(new Set(ids).size, 3, "each element gets its own marker");

  const win = run(out);
  const input = win.document.getElementById("i");
  input.value = "typed";
  input.dispatchEvent(new win.Event("input"));
  input.dispatchEvent(new win.Event("change"));
  assert.equal(input.dataset.i, "typed", "oninput sees the element's own properties by bare name");
  assert.equal(input.dataset.c, "1");
  win.document.getElementById("a").click();
  win.document.getElementById("b").click();
  assert.equal(win.hits, 11);
});

test("an existing data-localm-on value is not reused as a marker", () => {
  const out = srcdoc('<p data-localm-on="0">x</p><button id="b" onclick="this.textContent=\'ok\'">go</button>');
  const win = run(out);
  const b = win.document.getElementById("b");
  assert.notEqual(b.getAttribute("data-localm-on"), "0");
  b.click();
  assert.equal(b.textContent, "ok");
  assert.equal(win.document.querySelector("p").textContent, "x");
});

test("handler text with quotes and </script> stays inside its script", () => {
  const code = `this.textContent = "a</script><img src=x>" + '<!--<script>' + "\\"q\\""`;
  const out = srcdoc(`<button id="b" onclick='${code.replace(/'/g, "&#39;")}'>go</button>`);
  const doc = parse(out);
  assert.deepEqual(handlerAttrs(doc), [], out);
  assert.equal(doc.querySelectorAll("img").length, 0, `the payload became markup: ${out}`);
  const scripts = generatedScripts(doc);
  assert.equal(scripts.length, 1, out);
  assert.doesNotThrow(() => new vm.Script(scripts[0].textContent), "the generated script must parse");

  const win = run(out);
  const b = win.document.getElementById("b");
  b.click();
  assert.equal(b.textContent, 'a</script><img src=x><!--<script>"q"');
});

test("a handler with an unbalanced brace breaks only its own script", () => {
  const out = srcdoc(
    '<button id="bad" onclick="window.ranBad = 1 }">bad</button>'
    + '<button id="good" onclick="this.textContent = \'fine\'">good</button>');
  const doc = parse(out);
  const scripts = generatedScripts(doc);
  assert.equal(scripts.length, 2, out);
  assert.throws(() => new vm.Script(scripts[0].textContent), SyntaxError);
  assert.doesNotThrow(() => new vm.Script(scripts[1].textContent));

  const dom = new JSDOM(out, { runScripts: "dangerously", virtualConsole: new VirtualConsole() });
  frames.push(dom.window);
  const win = dom.window;
  win.document.getElementById("bad").click();
  assert.equal(win.ranBad, undefined, "a broken handler must not run");
  const good = win.document.getElementById("good");
  good.click();
  assert.equal(good.textContent, "fine");
});

test("return false from onsubmit cancels the submission", () => {
  const out = srcdoc(
    '<form id="f" onsubmit="this.dataset.seen = \'1\'; return false"><button>s</button></form>'
    + '<form id="g" onsubmit="this.dataset.seen = \'1\'"><button>s</button></form>');
  const win = run(out);
  const f = win.document.getElementById("f");
  const notCancelled = f.dispatchEvent(new win.Event("submit", { cancelable: true }));
  assert.equal(f.dataset.seen, "1");
  assert.equal(notCancelled, false, "return false must call preventDefault");
  const g = win.document.getElementById("g");
  assert.equal(g.dispatchEvent(new win.Event("submit", { cancelable: true })), true,
    "a handler that does not return false leaves the default alone");
});

test("body onload in a full document is attached to the window", async () => {
  const out = srcdoc(
    "<!doctype html><html lang='en'><head><title>t</title></head>"
    + "<body onload=\"document.getElementById('o').textContent = 'loaded'\"><p id='o'>no</p></body></html>");
  const doc = parse(out);
  assert.deepEqual(handlerAttrs(doc), [], out);
  assert.match(generatedScripts(doc)[0].textContent, /var t = window;/);
  assert.equal(doc.documentElement.getAttribute("lang"), "en");
  assert.equal(doc.title, "t");
  assert.equal(doc.compatMode, "CSS1Compat");

  const win = run(out);
  await new Promise((r) => win.addEventListener("load", r));
  assert.equal(win.document.getElementById("o").textContent, "loaded");
});

test("a full document without a doctype keeps its rendering mode", () => {
  const out = srcdoc("<html><body><button onclick='x()'>b</button></body></html>");
  assert.ok(!/<!doctype/i.test(out), out);
  assert.deepEqual(handlerAttrs(parse(out)), []);
});

test("svg element handlers are bound", () => {
  const out = srcdoc(
    '<svg viewBox="0 0 10 10"><rect id="r" width="10" height="10" '
    + 'onclick="this.setAttribute(\'fill\', \'red\')"/></svg>', "svg");
  const doc = parse(out);
  assert.deepEqual(handlerAttrs(doc), [], out);
  assert.equal(generatedScripts(doc).length, 1, out);
  const win = run(out);
  const r = win.document.getElementById("r");
  r.dispatchEvent(new win.MouseEvent("click", { bubbles: true }));
  assert.equal(r.getAttribute("fill"), "red");
});

test("the CSP meta stays the first element in <head> ahead of every script, in all shapes", () => {
  const shapes = {
    fragment: srcdoc('<script>a=1</script><button onclick="a++">b</button>'),
    "full document": srcdoc(
      "<!doctype html><html><script>a=1</script><head></head>"
      + "<body><button onclick='a++'>b</button></body></html>"),
    svg: srcdoc('<svg><circle r="1" onclick="a++"/></svg>', "svg"),
  };
  for (const [shape, out] of Object.entries(shapes)) {
    const doc = parse(out);
    const first = doc.head.firstElementChild;
    const csp = doc.querySelector('meta[http-equiv="Content-Security-Policy"]');
    assert.ok(first === csp || (first.getAttribute("charset") && first.nextElementSibling === csp),
      `${shape}: the CSP meta is not at the start of <head>: ${out}`);
    assert.match(csp.getAttribute("content"), /default-src 'none'/);
    for (const s of doc.querySelectorAll("script")) {
      assert.ok(csp.compareDocumentPosition(s) & 4, `${shape}: a script precedes the CSP: ${out}`);
      assert.equal(s.getAttribute("nonce"), NONCE, `${shape}: an unstamped script: ${out}`);
    }
    assert.deepEqual(handlerAttrs(doc), [], `${shape}: ${out}`);
  }
});

test("an artifact with no handler attribute is passed through without a re-parse", () => {
  const code = "<!doctype html><HTML><HEAD><title>t</title></HEAD><BODY>x</BODY></HTML>";
  const out = srcdoc(code);
  assert.ok(out.includes(code), out);
});

test("with no shell nonce the handler attributes are left as they are", () => {
  const { window: win } = loadApp();
  win.__LOCALM_CSP_NONCE__ = "";
  const out = win.artifactSrcdoc('<button onclick="go()">b</button>', "html");
  assert.match(out, /<button onclick="go\(\)">b<\/button>/);
});

test("a javascript: URL is left as it is", () => {
  const out = srcdoc('<a id="a" href="javascript:alert(1)">x</a>');
  assert.match(out, /href="javascript:alert\(1\)"/);
});

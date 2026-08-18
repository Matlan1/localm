// SPDX-License-Identifier: AGPL-3.0-or-later
// The LAST sentence of a spoken reply must actually be spoken.
//
// THE DEFECT, measured 2026-08-18 against the running product. tts.js's speak()
// passed a plain STRING to kokoro-js's stream(). That reads as the obvious call
// and is not equivalent to passing a stream: kokoro-js then builds a
// TextSplitterStream itself, pushes the text, and NEVER calls close(). The
// splitter only moves its trailing buffer into the sentence queue from flush(),
// and only close() calls flush(). So:
//
//   "One. Two. Three."  ->  yields "One.", "Two.", then awaits forever
//   "Just one sentence." ->  yields NOTHING AT ALL
//
// A short reply was therefore complete silence, a long one stopped one sentence
// short, and in both cases speak() never returned, so the provider's speaking()
// stayed true for the life of the page. No error was thrown or logged anywhere,
// which is why this survived: it is indistinguishable from a slow model.
//
// WHY THE ASSERTION IS ON THE VENDORED BUNDLE'S OWN BEHAVIOUR, not on a mock.
// The whole defect lives in an undocumented detail of the vendored library, so a
// stand-in for the splitter would encode whatever the author believed and pass
// either way. Reading the real artefact is the only thing that can say the fix is
// still needed and still sufficient after a re-vendor - and it costs no model
// load, so it runs in about a second.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";

const PLUGIN = new URL("../localm/plugins/builtin/tts/", import.meta.url);
const BUNDLE = new URL("static/vendor/kokoro.min.js", PLUGIN);
const THREE = "One. Two. Three.";

/** Drain a TextSplitterStream, giving up after `ms` so a hang is a RESULT rather
 *  than a test-runner timeout with nothing to report. */
async function drain(splitter, ms = 2000) {
  const got = [];
  const done = (async () => {
    for await (const s of splitter) got.push(s);
    return "completed";
  })();
  const outcome = await Promise.race([
    done,
    new Promise((r) => setTimeout(() => r("hung"), ms)),
  ]);
  return { outcome, got };
}

test("the vendored splitter withholds its last sentence until close()", async () => {
  // This is the fires-control, and it runs FIRST on purpose: it proves the
  // library still has the behaviour the fix exists for. If a future re-vendor
  // fixes it upstream, this test fails and tells the next reader that speak()'s
  // explicit close() is now belt-and-braces rather than load-bearing - which is
  // a thing they should learn from a red test, not by guessing.
  const { TextSplitterStream } = await import(BUNDLE);
  const s = new TextSplitterStream();
  s.push(THREE);
  const { outcome, got } = await drain(s);
  assert.equal(outcome, "hung",
    "the vendored TextSplitterStream now terminates without close(); "
    + "re-read speak()'s comment, the workaround may no longer be needed");
  assert.deepEqual(got, ["One.", "Two."],
    "expected the trailing sentence to be withheld, got " + JSON.stringify(got));
});

test("closing the splitter yields every sentence and terminates", async () => {
  const { TextSplitterStream } = await import(BUNDLE);
  const s = new TextSplitterStream();
  s.push(THREE);
  s.close();
  const { outcome, got } = await drain(s);
  assert.equal(outcome, "completed", "the closed splitter still did not terminate");
  assert.deepEqual(got, ["One.", "Two.", "Three."]);
});

test("a single-sentence reply is silent unless the splitter is closed", async () => {
  // The user-visible worst case, asserted on its own because it is the one that
  // produces NO audio whatsoever rather than merely a truncated reply.
  const { TextSplitterStream } = await import(BUNDLE);
  const open = new TextSplitterStream();
  open.push("Just one sentence.");
  const a = await drain(open);
  assert.deepEqual(a.got, [], "expected total silence from the unclosed splitter");

  const closed = new TextSplitterStream();
  closed.push("Just one sentence.");
  closed.close();
  const b = await drain(closed);
  assert.equal(b.outcome, "completed");
  assert.deepEqual(b.got, ["Just one sentence."]);
});

test("speak() feeds stream() a splitter it closed, never a bare string", () => {
  // The behavioural tests above prove what the LIBRARY does. This one pins what
  // OUR code does with it, because the two can drift apart without either of the
  // others failing: reverting speak() to k.stream(text, ...) leaves every
  // assertion above green while shipping the silence back to users.
  const src = fs.readFileSync(new URL("static/tts.js", PLUGIN)).toString("utf8");
  const code = src.split("\n").filter((l) => {
    const s = l.trim();
    return s && !s.startsWith("//") && !s.startsWith("*") && !s.startsWith("/*");
  }).join("\n");

  assert.ok(/new Splitter\(\)/.test(code) && /\.close\(\)/.test(code),
    "speak() no longer builds and closes its own TextSplitterStream, so the "
    + "last sentence of every reply will be dropped again");
  assert.ok(!/k\.stream\(\s*text\s*,/.test(code),
    "speak() passes the raw string to stream() again; kokoro-js will build a "
    + "splitter it never closes and the final sentence will never be spoken");
  assert.ok(/Splitter = mod\.TextSplitterStream/.test(code),
    "load() no longer captures TextSplitterStream from the vendored bundle");
});

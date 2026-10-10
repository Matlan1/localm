// SPDX-License-Identifier: AGPL-3.0-or-later
// Audio clips attached in the chat composer: picked or dropped as a file, kept
// as their own attachment kind, sent as OpenAI input_audio parts, shown in the
// transcript, and recovered from when the server refuses them.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp, runScript } from "./harness.mjs";

const WAV_BYTES = new Uint8Array([
  0x52, 0x49, 0x46, 0x46, 0x24, 0x00, 0x00, 0x00, 0x57, 0x41, 0x56, 0x45, 0x66, 0x6d, 0x74, 0x20,
]);
const WAV_B64 = Buffer.from(WAV_BYTES).toString("base64");

function setup({ rejectWith = null } = {}) {
  const fetchCalls = [];
  const ctl = { rejectWith };
  const impl = async (url, opts = {}) => {
    const u = String(url);
    let body;
    try { body = opts.body ? JSON.parse(opts.body) : null; } catch { body = null; }
    fetchCalls.push({ url: u, opts, body });
    if (u === "/v1/chat/completions" && ctl.rejectWith) {
      return { ok: false, status: ctl.rejectStatus || 400, headers: { get: () => null },
        json: async () => ({ detail: ctl.rejectWith }), text: async () => "" };
    }
    if (u === "/v1/chat/completions") {
      return { ok: true, status: 200, body: null, headers: { get: () => null },
        json: async () => ({}) };
    }
    if (u === "/api/rag/extract") {
      return { ok: true, status: 200, headers: { get: () => null },
        json: async () => ({ filename: "notes.txt", text: "hello", chars: 5, truncated: false }) };
    }
    return { ok: true, status: 200, headers: { get: () => null },
      json: async () => ({}), text: async () => "" };
  };
  const { window } = loadApp({ fetchImpl: impl });
  window.maybeCompactConversation = async () => {};
  const realWavSeconds = window.wavSeconds;
  window.wavSeconds = async (b) => (b.size === WAV_BYTES.length ? 4 : realWavSeconds(b));
  window.__blobs = [];
  window.URL.createObjectURL = (blob) => { window.__blobs.push(blob); return `blob:test/${window.__blobs.length}`; };
  window.__revoked = [];
  window.URL.revokeObjectURL = (url) => { window.__revoked.push(url); };
  window.readSSE = async (_r, onData) => {
    onData(JSON.stringify({ choices: [{ delta: { content: "a transcript" } }] }));
    onData(JSON.stringify({ choices: [{ delta: {}, finish_reason: "stop" }] }));
  };
  const doc = window.document;
  for (const id of ["p-speak", "p-memory", "p-web"]) doc.getElementById(id).checked = false;
  runScript(window, "modelCache.active = 'test-model';");
  return { window, fetchCalls, ctl };
}

const evalIn = (window, expr) => {
  window.__err = null;
  runScript(window,
    `try { window.__out = (${expr}); } catch (e) { window.__out = undefined; window.__err = String(e); }`);
  if (window.__err) throw new Error(`${expr}: ${window.__err}`);
  return window.__out;
};

function activate(window, conv) {
  window.__testConv = conv;
  runScript(window, "chat.conversations = [window.__testConv]; chat.activeId = window.__testConv.id;");
}

const wavFile = (window, name = "hello.wav", type = "audio/wav") =>
  new window.File([WAV_BYTES], name, { type });

async function waitFor(cond, what, ms = 3000) {
  const end = Date.now() + ms;
  while (Date.now() < end) {
    if (cond()) return;
    await new Promise((r) => setTimeout(r, 5));
  }
  assert.fail("timed out waiting for " + what);
}

const clipCount = (window) => evalIn(window, "chat.clips.length");
const toastText = (window) => window.document.getElementById("toast").textContent;
const streamed = (fetchCalls) => fetchCalls.filter(
  (c) => c.url === "/v1/chat/completions" && c.body && c.body.stream === true);

async function attachWav(window, file = wavFile(window)) {
  window.addAttachedFiles([file]);
  await waitFor(() => clipCount(window) === 1, "the clip to attach");
  await new Promise((r) => setTimeout(r, 0));
}

async function sendText(window, text) {
  runScript(window, "modelCache.active = 'test-model';");
  window.document.getElementById("chat-input").value = text;
  await window.sendChat();
  await waitFor(() => !evalIn(window, "chatBusy()"), "the chat to go idle");
}

test("format helpers: extension wins, MIME is the fallback, documents are not audio", () => {
  const { window } = setup();
  const f = (name, type) => ({ name, type });
  assert.equal(window.isAudioFile(f("a.WAV", "")), true);
  assert.equal(window.isAudioFile(f("voice", "audio/ogg")), true);
  assert.equal(window.isAudioFile(f("notes.txt", "text/plain")), false);
  assert.equal(window.isAudioFile(f("pic.png", "image/png")), false);
  assert.equal(window.audioFormat(f("a.MP3", "audio/mpeg")), "mp3");
  assert.equal(window.audioFormat(f("take", "audio/x-wav")), "wav");
  assert.equal(window.audioFormat(f("take", "audio/mpeg")), "mp3");
  assert.equal(window.audioFormat(f("take", "audio/webm")), "webm");
  assert.equal(window.audioMime("mp3"), "audio/mpeg");
  assert.equal(window.audioMime("wav"), "audio/wav");
  assert.equal(window.formatClipTime(75.4), "1:15");
});

test("attaching a WAV keeps it as an audio clip with a chip, not as a document or image", async () => {
  const { window, fetchCalls } = setup();
  await attachWav(window);

  assert.equal(evalIn(window, "chat.attachments.length"), 0, "not an image");
  assert.equal(evalIn(window, "chat.docs.length"), 0, "not a document");
  const clip = evalIn(window, "({...chat.clips[0]})");
  assert.equal(clip.name, "hello.wav");
  assert.equal(clip.format, "wav");
  assert.equal(clip.data, WAV_B64);
  assert.equal(clip.seconds, 4);
  assert.equal(fetchCalls.filter((c) => c.url === "/api/rag/extract").length, 0,
    "audio never reaches document extraction");

  const chip = window.document.querySelector("#attach-chips .chip");
  assert.ok(chip, "a chip is shown");
  assert.match(chip.textContent, /hello\.wav \(0:04\)/);
  assert.ok(chip.querySelector("svg"), "the chip carries an icon");

  chip.querySelector("button").click();
  assert.equal(clipCount(window), 0, "the chip's remove button drops the clip");
  assert.equal(window.document.querySelectorAll("#attach-chips .chip").length, 0);
});

test("a text file still goes through document extraction", async () => {
  const { window, fetchCalls } = setup();
  window.addAttachedFiles([new window.File(["hello"], "notes.txt", { type: "text/plain" })]);
  await waitFor(() => fetchCalls.some((c) => c.url === "/api/rag/extract"), "extraction");
  assert.equal(clipCount(window), 0);
});

test("a file over the server's size limit is refused before it is read", async () => {
  const { window } = setup();
  const big = wavFile(window, "huge.wav");
  Object.defineProperty(big, "size", { value: 50_000_001 });
  let reads = 0;
  const RealReader = window.FileReader;
  window.FileReader = class extends RealReader {
    readAsDataURL(f) { reads++; return super.readAsDataURL(f); }
  };
  window.addAttachedFiles([big]);
  await waitFor(() => /50 MB/.test(toastText(window)), "the refusal toast");
  assert.match(toastText(window), /huge\.wav is 50\.0 MB; an audio clip can be at most 50 MB/);
  assert.equal(window.document.getElementById("toast").className.includes("error"), true);
  assert.equal(clipCount(window), 0);
  assert.equal(reads, 0, "the file was never read");
});

/** A RIFF/WAVE byte array: *seconds* of 8 kHz mono 8-bit audio, an optional
 *  LIST chunk before the data, and the data size field overridden by *sizeField*. */
function wavBytes({ seconds, list = false, sizeField = null }) {
  const data = 8000 * seconds;
  const listChunk = list ? [0x4c, 0x49, 0x53, 0x54, 3, 0, 0, 0, 1, 2, 3, 0] : [];
  const out = new Uint8Array(12 + 24 + listChunk.length + 8 + data);
  const v = new DataView(out.buffer);
  out.set([0x52, 0x49, 0x46, 0x46], 0);
  out.set([0x57, 0x41, 0x56, 0x45], 8);
  out.set([0x66, 0x6d, 0x74, 0x20], 12);
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, 8000, true); v.setUint32(28, 8000, true);
  v.setUint16(32, 1, true); v.setUint16(34, 8, true);
  out.set(listChunk, 36);
  const at = 36 + listChunk.length;
  out.set([0x64, 0x61, 0x74, 0x61], at);
  v.setUint32(at + 4, sizeField ?? data, true);
  return out;
}

test("wavSeconds reads the length from the WAV header, with extra chunks and streamed sizes", async () => {
  const { window } = setup();
  const secs = (bytes) => window.wavSeconds(new window.Blob([bytes]));
  assert.equal(await secs(wavBytes({ seconds: 3 })), 3);
  assert.equal(await secs(wavBytes({ seconds: 3, list: true })), 3);
  assert.equal(await secs(wavBytes({ seconds: 3, sizeField: 0xFFFFFFFF })), 3, "an unknown data size uses the bytes present");
  assert.equal(await secs(WAV_BYTES.slice(0, 15)), 0, "a truncated header yields 0, not a guess");
  assert.equal(await secs(new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])), 0, "not a WAV");
});

test("an 11 minute WAV is refused from its header and its chip is removed", async () => {
  const { window } = setup();
  const file = new window.File([wavBytes({ seconds: 601 })], "long.wav", { type: "audio/wav" });
  window.addAttachedFiles([file]);
  await waitFor(() => /min/.test(toastText(window)), "the refusal toast");
  assert.match(toastText(window), /long\.wav is about 11 min long; an audio clip can be at most 10 min/);
  assert.equal(clipCount(window), 0);
  assert.equal(window.document.querySelectorAll("#attach-chips .chip").length, 0);
});

test("a short WAV shows its length from the header", async () => {
  const { window } = setup();
  window.addAttachedFiles([new window.File([wavBytes({ seconds: 75 })], "mid.wav", { type: "audio/wav" })]);
  await waitFor(() => clipCount(window) === 1 && evalIn(window, "chat.clips[0].seconds") === 75, "the length");
  assert.match(window.document.querySelector("#attach-chips .chip").textContent, /mid\.wav \(1:15\)/);
});

test("a file named .wav that is not a WAV file is refused before it is read in full", async () => {
  const { window } = setup();
  window.addAttachedFiles([new window.File(["this is not audio at all"], "bad.wav", { type: "audio/wav" })]);
  await waitFor(() => /not a valid WAV/.test(toastText(window)), "the refusal toast");
  assert.match(toastText(window), /bad\.wav is not a valid WAV file/);
  assert.equal(clipCount(window), 0);
});

test("an MP3 is attached without a WAV header check", async () => {
  const { window } = setup();
  window.addAttachedFiles([new window.File(["ID3 not checked"], "song.mp3", { type: "audio/mpeg" })]);
  await waitFor(() => clipCount(window) === 1, "the clip");
  assert.equal(evalIn(window, "chat.clips[0].format"), "mp3");
});

test("a clip whose length cannot be read stays attached for the server to judge", async () => {
  const { window } = setup();
  window.wavSeconds = async () => 0;
  await attachWav(window);
  assert.equal(evalIn(window, "chat.clips[0].seconds"), 0);
  assert.doesNotMatch(window.document.querySelector("#attach-chips .chip").textContent, /\(\d+:\d\d\)/);
});

test("sending sends exactly an input_audio part, shows a player, and clears the composer", async () => {
  const { window, fetchCalls } = setup();
  const conv = { id: "c1", title: "t", messages: [] };
  activate(window, conv);
  await attachWav(window);
  await sendText(window, "Transcribe this.");

  const calls = streamed(fetchCalls);
  assert.equal(calls.length, 1);
  const user = calls[0].body.messages.filter((m) => m.role === "user").pop();
  assert.deepEqual(user.content, [
    { type: "text", text: "Transcribe this." },
    { type: "input_audio", input_audio: { data: WAV_B64, format: "wav" } },
  ], "only the wire fields leave the browser");

  assert.equal(clipCount(window), 0, "the composer is cleared");
  assert.equal(window.document.querySelectorAll("#attach-chips .chip").length, 0);

  const player = window.document.querySelector("#chat-messages .msg-row.user audio");
  assert.ok(player, "the sent clip is playable in the transcript");
  assert.match(player.getAttribute("src"), /^blob:/, "the page CSP refuses data: media");
  assert.equal(window.__blobs.length, 1);
  assert.equal(window.__blobs[0].type, "audio/wav");
  assert.equal(window.__blobs[0].size, WAV_BYTES.length, "the blob holds the decoded bytes");
  assert.match(window.document.querySelector("#chat-messages .msg-clip-name").textContent, /hello\.wav/);
  assert.ok(conv.messages.some((m) => m.role === "assistant" && m.content === "a transcript"));
});

test("a clip with no typed text still sends, as the audio part alone", async () => {
  const { window, fetchCalls } = setup();
  const conv = { id: "c1", title: "t", messages: [] };
  activate(window, conv);
  await attachWav(window, wavFile(window, "memo.mp3", "audio/mpeg"));
  await sendText(window, "");

  const user = streamed(fetchCalls)[0].body.messages.filter((m) => m.role === "user").pop();
  assert.deepEqual(user.content, [
    { type: "input_audio", input_audio: { data: WAV_B64, format: "mp3" } },
  ]);
  assert.equal(conv.title, "memo.mp3", "the chat is titled after the clip");
});

test("a clip sent while the chat is busy is queued with its message", async () => {
  const { window } = setup();
  activate(window, { id: "c1", title: "t", messages: [] });
  await attachWav(window);
  runScript(window, "chat.busy = true; modelCache.active = 'test-model';");
  window.document.getElementById("chat-input").value = "later";
  await window.sendChat();

  assert.equal(evalIn(window, "chat.queue.length"), 1);
  assert.equal(evalIn(window, "chat.queue[0].clips.length"), 1);
  assert.equal(clipCount(window), 0);
});

test("a server refusal is shown as the server wrote it and the clip is not sent again", async () => {
  const detail = "The loaded model cannot hear audio. Load a model with an audio projector.";
  const { window, fetchCalls, ctl } = setup({ rejectWith: detail });
  const conv = { id: "c1", title: "t", messages: [] };
  activate(window, conv);
  await attachWav(window);
  await sendText(window, "Transcribe this.");

  assert.ok(toastText(window).includes(detail), "the server's own message reaches the user");
  assert.equal(window.document.getElementById("toast").className.includes("error"), true);
  assert.equal(conv.messages.length, 1, "no blank assistant turn is saved");
  assert.equal(typeof conv.messages[0].content, "string");
  assert.match(conv.messages[0].content, /Transcribe this\./);
  assert.match(conv.messages[0].content, /\[Audio removed/);
  assert.equal(window.document.querySelectorAll("#chat-messages audio").length, 0);

  ctl.rejectWith = null;
  await sendText(window, "never mind, just chat");
  const last = streamed(fetchCalls).pop();
  const resent = last.body.messages.some((m) => Array.isArray(m.content) &&
    m.content.some((p) => p.type === "input_audio"));
  assert.equal(resent, false, "the refused clip is gone from the next request");
  assert.ok(conv.messages.some((m) => m.role === "assistant" && m.content === "a transcript"),
    "the chat answers again");
});

const audioMsg = (text, extra, tag) => ({ role: "user", content: [
  { type: "text", text },
  ...extra,
  { type: "input_audio", input_audio: { data: tag + WAV_B64, format: "wav" }, name: `${tag}.wav`, seconds: 1 },
] });

test("a refused message with an image and a clip keeps the image and the text", async () => {
  const detail = "The loaded model cannot hear audio.";
  const { window } = setup({ rejectWith: detail });
  const conv = { id: "c1", title: "t", messages: [
    audioMsg("what is this?", [{ type: "image_url", image_url: { url: "data:image/png;base64,AAAA" } }], "AAAA"),
  ] };
  activate(window, conv);
  await window.runCompletion(conv);

  const parts = conv.messages[0].content;
  assert.ok(Array.isArray(parts), "the message keeps its parts");
  assert.equal(parts.some((p) => p.type === "input_audio"), false);
  assert.equal(parts.filter((p) => p.type === "image_url").length, 1, "the image survives");
  const text = parts.filter((p) => p.type === "text").map((p) => p.text).join("");
  assert.match(text, /what is this\?/);
  assert.match(text, /\[Audio removed/);
  assert.ok(toastText(window).includes(detail));
});

test("a 413 from a too-large request drops the clips like a 400", async () => {
  const { window, ctl } = setup({ rejectWith: "Request body too large" });
  ctl.rejectStatus = 413;
  const conv = { id: "c1", title: "t", messages: [audioMsg("hi", [], "BBBB")] };
  activate(window, conv);
  await window.runCompletion(conv);
  assert.equal(typeof conv.messages[0].content, "string");
  assert.match(conv.messages[0].content, /\[Audio removed/);
  assert.ok(toastText(window).includes("Request body too large"));
});

test("ten clips in one transcript each keep a live player and are not rebuilt on re-render", async () => {
  const { window } = setup();
  const conv = { id: "c1", title: "t", messages: [] };
  for (let i = 0; i < 10; i++) {
    conv.messages.push(audioMsg(`clip ${i}`, [], `T${i}`.padEnd(4, "A")));
  }
  activate(window, conv);
  window.renderChat();
  const srcs = [...window.document.querySelectorAll("#chat-messages audio")].map((a) => a.getAttribute("src"));
  assert.equal(srcs.length, 10);
  assert.equal(new Set(srcs).size, 10, "one URL per clip");
  assert.deepEqual(window.__revoked, [], "no URL is revoked while its player is on screen");
  assert.equal(window.__blobs.length, 10);

  window.renderChat();
  assert.equal(window.__blobs.length, 10, "a re-render reuses the blobs");
  assert.deepEqual(window.__revoked, []);

  const other = { id: "c2", title: "o", messages: [] };
  window.__testConv = other;
  runScript(window, "chat.conversations.push(window.__testConv); chat.activeId = 'c2';");
  window.renderChat();
  assert.equal(window.__revoked.length, 10, "switching away revokes every clip URL of the old conversation");
});

test("a stored clip that cannot be decoded does not break the transcript", async () => {
  const { window } = setup();
  const conv = { id: "c1", title: "t", messages: [
    { role: "user", content: [
      { type: "text", text: "broken" },
      { type: "input_audio", input_audio: { data: "!!!not base64!!!", format: "wav" }, name: "x.wav" },
    ] },
    { role: "user", content: "after" },
  ] };
  activate(window, conv);
  const errors = [];
  const orig = window.console.error;
  window.console.error = (...a) => errors.push(a.join(" "));
  try { window.renderChat(); } finally { window.console.error = orig; }
  const rows = [...window.document.querySelectorAll("#chat-messages .msg-row")];
  assert.equal(rows.length, 2, "the later message still renders");
  assert.match(rows[0].textContent, /cannot be played/);
  assert.equal(errors.length, 1);
});

test("wireParts drops an audio part that has no data instead of throwing", () => {
  const { window } = setup();
  const out = window.wireParts([
    { type: "text", text: "a" }, { type: "input_audio" }, { type: "input_audio", input_audio: {} },
    { type: "input_audio", input_audio: { data: "QQ==", format: "wav" }, name: "n", seconds: 1 },
  ]);
  assert.deepEqual(JSON.parse(JSON.stringify(out)), [
    { type: "text", text: "a" },
    { type: "input_audio", input_audio: { data: "QQ==", format: "wav" } },
  ]);
});

test("a length read that finishes after the clip was sent shows no refusal", async () => {
  const { window } = setup();
  let finish;
  window.wavSeconds = () => new Promise((resolve) => { finish = resolve; });
  window.addAttachedFiles([wavFile(window)]);
  await waitFor(() => clipCount(window) === 1, "the clip");
  runScript(window, "chat.clips = [];");
  window.document.getElementById("toast").textContent = "";
  finish(9999);
  await new Promise((r) => setTimeout(r, 20));
  assert.equal(toastText(window), "", "no toast for a clip that is already gone");
});

test("compaction archive counts audio clips as attachments not archived", async () => {
  const { window } = setup();
  const msg = { role: "user", content: [
    { type: "text", text: "what is said?" },
    { type: "input_audio", input_audio: { data: WAV_B64, format: "wav" }, name: "a.wav", seconds: 2 },
  ] };
  const copy = window.archiveCopy(msg);
  assert.equal(typeof copy.content, "string");
  assert.match(copy.content, /what is said\?/);
  assert.match(copy.content, /1 attachment\(s\) not archived/);
  assert.doesNotMatch(copy.content, new RegExp(WAV_B64));
});

test("the catalogs carry every audio string in both languages with the same placeholders", async () => {
  const { readFileSync } = await import("node:fs");
  const dir = new URL("../localm/plugins/gui/static/", import.meta.url);
  const de = JSON.parse(readFileSync(new URL("i18n/de.json", dir), "utf-8"));
  const enSrc = readFileSync(new URL("app/i18n-en.js", dir), "utf-8");
  const ph = (s) => (s.match(/\{\w+\}/g) || []).sort().join();
  for (const key of ["chat.audio.readError", "chat.audio.rejected", "chat.audio.tooLarge",
                     "chat.audio.tooLong", "chat.audio.notWav", "chat.status.encodingAudio"]) {
    const m = enSrc.match(new RegExp(`"${key.replaceAll(".", "\\.")}":\\s*"([^"]*)"`));
    assert.ok(m, `${key} is in the English catalog`);
    assert.ok(de[key], `${key} is in the German catalog`);
    assert.equal(ph(de[key]), ph(m[1]), `${key} keeps its placeholders in German`);
  }
});

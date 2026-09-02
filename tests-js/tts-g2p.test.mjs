// SPDX-License-Identifier: AGPL-3.0-or-later
// Non-English grapheme-to-phoneme for the Kokoro voices
// (localm/plugins/builtin/tts/static/g2p.js), driven against the REAL eSpeak-NG
// engine inside the vendored bundle and the REAL vendored dictionaries.
//
// The phonemes are checked against the REAL Kokoro tokenizer alphabet: its
// normaliser has no unknown-token and deletes anything outside a fixed
// whitelist, so a symbol this engine emits that Kokoro does not know would
// vanish silently rather than fail.

import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath, pathToFileURL } from "node:url";
import { dirname, join } from "node:path";

const HERE = dirname(fileURLToPath(import.meta.url));
const STATIC = join(HERE, "..", "localm", "plugins", "builtin", "tts", "static");
const VENDOR = join(STATIC, "vendor");
const BASE = pathToFileURL(join(STATIC, "tts.js")).href;

// Kokoro's phoneme alphabet, taken from the tokenizer that ships with the model
// repo named in tts.example.json. Kept as a literal so the test needs no
// network; the same 115 symbols are the vocab keys and the normaliser whitelist.
const KOKORO_ALPHABET = new Set([...(
  "$;:,.!?\u2014\u2026\"()\u201c\u201d \u0303\u02a3\u02a5\u02a6\u02a8\u1d5d\uab67AIOQSTWY\u1d4aabcdefhijklmnopqrstuvwxyz\u0251\u0250\u0252\u00e6\u03b2\u0254\u0255\u00e7\u0256\u00f0\u02a4\u0259\u025a\u025b\u025c\u025f\u0261\u0265\u0268\u026a\u029d\u026f\u0270\u014b\u0273\u0272\u0274\u00f8\u0278\u03b8\u0153\u0279\u027e\u027b\u0281\u027d\u0282\u0283\u0288\u02a7\u028a\u028b\u028c\u0263\u0264\u03c7\u028e\u0292\u0294\u02c8\u02cc\u02d0\u02b0\u02b2\u2193\u2192\u2197\u2198\u1d7b"
)]);

// Serve the vendored dictionaries from disk in place of the network.
globalThis.fetch = async (url) => {
  const path = fileURLToPath(url);
  const bytes = readFileSync(path);
  return {
    ok: true,
    status: 200,
    arrayBuffer: async () =>
      bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
  };
};

const kokoro = await import(pathToFileURL(join(VENDOR, "kokoro.min.js")).href);
const g2p = await import(pathToFileURL(join(STATIC, "g2p.js")).href);

// ---- the vendored bundle still exposes what g2p.js needs ---------------- //

test("the vendored bundle exports the eSpeak-NG worker and filesystem", async () => {
  assert.ok(kokoro.espeakWorker, "kokoro.min.js must export espeakWorker");
  assert.ok(kokoro.espeakFS, "kokoro.min.js must export espeakFS");
  const worker = await kokoro.espeakWorker;
  assert.equal(typeof worker.set_voice, "function");
  assert.equal(typeof worker.synthesize_ipa, "function");
  assert.equal(typeof kokoro.espeakFS.writeFile, "function");
});

// ---- the locale table -------------------------------------------------- //

test("every non-English Kokoro voice prefix maps to its language", () => {
  assert.equal(g2p.localeForVoice("ef_dora"), "es");
  assert.equal(g2p.localeForVoice("em_santa"), "es");
  assert.equal(g2p.localeForVoice("ff_siwis"), "fr");
  assert.equal(g2p.localeForVoice("hf_alpha"), "hi");
  assert.equal(g2p.localeForVoice("if_sara"), "it");
  assert.equal(g2p.localeForVoice("pf_dora"), "pt-br");
});

test("English voices are left to the bundle's own phonemizer", () => {
  for (const id of ["af_heart", "am_adam", "bf_emma", "bm_george"]) {
    assert.equal(g2p.localeForVoice(id), null, id);
    assert.equal(g2p.isNonEnglishVoice(id), false, id);
  }
});

test("a missing or malformed voice id is not treated as non-English", () => {
  for (const id of [undefined, null, "", 42, {}]) {
    assert.equal(g2p.localeForVoice(id), null);
  }
});

// Kokoro's own voice list documents pf_dora/pm_alex/pm_santa as BRAZILIAN
// Portuguese, and eSpeak-NG's "pt" is European: it reads "Rio" as ʁˈiʊ where
// "pt-br" reads it xˈiʊ. Picking "pt" here would be a different accent.
test("Portuguese uses the Brazilian locale, not European Portuguese", async () => {
  const worker = await kokoro.espeakWorker;
  await g2p.phonemize(kokoro, "Rio", "pt-br", BASE);
  const br = await g2p.phonemize(kokoro, "Rio Roberto", "pt-br", BASE);
  assert.equal(worker.set_voice("pt"), 0);
  const eu = g2p.normalizePhonemes(worker.synthesize_ipa("Rio Roberto").ipa);
  assert.notEqual(br, eu, "pt-br must not phonemize the same as pt");
  assert.match(br, /x/, "Brazilian Portuguese realises this r as x");
});

// ---- normalisation ----------------------------------------------------- //

test("normalizePhonemes maps the symbols Kokoro's alphabet lacks", () => {
  assert.equal(g2p.normalizePhonemes("aʲb"), "ajb");
  assert.equal(g2p.normalizePhonemes("aɬb"), "alb");
});

// eSpeak-NG marks a stretch it read in another language, e.g. a loanword in
// French text comes back as "(en)wˈɪski(fr)". Every character of that marker is
// in Kokoro's alphabet, so an unstripped marker is spoken rather than dropped.
test("normalizePhonemes strips eSpeak-NG language-switch markers", () => {
  assert.equal(g2p.normalizePhonemes("(en)wˈɪski(fr)"), "wˈɪski");
  assert.equal(g2p.normalizePhonemes("a(hi)b"), "ab");
  assert.equal(g2p.normalizePhonemes("(cmn-latn-pinyin)a"), "a");
});

test("normalizePhonemes keeps ordinary parentheses", () => {
  assert.equal(g2p.normalizePhonemes("(ˈola)"), "(ˈola)");
});

// ---- the phonemes Kokoro will actually receive ------------------------- //

const SENTENCES = {
  es: "Jorge y Rosa trabajan en Guadalajara. El perro corre rapido, hijo.",
  fr: "Bonjour, comment allez-vous? Portez ce vieux whisky au juge blond.",
  hi: "नमस्ते, आप कैसे हैं? राम और राधा।",
  it: "Ciao! Roma, Roberto, arrivederci. La rana rossa corre.",
  "pt-br": "Ola! Rio, carro, rapido. O rato roeu a roupa do rei.",
};

for (const [locale, text] of Object.entries(SENTENCES)) {
  test(`${locale}: every phoneme is in Kokoro's alphabet`, async () => {
    const phonemes = await g2p.phonemize(kokoro, text, locale, BASE);
    assert.ok(phonemes.length > 0);
    const unsupported = [...new Set(phonemes)]
      .filter((c) => c !== " " && !KOKORO_ALPHABET.has(c));
    assert.deepEqual(unsupported, [],
      `${locale} emitted symbols Kokoro would silently delete: ` +
      unsupported.map((c) => "U+" + c.codePointAt(0).toString(16)).join(" ") +
      ` in ${JSON.stringify(phonemes)}`);
  });
}

test("no language-switch marker survives into real phonemized output", async () => {
  for (const [locale, text] of Object.entries(SENTENCES)) {
    const phonemes = await g2p.phonemize(kokoro, text, locale, BASE);
    assert.doesNotMatch(phonemes, /\([a-z]{2,3}(?:-[a-z]{2,8})*\)/,
      `${locale}: ${phonemes}`);
  }
});

// The bundle's own phonemize path rewrites r->ɹ and x->k because that is right
// for English. Applying it here would say "Kio" for "Rio" and "Corque" for
// "Jorge": r is the Italian and Spanish trill, x is the Spanish jota and the
// Brazilian Portuguese r, and both are in Kokoro's alphabet.
test("test_preserves_r_and_x: the English-only fixups are not applied", async () => {
  const es = await g2p.phonemize(kokoro, "Jorge, Mexico, perro", "es", BASE);
  assert.match(es, /x/, `Spanish jota must survive as x: ${es}`);
  assert.match(es, /r/, `Spanish trill must survive as r: ${es}`);
  assert.doesNotMatch(es, /kˈokɹe/, "Jorge must not become kˈoɹke");

  const it = await g2p.phonemize(kokoro, "Roma, Roberto", "it", BASE);
  assert.match(it, /r/, `Italian trill must survive as r: ${it}`);

  const pt = await g2p.phonemize(kokoro, "Rio, carro", "pt-br", BASE);
  assert.match(pt, /x/, `Brazilian r must survive as x: ${pt}`);
  assert.doesNotMatch(pt, /kˈiʊ/, "Rio must not become kˈiʊ");
});

// ---- the two ways eSpeak-NG fails without saying so -------------------- //

// A rejected set_voice leaves the PREVIOUS language selected and reports
// nothing, so an unusable locale reads as fluent output in the wrong language.
// "fr-fr" is listed in the engine's own voice metadata and is still rejected.
// The engine is driven through a stub here on purpose: every locale g2p.js
// actually ships is settable, so nothing real can reach this guard. Without the
// stub the call is refused one step earlier, by the dictionary table, and the
// guard itself is never executed.
test("test_rejects_unsettable_locale: a non-zero set_voice status throws", async () => {
  const stub = {
    espeakFS: { analyzePath: () => ({ exists: true }), writeFile: () => {} },
    espeakWorker: Promise.resolve({
      set_voice: () => 2,
      synthesize_ipa: () => ({ ipa: "output in the previously selected language" }),
    }),
  };
  g2p._resetDictionaryCache();
  try {
    await assert.rejects(
      () => g2p.phonemize(stub, "Bonjour", "fr", BASE),
      /rejected locale "fr" \(status 2\)/);
  } finally {
    g2p._resetDictionaryCache();
  }
});

// Why the table maps prefix "f" to "fr" and not "fr-fr": the engine lists
// "fr-fr" among its voices' languages and still refuses to select it.
test("the locale table avoids the identifiers the engine refuses", async () => {
  const worker = await kokoro.espeakWorker;
  assert.notEqual(worker.set_voice("fr-fr"), 0, "the engine must reject fr-fr");
  for (const id of ["es", "fr", "hi", "it", "pt-br"]) {
    assert.equal(worker.set_voice(id), 0, `the engine must accept ${id}`);
  }
});

test("a locale with no vendored dictionary throws", async () => {
  await assert.rejects(
    () => g2p.phonemize(kokoro, "hello", "en-gb", BASE),
    /No dictionary is vendored/);
});

test("a locale whose dictionary cannot be fetched throws", async () => {
  const realFetch = globalThis.fetch;
  globalThis.fetch = async () => ({ ok: false, status: 404 });
  g2p._resetDictionaryCache();
  try {
    // "it" is only fetched once per page; remove it so the fetch is retried.
    const target = "/usr/share/espeak-ng-data/it_dict";
    if (kokoro.espeakFS.analyzePath(target).exists) kokoro.espeakFS.unlink(target);
    await assert.rejects(
      () => g2p.phonemize(kokoro, "Ciao", "it", BASE), /it_dict: HTTP 404/);
  } finally {
    globalThis.fetch = realFetch;
    g2p._resetDictionaryCache();
  }
});

// ---- the English path is untouched ------------------------------------- //

test("the bundle's own English phonemizer still works after g2p.js runs", async () => {
  const worker = await kokoro.espeakWorker;
  for (const locale of Object.keys(SENTENCES)) {
    await g2p.phonemize(kokoro, "test", locale, BASE).catch(() => {});
  }
  assert.equal(worker.set_voice("en-us"), 0);
  const en = worker.synthesize_ipa("The quick brown fox jumps over the lazy dog.")
    .ipa.trim().replace(/\n/g, " ");
  assert.equal(en,
    "ðə kwˈɪk bɹˈaʊn fˈɑːks " +
    "dʒˈʌmps ˌoʊvɚ ðə lˈeɪzi " +
    "dˈɑːɡ");
});

// ---- the picker order must not decide the default voice ---------------- //

// Ordering voices.json for the picker moved af_heart out of first position, and
// the default used to be voiceList[0]. Selecting it by name keeps the shipped
// default independent of however the list is later sorted.
test("test_falls_back_to_af_heart: an unknown configured voice falls back by name", async () => {
  const { JSDOM } = await import("jsdom");
  const TTS_JS = join(STATIC, "tts.js");
  const dom = new JSDOM(
    `<!DOCTYPE html><html><body>
       <div id="modal" style="display:none">
         <div id="modal-title"></div><div id="modal-body"></div>
       </div></body></html>`, { url: "http://localhost:8642/" });
  const win = dom.window;
  win.$ = (id) => win.document.getElementById(id);
  win.el = (t) => win.document.createElement(t);
  win.openModal = () => {};
  win.Audio = class {};
  global.window = win; global.document = win.document; global.Audio = win.Audio;
  delete global.confirm;

  const realVoices = readFileSync(join(VENDOR, "voices.json"), "utf8");
  const prevFetch = globalThis.fetch;
  win.fetch = async (url) => {
    const u = String(url);
    if (u.includes("/api/tts/config"))
      return { ok: true, status: 200,
               json: async () => ({ net_mode: "off", voice: "zz_not_a_real_voice" }) };
    if (u.includes("voices.json"))
      return { ok: true, status: 200, json: async () => JSON.parse(realVoices) };
    return { ok: false, status: 404, json: async () => ({}) };
  };
  global.fetch = win.fetch;

  try {
    const registered = [];
    const mod = await import(pathToFileURL(TTS_JS).href + "?t=" + Date.now());
    await mod.register({ authHeaders: () => ({}), toast: () => {},
                         registerTTS: (p) => registered.push(p) });
    const provider = registered[0];
    assert.ok(provider, "register() must hand a provider to registerTTS");

    const list = provider.voices();
    assert.equal(list.length, 41);
    assert.notEqual(list[0].id, "af_heart",
      "precondition: the picker order must NOT start with af_heart, or this " +
      "test cannot tell a by-name fallback from a first-entry fallback");
    assert.equal(provider.getVoice(), "af_heart");

    // grouped by language, English first
    const langs = list.map((v) => v.language).filter((l, i, a) => l !== a[i-1]);
    assert.deepEqual(langs, ["en-gb", "en-us", "es", "fr", "hi", "it", "pt-br"]);

    // upstream grades only: the voices upstream does not grade show none
    const ungraded = list.filter((v) => !/, [A-F][+-]?\)$/.test(v.label)).map((v) => v.id);
    assert.deepEqual(ungraded.sort(),
      ["ef_dora", "em_alex", "em_santa", "pf_dora", "pm_alex", "pm_santa"]);
  } finally {
    globalThis.fetch = prevFetch;
    delete global.window; delete global.document; delete global.Audio;
  }
});

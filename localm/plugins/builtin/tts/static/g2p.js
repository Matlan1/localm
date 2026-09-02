/**
 * Non-English grapheme-to-phoneme for the Kokoro voices.
 *
 * The vendored `kokoro.min.js` carries a complete eSpeak-NG WASM build, but its
 * own phonemize path pins the locale to English and then applies English-only
 * phoneme fixups. This module drives the same engine directly for the
 * non-English voices and produces the IPA their Kokoro voice expects.
 *
 * `kokoro.min.js` exports `espeakWorker` and `espeakFS` for this module; a
 * re-vendor that drops those exports is caught by tests-js/tts-g2p.test.mjs.
 */

/** Kokoro voice-id prefix -> eSpeak-NG locale. Prefixes absent here are English. */
const VOICE_LOCALE = Object.freeze({
  e: "es",
  f: "fr",
  h: "hi",
  i: "it",
  p: "pt-br",
});

/** eSpeak-NG locale -> the `<name>_dict` file that locale loads. */
const LOCALE_DICT = Object.freeze({
  es: "es_dict",
  fr: "fr_dict",
  hi: "hi_dict",
  it: "it_dict",
  "pt-br": "pt_dict",
});

const DATA_DIR = "/usr/share/espeak-ng-data";

/** eSpeak-NG language-switch markers, e.g. the "(en)" around a loanword. */
const LANG_SWITCH = /\([a-z]{2,3}(?:-[a-z]{2,8})*\)/g;

/**
 * The locale for a Kokoro voice id, or null when the voice is English and the
 * bundle's own phonemizer should handle it.
 */
export function localeForVoice(voiceId) {
  if (typeof voiceId !== "string" || voiceId.length === 0) return null;
  return VOICE_LOCALE[voiceId[0]] ?? null;
}

/** True when this module handles the voice. */
export function isNonEnglishVoice(voiceId) {
  return localeForVoice(voiceId) !== null;
}

/**
 * Normalise raw eSpeak-NG IPA into Kokoro's phoneme alphabet.
 *
 * Kokoro's tokenizer has no unknown-token: its normaliser deletes any character
 * outside a 115-symbol whitelist, so anything left here that Kokoro does not
 * know is dropped without a trace. `j` and `l` are the in-vocabulary spellings
 * of the two symbols eSpeak-NG emits that Kokoro lacks.
 *
 * The English fixups in the bundle map `r`->`ɹ` and `x`->`k`, which is wrong for
 * these languages: `r` is the Italian and Spanish trill and `x` is both the
 * Spanish jota and the Brazilian Portuguese r. Both are in Kokoro's vocabulary
 * and are what eSpeak-NG produced for the recordings these voices were trained
 * on, so they are kept verbatim. See test_preserves_r_and_x.
 */
export function normalizePhonemes(ipa) {
  return ipa
    .replace(LANG_SWITCH, "")
    .replace(/ʲ/g, "j")
    .replace(/ɬ/g, "l")
    .replace(/-/g, "")
    .replace(/\s+/g, " ")
    .trim();
}

/** Per-locale dictionary loads, settled or in flight, keyed by locale. */
const dictLoads = new Map();

/**
 * Fetch a locale's dictionary and write it into the running WASM filesystem.
 *
 * eSpeak-NG ships every language's rules and voice files but only the English
 * dictionary, so the dictionary is fetched from our own server on first use of
 * a language and never for a user who only speaks English.
 */
async function ensureDictionary(mod, locale, baseURL) {
  const dict = LOCALE_DICT[locale];
  if (!dict) throw new Error(`No dictionary is vendored for locale "${locale}".`);
  let load = dictLoads.get(locale);
  if (load) return load;

  load = (async () => {
    const fs = mod.espeakFS;
    const target = `${DATA_DIR}/${dict}`;
    if (fs.analyzePath(target).exists) return;
    const url = new URL(`vendor/espeak-ng-data/${dict}`, baseURL);
    const res = await fetch(url);
    if (!res.ok) throw new Error(`${dict}: HTTP ${res.status}`);
    const bytes = new Uint8Array(await res.arrayBuffer());
    if (bytes.length === 0) throw new Error(`${dict}: empty response`);
    fs.writeFile(target, bytes);
  })();

  dictLoads.set(locale, load);
  try {
    await load;
  } catch (e) {
    dictLoads.delete(locale);
    throw e;
  }
  return load;
}

/**
 * Phonemize one chunk of text for a non-English locale.
 *
 * Throws on both of eSpeak-NG's silent failures.
 *
 * `set_voice` answers 0 on success and non-zero on rejection, and a rejected
 * call leaves the PREVIOUS language selected rather than reporting an error, so
 * an unusable locale reads as fluent output in the wrong language. Not every
 * language name it lists is a settable identifier: "fr-fr" appears in the voice
 * metadata and is rejected, while "fr" is accepted. See test_rejects_unsettable_locale.
 *
 * A dictionary it cannot read is likewise not an error: it writes a message to
 * the console and returns an empty string with a success code, which reaches
 * the model as an empty token sequence and plays as silence.
 */
export async function phonemize(mod, text, locale, baseURL) {
  await ensureDictionary(mod, locale, baseURL);
  const worker = await mod.espeakWorker;
  const status = worker.set_voice(locale);
  if (status !== 0) {
    throw new Error(
      `eSpeak-NG rejected locale "${locale}" (status ${status}); refusing to ` +
      `phonemize, which would silently use the previously selected language.`);
  }
  const raw = worker.synthesize_ipa(text).ipa ?? "";
  const joined = raw.split("\n").filter((line) => line.length > 0).join(" ");
  const phonemes = normalizePhonemes(joined);
  if (phonemes.length === 0) {
    throw new Error(
      `eSpeak-NG produced no phonemes for locale "${locale}"; the ` +
      `${LOCALE_DICT[locale]} dictionary is missing or unreadable.`);
  }
  return phonemes;
}

/**
 * Stream sentences of `text` as `{ text, phonemes, audio }`, matching the shape
 * `KokoroTTS.stream` yields so both paths feed one consumer.
 */
export async function* streamNonEnglish(mod, tts, source, { voice, speed, baseURL }) {
  const locale = localeForVoice(voice);
  if (!locale) throw new Error(`Voice "${voice}" is not a non-English voice.`);
  const sentences = typeof source === "string" ? [source] : source;
  for await (const sentence of sentences) {
    const phonemes = await phonemize(mod, sentence, locale, baseURL);
    const { input_ids } = tts.tokenizer(phonemes, { truncation: true });
    const audio = await tts.generate_from_ids(input_ids, { voice, speed });
    yield { text: sentence, phonemes, audio };
  }
}

/** Test seam: forget cached dictionary loads. */
export function _resetDictionaryCache() {
  dictLoads.clear();
}

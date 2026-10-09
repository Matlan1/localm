# SPDX-License-Identifier: AGPL-3.0-or-later
"""Output formats for a detailed transcription: plain JSON, text, SRT, WebVTT
and verbose JSON.

Pure functions over the plain-dict transcription detail the speech worker
returns (see ``localm.voice.transcribe_detailed``); nothing here imports the
speech stack, so it is safe to import in the server process.
"""

from __future__ import annotations

import re
from typing import Iterable, Optional

RESPONSE_FORMATS = ("json", "text", "srt", "vtt", "verbose_json")

LANGUAGE_NAMES = {
    "en": "english", "zh": "chinese", "de": "german", "es": "spanish",
    "ru": "russian", "ko": "korean", "fr": "french", "ja": "japanese",
    "pt": "portuguese", "tr": "turkish", "pl": "polish", "ca": "catalan",
    "nl": "dutch", "ar": "arabic", "sv": "swedish", "it": "italian",
    "id": "indonesian", "hi": "hindi", "fi": "finnish", "vi": "vietnamese",
    "he": "hebrew", "uk": "ukrainian", "el": "greek", "ms": "malay",
    "cs": "czech", "ro": "romanian", "da": "danish", "hu": "hungarian",
    "ta": "tamil", "no": "norwegian", "th": "thai", "ur": "urdu",
    "hr": "croatian", "bg": "bulgarian", "lt": "lithuanian", "la": "latin",
    "mi": "maori", "ml": "malayalam", "cy": "welsh", "sk": "slovak",
    "te": "telugu", "fa": "persian", "lv": "latvian", "bn": "bengali",
    "sr": "serbian", "az": "azerbaijani", "sl": "slovenian", "kn": "kannada",
    "et": "estonian", "mk": "macedonian", "br": "breton", "eu": "basque",
    "is": "icelandic", "hy": "armenian", "ne": "nepali", "mn": "mongolian",
    "bs": "bosnian", "kk": "kazakh", "sq": "albanian", "sw": "swahili",
    "gl": "galician", "mr": "marathi", "pa": "punjabi", "si": "sinhala",
    "km": "khmer", "sn": "shona", "yo": "yoruba", "so": "somali",
    "af": "afrikaans", "oc": "occitan", "ka": "georgian", "be": "belarusian",
    "tg": "tajik", "sd": "sindhi", "gu": "gujarati", "am": "amharic",
    "yi": "yiddish", "lo": "lao", "uz": "uzbek", "fo": "faroese",
    "ht": "haitian creole", "ps": "pashto", "tk": "turkmen", "nn": "nynorsk",
    "mt": "maltese", "sa": "sanskrit", "lb": "luxembourgish", "my": "myanmar",
    "bo": "tibetan", "tl": "tagalog", "mg": "malagasy", "as": "assamese",
    "tt": "tatar", "haw": "hawaiian", "ln": "lingala", "ha": "hausa",
    "ba": "bashkir", "jw": "javanese", "su": "sundanese", "yue": "cantonese",
}


def language_name(code: Optional[str]) -> str:
    """The English name of a Whisper language code (``en`` -> ``english``), or
    the code itself when it has no entry; empty for no code."""
    if not code:
        return ""
    return LANGUAGE_NAMES.get(code, code)


def _stamp(seconds: float, sep: str) -> str:
    total_ms = max(0, round(float(seconds) * 1000))
    hours, rest = divmod(total_ms, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    secs, ms = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{sep}{ms:03d}"


def _one_line(text: str) -> str:
    return re.sub(r"\s*[\r\n]+\s*", " ", text).strip()


def to_srt(segments: Iterable[dict]) -> str:
    """SubRip text: numbered cues with ``HH:MM:SS,mmm`` times. Cue text is kept
    on one line because a blank line ends a cue."""
    out = []
    for n, seg in enumerate(segments, start=1):
        out.append(f"{n}\n{_stamp(seg['start'], ',')} --> "
                   f"{_stamp(seg['end'], ',')}\n{_one_line(seg['text'])}\n")
    return "\n".join(out)


def _vtt_text(text: str) -> str:
    text = _one_line(text)
    return (text.replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;"))


def to_vtt(segments: Iterable[dict]) -> str:
    """WebVTT text: a ``WEBVTT`` header, then cues with ``HH:MM:SS.mmm`` times.
    Cue text has ``&``, ``<`` and ``>`` escaped, which a cue would otherwise
    parse as markup."""
    out = ["WEBVTT\n"]
    for seg in segments:
        out.append(f"{_stamp(seg['start'], '.')} --> "
                   f"{_stamp(seg['end'], '.')}\n{_vtt_text(seg['text'])}\n")
    return "\n".join(out)


def to_verbose_json(detail: dict, *, include_segments: bool,
                    include_words: bool) -> dict:
    """The ``verbose_json`` body: task, language name, duration, text, and
    optionally the segments and the flat word list."""
    body = {
        "task": "transcribe",
        "language": language_name(detail.get("language")),
        "duration": round(float(detail.get("duration") or 0.0), 3),
        "text": detail["text"],
    }
    segments = detail.get("segments") or []
    if include_words:
        body["words"] = [
            {"word": w["word"], "start": w["start"], "end": w["end"]}
            for seg in segments for w in (seg.get("words") or [])]
    if include_segments:
        body["segments"] = [
            {k: seg[k] for k in ("id", "seek", "start", "end", "text",
                                 "tokens", "temperature", "avg_logprob",
                                 "compression_ratio", "no_speech_prob")}
            for seg in segments]
    return body

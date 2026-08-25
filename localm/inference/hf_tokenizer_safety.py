# SPDX-License-Identifier: AGPL-3.0-or-later
"""Load-time ReDoS gate for a pulled HuggingFace model's ``tokenizer.json`` (NEW-HF-TOKENIZER-REDOS; maintainer release-gate decision Q1-Q3, 2026-07-30)."""

from __future__ import annotations

import importlib.util
import json
import queue
import subprocess
import threading
from pathlib import Path
from typing import Iterator, List

# Real pre_tokenizer/normalizer/decoder patterns are tiny (the GPT-2 pattern
# above is 74 characters). Checked BEFORE anything else - including the
# recursive JSON walk that finds these patterns in the first place is cheap,
# but there is no legitimate reason a load-bearing tokenizer pattern needs to
# be large, so an oversized one is refused on sight rather than spent any
# further work on. Same order of magnitude as gbnf.MAX_TRIGGER_PATTERN_BYTES.
MAX_TOKENIZER_REGEX_BYTES = 4096

# Per-pattern read timeout against the probe subprocess. One flat value for
# every pattern (including the first): unlike _trigger_probe.py's persistent
# daemon, there is no "steady state" here - every call is a fresh cold start,
# so every line pays whatever the first one would. Generous relative to a
# safe pattern (measured: the full ~10-probe battery, including a Oniguruma
# cold import, finishes in well under a second) and relative to how long a
# once-per-model-load check may reasonably take before the load itself
# (already seconds to minutes for real weight files) notices the extra wait.
_PROBE_TIMEOUT_SECONDS = 8.0


def _iter_regex_patterns(node) -> Iterator[str]:
    """Yield every string value found under a JSON key literally named 'Regex', anywhere in *node* - the exact shape ``tokenizers`` serializes a ``Regex``-typed ``Split``/``Replace`` pattern as (verified empirically by constructing a real Tokenizer with Split+Regex and Replace+Regex steps in the normalizer..."""
    if isinstance(node, dict):
        pattern = node.get("Regex")
        if isinstance(pattern, str):
            yield pattern
        for value in node.values():
            yield from _iter_regex_patterns(value)
    elif isinstance(node, list):
        for item in node:
            yield from _iter_regex_patterns(item)


def _extract_unique_patterns(model_path: str) -> List[str]:
    """Unique Regex pattern strings from ``<model_path>/tokenizer.json``, in first-occurrence order."""
    path = Path(model_path) / "tokenizer.json"
    if not path.is_file():
        return []
    try:
        text = path.read_text(encoding="utf-8")
        data = json.loads(text)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        raise RuntimeError(
            f"'{Path(model_path).name}' ships a tokenizer.json that could not "
            f"be read as valid JSON ({type(e).__name__}: {e}), so its regex "
            "patterns cannot be verified safe; refusing to load.") from e
    return list(dict.fromkeys(_iter_regex_patterns(data)))


def _readline_with_timeout(stream, timeout: float):
    """``stream.readline()`` bounded by *timeout*."""
    q: "queue.Queue" = queue.Queue(maxsize=1)

    def _reader():
        try:
            q.put(stream.readline())
        except Exception:
            q.put(None)

    threading.Thread(target=_reader, daemon=True).start()
    try:
        line = q.get(timeout=timeout)
    except queue.Empty:
        return None
    return line or None


def _run_probe_subprocess(patterns: List[str]) -> "List[str] | None":
    """Spawn the isolated probe subprocess, feed it *patterns*, and return one verdict string per pattern successfully read ('OK' or 'BAD <reason>')."""
    import json as _json

    from localm._mp_spawn import interpreter_for_localm_children

    try:
        proc = subprocess.Popen(
            [interpreter_for_localm_children(), "-u", "-m",
             "localm.inference._hf_tokenizer_probe"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1,
        )
    except OSError:
        return None

    try:
        proc.stdin.write(_json.dumps({"patterns": patterns}) + "\n")
        proc.stdin.flush()
    except OSError:
        proc.kill()
        return None

    verdicts: List[str] = []
    try:
        for _ in patterns:
            line = _readline_with_timeout(proc.stdout, _PROBE_TIMEOUT_SECONDS)
            if line is None:
                break   # hang or crash on this pattern - caller kills below
            verdicts.append(line.strip())
    finally:
        proc.kill()
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            pass   # best-effort reap; a lingering zombie here is not a safety issue

    return verdicts


def validate_tokenizer_json(model_path: str) -> None:
    """Raise ``RuntimeError`` if ``<model_path>/tokenizer.json`` carries a Regex-typed normalizer/pre_tokenizer/decoder pattern that is unsafe to compile-and-run against arbitrary caller text, BEFORE ``HFBackend.load()`` hands the directory to ``transformers``."""
    if importlib.util.find_spec("tokenizers") is None:
        return

    patterns = _extract_unique_patterns(model_path)
    if not patterns:
        return

    oversized = [p for p in patterns if len(p.encode("utf-8", "surrogatepass")) >
                 MAX_TOKENIZER_REGEX_BYTES]
    if oversized:
        raise RuntimeError(
            f"'{Path(model_path).name}' ships a tokenizer.json with a regex "
            f"pattern of {len(oversized[0])} bytes, over the "
            f"{MAX_TOKENIZER_REGEX_BYTES}-byte limit for a tokenizer pattern; "
            "refusing to load. A legitimate pre_tokenizer/normalizer pattern "
            "is tens to a few hundred bytes.")

    verdicts = _run_probe_subprocess(patterns)
    if verdicts is None:
        raise RuntimeError(
            f"'{Path(model_path).name}' ships a tokenizer.json with regex "
            "patterns that could not be verified safe: the Oniguruma safety "
            "probe process could not be started; refusing to load.")
    if len(verdicts) < len(patterns):
        culprit = patterns[len(verdicts)]
        raise RuntimeError(
            f"'{Path(model_path).name}' ships a tokenizer.json whose regex "
            f"pattern {culprit[:80]!r} did not pass the Oniguruma safety probe "
            f"within {_PROBE_TIMEOUT_SECONDS:.0f}s (the probe process hung or "
            "crashed while checking it) - likely catastrophic native regex "
            "backtracking against ordinary chat text; refusing to load.")

    for pattern, verdict in zip(patterns, verdicts):
        if verdict == "OK":
            continue
        reason = verdict[4:] if verdict.startswith("BAD ") else verdict
        raise RuntimeError(
            f"'{Path(model_path).name}' ships a tokenizer.json whose regex "
            f"pattern {pattern[:80]!r} was rejected by the Oniguruma safety "
            f"probe: {reason}; refusing to load.")

# SPDX-License-Identifier: AGPL-3.0-or-later
"""Load-time ReDoS gate for a pulled HuggingFace model's ``tokenizer.json``.

``HFBackend.load()`` (inference/backends/hf.py) resolves a model directory's
tokenizer via ``transformers.AutoProcessor``/``AutoTokenizer.from_pretrained``,
which - for any "fast" tokenizer, i.e. one shipping a ``tokenizer.json`` -
loads through HuggingFace's ``tokenizers`` Rust extension. That extension's
normalizer/pre_tokenizer/decoder pipeline can include a ``Split`` or
``Replace`` step whose pattern is a ``tokenizers.Regex``: a real native regex,
compiled once at load time and then run, on every subsequent request, against
the raw text of every caller's chat message (``HFBackend.count_tokens`` /
``count_messages_tokens``, and generation itself). ``tokenizer.json`` is
attacker-reachable end to end: ``pull_model`` accepts an unrestricted repo id
(reachable by a prompt-injected model calling the MCP server), fetches the
WHOLE repo including ``tokenizer.json`` verbatim from whatever the repo owner
put there, and loads it immediately by default. This module validates that
file's regex patterns before the load.

THE PROBE RUNS AGAINST ONIGURUMA, NOT PYTHON ``re``. ``tokenizers.Regex``
wraps Oniguruma (the ``onig`` crate), a different native backtracking engine
from Python's ``re``, and the two have different catastrophic sets, neither
contained in the other:

  * ``(a|a)*b`` is catastrophic in Python ``re`` and cheap in Oniguruma.
  * ``((?=(a+))a)+b`` against ``"a" * 2000`` is under a second in ``re`` and
    unbounded in Oniguruma. ``re.compile`` accepts it, so an ``re``-based
    validator scores it safe while Oniguruma is already hung.
  * ``re.compile`` outright REJECTS the real, widely-shipped GPT-2 byte-level
    pre-tokenizer pattern with ``re.error: bad escape`` (Python's ``re`` has no
    ``\\p{...}`` Unicode-property support), while Oniguruma compiles and runs it
    in low milliseconds - so an ``re``-based validator would refuse to load
    real, unmodified tokenizers.
  * Oniguruma also has an internal backtrack ceiling that classic
    nested-quantifier shapes (``(a+)+b`` and friends) hit almost immediately,
    flat regardless of input length, rather than hanging - but it surfaces as
    ``pyo3_runtime.PanicException``, whose MRO is
    ``(PanicException, BaseException, object)``: NOT an ``Exception`` subclass.
    See ``_hf_tokenizer_probe.py``'s docstring - the probe must catch
    ``BaseException``, since an uncaught instance reaching the SERVER's own
    in-process tokenizer call (HFBackend has no worker isolation) is a Rust
    panic crossing the FFI boundary in the main process.

``gbnf.py``'s ``_static_shape_rejection`` (the cheap in-process heuristic used
for ``grammar_triggers``) does not catch the lookahead pattern above: the
nested ``+`` inside ``(a+)`` sits two paren-levels below the outer ``(...)+``
(the lookahead ``(?=...)`` opens a level in between), so its "quantifier
immediately inside an immediately-quantified group" scan never attributes it to
the outer group. It is not reused here. The empirical probe below is the sole
source of truth; only a plain length cap is applied first.

There is NO persistent probe daemon here, unlike ``_trigger_probe.py``. That
daemon exists because ``grammar_triggers`` is validated on every
``/v1/chat/completions`` request with the field set. Validating a model's
``tokenizer.json`` happens once per ``HFBackend.load()``, so a fresh subprocess
per validation call is a negligible addition and needs none of that daemon's
lock/cache/prewarm machinery.
"""

from __future__ import annotations

import importlib.util
import json
import queue
import subprocess
import threading
from pathlib import Path
from typing import Iterator, List

# Real pre_tokenizer/normalizer/decoder patterns are tiny (the GPT-2 pattern
# above is 74 characters). Checked BEFORE anything else, including the recursive
# JSON walk that finds these patterns: an oversized pattern is refused on sight.
# Same order of magnitude as gbnf.MAX_TRIGGER_PATTERN_BYTES.
MAX_TOKENIZER_REGEX_BYTES = 4096

# Per-pattern read timeout against the probe subprocess. One flat value for
# every pattern, including the first: unlike _trigger_probe.py's persistent
# daemon there is no steady state here, so every call is a fresh cold start and
# every line pays what the first one would. Generous relative to a safe pattern
# and to how long a once-per-model-load check may take before the load itself,
# already seconds to minutes for real weight files, notices the extra wait.
_PROBE_TIMEOUT_SECONDS = 8.0


def _iter_regex_patterns(node) -> Iterator[str]:
    """Yield every string value found under a JSON key literally named
    "Regex", anywhere in *node* - the exact shape ``tokenizers`` serializes a
    ``Regex``-typed ``Split``/``Replace`` pattern as. In the normalizer,
    pre_tokenizer and decoder alike it is a ``{"pattern": {"Regex": "<source>"}}``
    object, whether at the top level or nested inside a ``Sequence`` - which
    itself uses three different list-key names across those three components:
    "normalizers", "pretokenizers" (no underscore), "decoders".

    Walks the WHOLE parsed document for a bare ``{"Regex": ...}`` shape rather
    than hand-picking "normalizer"/"pre_tokenizer"/"decoder"/"post_processor"
    and their known nesting, so it needs no per-key knowledge of a schema that
    varies its own list-key naming between components. A full walk of even a
    very large tokenizer.json is a linear scan of cheap isinstance checks (a
    large vocab is a flat dict of str->int, merges a flat list of str)."""
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
    """Unique Regex pattern strings from ``<model_path>/tokenizer.json``, in
    first-occurrence order. Returns ``[]`` when the file is absent (a "slow"
    /legacy/sentencepiece-only tokenizer ships no ``tokenizer.json`` and has
    no Oniguruma-backed pipeline for this check to examine at all).

    Malformed JSON is NOT treated as "nothing to check": ``tokenizer.json`` is
    always produced by ``tokenizers``' own strict JSON serializer, so a real
    model's file parses cleanly here exactly as it does for ``transformers``'
    own (also strict-JSON) loader. A file that fails to parse cannot be proven
    safe, and "cannot prove safe" is refused, the same fail-closed posture
    ``_check_one`` uses in both probe daemons for a syntactically invalid
    pattern."""
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
    """``stream.readline()`` bounded by *timeout*. Windows pipes offer no
    ``select()``-style wait, so the read runs in a daemon thread and this waits
    on it via a bounded queue (see gbnf.py's own copy of this helper). On
    timeout the reader thread is abandoned, not joined - it exits on its own
    once the read unblocks or the pipe closes, which killing the probe process
    (done by the caller immediately after a timeout) guarantees. Returns None on
    timeout, EOF, or any read error."""
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
    """Spawn the isolated probe subprocess, feed it *patterns*, and return one
    verdict string per pattern successfully read ("OK" or "BAD <reason>").

    Returns ``None`` - distinct from an empty or short list - when the process
    could not even be started or handed its input: an infrastructure failure
    that happens immediately, before any per-pattern wait. Returns a list with
    FEWER entries than *patterns* when a line could not be read within the
    per-pattern timeout (a hang) or the process died mid-stream (a crash, e.g.
    an uncaught BaseException - see _hf_tokenizer_probe.py's docstring). Only
    that second case genuinely waited up to _PROBE_TIMEOUT_SECONDS, so the two
    are kept apart for an accurate error message. The caller treats "could not
    verify" as unsafe either way."""
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
    """Raise ``RuntimeError`` if ``<model_path>/tokenizer.json`` carries a
    Regex-typed normalizer/pre_tokenizer/decoder pattern that is unsafe to
    compile-and-run against arbitrary caller text, BEFORE ``HFBackend.load()``
    hands the directory to ``transformers``. A no-op (returns immediately)
    when there is nothing to check: no ``tokenizer.json`` at all, no
    Regex-typed pattern within it, or ``tokenizers`` itself is not importable
    (in which case ``_require_transformers()`` a few lines later in
    ``HFBackend.load()`` gives the correct, clearer "transformers not
    installed" refusal - this function does not pre-empt that with a
    confusing message about its own probing machinery instead).

    Every extracted pattern must be proven safe or this call raises - a
    pattern the probe subprocess could not reach a verdict for (it hung, or
    the process died) is treated identically to one it explicitly rejected,
    matching gbnf.py's own three-way "unsafe" reasoning: unproven is unsafe."""
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

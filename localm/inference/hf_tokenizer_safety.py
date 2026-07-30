# SPDX-License-Identifier: AGPL-3.0-or-later
"""Load-time ReDoS gate for a pulled HuggingFace model's ``tokenizer.json``
(NEW-HF-TOKENIZER-REDOS; maintainer release-gate decision Q1-Q3, 2026-07-30).

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
put there, and loads it immediately by default. Nothing validated that file's
regex patterns before this fix - unlike ``grammar_triggers``
(GitHub #928/#833, see gbnf.py), which is the same class of bug in a
different field, already fixed there.

WHY THIS PROBES ONIGURUMA, NOT PYTHON ``re`` (Q3): ``tokenizers.Regex`` wraps
Oniguruma (the ``onig`` crate), a DIFFERENT native backtracking engine from
Python's ``re`` - and the two have measurably DIFFERENT catastrophic sets,
neither contained in the other (the same lesson
``tests/test_model_supplied_regex_bound.py`` already documents for a
different pair of engines - ``re`` vs the ``regex`` module). MEASURED while
building this validator (same venv, python 3.12.13, tokenizers 0.22.2):

  * The direction already known: ``(a|a)*b`` on 26 chars costs 9.5s+ in
    Python ``re`` and ~10ms in Oniguruma (963x apart per the release-gate
    note) - reusing ``_trigger_probe.py``'s ``re``-based probe here would
    validate the WRONG engine and its verdict would carry no information.
  * The direction that was NOT yet measured when Q3 was written: a
    lookahead-in-a-loop pattern, ``((?=(a+))a)+b``, fed ``"a" * N``. Clean
    timing (imports excluded from the measured region):

        N      re match      onig match
        300      3.1ms         46.1ms
        1000   104.9ms       1801.0ms
        1800   461.1ms      10383.9ms
        2000   705.3ms       HANG (>12s)

    At N=2000 (an ordinary chat-message/paste length) Python ``re`` is still
    under a second while Oniguruma has ALREADY exceeded 12 seconds unbounded -
    the exact "cheap in re, catastrophic in Oniguruma" direction the release
    note asked to be measured first. ``re.compile`` accepts this pattern (no
    syntax rejection), so a ``re``-based validator would score it "safe" and
    completely miss that Oniguruma is already hung on the identical input.
  * Beyond danger-detection: ``re.compile`` outright REJECTS the real,
    widely-shipped GPT-2 byte-level pre-tokenizer pattern -
    ``'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+``
    - with ``re.error: bad escape \\p`` (Python's ``re`` has no ``\\p{...}``
    Unicode-property support). Oniguruma compiles and runs it in low
    milliseconds. A ``re``-based validator would therefore not just miss a
    danger class, it would refuse to load real, legitimate, unmodified
    tokenizers outright - a functional regression stacked on the security gap.
  * Oniguruma ALSO has an internal backtrack ceiling that classic
    nested-quantifier shapes (``(a+)+b`` and friends) hit almost immediately
    (a few hundred ms, flat regardless of N) rather than hanging - but it
    surfaces as ``pyo3_runtime.PanicException``, whose MRO is
    ``(PanicException, BaseException, object)``: NOT an ``Exception``
    subclass. See ``_hf_tokenizer_probe.py``'s docstring for why this means
    the probe must catch ``BaseException``, and why that distinction matters
    well beyond this validator (an uncaught instance reaching the SERVER's
    own in-process tokenizer call - HFBackend has no worker isolation,
    Q1/Q2 - would be a Rust panic crossing the FFI boundary in the main
    process, not merely a hang).

``gbnf.py``'s ``_static_shape_rejection`` (the cheap in-process heuristic used
for ``grammar_triggers``) was checked against the lookahead pattern above and
does NOT catch it: the nested ``+`` inside ``(a+)`` sits two paren-levels
below the outer ``(...)+`` (the lookahead ``(?=...)`` opens a level in
between), so the heuristic's "quantifier immediately inside an immediately-
quantified group" scan never attributes it to the outer group. That heuristic
is therefore not reused here even as a "cheap first layer" - it is proven
incomplete independent of the engine question, and unlike the per-request
``grammar_triggers`` hot path, THIS check runs once per model load (already a
multi-second-to-minutes operation), so there is no latency pressure to accept
a heuristic's blind spots in exchange for speed. The empirical probe below is
the sole source of truth; only a plain length cap is applied first (cheap,
and carries no false shape-based confidence).

WHY NO PERSISTENT DAEMON (unlike ``_trigger_probe.py``): that daemon exists
because ``grammar_triggers`` is validated on every ``/v1/chat/completions``
request with the field set - a hot path where per-request Python startup cost
was rejected. Validating a model's ``tokenizer.json`` happens once per
``HFBackend.load()`` - a cold, infrequent operation - so a fresh subprocess
per validation call is a negligible addition and needs none of that daemon's
lock/cache/prewarm machinery, which exists solely to keep a hot path cheap.
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
    """Yield every string value found under a JSON key literally named
    "Regex", anywhere in *node* - the exact shape ``tokenizers`` serializes a
    ``Regex``-typed ``Split``/``Replace`` pattern as (verified empirically by
    constructing a real Tokenizer with Split+Regex and Replace+Regex steps in
    the normalizer, pre_tokenizer AND decoder, then inspecting
    ``Tokenizer.to_str()``'s JSON: every one of them is a
    ``{"pattern": {"Regex": "<source>"}}`` object, whether at the top level or
    nested inside a ``Sequence`` - which itself uses THREE DIFFERENT list-key
    names across those three components: "normalizers", "pretokenizers"
    (no underscore), "decoders").

    Walking the WHOLE parsed document for a bare ``{"Regex": ...}`` shape,
    rather than hand-picking "normalizer"/"pre_tokenizer"/"decoder"/
    "post_processor" and their known nesting, is deliberate: it needs no
    per-key knowledge of a schema that already varies its own list-key naming
    convention between components, so it can't be defeated by a nesting shape
    this function's author did not anticipate, and it costs nothing extra
    that matters (a full walk of even a very large tokenizer.json - a large
    vocab is a flat dict of str->int, merges a flat list of str, neither
    nested - is a linear scan of cheap isinstance checks, negligible next to
    the multi-GB weight load this check is a preamble to)."""
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

    Malformed JSON is NOT treated as "nothing to check": ``tokenizer.json` is
    always produced by ``tokenizers``' own strict JSON serializer, so a real
    model's file parses cleanly here exactly as it does for
    ``transformers``' own (also strict-JSON) loader - a file that fails to
    parse cannot be proven safe, and "cannot prove safe" is refused, not
    waved through (the same fail-closed posture ``_check_one`` in both probe
    daemons uses for a syntactically invalid pattern)."""
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
    ``select()``-style wait, so the read runs in a daemon thread and this
    waits on it via a bounded queue - reimplemented here rather than imported
    across the private gbnf.py/_loader.py module boundary, matching the
    existing convention (see gbnf.py's own copy of this same helper). On
    timeout the reader thread is abandoned, not joined - it exits on its own
    once the read unblocks or the pipe closes, which killing the probe
    process (done by the caller immediately after a timeout) guarantees.
    Returns None on timeout, EOF, or any read error."""
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

    Returns ``None`` - distinct from an empty or short list - when the
    process could not even be started or handed its input: an infrastructure
    failure that happens immediately, before any per-pattern wait. Returns a
    list with FEWER entries than *patterns* when a line could not be read
    within the per-pattern timeout (a hang) or the process died mid-stream (a
    crash, e.g. an uncaught BaseException - see _hf_tokenizer_probe.py's
    docstring) - THAT case genuinely waited up to _PROBE_TIMEOUT_SECONDS, so
    the caller's message naming that timeout is accurate for it and would NOT
    be for a spawn failure, which is why the two are not collapsed into the
    same return shape. Either way the caller treats "could not verify" as
    unsafe - the distinction is for an accurate error message, not the
    safety verdict, which is identical (refuse) in both cases."""
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

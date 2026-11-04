#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Audit localm debug logs against what a run asked for.

Reads one or more ``localm --debug`` logs (``<data dir>/logs/localm_*.log``)
and a per-run manifest, and prints every event of these kinds, each marked
EXPECTED (a manifest entry matches it) or UNEXPLAINED:

  warning     a WARNING record, or a Python ``...py:N: SomeWarning:`` line
  error       an ERROR or CRITICAL record; a native llama.cpp / ggml / mtmd
              line reporting an error or failure; a backend error (``ROCm
              error``, ``CUDA error``, ...); a ggml abort (``file.cu:88: ...``);
              ``GGML_ASSERT``; ``Fatal Python error``, ``Windows fatal
              exception``, ``Segmentation fault``, ``Exception in thread``,
              ``Exception ignored in``; a ``Traceback`` that does not start on
              the line directly after a WARNING-or-higher record (a chained
              exception of that record's own traceback stays part of it, so a
              raw traceback that lands on exactly that line is read as the
              record's own); and, with ``--home``, every non-empty crash or hang
              trace the run left in its data dir (``run/server-crash-trace*.txt``,
              ``logs/crash_*.txt``, ``logs/hang_*.log``) and every server crash
              marker or stopping record whatever its size
              (``run/server-crash*.marker``, ``run/server-crash*.stopping``:
              a server that was killed or crashed leaves one even with no trace)
  toast       a GUI report sent through ``POST /api/gui-error`` (logged at
              WARNING as ``gui: [source page] message``)
  model-load  a load localm's own log reports: a llama.cpp model loader banner
              (``loaded meta data ... from``), an mtmd projector load (``loaded N
              tensors from``), ``embedding model ready: ...``, ``switch_engine:
              loaded '...'``, or ``... ready: <model>`` from the HF, ExLlama,
              sd.cpp (model and upscaler), speech, reranker and Whisper loaders.
              ComfyUI and koboldcpp load inside their own processes, so for them
              this is a job naming a model (``comfy: job uses model ...``,
              ``koboldcpp music: job uses model ...``), logged whether or not that
              process already held it. A model a plugin, an external server or
              any other process loads without one of these lines is not seen.
  download    ``downloading ... model ... (one-time)``, ``pull: downloading
              ...`` (every remote pull: CLI, MCP, GUI), ``Model pull ...
              started`` (a GUI pull job) or a tqdm progress bar (``name:  42%|``)

Consecutive load lines naming the same model are one model-load event, and
consecutive progress lines of one file are one download event. A WARNING record
that also reports a load is both a warning event and a model-load event. A
WARNING-or-higher record whose first line is empty takes its text from the next
line.

A log that simply stops (the process was killed) is not detectable from the log
alone; pass ``--home`` so a crash leaves an event, and audit after the server has
stopped, since a running server holds a marker. The marker belongs to a server
(``localm serve``, ``localm gui``), so a one-shot command that dies leaves none. A
record glued behind a partial native line is read as a record, and the text before
it as a line of its own.

Manifest (JSON object; every key optional, any other key is an error):

    {
      "run": "free text naming the run",
      "models":    ["Qwen3-0.6B-Q8_0", {"match": "bge-small-*", "why": "step 4 indexes a file"}],
      "downloads": ["CompendiumLabs/bge-small-en-v1.5-gguf/*"],
      "warnings":  [{"match": "gpu layers auto: *", "why": "the MoE row loads past VRAM"}],
      "errors":    [],
      "toasts":    [{"match": "[toast /] Model switched to *"}]
    }

Each entry is a glob matched case-insensitively against the whole event subject;
only ``*`` and ``?`` are wildcards, so ``[`` is literal (a toast starts with one): the model name (file name without ``.gguf``) for model-load, the
repository/file or pull spec for download, the message text after the logger
name for warning and error, and the text after ``gui: `` for toast.

Exit codes: 0 every event is EXPECTED; 1 at least one is UNEXPLAINED (or, with
``--strict``, a ``models``/``downloads`` entry matched nothing); 2 the audit
could not run (a log missing, unreadable or holding no localm log record, a
``--home`` that is not a folder, or a malformed manifest). 2 is never reported
as 0.

Usage:
    python scripts/log_audit.py --manifest run.json <log> [<log> ...]
    python scripts/log_audit.py --manifest run.json --home <data dir> <data dir>/logs/localm_*.log
    python scripts/log_audit.py --manifest run.json --json audit.json <log>
    python scripts/log_audit.py --manifest run.json --strict <log>
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

MANIFEST_KEYS = ("run", "models", "downloads", "warnings", "errors", "toasts")
KIND_KEYS = {"model-load": "models", "download": "downloads", "warning": "warnings",
             "error": "errors", "toast": "toasts"}

RECORD_RE = re.compile(
    r"^(?P<prefix>.*?)(?P<time>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) "
    r"(?P<level>DEBUG|INFO|WARNING|ERROR|CRITICAL)\s+(?P<logger>[\w.\-]+):(?: (?P<msg>.*))?$")
GUI_TOAST_RE = re.compile(r"^gui: (?P<text>.*)$")
EMBED_READY_RE = re.compile(r"^embedding model ready: (?P<model>.+?) \(dim=")
SWITCH_LOADED_RE = re.compile(r"^switch_engine: loaded '(?P<model>[^']+)'")
READY_RE = re.compile(r"^(?:hf model|exllama model|sd\.cpp model|sd\.cpp upscaler|speech model"
                      r"|reranker|whisper model) ready: (?P<model>.+?)(?: \(.*\))?\s*$")
JOB_MODEL_RE = re.compile(r"^(?:comfy|koboldcpp music): job uses model (?P<model>.+?)\s*$")
DOWNLOADING_RE = re.compile(r"^downloading .*?model (?P<id>\S+) \(one-time\)")
PULL_STARTED_RE = re.compile(r"^Model pull (?P<id>.+?) started\b")
PULL_LOG_RE = re.compile(r"^pull: downloading (?P<id>.+?)\s*$")

NATIVE_PREFIX = r"(?:llama|ggml|clip|mtmd|gguf|whisper|sd|common|load|sched|graph|init|decode|print_info|alloc)\w*"
LLAMA_LOADER_RE = re.compile(r"^llama_model_loader: loaded meta data with .* from (?P<path>.+?) \(version")
MTMD_LOADED_RE = re.compile(rf"^{NATIVE_PREFIX}: loaded \d+ tensors from (?P<path>.+?)\s*$")
NATIVE_ERROR_RE = re.compile(rf"^{NATIVE_PREFIX}: .*\b(?:error|failed|failure)\b", re.IGNORECASE)
BACKEND_ERROR_RE = re.compile(r"\b(?:ROCm|CUDA|HIP|Vulkan|Metal|SYCL|cuBLAS|hipBLAS) error\b")
NATIVE_ABORT_RE = re.compile(r"^\S+\.(?:c|cc|cpp|cu|h|hpp|m|mm):\d+: ")
FATAL_RE = re.compile(r"GGML_ASSERT|^Exception in thread |^Exception ignored in|^Fatal Python error"
                      r"|^Windows fatal exception|^Segmentation fault")
TRACEBACK_RE = re.compile(r"^Traceback \(most recent call last\):")
CHAINED_RE = re.compile(r"^(?:During handling of the above exception, another exception occurred:"
                        r"|The above exception was the direct cause of the following exception:)$")
PY_WARNING_RE = re.compile(r"^.+\.py:\d+: (?P<cat>\w*Warning): (?P<msg>.*)$")
TQDM_RE = re.compile(r"^(?P<name>[^\s|][^|]*?):\s+\d{1,3}%\|")

# Files localm writes only when something crashed or hung, relative to its data dir.
CRASH_ARTEFACT_GLOBS = ("run/server-crash-trace*.txt", "logs/crash_*.txt", "logs/hang_*.log")
# Files a server leaves behind when it did not finish a clean stop; empty or not.
CRASH_MARKER_GLOBS = ("run/server-crash*.marker", "run/server-crash*.stopping")


class AuditError(Exception):
    """The audit cannot run (bad manifest, unreadable or non-localm log)."""


@dataclass
class Event:
    kind: str
    subject: str
    log: str
    line: int
    time: str = ""
    lines: list[int] = field(default_factory=list)
    verdict: str = "UNEXPLAINED"
    why: str = ""
    matched: str = ""


@dataclass
class Rule:
    key: str
    pattern: str
    why: str = ""
    hits: int = 0


def load_manifest(path: Path) -> dict[str, list[Rule]]:
    """The manifest's rules per key. Raises AuditError on a file that is not a
    JSON object, an unknown key, or an entry that is neither a glob string nor
    an object with a string ``match``."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as e:
        raise AuditError(f"manifest {path} is not readable JSON: {e}") from e
    if not isinstance(data, dict):
        raise AuditError(f"manifest {path} is not a JSON object")
    unknown = sorted(set(data) - set(MANIFEST_KEYS))
    if unknown:
        raise AuditError(f"manifest {path} has unknown key(s): {', '.join(unknown)}")
    rules: dict[str, list[Rule]] = {k: [] for k in KIND_KEYS.values()}
    for key in KIND_KEYS.values():
        entries = data.get(key, [])
        if not isinstance(entries, list):
            raise AuditError(f"manifest {key!r} must be a list")
        for i, ent in enumerate(entries):
            if isinstance(ent, str) and ent:
                rules[key].append(Rule(key, ent))
            elif isinstance(ent, dict) and isinstance(ent.get("match"), str) and ent["match"]:
                extra = sorted(set(ent) - {"match", "why"})
                if extra:
                    raise AuditError(f"manifest {key}[{i}] has unknown field(s): {', '.join(extra)}")
                rules[key].append(Rule(key, ent["match"], str(ent.get("why") or "")))
            else:
                raise AuditError(f"manifest {key}[{i}] is not a glob or {{\"match\": ...}}")
    return rules


def _model_name(path_or_name: str) -> str:
    name = re.split(r"[\\/]", path_or_name.strip().strip("'\""))[-1]
    return name[:-5] if name.lower().endswith(".gguf") else name


def parse_log(path: Path) -> tuple[list[Event], int, int]:
    """``(events, line_count, record_count)`` for one log. Raises AuditError
    when the file cannot be read or holds no localm log record."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        raise AuditError(f"cannot read log {path}: {e}") from e
    lines = re.split(r"\r\n|\r|\n", text)
    events: list[Event] = []
    records = 0
    cur_time = ""
    # The next non-empty line may be the traceback of the record before it.
    traceback_owned = False
    in_owned_traceback = False
    pending: Optional[Event] = None
    last_load: Optional[Event] = None
    last_tqdm: Optional[Event] = None
    label = str(path)

    def add(kind: str, subject: str, n: int) -> Event:
        ev = Event(kind, subject, label, n, cur_time, [n])
        events.append(ev)
        return ev

    def add_load(subject: str, n: int) -> None:
        nonlocal last_load
        if last_load is not None and last_load.subject.lower() == subject.lower():
            last_load.lines.append(n)
            return
        last_load = add("model-load", subject, n)

    def scan_raw(line: str, n: int) -> None:
        nonlocal pending, last_tqdm, traceback_owned, in_owned_traceback
        if not line:
            if not in_owned_traceback:
                traceback_owned = False
            return
        if pending is not None:
            pending.subject = f"{pending.subject} {line.strip()}".strip()
            pending.lines.append(n)
            pending = None
        if TRACEBACK_RE.match(line):
            in_owned_traceback = traceback_owned
            traceback_owned = False
            if not in_owned_traceback:
                add("error", line, n)
                return
        elif in_owned_traceback and CHAINED_RE.match(line):
            traceback_owned = True
            return
        else:
            traceback_owned = False
        if BACKEND_ERROR_RE.search(line) or NATIVE_ABORT_RE.match(line) or FATAL_RE.search(line):
            add("error", line, n)
            return
        lm = LLAMA_LOADER_RE.match(line) or MTMD_LOADED_RE.match(line)
        if lm:
            add_load(_model_name(lm.group("path")), n)
            return
        tq = TQDM_RE.match(line)
        if tq:
            name = tq.group("name").strip()
            if last_tqdm is not None and last_tqdm.subject == name and \
                    last_tqdm.lines[-1] >= n - 3:
                last_tqdm.lines.append(n)
            else:
                last_tqdm = add("download", name, n)
            return
        pw = PY_WARNING_RE.match(line)
        if pw:
            add("warning", f"{pw.group('cat')}: {pw.group('msg')}", n)
            return
        if NATIVE_ERROR_RE.match(line):
            add("error", line, n)

    for n, raw in enumerate(lines, start=1):
        m = RECORD_RE.match(raw)
        if m:
            records += 1
            if m.group("prefix").strip():
                scan_raw(m.group("prefix").rstrip(), n)
            pending = None
            cur_time = m.group("time")
            level, msg = m.group("level"), (m.group("msg") or "").rstrip()
            flagged = level in ("WARNING", "ERROR", "CRITICAL")
            traceback_owned = flagged
            in_owned_traceback = False
            if flagged:
                toast = GUI_TOAST_RE.match(msg)
                if toast:
                    ev = add("toast", toast.group("text"), n)
                else:
                    ev = add("warning" if level == "WARNING" else "error", msg, n)
                if not ev.subject.strip():
                    pending = ev
            for rx in (EMBED_READY_RE, SWITCH_LOADED_RE, READY_RE, JOB_MODEL_RE):
                lm = rx.match(msg)
                if lm:
                    add_load(_model_name(lm.group("model")), n)
            for rx in (DOWNLOADING_RE, PULL_STARTED_RE, PULL_LOG_RE):
                dm = rx.match(msg)
                if dm:
                    add("download", dm.group("id"), n)
            continue
        scan_raw(raw.rstrip(), n)
    if records == 0:
        raise AuditError(f"{path} holds no localm log record (not a --debug log?)")
    return events, len(lines), records


def crash_artefacts(home: Path) -> list[Event]:
    """One error event per non-empty crash or hang trace in the localm data dir
    *home* (``CRASH_ARTEFACT_GLOBS``) and per server crash marker or stopping
    record there, whatever its size (``CRASH_MARKER_GLOBS``). Raises AuditError
    when *home* is not a folder."""
    if not home.is_dir():
        raise AuditError(f"--home {home} is not a folder")
    out = []
    for pattern in CRASH_ARTEFACT_GLOBS:
        for p in sorted(home.glob(pattern)):
            try:
                size = p.stat().st_size
            except OSError as e:
                raise AuditError(f"cannot read {p}: {e}") from e
            if size > 0:
                out.append(Event("error", f"crash artefact {p.relative_to(home).as_posix()}"
                                          f" ({size} bytes)", str(p), 0, "", [0]))
    for pattern in CRASH_MARKER_GLOBS:
        for p in sorted(home.glob(pattern)):
            out.append(Event("error", f"crash marker {p.relative_to(home).as_posix()}",
                             str(p), 0, "", [0]))
    return out


def judge(events: list[Event], rules: dict[str, list[Rule]]) -> None:
    """Mark each event EXPECTED when a rule of its kind matches its subject."""
    for ev in events:
        subject = ev.subject.lower()
        for rule in rules[KIND_KEYS[ev.kind]]:
            if fnmatch.fnmatchcase(subject, rule.pattern.lower().replace("[", "[[]")):
                ev.verdict = "EXPECTED"
                ev.why = rule.why
                ev.matched = rule.pattern
                rule.hits += 1
                break


def audit(logs: list[Path], manifest: Path, homes: tuple[Path, ...] = ()
          ) -> tuple[list[Event], dict[str, list[Rule]], list[dict]]:
    """Parse and judge every log, plus the crash artefacts of each data dir in
    *homes*. Returns ``(events, rules, per_log_info)``."""
    rules = load_manifest(manifest)
    events: list[Event] = []
    info = []
    for log in logs:
        evs, n_lines, n_records = parse_log(log)
        info.append({"log": str(log), "lines": n_lines, "records": n_records,
                     "events": len(evs)})
        events.extend(evs)
    for home in homes:
        events.extend(crash_artefacts(home))
    judge(events, rules)
    return events, rules, info


def _short(text: str, limit: int = 220) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("logs", nargs="+", type=Path, help="localm debug log file(s)")
    ap.add_argument("--manifest", required=True, type=Path, help="per-run manifest (JSON)")
    ap.add_argument("--strict", action="store_true",
                    help="also fail when a models or downloads entry matched nothing")
    ap.add_argument("--json", type=Path, help="also write the events as JSON to this file")
    ap.add_argument("--home", type=Path, action="append", default=[],
                    help="localm data dir of the run; its crash and hang traces and server crash"
                         " markers become error events; audit after the server stopped (repeatable)")
    args = ap.parse_args(argv)
    try:
        events, rules, info = audit(args.logs, args.manifest, tuple(args.home))
    except AuditError as e:
        print(f"log audit could not run: {e}")
        return 2
    for item in info:
        print(f"log: {item['log']} ({item['lines']} lines, {item['records']} records,"
              f" {item['events']} events)")
    for ev in events:
        where = f"{Path(ev.log).name}:{ev.line}"
        if len(ev.lines) > 1:
            where += f"(+{len(ev.lines) - 1})"
        note = f"  [{ev.why or ev.matched}]" if ev.verdict == "EXPECTED" else ""
        print(f"{ev.verdict:<11}  {ev.kind:<10}  {where:<40}  {ev.time[11:19] if ev.time else '--:--:--'}"
              f"  {_short(ev.subject)}{note}")
    unseen = [r for key in ("models", "downloads") for r in rules[key] if r.hits == 0]
    for r in unseen:
        print(f"{'NOT SEEN':<11}  {r.key:<10}  manifest entry {r.pattern!r} matched no event")
    bad = [e for e in events if e.verdict == "UNEXPLAINED"]
    counts = {}
    for e in bad:
        counts[e.kind] = counts.get(e.kind, 0) + 1
    print(f"{len(events)} event(s): {len(events) - len(bad)} expected, {len(bad)} unexplained"
          + (" (" + ", ".join(f"{v} {k}" for k, v in sorted(counts.items())) + ")" if counts else "")
          + (f"; {len(unseen)} manifest model/download entr{'y' if len(unseen) == 1 else 'ies'}"
             " not seen" if unseen else ""))
    if args.json:
        args.json.write_text(json.dumps({
            "logs": info,
            "events": [{"verdict": e.verdict, "kind": e.kind, "subject": e.subject,
                        "log": e.log, "line": e.line, "lines": e.lines, "time": e.time,
                        "matched": e.matched, "why": e.why} for e in events],
            "not_seen": [{"key": r.key, "match": r.pattern} for r in unseen],
        }, indent=2), encoding="utf-8")
    if bad or (args.strict and unseen):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

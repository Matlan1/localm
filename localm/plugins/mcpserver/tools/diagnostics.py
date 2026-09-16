# SPDX-License-Identifier: AGPL-3.0-or-later
"""MCP tools that report on this install: ``server_activity``,
``system_stats`` and ``run_doctor``."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict

from ..server import _text_result


def _child_identity_env() -> dict:
    """Env for a ``-m localm`` helper process so it is THE SAME localm as this
    server: same data home, same code.

    Both are otherwise re-resolved from ambient state at every process
    boundary: the data home falls back to a contained default derived from the
    running code's location when nothing is configured, and ``-m`` puts the
    child's cwd first on ``sys.path``, where a ``localm/`` directory in that
    cwd (any other checkout) silently swaps which CODE runs.

    LOCALM_HOME pins the data home; PYTHONSAFEPATH stops ``-m`` from putting
    the child's cwd on ``sys.path``; the PYTHONPATH entry keeps this server's
    own package importable regardless of cwd (PYTHONSAFEPATH only drops the
    implicit cwd entry, explicit PYTHONPATH entries still apply)."""
    import localm as _pkg
    from localm.config import home_dir
    env = dict(os.environ)
    env["LOCALM_HOME"] = str(home_dir())
    env["PYTHONSAFEPATH"] = "1"
    pkg_root = str(Path(_pkg.__file__).resolve().parent.parent)
    prior = env.get("PYTHONPATH")
    env["PYTHONPATH"] = pkg_root + ((os.pathsep + prior) if prior else "")
    return env


def build() -> Dict[str, dict]:
    """``server_activity``, ``system_stats`` and ``run_doctor``."""
    def server_activity(args: dict) -> dict:
        """What any running localm server of this install is doing.

        This MCP server is a SEPARATE PROCESS from the HTTP/GUI server and
        shares no memory with it, so it finds the running instances on disk and
        asks each one over HTTP.

        The states are kept apart. "No server is running" is not "nothing is
        running" - there is nothing to ask. "Could not reach it" is not "it is
        idle". Only a server that actually answered can report an empty list,
        and only that case says nothing is running.
        """
        from localm import instances
        from localm.config import home_dir
        from localm.selfclient import read_activity

        # include_token=True: this call ASKS each discovered instance over HTTP
        # (an internal, non-display use), so it needs the attach token a
        # genuinely open (keyless) instance's middleware requires. Never do this
        # for anything a human reads (e.g. `localm ps`, which keeps the
        # default-stripped snapshot()).
        rows = instances.snapshot(home_dir(), include_token=True)
        if not rows:
            return _text_result(
                "No localm server of this install is running, so there is "
                "nothing to ask. This is not the same as a server reporting "
                "that it is idle, and a server started from a different localm "
                "install keeps its own data directory and is not asked here.")
        lines = []
        for e in rows:
            from localm.bindhost import self_connect_host, url_host
            _h = url_host(self_connect_host(e.get("host")))
            where = f"{e.get('scheme', 'http')}://{_h}:{e.get('port')}"
            if not e.get("alive"):
                lines.append(f"{where}: registered but not responding; "
                             f"its activity is unknown.")
                continue
            state, payload = read_activity(
                e.get("scheme", "http"), e.get("port"), e.get("token"),
                e.get("host"))
            if state == "unreachable":
                lines.append(f"{where}: could not be reached ({payload}); "
                             f"its activity is unknown.")
            elif state == "unauthorized":
                # Matches the "could not be X" register the other failure
                # branches use, not a "needs a key" requirement statement: this
                # process cannot tell what the server is doing right now.
                lines.append(f"{where}: could not be asked (it requires an "
                             f"API key this process does not have); its "
                             f"activity is unknown.")
            elif state == "unsupported":
                lines.append(f"{where}: does not report activity (older "
                             f"localm); its activity is unknown.")
            elif state != "ok":
                lines.append(f"{where}: could not be read (HTTP {payload}); "
                             f"its activity is unknown.")
                continue
            else:
                ops = (payload or {}).get("operations") or []
                now = (payload or {}).get("now")
                if not ops:
                    lines.append(f"{where}: idle, nothing running.")
                    continue
                lines.append(f"{where}: {len(ops)} operation(s)")
                for op in ops:
                    label = op.get("label") or op.get("kind") or "operation"
                    bits = [op.get("status") or "?"]
                    pct = op.get("pct")
                    # Absent, not zero: an operation that has reported no
                    # progress is at an unknown percentage.
                    if isinstance(pct, (int, float)):
                        bits.append(f"{pct:.0f}%")
                    created = op.get("created_at")
                    # Age against the SERVER's clock; this process may not
                    # share it.
                    if isinstance(now, (int, float)) and isinstance(created, (int, float)):
                        bits.append(f"{int(max(0, now - created))}s elapsed")
                    lines.append(f"  - {label} [{', '.join(bits)}]")
        return _text_result("\n".join(lines))

    def system_stats(args: dict) -> dict:
        from localm.sysstats import system_stats as _stats
        # A ONE-SHOT call, unlike the GUI's repeating poll: without
        # wait_first_vram the "Live ... VRAM" promise below omits VRAM on a cold
        # first call while the background probe is still running, because there
        # is no later poll here to pick up the landed reading. MCP stdio serves
        # one request at a time with no event loop to stall.
        return _text_result(json.dumps(_stats(wait_first_vram=True)))

    def run_doctor(args: dict) -> dict:
        cmd = [sys.executable, "-m", "localm", "doctor"]
        try:
            # env=_child_identity_env(): a doctor that re-resolves home/code
            # from ambient state reports on the WRONG install whenever this
            # server's home came from its own location (source-checkout setup).
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60,
                                  env=_child_identity_env())
            output = proc.stdout
            if proc.stderr:
                output += "\n\nStderr:\n" + proc.stderr
            return _text_result(output)
        except subprocess.TimeoutExpired:
            return _text_result("Doctor task timed out after 60s", is_error=True)
        except Exception as e:
            return _text_result(f"Failed to run doctor: {e}", is_error=True)

    return {
        "server_activity": {
            "description": (
                "What any running localm server of this install is currently "
                "doing: model downloads, indexing, media generation. Check this "
                "BEFORE starting a long operation - a pull started from the "
                "browser or another client is otherwise invisible here, and "
                "starting a second one wastes bandwidth and disk. Distinguishes "
                "'no server is running' and 'could not reach it' from 'the "
                "server says it is idle'; only the last one means nothing is "
                "happening."
            ),
            "inputSchema": {"type": "object", "properties": {}},
            "annotations": {"readOnlyHint": True, "title": "Server activity"},
            "handler": server_activity,
        },
        "system_stats": {
            "description": (
                "Live CPU/RAM/VRAM/GPU load. Use this BEFORE picking a model or "
                "quant for a task: if VRAM is tight, prefer a smaller quant "
                "(Q4/Q6 over Q8) rather than skipping the task or degrading "
                "quality - and prefer evicting the current model over settling "
                "for a worse-fit one when the task genuinely needs it."
            ),
            "inputSchema": {"type": "object", "properties": {}},
            "annotations": {"readOnlyHint": True, "title": "System stats"},
            "handler": system_stats,
        },
        "run_doctor": {
            "description": "Check system requirements and report any issues (runs localm doctor).",
            "inputSchema": {"type": "object", "properties": {}},
            "annotations": {"readOnlyHint": True, "title": "Run doctor"},
            "handler": run_doctor,
        },
    }

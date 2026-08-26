# SPDX-License-Identifier: AGPL-3.0-or-later
"""The scheduled-jobs VRAM gate must not swallow failures silently.

runner._load_engine wrapping its whole live-engine-reuse + VRAM-gate block in a
bare `try/except Exception: pass` mutes the cause entirely. Falling through to a
fresh load when the gate cannot run is a legitimate best-effort path, but the
cause has to be surfaced - a log line is the right altitude, not silence, and
not a hard failure either.
"""

import pytest

from localm.plugins.builtin.jobs import runner


class LiveEngine:
    loaded = True
    display_name = "already-loaded-other"

    def unload(self):
        pass


def test_vram_gate_failure_is_logged_not_swallowed(monkeypatch):
    import localm.inference.http_server as hs
    monkeypatch.setattr(hs, "_engine", LiveEngine())

    # Make the VRAM probe inside the gate raise, so the block fails.
    def boom():
        raise RuntimeError("vram probe unavailable")
    monkeypatch.setattr("localm.discover.vram_info", boom)
    # The fall-through path then fails to resolve the model: the gate degraded
    # rather than crashed on the swallowed exception.
    monkeypatch.setattr("localm.model_manager.get_model_info", lambda n: None)

    calls = []
    from localm.debuglog import logger as dbg
    monkeypatch.setattr(dbg, "debug", lambda *a, **k: calls.append(a))

    with pytest.raises(RuntimeError, match="model not found"):
        runner._load_engine("wanted-model")

    assert calls, "the swallowed VRAM-gate failure must be logged, not silently passed"
    joined = " ".join(str(a) for a in calls).lower()
    assert "vram" in joined or "gate" in joined or "reuse" in joined

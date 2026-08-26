# SPDX-License-Identifier: AGPL-3.0-or-later
"""NEW-DEBUG-LOG-PROMISED-BUT-ABSENT: `_runner.py`, `_hf_runner.py` and
`_embedder_runner.py` all appended "(full trace in the debug log)" to a
native-fault message UNCONDITIONALLY. The trace itself goes to
`logger.error`, but a debug log FILE only exists once `enable_debug()` has
run, which is off by default - so on a default install the message named a
file that was never created, misdirecting every default-mode user chasing
a crash. Verified live: a positive control with `--debug` on DOES produce
that log line; a default-mode run does not.
"""

from pathlib import Path

import localm.debuglog as debuglog


class TestNativeFaultHint:
    def test_debug_off_does_not_promise_a_debug_log(self, monkeypatch):
        monkeypatch.delenv("LOCALM_DEBUG", raising=False)
        hint = debuglog.native_fault_hint()
        assert "debug log" not in hint, (
            "must not claim a debug log exists when debug_enabled() is False")
        assert "--debug" in hint, "must say how to actually get one"

    def test_debug_on_names_the_debug_log(self, monkeypatch):
        monkeypatch.setenv("LOCALM_DEBUG", "1")
        assert debuglog.native_fault_hint() == "full trace in the debug log"


class TestAllThreeRunnersUseTheSharedHint:
    """Source-level guard: the three sites this bug was found in must all
    route through native_fault_hint() rather than re-hardcoding the old
    unconditional text (or a future fourth site being added the old way)."""

    FILES = [
        "localm/inference/_embedder_runner.py",
        "localm/inference/backends/llamacpp/_runner.py",
        "localm/inference/backends/_hf_runner.py",
    ]

    def _repo_root(self) -> Path:
        return Path(__file__).resolve().parents[1]

    def test_no_hardcoded_unconditional_debug_log_claim(self):
        for rel in self.FILES:
            body = (self._repo_root() / rel).read_text(encoding="utf-8")
            assert "(full trace in the debug log)" not in body, (
                f"{rel} hardcodes the unconditional claim again - use "
                "native_fault_hint() instead")

    def test_every_site_calls_native_fault_hint(self):
        for rel in self.FILES:
            body = (self._repo_root() / rel).read_text(encoding="utf-8")
            assert "native_fault_hint()" in body, (
                f"{rel} no longer routes its native-fault message through "
                "native_fault_hint()")

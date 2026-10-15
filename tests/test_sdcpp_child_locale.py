# SPDX-License-Identifier: AGPL-3.0-or-later
"""The stable-diffusion.cpp worker keeps non-ASCII prompt bytes intact through
the native tokenizer's byte-wise tolower."""

import subprocess
import sys
import textwrap

import pytest

from localm.media.sdcpp import _child

_PROMPT = "Ein weißer Fuchs läuft über einen vereisten See"

_CRT_PROBE = textwrap.dedent("""
    import ctypes, locale, sys
    from localm.media.sdcpp import _child
    crt = ctypes.CDLL("ucrtbase")
    raw = sys.argv[1].encode("utf-8")

    def lowered():
        return bytes(crt.tolower(b) & 0xFF for b in raw)

    locale.setlocale(locale.LC_CTYPE, ".1252")
    before = lowered()
    _child._use_c_ctype_locale()
    after = lowered()
    print(before.hex())
    print(after.hex())
""")


@pytest.mark.skipif(sys.platform != "win32", reason="the code-page C runtime locale is Windows-only")
def test_the_worker_keeps_utf8_prompt_bytes_through_tolower():
    out = subprocess.run([sys.executable, "-c", _CRT_PROBE, _PROMPT],
                         capture_output=True, text=True, timeout=60, check=True)
    before_hex, after_hex = out.stdout.split()
    before, after = bytes.fromhex(before_hex), bytes.fromhex(after_hex)
    with pytest.raises(UnicodeDecodeError):
        before.decode("utf-8")
    assert after.decode("utf-8") == _PROMPT.lower()


def test_the_worker_entry_point_sets_the_locale_before_serving(monkeypatch):
    calls = []
    monkeypatch.setattr(_child, "_use_c_ctype_locale", lambda: calls.append("locale"))
    monkeypatch.setattr(_child, "_arm_crash_trace", lambda path: None)
    import localm._mp_spawn as spawn
    import localm.debuglog as debuglog
    for name in ("install_parent_death_watchdog", "ignore_interrupt_signals",
                 "suppress_native_error_dialogs"):
        monkeypatch.setattr(spawn, name, lambda: None)
    monkeypatch.setattr(debuglog, "attach_child_logging", lambda: None)

    class _Queue:
        def get(self):
            calls.append("get")
            return None

    import threading
    _child.worker_main(_Queue(), _Queue(), threading.Event())
    assert calls[:2] == ["locale", "get"]


def test_other_platforms_leave_the_locale_alone(monkeypatch):
    import locale
    seen = []
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(locale, "setlocale", lambda *a: seen.append(a))
    _child._use_c_ctype_locale()
    assert seen == []

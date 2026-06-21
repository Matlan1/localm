# SPDX-License-Identifier: AGPL-3.0-or-later
"""The launcher window must keep a FIXED width: a long status line or a long
model name in the dropdown must never widen it (issues.txt)."""

import importlib.util
from pathlib import Path

import pytest

_LAUNCHER = Path(__file__).resolve().parents[1] / "launcher.pyw"


def _load_launcher():
    spec = importlib.util.spec_from_file_location("localm_launcher_mod", _LAUNCHER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_ellipsize_clips_long_text():
    mod = _load_launcher()
    short = "all good"
    assert mod._ellipsize(short) == short
    long = "No models yet - Import one, or just launch the Web GUI and add one"
    out = mod._ellipsize(long, 46)
    assert len(out) <= 46
    assert out.endswith("…")


def test_ellipsize_boundary():
    mod = _load_launcher()
    exact = "x" * 46
    assert mod._ellipsize(exact, 46) == exact          # fits exactly, untouched
    assert mod._ellipsize("y" * 47, 46).endswith("…")  # one over -> clipped


def test_window_width_is_fixed_against_long_content():
    """Build the real launcher and confirm a huge status message + a huge model
    name leave the window width unchanged. Skips when no display is available."""
    mod = _load_launcher()
    tk = mod.tk
    try:
        app = mod.Launcher()
    except tk.TclError as e:
        pytest.skip(f"no display for tkinter: {e}")
    try:
        app.withdraw()
        app.update_idletasks()
        base = app.winfo_reqwidth()

        # Slam in pathological content the user complained about.
        app.status_msg("X" * 400)
        app.model.set("Llama3.3-8B-Instruct-Thinking-Heretic-Uncensored-"
                      "Claude-4.5-Opus-High-Reasoning.i1-Q6_K" * 3)
        app.model_box["values"] = [app.model.get()]
        app.update_idletasks()

        assert app.winfo_reqwidth() == base, "window width must not grow"
        # maxsize width is pinned to the base width.
        assert app.maxsize()[0] == base
        assert app.minsize()[0] == base
    finally:
        app.destroy()

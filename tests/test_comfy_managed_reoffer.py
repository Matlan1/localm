# SPDX-License-Identifier: AGPL-3.0-or-later
"""Oracle for the managed-ComfyUI re-offer: the __func__ regression re-offers
localm's own managed ComfyUI ONCE.

When T1's ``__func__`` detection fires AND no managed ComfyUI is installed AND
localm has not already made this offer, localm ALSO surfaces a one-time offer to
set up its OWN managed, patched ComfyUI (``localm comfy setup``) as the durable
fix, alongside T1's fix-this-run shim offer. Each check carries a built-in
negative case so it fails on known-bad work:

  - GATE: offered only for the known __func__ regression, only when NO managed
    instance is installed, only when not already offered.
  - ONCE: the persisted ``comfy_managed_setup_offered`` flag makes it never nag.
  - MOOT: a managed instance already installed -> no offer at all.
  - T1 INTACT: the shim intro + prompt still fire; the addition does not break it.

Config + managed-comfy paths are pinned to a throwaway LOCALM_HOME so no test
touches real data (config.py freezes CONFIG_FILE at import; managed_comfy uses
the lazy home_dir()).
"""

import pytest


_REGRESSION = "ComfyUI execution failed: 'function' object has no attribute '__func__'"
_UNRELATED = "ComfyUI execution failed: CUDA out of memory (node KSampler)"


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A throwaway LOCALM_HOME with config.py's frozen paths redirected to it, so
    load_config/save_config AND managed_comfy_paths() resolve to the same tmp dir."""
    import localm.config as cfg
    h = tmp_path / ".localm"
    h.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LOCALM_HOME", str(h))          # managed_comfy home_dir() (lazy)
    monkeypatch.setattr(cfg, "HOME_DIR", h)            # load/save_config (frozen)
    monkeypatch.setattr(cfg, "MODELS_DIR", h / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", h / "config.json")
    return h


def _install_managed():
    """Create the on-disk markers is_managed_comfy_installed() checks (main.py,
    the managed venv interpreter, and the completion marker - the first two alone
    mean "still installing"), so the REAL detector reports installed."""
    from localm.media.managed_comfy import managed_comfy_paths
    from localm.media.managed_comfy_provision import MARKER_FILENAME
    p = managed_comfy_paths()
    p.main_py.parent.mkdir(parents=True, exist_ok=True)
    p.main_py.write_text("# fake ComfyUI entry point\n", encoding="utf-8")
    p.venv_python.parent.mkdir(parents=True, exist_ok=True)
    p.venv_python.write_text("", encoding="utf-8")
    (p.root / MARKER_FILENAME).write_text("{}", encoding="utf-8")


# ------------------------------ pure helpers --------------------------------------

def test_offer_gated_on_regression_no_managed_not_offered(home):
    from localm.media.comfy_client import should_offer_managed_comfy_setup
    # regression + no managed install + not yet offered -> offer
    assert should_offer_managed_comfy_setup(_REGRESSION) is True
    # NEGATIVE: an unrelated error is never the durable-fix trigger
    assert should_offer_managed_comfy_setup(_UNRELATED) is False
    assert should_offer_managed_comfy_setup("") is False
    assert should_offer_managed_comfy_setup(None) is False


def test_offer_false_when_managed_already_installed(home):
    from localm.media.comfy_client import should_offer_managed_comfy_setup
    from localm.media.managed_comfy import is_managed_comfy_installed
    _install_managed()
    assert is_managed_comfy_installed() is True          # real detector, not mocked
    # MOOT: localm already routes to its patched managed instance -> no offer
    assert should_offer_managed_comfy_setup(_REGRESSION) is False


def test_offer_marked_once_then_never_again(home):
    from localm.media.comfy_client import (mark_managed_comfy_setup_offered,
                                           should_offer_managed_comfy_setup)
    from localm.config import load_config
    assert should_offer_managed_comfy_setup(_REGRESSION) is True
    mark_managed_comfy_setup_offered()
    # persisted across a fresh load (round-trip), and never offered again
    assert load_config().get("comfy_managed_setup_offered") is True
    assert should_offer_managed_comfy_setup(_REGRESSION) is False


def test_offer_message_points_at_the_setup_command():
    from localm.media.comfy_client import managed_comfy_setup_offer_message
    msg = managed_comfy_setup_offer_message()
    assert "localm comfy setup" in msg
    # frames it as the durable fix, not the fix-this-run shim
    assert "managed" in msg.lower()


# ------------------------------ CLI offer point -----------------------------------

def _drive_cli(monkeypatch, answer, message, *, spawned_pid=None):
    """Run _maybe_apply_func_shim_and_retry with an interactive console, capturing
    everything printed and whether the shim prompt (console.input) was shown."""
    from localm.cli import media
    from localm.media import comfy_client as cc
    cc._func_shim_once = False

    printed = []
    prompted = {"v": False}

    def _cap_print(*args, **kw):
        printed.append(" ".join(str(a) for a in args))

    def _cap_input(*a, **k):
        prompted["v"] = True
        return answer

    monkeypatch.setattr(media, "_is_interactive", lambda: True)
    monkeypatch.setattr(media.console, "print", _cap_print)
    monkeypatch.setattr(media.console, "input", _cap_input)
    monkeypatch.setattr(cc, "spawned_pid", lambda *a, **k: spawned_pid)
    monkeypatch.setattr(cc, "restart_comfy", lambda *a, **k: (True, "ok"))
    monkeypatch.setattr(media, "_remember_func_shim", lambda: None)

    retried = {"n": 0}

    def retry():
        retried["n"] += 1
        return True, "Saved"

    ok, out = media._maybe_apply_func_shim_and_retry(
        message, "http://127.0.0.1:8188", retry)
    cc._func_shim_once = False
    return {"ok": ok, "out": out, "printed": "\n".join(printed),
            "prompted": prompted["v"], "retried": retried["n"]}


def test_cli_reoffer_surfaced_alongside_shim_and_marks_offered(home, monkeypatch):
    from localm.config import load_config
    # __func__ + no managed + not offered, user applies the shim ("o"nce)
    r = _drive_cli(monkeypatch, "o", _REGRESSION, spawned_pid=123)
    # the durable-fix offer appeared...
    assert "localm comfy setup" in r["printed"]
    # ...alongside the shim offer (prompt shown + retried)
    assert r["prompted"] is True
    assert r["retried"] == 1
    assert r["ok"] and r["out"] == "Saved"
    # and the one-time flag is now persisted
    assert load_config().get("comfy_managed_setup_offered") is True


def test_cli_reoffer_not_repeated_but_shim_still_offered(home, monkeypatch):
    from localm.media.comfy_client import mark_managed_comfy_setup_offered
    mark_managed_comfy_setup_offered()          # already offered once
    r = _drive_cli(monkeypatch, "n", _REGRESSION)
    # NO repeat of the durable-fix offer (never nags)...
    assert "localm comfy setup" not in r["printed"]
    # ...but the shim offer/prompt is still presented
    assert r["prompted"] is True
    assert "12116" in r["printed"] or "__func__" in r["printed"]


def test_cli_reoffer_skipped_when_managed_installed(home, monkeypatch):
    from localm.config import load_config
    _install_managed()
    r = _drive_cli(monkeypatch, "n", _REGRESSION)
    assert "localm comfy setup" not in r["printed"]      # moot -> no offer
    # the flag was not consumed (nothing to offer), shim still offered
    assert load_config().get("comfy_managed_setup_offered") in (None, False)
    assert r["prompted"] is True


def test_cli_reoffer_absent_for_unrelated_error(home, monkeypatch):
    r = _drive_cli(monkeypatch, "o", _UNRELATED)
    # neither offer for an unrelated error
    assert "localm comfy setup" not in r["printed"]
    assert r["prompted"] is False        # T1 does not even prompt for a non-__func__ error
    assert r["retried"] == 0
    assert r["ok"] is False

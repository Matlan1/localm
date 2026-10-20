# SPDX-License-Identifier: AGPL-3.0-or-later
"""`localm doctor` names the llama.cpp runtime localm loads when an override
supplies it, and the user's chosen backend when a different one is installed,
with the reason and both ways forward."""

from __future__ import annotations

import importlib

from localm import setup_llama

doctor_mod = importlib.import_module("localm.cli.doctor")


def _flat(capsys) -> str:
    return " ".join(capsys.readouterr().out.split())


def _set(monkeypatch, *, choice=None, use=None):
    monkeypatch.setattr(setup_llama, "backend_choice", lambda: choice)
    monkeypatch.setattr(setup_llama, "runtime_in_use",
                        lambda: use or {"dir": "d", "source": "provisioned", "backend": "vulkan"})


def test_silent_when_choice_installed_and_loaded_agree(monkeypatch, capsys):
    _set(monkeypatch)
    doctor_mod._check_backend_choice()
    assert capsys.readouterr().out == ""


def test_names_the_chosen_backend_the_installed_one_and_why(monkeypatch, capsys):
    _set(monkeypatch, choice={"requested": "cuda", "installed": "vulkan",
                              "reason": "the cuda build could not be used here: [no GPU]"})
    doctor_mod._check_backend_choice()
    out = _flat(capsys)
    assert "backend chosen: cuda, but vulkan is installed" in out
    assert "[no GPU]" in out
    assert "localm setup-llama --backend cuda --force" in out
    assert "localm setup-llama --backend vulkan" in out


def test_names_a_binary_dir_override(monkeypatch, capsys):
    _set(monkeypatch, use={"dir": "D:/own/build", "source": "binary_dir", "backend": "vulkan"})
    doctor_mod._check_backend_choice()
    out = _flat(capsys)
    assert "llama.cpp runtime in use: vulkan from the binary_dir setting" in out
    assert "D:/own/build" in out


def test_a_failing_read_is_reported_not_raised(monkeypatch, capsys):
    def boom():
        raise OSError("unreadable")
    monkeypatch.setattr(setup_llama, "runtime_in_use", boom)
    doctor_mod._check_backend_choice()
    assert "could not read the backend choice" in _flat(capsys)

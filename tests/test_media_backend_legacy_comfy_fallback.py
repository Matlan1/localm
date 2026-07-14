# SPDX-License-Identifier: AGPL-3.0-or-later
"""Regression pin (confirmed by the conserve-mode review, 2026-07-13): the real
GUI/CLI generation path (image/music/video backend.py's settings()) used to
resolve `workdir`/`launch_cmd` from the LEGACY GLOBAL `comfy_workdir`/
`comfy_launch_cmd` config keys unconditionally, then pass them as an explicit
override into `comfy_client.ensure_comfy()`. Because ensure_comfy()'s managed-
instance auto-routing (#621) only activates when the CALLER passed no explicit
workdir/launch_cmd ("caller override wins"), any user who ever set the legacy
global comfy_workdir - even long before a managed instance existed - had that
stale value silently defeat managed routing forever, through the ONE realistic
end-to-end path real users actually hit (Settings -> Media / the Generate
button), even though `resolve_comfy_target()`/`default_api_url()` correctly
resolved the managed API URL. `legacy_comfy_value()` (managed_comfy.py) closes
this: it suppresses the legacy global fallback specifically while managed
routing is active, while a genuine PER-PLUGIN override (comfy_blk) still
always wins, exactly like before.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from localm.media import comfy_client
from localm.media import managed_comfy as mc
from localm.plugins.builtin.image import backend as image_backend
from localm.plugins.builtin.music import backend as music_backend
from localm.plugins.builtin.video import backend as video_backend

BACKENDS = [image_backend, music_backend, video_backend]


def _mock_managed_active(monkeypatch, active: bool):
    monkeypatch.setattr(mc, "managed_comfy_active", lambda cfg=None: active)


@pytest.mark.parametrize("backend_mod", BACKENDS)
def test_legacy_workdir_defeats_managed_routing_without_the_fix(backend_mod, monkeypatch):
    """Pin the BROKEN (pre-fix) behavior directly against legacy_comfy_value()
    in isolation, proving the bug this fix closes was real: honouring the
    legacy global value unconditionally reproduces the exact failure."""
    full_config = {"comfy_target": "own", "comfy_workdir": "C:\\Users\\test\\MyOwnComfyUI"}
    _mock_managed_active(monkeypatch, True)
    # The OLD behavior every backend.py used to have, inlined here to prove the
    # new legacy_comfy_value() genuinely changes the outcome for this exact input.
    old_behavior = full_config.get("comfy_workdir", "") or ""
    assert old_behavior == "C:\\Users\\test\\MyOwnComfyUI"
    # The FIX: suppressed to "" while managed routing is active.
    assert mc.legacy_comfy_value("comfy_workdir", full_config) == ""


@pytest.mark.parametrize("backend_mod", BACKENDS)
def test_settings_suppresses_legacy_workdir_when_managed_active(backend_mod, monkeypatch):
    """The real end-to-end path: backend.settings() must NOT leak the stale
    legacy comfy_workdir/comfy_launch_cmd into ensure_comfy() as an explicit
    override while the managed instance is active - this is exactly the
    scenario the review's live repro reproduced (an existing comfy_workdir
    survives `localm comfy setup`, comfy_target stays at its default 'own')."""
    _mock_managed_active(monkeypatch, True)
    full_config = {
        "comfy_target": "own",
        "comfy_workdir": "C:\\Users\\test\\MyOwnComfyUI",
        "comfy_launch_cmd": "C:\\Users\\test\\MyOwnComfyUI\\launch.bat",
    }
    s = backend_mod.settings(full_config)
    assert s["workdir"] == "", "the stale legacy workdir must not leak through"
    assert s["launch_cmd"] == "", "the stale legacy launch_cmd must not leak through"


@pytest.mark.parametrize("backend_mod", BACKENDS)
def test_settings_still_honours_legacy_workdir_when_managed_inactive(backend_mod, monkeypatch):
    """No managed instance active (the ordinary case today) -> the legacy
    global fallback must keep working exactly as before - no regression for
    the common case of a user pointing localm at their own ComfyUI."""
    _mock_managed_active(monkeypatch, False)
    full_config = {
        "comfy_target": "user",
        "comfy_workdir": "C:\\Users\\test\\MyOwnComfyUI",
        "comfy_launch_cmd": "C:\\Users\\test\\MyOwnComfyUI\\launch.bat",
    }
    s = backend_mod.settings(full_config)
    assert s["workdir"] == "C:\\Users\\test\\MyOwnComfyUI"
    assert s["launch_cmd"] == "C:\\Users\\test\\MyOwnComfyUI\\launch.bat"


@pytest.mark.parametrize("backend_mod", BACKENDS)
def test_settings_per_plugin_override_still_wins_even_when_managed_active(backend_mod, monkeypatch):
    """A genuine PER-PLUGIN comfy.workdir override is a deliberate choice, not
    a stale global default - it must still win even while managed routing is
    active, unlike the legacy global fallback."""
    _mock_managed_active(monkeypatch, True)
    plugin_name = {
        image_backend: "image", music_backend: "music", video_backend: "video",
    }[backend_mod]
    full_config = {
        "comfy_target": "own",
        "comfy_workdir": "C:\\Users\\test\\StaleGlobal",
        "plugins": {plugin_name: {"comfy": {"workdir": "C:\\Users\\test\\DeliberatePerPluginChoice"}}},
    }
    s = backend_mod.settings(full_config)
    assert s["workdir"] == "C:\\Users\\test\\DeliberatePerPluginChoice"


@pytest.mark.parametrize("backend_mod", BACKENDS)
def test_backend_ensure_available_launches_managed_instance_end_to_end(backend_mod, tmp_path):
    """The REAL per-plugin call path (settings() -> _comfy_ensure_available() ->
    ensure_comfy()), not ensure_comfy() called directly - proving managed
    auto-launch actually works through image AND music AND video, not just
    image (the only one previously exercised end to end; a gap found by the
    conserve-mode review, since all three share byte-identical settings()
    resolution but nothing had ever proven that in a real test)."""
    comfy_client._confirmed_alive.clear()
    managed_root = tmp_path / "comfyui"
    fake_paths = mc.ManagedComfyPaths(
        root=managed_root,
        models_dir=tmp_path / "comfyui-models",
        main_py=managed_root / "main.py",
        venv_python=managed_root / "venv" / "Scripts" / "python.exe",
        extra_model_paths=managed_root / "extra_model_paths.yaml",
    )
    full_config = {"comfy_target": "own", "comfy_launch_timeout": 30}
    alive = iter([False, True])
    spawned = {}

    def fake_popen(argv, cwd=None, **kw):
        spawned["argv"], spawned["cwd"] = argv, cwd
        proc = MagicMock()
        proc.poll.return_value = None
        return proc

    with patch("localm.config.load_config", return_value=full_config), \
         patch.object(comfy_client, "_comfy_alive", side_effect=lambda *a, **k: next(alive)), \
         patch.object(mc, "managed_comfy_active", return_value=True), \
         patch.object(mc, "managed_comfy_paths", return_value=fake_paths), \
         patch("subprocess.Popen", side_effect=fake_popen):
        s = backend_mod.settings(full_config)
        ok, msg = backend_mod._comfy_ensure_available(s)

    assert ok is True, msg
    assert spawned["cwd"] == str(managed_root)
    assert str(fake_paths.venv_python) in spawned["argv"]
    assert str(fake_paths.main_py) in spawned["argv"]

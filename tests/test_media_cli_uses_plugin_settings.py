# SPDX-License-Identifier: AGPL-3.0-or-later
"""`localm image` / `music` / `video` must honour the PER-PLUGIN ComfyUI URL.

The GUI and `localm plugin config` write a per-plugin comfy.api_url, and that
setting's own help says it is where THIS plugin's ComfyUI listens. The CLI used
to call default_api_url(), which only ever reads the shared key, so a per-plugin
target was ignored and the command line could talk to a different ComfyUI than
the GUI for the same plugin.

The CLI resolves through the plugin's own backend now, the same call its routes
make, so the two cannot drift apart.
"""

from __future__ import annotations

import pytest

from localm.cli import media as media_cli


@pytest.fixture
def cfg_home(tmp_path, monkeypatch):
    home = tmp_path / ".localm"
    monkeypatch.setenv("LOCALM_HOME", str(home))
    import localm.config as _cfg
    monkeypatch.setattr(_cfg, "HOME_DIR", home)
    monkeypatch.setattr(_cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(_cfg, "REGISTRY_FILE", home / "registry.json")
    monkeypatch.setattr(_cfg, "MODELS_DIR", home / "models")
    _cfg.ensure_dirs()
    return _cfg


@pytest.mark.parametrize("plugin", ["image", "music", "video"])
def test_the_cli_uses_the_per_plugin_comfy_url(plugin, cfg_home, monkeypatch):
    from localm.config import load_config, save_config

    cfg = load_config()
    cfg.setdefault("plugins", {}).setdefault(plugin, {})["comfy"] = {
        "api_url": "http://127.0.0.1:9999",
    }
    # A DIFFERENT shared default, so using the wrong one is visible.
    cfg["comfy_api_url"] = "http://127.0.0.1:8188"
    save_config(cfg)

    url = media_cli._plugin_api_url(plugin)
    assert url.rstrip("/") == "http://127.0.0.1:9999", (
        f"{plugin} on the command line ignored its own ComfyUI url and used "
        f"{url!r}; the GUI would have used the per-plugin one")


def test_a_plugin_with_no_own_url_falls_back_to_the_shared_default(
        cfg_home, monkeypatch):
    from localm.config import load_config, save_config

    cfg = load_config()
    cfg["comfy_api_url"] = "http://127.0.0.1:8188"
    cfg.setdefault("plugins", {}).setdefault("image", {})["comfy"] = {"api_url": ""}
    save_config(cfg)

    url = media_cli._plugin_api_url("image")
    assert url.rstrip("/") == "http://127.0.0.1:8188", (
        f"a blank per-plugin url must fall back to the shared default, got {url!r}")


def test_an_unimportable_plugin_still_answers_with_the_shared_default(cfg_home):
    # Never raise into a generate command just because a plugin is absent.
    url = media_cli._plugin_api_url("definitely-not-a-plugin")
    assert url.startswith("http"), f"no usable fallback url: {url!r}"


# --------------------------------------------------------------------------- #
#  The WIRING, not just the helper. Reverting the call sites alone left the    #
#  helper tests above green, which is exactly the defect-one-layer-away shape. #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("plugin,argv", [
    ("image", ["a cat"]),
    ("music", ["upbeat"]),
    ("video", ["a wave"]),
])
def test_the_generate_commands_pass_the_per_plugin_url(plugin, argv, cfg_home,
                                                       monkeypatch):
    from click.testing import CliRunner

    from localm.config import load_config, save_config

    cfg = load_config()
    cfg.setdefault("plugins", {}).setdefault(plugin, {})["comfy"] = {
        "api_url": "http://127.0.0.1:9999"}
    cfg["comfy_api_url"] = "http://127.0.0.1:8188"
    save_config(cfg)

    seen = []

    def _capture(api_url, run):
        seen.append(api_url)
        return True, "stubbed"

    monkeypatch.setattr(media_cli, "_generate_or_abort", _capture)

    cmd = getattr(media_cli, plugin + "_cmd")
    CliRunner().invoke(cmd, argv)

    assert seen, f"the {plugin} command never reached the generate step"
    assert seen[0].rstrip("/") == "http://127.0.0.1:9999", (
        f"`localm {plugin}` used {seen[0]!r} instead of the per-plugin ComfyUI "
        f"url the GUI would have used")

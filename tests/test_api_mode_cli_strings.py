# SPDX-License-Identifier: AGPL-3.0-or-later
"""NEW-API-MODE-PRINTS-GUI-STRINGS: `--api-mode` deliberately does not mount a
GUI (`if not api_mode: manager = attach_gui(...)`), but several printed lines
used to point at it regardless: "Open the GUI: <url>", "add one on the Models
page", "Settings > Server > Bind address", and more. All name a page that is
not being served in that mode.
"""

from localm.plugins.gui.cli import (
    _console_url_line,
    _empty_registry_hint,
    _engine_load_failed_hint,
    _model_less_hint,
    _no_loadable_model_hint,
    _no_model_flag_hint,
    _phone_lan_hint,
)


class TestModelLessHint:
    def test_gui_mode_points_at_the_models_page(self):
        assert "Models page" in _model_less_hint(False)

    def test_api_mode_points_at_a_cli_route_instead(self):
        hint = _model_less_hint(True)
        assert "Models page" not in hint, (
            "api_mode does not serve a Models page - must not point at one")
        assert "localm pull" in hint


class TestConsoleUrlLine:
    BASE = "https://127.0.0.1:8443/"
    OPEN = "https://127.0.0.1:8443/?view=models&pull=foo&localm_token=secret"

    def test_gui_mode_shows_the_gui_deep_link(self):
        label, url = _console_url_line(False, self.BASE, self.OPEN)
        assert label == "Open the GUI"
        assert url == self.OPEN

    def test_api_mode_shows_the_plain_api_base_not_the_gui_deep_link(self):
        label, url = _console_url_line(True, self.BASE, self.OPEN)
        assert label == "API base"
        assert url == self.BASE
        assert "view=models" not in url, (
            "api_mode must not print a link into a GUI it never mounted")


class TestPhoneLanHint:
    def test_gui_mode_offers_the_settings_page_alternative(self):
        hint = _phone_lan_hint(False)
        assert "Settings > Server > Bind address" in hint
        assert "localm gui -H 0.0.0.0" in hint

    def test_api_mode_does_not_mention_settings(self):
        hint = _phone_lan_hint(True)
        assert "Settings" not in hint, (
            "api_mode does not serve a Settings page - must not point at one")
        assert "localm gui -H 0.0.0.0" in hint
        assert "docs/phone.md" in hint


class TestNoModelFlagHint:
    """--no-model explicit startup (cli.py: `if no_model:`)."""

    def test_gui_mode_points_at_the_models_page(self):
        assert "Models page" in _no_model_flag_hint(False)

    def test_api_mode_does_not_mention_the_models_page(self):
        hint = _no_model_flag_hint(True)
        assert "Models page" not in hint, (
            "api_mode does not serve a Models page - must not point at one")
        assert "no model loaded" in hint.lower()


class TestEmptyRegistryHint:
    """Fresh install, nothing registered (cli.py: `if not registry:`)."""

    def test_gui_mode_points_at_the_models_page(self):
        hint = _empty_registry_hint(False, None)
        assert "GUI" in hint
        assert "Models page" in hint

    def test_gui_mode_with_pull_spec_notes_the_download(self):
        hint = _empty_registry_hint(False, "org/repo:file.gguf")
        assert "download starting" in hint

    def test_api_mode_does_not_mention_gui_or_models_page(self):
        hint = _empty_registry_hint(True, None)
        assert "GUI" not in hint
        assert "Models page" not in hint, (
            "api_mode does not serve a Models page - must not point at one")
        assert "No models registered yet" in hint

    def test_api_mode_with_pull_spec_still_notes_the_download(self):
        hint = _empty_registry_hint(True, "org/repo:file.gguf")
        assert "Download starting" in hint


class TestNoLoadableModelHint:
    """Registry has entries but none is a loadable chat model."""

    def test_gui_mode_points_at_the_models_page(self):
        hint = _no_loadable_model_hint(False)
        assert "GUI" in hint
        assert "Models page" in hint

    def test_api_mode_does_not_mention_gui_or_models_page(self):
        hint = _no_loadable_model_hint(True)
        assert "GUI" not in hint
        assert "Models page" not in hint, (
            "api_mode does not serve a Models page - must not point at one")
        assert "No loadable chat models" in hint


class TestEngineLoadFailedHint:
    """A named model's engine construction raised."""

    def test_gui_mode_points_at_the_models_page(self):
        hint = _engine_load_failed_hint(False)
        assert "GUI" in hint
        assert "Models page" in hint

    def test_api_mode_does_not_mention_gui_or_models_page(self):
        hint = _engine_load_failed_hint(True)
        assert "GUI" not in hint
        assert "Models page" not in hint, (
            "api_mode does not serve a Models page - must not point at one")
        assert "Continuing without a model loaded" in hint

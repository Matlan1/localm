# SPDX-License-Identifier: AGPL-3.0-or-later
"""NEW-API-MODE-PRINTS-GUI-STRINGS: `--api-mode` deliberately does not mount a
GUI (`if not api_mode: manager = attach_gui(...)`), but two printed lines used
to point at it regardless: "Open the GUI: <url>" and "add one on the Models
page". Both name a page that is not being served in that mode.
"""

from localm.plugins.gui.cli import _console_url_line, _model_less_hint


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

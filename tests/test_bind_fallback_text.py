# SPDX-License-Identifier: AGPL-3.0-or-later
"""A configured network bind that is not applied for want of a strong key names
the Settings card that fixes it: Keys & devices on a server with no key (the
first key made there also sets the owner key), the Owner key card when the key
is too short."""

from __future__ import annotations

import pytest

from localm.plugins.gui import cli as gui_cli


class _Console:
    def print(self, *args, **kwargs):
        pass


@pytest.mark.parametrize("keyed, card", [
    (False, "Settings > Security > Keys & devices: the first key created there also sets the owner key"),
    (True, "Settings > Security > Owner key;"),
])
def test_fallback_names_the_card_that_sets_a_key(monkeypatch, keyed, card):
    monkeypatch.setattr(gui_cli, "_gui_bind_warning", lambda host: "binding past loopback")
    monkeypatch.setattr("localm.auth.any_key_configured", lambda: keyed)
    plan = gui_cli._BindPlan(host="0.0.0.0", from_config=True)
    gui_cli._guard_unauthenticated_bind(_Console(), plan, insecure=False)
    assert plan.host == "127.0.0.1"
    assert card in plan.fallback
    other = "Owner key;" if not keyed else "Keys & devices"
    assert other not in plan.fallback

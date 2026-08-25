# SPDX-License-Identifier: AGPL-3.0-or-later
"""localm.bindhost: the loopback-host predicate hoisted out of five copies (2026-07-09 quality audit finding #1) - one of which had already drifted to a non-identical inline variant (routes/system.py)."""

import pytest

from localm.bindhost import is_loopback_host


@pytest.mark.parametrize("host,loop", [
    ("127.0.0.1", True), ("::1", True), ("localhost", True),
    ("127.0.0.5", True), ("10.0.0.7", False), ("192.168.1.4", False),
    ("0.0.0.0", False), ("testclient", False), ("", False),
])
def test_is_loopback_host(host, loop):
    assert is_loopback_host(host) is loop


def test_all_former_call_sites_reexport_the_same_function():
    import localm.inference.http_server as http_server
    import localm.inference.routes.keys as keys_routes
    import localm.plugins.deps_task as deps_task
    import localm.plugins.gui.web as gui_web

    assert http_server._is_loopback_host is is_loopback_host
    assert keys_routes._is_loopback is is_loopback_host
    assert deps_task.is_loopback_host is is_loopback_host
    assert gui_web._is_loopback_host is is_loopback_host


def test_system_route_uses_shared_predicate_not_a_drifted_inline_copy():
    # routes/system.py's whoami() previously had a NON-IDENTICAL inline
    # variant (checked a 3-item tuple first, ipaddress only as a fallback) -
    # confirm it now calls the shared helper for a loopback-shaped host that
    # only the ipaddress-based check recognizes.
    from localm.inference.routes import system as system_routes
    assert system_routes.is_loopback_host is is_loopback_host
    assert system_routes.is_loopback_host("127.0.0.5") is True

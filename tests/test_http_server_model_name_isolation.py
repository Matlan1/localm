# SPDX-License-Identifier: AGPL-3.0-or-later
"""_resolve_unnamed_model_name() must never resolve to a name published by an
EARLIER test's create_app(engine) call, once that test has ended - regardless
of whether the later test names an engine of its own.

The two tests below are ORDER-DEPENDENT BY DESIGN: test_a leaves the module
dirty and performs no cleanup of its own on purpose, so only conftest.py's
autouse ``_reset_http_server_model_name_state`` fixture stands between it and
test_b. pytest runs a module's tests in source-code definition order (no
pytest-randomly is installed in this project), which is what test_a/test_b
naming keeps aligned with.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from localm.inference.http_server import create_app, _resolve_unnamed_model_name


def test_a_names_an_engine_and_leaves_it_running():
    engine = MagicMock()
    engine.display_name = "leak-probe-model"
    create_app(engine)
    assert _resolve_unnamed_model_name() == "leak-probe-model"


def test_b_never_named_anything_of_its_own():
    assert _resolve_unnamed_model_name() is None

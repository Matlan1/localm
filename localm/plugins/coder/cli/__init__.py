# SPDX-License-Identifier: AGPL-3.0-or-later
"""localcoder CLI package.

The ``main`` Click command lives in ``_main`` (the localcoder entry point
``localm.plugins.coder.cli:main`` resolves via this re-export); the goal-mode,
estimate, and REPL helpers live in sibling modules.

The names imported from or monkeypatched on ``localm.plugins.coder.cli`` are
re-exported here: ``main`` (entry point plus the ``localm coder`` route in
localm/cli/maintenance.py), ``_run_goal_loop`` / ``_goal_task_wrap`` /
``_warn_sensitive_changes``, and the patched helpers ``_run_verify`` and
``make_localm_backend``. The call sites that consume a patched name resolve it
back from this package at call time (``import localm.plugins.coder.cli as _cli;
_cli.<name>``), so a patch is seen."""

# Patched + re-exported helper from backends (callers read it live via _cli).
from ..backends.http import make_localm_backend  # noqa: F401

from .goal import (  # noqa: F401
    _goal_feedback, _goal_task_wrap, _run_goal_loop, _run_verify,
)
from .estimate import _run_estimate  # noqa: F401
from .repl import _handle_command, _repl, _setup_readline  # noqa: F401
from ._main import (  # noqa: F401
    main, console_main, _complete_model, _warn_sensitive_changes,
)

__all__ = ["main", "console_main"]

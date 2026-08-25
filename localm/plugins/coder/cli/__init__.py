# SPDX-License-Identifier: AGPL-3.0-or-later
"""localcoder CLI package."""

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

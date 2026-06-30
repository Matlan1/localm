# SPDX-License-Identifier: AGPL-3.0-or-later
"""The coder Agent package.

This package replaces the former single-file ``localm/plugins/coder/agent.py``.
The Agent class is assembled in ``core.py`` from concern mixins (loop / execution
/ context / persistence / session); module-level helpers live in ``constants``,
``scope``, ``checkpoint``, and ``tooldefs``.

The public surface the rest of the coder and the test suite import from
``localm.plugins.coder.agent`` is preserved by name here: the ``Agent`` class
(re-exported so a test that monkeypatches ``agent.Agent`` is honoured by the
call-time ``from ...agent import Agent`` in spawn_agent / the jobs runner), the
module-level checkpoint probes, the scope-tool set, and the compaction-ratio
constants the tests read.

The single-file agent.py also imported several helpers as module globals that the
test suite monkeypatches via ``patch("localm.plugins.coder.agent.<name>")``
(ProjectMap, load_memory, make_audit_log, parse_tool_calls, confirm, confirm_diff,
print_diff_preview, print_warning, TOOL_REGISTRY). Those are re-exposed here so the patch target
still resolves; the mixin methods that use them read them back via live-attribute
access (``import localm.plugins.coder.agent as _agent; _agent.<name>``) so a patch
on this package is seen, exactly as it was when they were globals of agent.py.
(F401: re-exported for the patch surface, not used in this module.)"""

# Patch surface re-exports (see module docstring).
from ..display import (  # noqa: F401
    confirm, confirm_diff, print_diff_preview, print_warning,
)
from ..indexer import ProjectMap  # noqa: F401
from ..memory import load_memory  # noqa: F401
from ..audit import make_audit_log  # noqa: F401
from ..parser import parse_tool_calls  # noqa: F401
from ..tools import TOOL_REGISTRY  # noqa: F401

from .core import Agent
from .checkpoint import checkpoint_info, _checkpoint_path_for
from .constants import (
    _COMPACT_AUTO_RATIO, _COMPACT_WARN_RATIO, _DEFAULT_CTX_TOKENS, _SCOPED_TOOLS,
)

__all__ = [
    "Agent", "checkpoint_info", "_checkpoint_path_for", "_SCOPED_TOOLS",
    "_COMPACT_WARN_RATIO", "_COMPACT_AUTO_RATIO", "_DEFAULT_CTX_TOKENS",
]

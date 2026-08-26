# SPDX-License-Identifier: AGPL-3.0-or-later
"""The coder Agent package.

The Agent class is assembled in ``core.py`` from concern mixins (loop / execution
/ context / persistence / session); module-level helpers live in ``constants``,
``scope``, ``checkpoint``, and ``tooldefs``.

``localm.plugins.coder.agent`` re-exports the public surface by name: the
``Agent`` class (so a monkeypatch of ``agent.Agent`` is honoured by the
call-time ``from ...agent import Agent`` in spawn_agent / the jobs runner), the
module-level checkpoint probes, the scope-tool set, and the compaction-ratio
constants.

Several helpers (ProjectMap, load_memory, make_audit_log, parse_tool_calls,
confirm, confirm_diff, print_diff_preview, print_warning, TOOL_REGISTRY) are
re-exposed here as module attributes, and the mixin methods that use them read
them back via live-attribute access (``import localm.plugins.coder.agent as
_agent; _agent.<name>``), so a patch of ``localm.plugins.coder.agent.<name>`` is
seen. (F401: re-exported for that surface, not used in this module.)"""

# Patch surface re-exports (see module docstring).
from ..display import (  # noqa: F401
    confirm, confirm_diff, print_diff_preview, print_warning,
)
from ..indexer import ProjectMap  # noqa: F401
from ..memory import (  # noqa: F401
    cap_user_instructions, custom_instructions_warning, load_custom_instructions,
    load_memory, memory_warning,
)
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

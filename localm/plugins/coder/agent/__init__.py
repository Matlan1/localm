# SPDX-License-Identifier: AGPL-3.0-or-later
"""The coder Agent package."""

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

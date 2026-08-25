# SPDX-License-Identifier: AGPL-3.0-or-later
"""Pure-Python ctypes wrapper around llama.dll - our own llama-cpp-python."""

from .llama import LlamaCpp

__all__ = ["LlamaCpp"]

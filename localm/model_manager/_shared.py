# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared, low-level state for the model_manager package: the Rich console, the
GUI progress sentinel, and the one-time Windows stdout/stderr UTF-8 reconfigure
(an import-time side effect preserved from the original module)."""

import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from rich.console import Console

console = Console()

# GUI download progress: the GUI runs ``localm pull`` as a subprocess and
# parses its stdout. When LOCALM_PROGRESS_JSON=1 the downloader streams
# structured progress lines (this sentinel + JSON) that the GUI renders as a
# progress bar; interactive CLI use keeps huggingface_hub's own tqdm bars.
PROGRESS_SENTINEL = "\x1flocalm-progress\x1f"

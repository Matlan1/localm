# SPDX-License-Identifier: AGPL-3.0-or-later
"""Standalone helper spawned as a REAL subprocess by test_setup_llama_provisioning_lock.py's cross-process test."""

import sys
import time
from pathlib import Path

from localm import setup_llama as sl


def main() -> int:
    target = Path(sys.argv[1])
    marker = Path(sys.argv[2])
    hold_s = float(sys.argv[3])
    # Printed so a caller reading stdout on failure can tell "the lock logic
    # was wrong" apart from "this process imported a DIFFERENT localm tree
    # than the one under test" - the two look identical otherwise.
    print(f"holder using localm from {sl.__file__}", flush=True)
    with sl._provisioning_lock(target):
        marker.write_text("holding", encoding="utf-8")
        time.sleep(hold_s)
    return 0


if __name__ == "__main__":
    sys.exit(main())

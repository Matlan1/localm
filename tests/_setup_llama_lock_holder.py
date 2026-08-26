# SPDX-License-Identifier: AGPL-3.0-or-later
"""Standalone helper spawned as a REAL subprocess by the cross-process
provisioning-lock test. Not collected by pytest (no test_ prefix). Takes
<target-dir> <marker-file> <hold-seconds> on argv, acquires
setup_llama._provisioning_lock on target-dir, writes marker to signal it is
holding the lock, sleeps for hold-seconds, then releases and exits 0.
"""

import sys
import time
from pathlib import Path

from localm import setup_llama as sl


def main() -> int:
    target = Path(sys.argv[1])
    marker = Path(sys.argv[2])
    hold_s = float(sys.argv[3])
    # Report which localm tree this process imported.
    print(f"holder using localm from {sl.__file__}", flush=True)
    with sl._provisioning_lock(target):
        marker.write_text("holding", encoding="utf-8")
        time.sleep(hold_s)
    return 0


if __name__ == "__main__":
    sys.exit(main())

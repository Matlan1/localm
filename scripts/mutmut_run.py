#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Runs mutmut after pre-importing cryptography's compiled extension.

Run:  python scripts/mutmut_run.py <mutmut args>, e.g.
python scripts/mutmut_run.py run
"""

from __future__ import annotations

import sys

# These imports are used only for their side effect: keeping cryptography's
# compiled extension loaded before mutmut's own process starts. See
# tests/test_mutmut_crypto_reload.py.
import cryptography.exceptions  # noqa: F401
import cryptography.hazmat.primitives.asymmetric.padding  # noqa: F401
import cryptography.hazmat.primitives.asymmetric.rsa  # noqa: F401
import cryptography.hazmat.primitives.hashes  # noqa: F401
import cryptography.hazmat.primitives.serialization  # noqa: F401
import cryptography.x509  # noqa: F401
import cryptography.x509.oid  # noqa: F401

from mutmut.__main__ import cli

if __name__ == "__main__":
    sys.exit(cli())

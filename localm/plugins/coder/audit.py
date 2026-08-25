# SPDX-License-Identifier: AGPL-3.0-or-later
"""Back-compat shim - the session-mode/audit machinery moved to :mod:`localm.audit` so the core CLI and HTTP server can use it too."""

import sys

from localm import audit as _audit

sys.modules[__name__] = _audit

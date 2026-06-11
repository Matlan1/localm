"""
Back-compat shim — the session-mode/audit machinery moved to
:mod:`localm.audit` so the core CLI and HTTP server can use it too.

The module-alias trick below makes this name point at the SAME module
object, so existing imports and test patches
(``patch("localm.plugins.coder.audit._SESSIONS_DIR", ...)``) keep working.
"""

import sys

from localm import audit as _audit

sys.modules[__name__] = _audit

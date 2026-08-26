# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Back-compat shim - the session-mode/audit machinery lives in
:mod:`localm.audit`.

The module-alias trick below makes this name point at the SAME module
object, so existing imports and test patches keep working.
"""

import sys

from localm import audit as _audit

sys.modules[__name__] = _audit

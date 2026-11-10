# SPDX-License-Identifier: AGPL-3.0-or-later

from tests import _no_console_window, _no_window

# psutil is imported here, under the real sys.platform, before any test patches
# sys.platform. See test_psutil_survives_a_failed_import_under_a_patched_platform.
try:
    import psutil  # noqa: F401
except ImportError:
    pass

_no_console_window.arm()
_no_window.arm()

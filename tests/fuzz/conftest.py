# SPDX-License-Identifier: AGPL-3.0-or-later
"""Hypothesis profiles for the fuzz suite.

``LOCALM_FUZZ_PROFILE`` picks one:

``pr`` (default)  small, derandomized, no example database: a pull request run
                  is reproducible and cannot go red on a seed nobody can replay.
``nightly``       large and random, with a persistent example database under
                  ``LOCALM_FUZZ_DB`` (default ``.hypothesis/examples``) that the
                  scheduled workflow caches between runs, so a failing input is
                  replayed first on the next run.
``explore``       a large local budget for hunting.

``LOCALM_FUZZ_EXAMPLES`` overrides ``max_examples`` of any profile.

The example database is never committed: a failing input found by a run becomes
an explicit ``@example`` or a plain regression test next to the fix.
"""
from __future__ import annotations

import os

try:
    from hypothesis import HealthCheck, settings
    from hypothesis.database import DirectoryBasedExampleDatabase
except ImportError:
    pass
else:
    _examples = os.environ.get("LOCALM_FUZZ_EXAMPLES")
    _common = dict(
        deadline=None,
        print_blob=True,
        suppress_health_check=[HealthCheck.function_scoped_fixture,
                               HealthCheck.too_slow, HealthCheck.data_too_large],
    )

    def _n(default: int) -> int:
        return int(_examples) if _examples else default

    settings.register_profile(
        "pr", max_examples=_n(40), derandomize=True, database=None, **_common)
    settings.register_profile(
        "nightly", max_examples=_n(2000), derandomize=False,
        database=DirectoryBasedExampleDatabase(
            os.environ.get("LOCALM_FUZZ_DB", ".hypothesis/examples")),
        **_common)
    settings.register_profile(
        "explore", max_examples=_n(3000), derandomize=False, database=None, **_common)
    settings.load_profile(os.environ.get("LOCALM_FUZZ_PROFILE", "pr"))

# SPDX-License-Identifier: AGPL-3.0-or-later
"""Turn "an optional prerequisite is missing" from a skip into a failure.

A test that skips because a package or resource is absent reports green while
checking nothing. A CI job that is supposed to cover that package names it in
``LOCALM_REQUIRE_OPTIONAL`` (comma separated), and any skip of that kind in the
run then fails instead of passing quietly:

* a package name (``torch``, ``transformers``, ``psutil``, ...): matches the
  ``could not import '<name>'`` skip that ``pytest.importorskip`` raises, at
  collection or inside a test;
* a resource-gate marker (``real_gguf``, ``real_browser``, ...): matches the
  ``<marker>: <reason>`` skip that ``tests/conftest.py`` raises for it;
* ``*``: every skip of either kind.

Skips for any other reason (a Windows-only test on Linux, no display, no GPU)
are untouched: they are platform or hardware facts, not a missing install.
Unset, nothing changes.
"""
from __future__ import annotations

import os
import re
from typing import Optional

import pytest

REQUIRE_ENV = "LOCALM_REQUIRE_OPTIONAL"

_IMPORT_SKIP = re.compile(r"could not import '([\w.]+)'")
_GATE_SKIP = re.compile(r"^(?:Skipped: )?(real_[a-z0-9_]+): ")


def required_names() -> frozenset:
    raw = os.environ.get(REQUIRE_ENV, "")
    return frozenset(part.strip() for part in raw.split(",") if part.strip())


def skip_subject(reason: str) -> Optional[str]:
    """The package or resource-gate marker that a skip *reason* says is missing,
    or None when the reason is about something else."""
    match = _IMPORT_SKIP.search(reason)
    if match:
        return match.group(1)
    match = _GATE_SKIP.match(reason)
    if match:
        return match.group(1)
    return None


def forbidden_subject(reason: str, required: frozenset) -> Optional[str]:
    """The missing package or marker when *reason* is a skip the environment
    forbids, else None."""
    subject = skip_subject(reason)
    if subject is None or not required:
        return None
    if "*" in required or subject in required or subject.split(".")[0] in required:
        return subject
    return None


def _reason(report) -> str:
    longrepr = report.longrepr
    if isinstance(longrepr, tuple) and len(longrepr) == 3:
        return str(longrepr[2])
    return str(longrepr)


def _fail_if_forbidden(report) -> None:
    if not report.skipped:
        return
    subject = forbidden_subject(_reason(report), required_names())
    if subject is None:
        return
    report.outcome = "failed"
    report.longrepr = (
        f"{REQUIRE_ENV} requires '{subject}', but this test was skipped "
        f"because it is missing: {_reason(report)}")


@pytest.hookimpl(hookwrapper=True)
def pytest_make_collect_report(collector):
    outcome = yield
    _fail_if_forbidden(outcome.get_result())


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    _fail_if_forbidden(outcome.get_result())

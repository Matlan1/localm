"""Shared pytest fixtures.

Hermetic data dir: every test gets its own ``LOCALM_HOME`` under the test's
``tmp_path``. Without this, ``_detect_home()`` falls through to *portable mode*
(a ``home/`` directory next to the installed package) or the real ``~/.localm``
- so a developer who has run the GUI in portable mode would have the GUI tests
read, write, and DELETE their actual conversations/images while the suite runs.
``LOCALM_HOME`` takes priority over portable mode in ``_detect_home()``, so
pinning it here isolates every test. Tests that need a specific home override
this with their own ``monkeypatch.setenv`` (which runs after this autouse
fixture).
"""

import pytest


@pytest.fixture(autouse=True)
def _isolate_localm_home(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path / ".localm"))

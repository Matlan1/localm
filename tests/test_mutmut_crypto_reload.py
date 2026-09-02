# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for scripts/mutmut_run.py and the defect it works around.

mutmut's mutate_only_covered_lines pass unloads every module first imported
during its own coverage-gathering run, cryptography's compiled extension
included, then reimports it for the next pass. The reimported module is a
distinct object from the one the extension's own internal state was built
against, so a later `.sign()` call raises "Algorithm must be a registered
hash algorithm." These tests reproduce that unload/reimport cycle directly
against cryptography, without running mutmut, and confirm that a module
already present before the unload decision survives it.
"""

from __future__ import annotations

import importlib
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _build_and_sign_a_cert(hashes_module) -> None:
    from cryptography import x509
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "test")])
    now = datetime.now(timezone.utc)
    (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=5))
        .not_valid_after(now + timedelta(days=1))
        .sign(key, hashes_module.SHA256())
    )


def _pop_and_reimport_cryptography(snapshot: set[str]):
    """Removes every sys.modules entry named after `snapshot`, then
    reimports cryptography.hazmat.primitives.hashes fresh. Returns the
    freshly imported module."""
    for name in list(sys.modules):
        if name.startswith("cryptography") and name not in snapshot:
            del sys.modules[name]
    importlib.invalidate_caches()
    import cryptography.hazmat.primitives.hashes as hashes_module

    return hashes_module


class TestCryptographyRejectsAReimportedHashAlgorithmClass:
    def test_signing_with_a_reimported_hashes_module_raises_type_error(self):
        import cryptography.hazmat.primitives.hashes as hashes_before

        _build_and_sign_a_cert(hashes_before)  # establishes the extension's own state

        snapshot = {n for n in sys.modules if not n.startswith("cryptography")}
        saved = {n: m for n, m in sys.modules.items() if n.startswith("cryptography")}
        try:
            hashes_after = _pop_and_reimport_cryptography(snapshot)
            assert hashes_after.SHA256 is not hashes_before.SHA256
            with pytest.raises(TypeError, match="registered hash algorithm"):
                _build_and_sign_a_cert(hashes_after)
        finally:
            for name in list(sys.modules):
                if name.startswith("cryptography") and name not in saved:
                    del sys.modules[name]
            sys.modules.update(saved)
            importlib.invalidate_caches()

    def test_a_module_present_before_the_unload_decision_is_never_reimported(self):
        import cryptography.hazmat.primitives.hashes as hashes_before

        _build_and_sign_a_cert(hashes_before)

        # The module is already in the pre-unload snapshot this time, so the
        # pop-by-name loop below removes nothing under it.
        snapshot = set(sys.modules)
        saved = dict(sys.modules)
        try:
            hashes_after = _pop_and_reimport_cryptography(snapshot)
            assert hashes_after.SHA256 is hashes_before.SHA256
            _build_and_sign_a_cert(hashes_after)  # does not raise
        finally:
            for name in list(sys.modules):
                if name not in saved:
                    del sys.modules[name]
            sys.modules.update(saved)
            importlib.invalidate_caches()


class TestMutmutRunPreimportsEveryCryptographySubmoduleTlsUses:
    def test_every_cryptography_import_in_tls_py_is_preimported_by_the_wrapper(self):
        tls_source = (REPO_ROOT / "localm" / "tls.py").read_text(encoding="utf-8")
        wrapper_source = (REPO_ROOT / "scripts" / "mutmut_run.py").read_text(encoding="utf-8")

        imported_from = set(re.findall(r"from (cryptography[\w.]*) import", tls_source))
        assert imported_from, "expected tls.py to still import from cryptography"

        for module in sorted(imported_from):
            assert module in wrapper_source, (
                f"tls.py imports from {module!r}, but scripts/mutmut_run.py "
                "does not pre-import it"
            )

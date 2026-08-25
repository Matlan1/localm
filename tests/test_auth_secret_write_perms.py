# SPDX-License-Identifier: AGPL-3.0-or-later
"""A credential file must never exist unrestricted, not even for one rename."""

import json
import os
import subprocess

import pytest


@pytest.fixture
def auth(tmp_path, monkeypatch):
    """localm.auth against a throwaway data dir."""
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path)
    monkeypatch.setattr(cfg, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", tmp_path / "registry.json")
    import localm.auth as a
    a._forget_cached_digests()
    yield a
    a._forget_cached_digests()


def _perm_fingerprint(path):
    """A comparable description of *path*'s permissions on this platform."""
    if os.name == "posix":
        return oct(os.stat(path).st_mode & 0o777)
    out = subprocess.run(["icacls", str(path)], capture_output=True,
                         check=False).stdout.decode("utf-8", "replace")
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    return sorted(ln.replace(str(path), "").strip()
                  for ln in lines if "Successfully processed" not in ln)


def _install_spies(monkeypatch):
    """Record, in order, every ``restrict_file_perms`` call and every ``os.replace``, capturing the SOURCE file's permissions and content at the instant of the rename (afterwards the temp file no longer exists)."""
    import localm.config as cfg
    events = []
    real_restrict = cfg.restrict_file_perms
    real_replace = os.replace

    def spy_restrict(path, **kwargs):
        events.append(("restrict", str(path)))
        return real_restrict(path, **kwargs)

    def spy_replace(src, dst, *a, **kw):
        try:
            captured = (_perm_fingerprint(src),
                        open(src, "r", encoding="utf-8-sig").read())
        except OSError:                      # not one of ours; stay transparent
            captured = (None, None)
        events.append(("replace", str(src), str(dst), *captured))
        return real_replace(src, dst, *a, **kw)

    monkeypatch.setattr(cfg, "restrict_file_perms", spy_restrict)
    monkeypatch.setattr(os, "replace", spy_replace)
    return events


def _rename_of(events, dest_path):
    """The single recorded rename onto *dest_path*."""
    hits = [e for e in events if e[0] == "replace" and e[2] == str(dest_path)]
    assert len(hits) == 1, (
        f"expected exactly one rename onto {dest_path}, saw {len(hits)}: "
        f"{[e[:3] for e in events]}")
    return hits[0]


KEY = "spy-on-the-rename-0123456789"


def test_every_credential_writer_restricts_its_temp_before_the_rename(
        auth, tmp_path, monkeypatch):
    """auth.key, the owner-KDF file and auth.json, in one pass."""
    from localm import config as cfg

    # The reference: what "restricted" actually looks like in THIS directory on
    # THIS box, produced by the real helper before any spy is installed.
    home = auth.key_file().parent
    home.mkdir(parents=True, exist_ok=True)
    reference = home / "reference.txt"
    reference.write_text("x", encoding="utf-8")
    assert cfg.restrict_file_perms(reference) is True
    restricted = _perm_fingerprint(reference)

    if os.name == "posix":
        # Fires-control for the INSTRUMENT: prove the fingerprint can report
        # "not restricted" at all. Pinned to 0o644 rather than left to the
        # ambient umask, because under umask 0077 a fresh file is ALREADY 0600
        # and the control would silently collapse (the lesson recorded in
        # tests/test_auth_kdf.py).
        loose = home / "loose.txt"
        loose.write_text("x", encoding="utf-8")
        os.chmod(loose, 0o644)
        assert _perm_fingerprint(loose) != restricted
        assert restricted == "0o600"

    events = _install_spies(monkeypatch)
    auth.set_api_key(KEY)
    auth._save_keystore([{"id": "k1", "alg": "sha256"}])

    expected_payload = {
        auth.key_file(): KEY,
        auth.owner_kdf_file(): "records",
        auth.keystore_file(): "sha256",
    }
    for dest, marker in expected_payload.items():
        _, src, _dst, fingerprint, content = _rename_of(events, dest)

        # The temp file we fingerprinted really did hold the secret. Without
        # this the test could pass by measuring an empty or unrelated file.
        assert marker in content, (dest, content[:120])

        # ORDERING - the platform-independent half.
        order = [e for e in events
                 if (e[0] == "restrict" and e[1] == src)
                 or (e[0] == "replace" and e[1] == src)]
        assert order and order[0][0] == "restrict", (
            f"{dest.name}: its temp file was renamed into place before "
            f"anything restricted it - the payload was readable at the "
            f"directory default for the whole write. Order seen: "
            f"{[e[0] for e in order]}")

        # THE PROPERTY - the temp file was already locked down at rename time.
        assert fingerprint == restricted, (
            f"{dest.name}: temp permissions at rename time {fingerprint} "
            f"differ from a genuinely restricted file {restricted}")

        # AND THE END STATE IS NOT WEAKENED. Restricting the temp instead of the
        # destination relies on os.replace carrying the source's ACL/mode across
        # (config.py:766-773, MEASURED there and the basis of the same pattern in
        # sessions.py, instances.py and gpu_registry.py). The previous code
        # tightened the destination directly, so if that carry-over were ever
        # untrue on some filesystem this refactor would silently weaken the
        # finished file. Pinned here rather than assumed.
        assert _perm_fingerprint(dest) == restricted, (
            f"{dest.name}: the finished file is not restricted - os.replace did "
            f"not carry the temp file's permissions onto it")


def test_a_failed_restriction_still_persists_the_key_and_never_raises(
        auth, monkeypatch):
    """The contract stays BEST-EFFORT. ``restrict_file_perms`` returns False on a non-NTFS volume or when icacls is unavailable; that must not stop a user setting a key, and it must fall back to retrying the destination rather than leaving the tightening to a call that already reported failure."""
    from localm import config as cfg
    calls = []

    def always_fails(path, **kwargs):
        calls.append(str(path))
        return False

    monkeypatch.setattr(cfg, "restrict_file_perms", always_fails)
    auth.set_api_key(KEY)

    assert auth.get_api_key() == KEY
    key_path = str(auth.key_file())
    assert key_path + ".tmp" in calls, "the temp file was never restricted"
    assert key_path in calls, (
        "the destination was not retried after the temp attempt reported "
        "failure - one failure would then be the single point of failure")


def test_the_written_bytes_are_unchanged_by_the_permission_fix(auth, tmp_path):
    """A permissions fix must not quietly rewrite every credential file."""
    payloads = ["a-key-value\n", json.dumps([{"id": "k1"}], indent=2)]
    for i, payload in enumerate(payloads):
        old = tmp_path / f"old{i}.txt"
        new = tmp_path / f"new{i}.txt"
        old.write_text(payload, encoding="utf-8")     # the replaced idiom
        auth._atomic_write_private(new, payload)
        assert new.read_bytes() == old.read_bytes(), payload
        assert not (tmp_path / f"new{i}.txt.tmp").exists()

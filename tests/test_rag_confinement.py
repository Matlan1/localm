# SPDX-License-Identifier: AGPL-3.0-or-later
"""The RAG indexing API must not be trickable into reading and serving back files
outside the allowed roots (an OS file elsewhere on the disk) or THIRD-PARTY
credential folders (.ssh, .aws, ...) that a caller other than the owner should
never be able to reach through the API.

Note on targets: every "outside the allowed roots" path in this file is a real
but DISPOSABLE file the test creates under its own tmp_path, never an actual
system path. confine_index_path() resolve()s what it is handed BEFORE it decides
anything, so handing it a real OS file would make the test suite itself touch
one. An outside-the-roots temp file exercises the identical code path.

The confinement is MODE-based (whitelist / blacklist) with an always-on HARD
FLOOR of well-known third-party credential folders, refused in every mode. The
localm data directory (LOCALM_HOME) is NOT part of that hard floor and is
subject to the SAME whitelist/blacklist rules as any other location. The CLI
stays unconfined except the credential-folder hard floor: the mode confinement
engages only when a policy is passed, which the HTTP route always does and the
CLI never does. A whitelist MISS is offered back to the owner as 'add and
continue' (409), not a dead-end error.
"""

import base64
import os
import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from localm.rag.store import (Collection, ConfinementError, confine_index_path,
                              indexing_policy)


def _wl(*allowed):
    """A whitelist policy allowing (home + cwd, always) plus *allowed*."""
    return {"mode": "whitelist", "allowed": [Path(a) for a in allowed], "denied": []}


def _bl(*denied):
    """A blacklist policy denying *denied* (everything else allowed)."""
    return {"mode": "blacklist", "allowed": [], "denied": [Path(d) for d in denied]}


@pytest.fixture
def home_env(tmp_path, monkeypatch):
    """A controlled user home (Path.home) with the localm data dir under it."""
    home = tmp_path / "userhome"
    (home / "docs").mkdir(parents=True)
    localm = home / ".localm"
    localm.mkdir()
    monkeypatch.setenv("LOCALM_HOME", str(localm))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", localm)
    monkeypatch.setattr(cfg, "MODELS_DIR", localm / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", localm / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", localm / "registry.json")
    return home, localm


# --------------------------------------------------------------------------- #
#  Whitelist mode                                                             #
# --------------------------------------------------------------------------- #

class TestWhitelist:
    def test_in_home_ok(self, home_env):
        home, _ = home_env
        f = home / "docs" / "a.txt"
        f.write_text("hi", encoding="utf-8")
        assert confine_index_path(f, _wl()) == f.resolve()   # home always allowed

    def test_added_root_ok(self, home_env, tmp_path):
        extra = tmp_path / "shared"
        extra.mkdir()
        f = extra / "a.txt"
        f.write_text("hi", encoding="utf-8")
        assert confine_index_path(f, _wl(extra)) == f.resolve()

    def test_outside_rejected_with_reason(self, home_env, tmp_path):
        outside = tmp_path / "outside.txt"
        outside.write_text("secret", encoding="utf-8")
        with pytest.raises(ConfinementError) as ei:
            confine_index_path(outside, _wl())
        assert ei.value.reason == "outside_allowed"
        assert "outside" in str(ei.value).lower()

    def test_absolute_string_outside_allowed_rejected(self, home_env,
                                                      tmp_path_factory):
        """The out-of-roots refusal, exercised against a disposable stand-in for
        an OS file elsewhere on the disk.

        Distinct from test_outside_rejected_with_reason above in two ways: the
        target is passed as a plain str (what an API caller sends, not a Path),
        and it sits in a temp tree of its own rather than under the same tmp_path
        as the home folder. The refusal is by root containment."""
        outside = tmp_path_factory.mktemp("outside_roots") / "notes.txt"
        outside.write_text("stand-in for a file outside every allowed root\n",
                           encoding="utf-8")
        with pytest.raises(ConfinementError) as ei:
            confine_index_path(str(outside), _wl())
        assert ei.value.reason == "outside_allowed"

    def test_ordinary_dotdir_under_home_indexable(self, home_env):
        # A non-credential dotted folder (.github) is not blocked.
        home, _ = home_env
        wf = home / "repo" / ".github" / "ci.yml"
        wf.parent.mkdir(parents=True)
        wf.write_text("on: push", encoding="utf-8")
        assert confine_index_path(wf, _wl()) == wf.resolve()


# --------------------------------------------------------------------------- #
#  Blacklist mode                                                             #
# --------------------------------------------------------------------------- #

class TestBlacklist:
    def test_outside_home_allowed(self, home_env, tmp_path):
        # blacklist: anywhere not denied is fine, even far outside home.
        outside = tmp_path / "elsewhere" / "a.txt"
        outside.parent.mkdir(parents=True)
        outside.write_text("x", encoding="utf-8")
        assert confine_index_path(outside, _bl()) == outside.resolve()

    def test_denied_root_rejected(self, home_env, tmp_path):
        denied = tmp_path / "secret"
        deep = denied / "deep"
        deep.mkdir(parents=True)
        f = deep / "a.txt"
        f.write_text("x", encoding="utf-8")
        with pytest.raises(ConfinementError) as ei:
            confine_index_path(f, _bl(denied))
        assert ei.value.reason == "denied"

    def test_hard_floor_still_applies(self, home_env):
        # Credential folders are refused even in blacklist mode; the data dir is
        # NOT part of the hard floor (it is not special-cased at all).
        home, localm = home_env
        (localm / "registry.json").write_text("{}", encoding="utf-8")
        assert confine_index_path(localm / "registry.json", _bl()) == \
            (localm / "registry.json").resolve()
        ssh = home / ".ssh"
        ssh.mkdir()
        (ssh / "id").write_text("k", encoding="utf-8")
        with pytest.raises(ConfinementError) as ei2:
            confine_index_path(ssh / "id", _bl())
        assert ei2.value.reason == "credential"


    def test_string_policy_entries_are_coerced(self, home_env, tmp_path):
        # A policy carrying str (not Path) entries does not crash;
        # indexing_policy() returns Paths, but confine does not rely on that.
        outside = tmp_path / "z"
        outside.mkdir()
        f = outside / "a.txt"
        f.write_text("x", encoding="utf-8")
        assert confine_index_path(
            f, {"mode": "whitelist", "allowed": [str(outside)], "denied": []}
        ) == f.resolve()
        with pytest.raises(ConfinementError, match="denied"):
            confine_index_path(
                f, {"mode": "blacklist", "allowed": [], "denied": [str(outside)]})


# --------------------------------------------------------------------------- #
#  Hard floor (both modes + the unconfined CLI)                               #
# --------------------------------------------------------------------------- #

class TestHardFloor:
    def test_data_dir_not_blocked_by_role(self, home_env):
        # The localm data directory is not special-cased: it indexes like any
        # other file under an allowed root (here, the home folder).
        home, localm = home_env
        keyfile = localm / "keys.json"
        keyfile.write_text("{}", encoding="utf-8")
        assert confine_index_path(keyfile, _wl()) == keyfile.resolve()

    def test_credential_dir_rejected(self, home_env):
        home, _ = home_env
        ssh = home / ".ssh"
        ssh.mkdir()
        key = ssh / "id_rsa"
        key.write_text("PRIVATE KEY", encoding="utf-8")
        with pytest.raises(ConfinementError) as ei:
            confine_index_path(key, _wl())
        assert ei.value.reason == "credential"

    def test_nested_credential_under_home_rejected(self, home_env):
        home, _ = home_env
        nested = home / "proj" / ".ssh"
        nested.mkdir(parents=True)
        key = nested / "id_rsa"
        key.write_text("PRIVATE KEY", encoding="utf-8")
        with pytest.raises(ConfinementError, match="credential"):
            confine_index_path(key, _wl())

    def test_credential_name_case_insensitive(self, home_env):
        home, _ = home_env
        d = home / "docs" / ".SSH"
        d.mkdir(parents=True)
        f = d / "key"
        f.write_text("x", encoding="utf-8")
        with pytest.raises(ConfinementError, match="credential"):
            confine_index_path(f, _wl())

    def test_cli_unconfined_but_credential_floor_holds(self, home_env, tmp_path):
        # policy=None: an ordinary path anywhere is allowed (the CLI contract),
        # INCLUDING the data dir (not special-cased) - only the credential-folder
        # hard floor still denies anything.
        home, localm = home_env
        ok = tmp_path / "anywhere.txt"
        ok.write_text("x", encoding="utf-8")
        assert confine_index_path(ok, None) == ok.resolve()
        assert confine_index_path(localm / "registry.json", None) == \
            (localm / "registry.json").resolve()
        ssh = home / ".ssh"
        ssh.mkdir()
        (ssh / "id").write_text("k", encoding="utf-8")
        with pytest.raises(ConfinementError):
            confine_index_path(ssh / "id", None)


# --------------------------------------------------------------------------- #
#  UNC / device path guard                                                    #
# --------------------------------------------------------------------------- #

# A non-routable RFC5737 (TEST-NET-1) address, so it never reaches a real host.
_UNC = r"\\192.0.2.1\share"
_UNC_FWD = "//192.0.2.1/share"
_DEVICE = r"\\.\PhysicalDrive0"


def _is_unc_or_device(s: str) -> bool:
    """pathsafe.is_unc_or_device_path's forbidden-prefix check, judged by
    Windows rules on every host, not gated on os.name - confine_index_path's
    `p` is HTTP-API-reachable, so the guard refuses `//host/share` everywhere."""
    return s[:2] in ("\\\\", "//", "\\/", "/\\")


class TestUncDeviceGuard:
    """confine_index_path() applies a LEXICAL UNC/device check before
    ``expanduser().resolve()``. Every credential/policy check in that function
    runs after resolve(), so none of them can prevent an SMB dial."""

    @pytest.mark.parametrize("bad", [_UNC, _UNC_FWD, _DEVICE])
    def test_rejected_without_touching_the_filesystem(self, home_env, monkeypatch, bad):
        real_resolve = Path.resolve
        seen: list = []

        def spy(self, *a, **kw):
            s = str(self)
            seen.append(s)
            if _is_unc_or_device(s):
                raise AssertionError(
                    f"Path.resolve() reached the filesystem with a UNC/device "
                    f"string: {s!r} - this is the SMB dial (and the "
                    "net-NTLMv2 leak), which happens before any exception is "
                    "raised")
            return real_resolve(self, *a, **kw)

        monkeypatch.setattr(Path, "resolve", spy)
        with pytest.raises(ConfinementError) as ei:
            confine_index_path(bad, _wl())
        assert ei.value.reason == "unc_or_device"
        assert not any(_is_unc_or_device(s) for s in seen), (
            "the UNC/device string reached Path.resolve() - the whole finding "
            "is that this syscall happens before the confinement checks below "
            "it get a chance to refuse it")

    def test_rejected_in_blacklist_mode_and_cli_mode_too(self, home_env):
        # Mode-independent, like the credential/secret-file hard floor: refused in
        # blacklist mode (denial matching needs resolve() to have already run) and
        # for the unconfined CLI (policy=None).
        with pytest.raises(ConfinementError) as ei:
            confine_index_path(_UNC, _bl())
        assert ei.value.reason == "unc_or_device"
        with pytest.raises(ConfinementError) as ei2:
            confine_index_path(_UNC, None)
        assert ei2.value.reason == "unc_or_device"

    def test_ordinary_path_still_resolves(self, home_env):
        """Control: an ordinary local path (not UNC/device syntax) still reaches
        resolve() and is confined normally."""
        home, _ = home_env
        f = home / "docs" / "a.txt"
        f.write_text("hi", encoding="utf-8")
        assert confine_index_path(f, _wl()) == f.resolve()

    @pytest.mark.skipif(
        os.name != "nt",
        reason="HOMEDRIVE/HOMEPATH take precedence over USERPROFILE and this "
               "repro clears them so USERPROFILE wins; POSIX expands ~ from "
               "the password database and ignores USERPROFILE entirely, so "
               "this mechanism is Windows-specific")
    def test_rejects_a_path_that_expands_into_unc_via_userprofile(
            self, home_env, monkeypatch):
        """The UNC check runs on the EXPANDED path, not the raw string: a
        `~`-prefixed path is not UNC-shaped as written and only becomes one
        after expanduser() resolves ~ against the server's own USERPROFILE
        (a roaming profile pointing the home directory at a network share).
        Every other UNC test in this class uses an already-UNC-shaped raw
        string."""
        monkeypatch.setenv("USERPROFILE", _UNC)
        monkeypatch.delenv("HOMEDRIVE", raising=False)
        monkeypatch.delenv("HOMEPATH", raising=False)
        expanded = str(Path("~/proj").expanduser())
        assert _is_unc_or_device(expanded), (
            f"test setup did not produce a UNC path: expanduser() gave {expanded!r}")

        real_resolve = Path.resolve
        seen: list = []

        def spy(self, *a, **kw):
            s = str(self)
            seen.append(s)
            if _is_unc_or_device(s):
                raise AssertionError(
                    f"Path.resolve() reached the filesystem with a UNC/device "
                    f"string: {s!r}")
            return real_resolve(self, *a, **kw)

        monkeypatch.setattr(Path, "resolve", spy)
        with pytest.raises(ConfinementError) as ei:
            confine_index_path("~/proj", _wl())
        assert ei.value.reason == "unc_or_device"
        assert not any(_is_unc_or_device(s) for s in seen)


# --------------------------------------------------------------------------- #
#  indexing_policy()                                                          #
# --------------------------------------------------------------------------- #

class TestIndexingPolicy:
    def test_default_mode_is_whitelist(self, home_env):
        assert indexing_policy()["mode"] == "whitelist"

    def test_reads_mode_and_both_lists(self, home_env, tmp_path, monkeypatch):
        a = tmp_path / "a"
        d = tmp_path / "d"
        a.mkdir()
        d.mkdir()
        import localm.config as cfg
        monkeypatch.setattr(cfg, "load_config", lambda: {
            "rag_indexing_mode": "blacklist",
            "rag_allowed_roots": [str(a)],
            "rag_denied_roots": [str(d)]})
        pol = indexing_policy()
        assert pol["mode"] == "blacklist"
        assert a.resolve() in pol["allowed"]
        assert d.resolve() in pol["denied"]

    def test_bad_mode_falls_back_to_whitelist(self, home_env, monkeypatch):
        import localm.config as cfg
        monkeypatch.setattr(cfg, "load_config",
                            lambda: {"rag_indexing_mode": "bogus"})
        assert indexing_policy()["mode"] == "whitelist"

    def test_allow_network_drives_defaults_true(self, home_env):
        """Matches config.py's DEFAULT_CONFIG: a mapped drive works like a local
        one unless a caller explicitly turns it off."""
        assert indexing_policy()["allow_network_drives"] is True

    def test_allow_network_drives_reads_config(self, home_env, monkeypatch):
        import localm.config as cfg
        monkeypatch.setattr(cfg, "load_config",
                            lambda: {"allow_network_drives": False})
        assert indexing_policy()["allow_network_drives"] is False


# --------------------------------------------------------------------------- #
#  Mapped network drive guard (GetDriveTypeW)                                 #
# --------------------------------------------------------------------------- #

class TestNetworkDriveGuard:
    """A mapped Windows drive letter (Z:\\...) is syntactically an ordinary
    local path - is_unc_or_device_path correctly returns False for it, since
    "Z:" IS the local-drive form. confine_index_path's own network-drive
    check (pathsafe.is_mapped_network_drive) is what tells the two apart, and
    unlike the UNC guard above it is a config-gated PREFERENCE, not an
    unconditional hard floor: it must never fire unless GetDriveTypeW itself
    says the drive is remote AND allow_network_drives is off."""

    @pytest.mark.skipif(os.name != "nt", reason="GetDriveTypeW is Windows-only")
    def test_rejected_in_whitelist_and_blacklist_mode_when_disallowed(
            self, home_env, tmp_path, monkeypatch):
        import ctypes
        monkeypatch.setattr(ctypes.windll.kernel32, "GetDriveTypeW",
                            lambda root: 4)   # DRIVE_REMOTE
        f = tmp_path / "shared" / "a.txt"
        f.parent.mkdir(parents=True)
        f.write_text("x", encoding="utf-8")
        for pol in ({**_wl(f.parent), "allow_network_drives": False},
                    {**_bl(), "allow_network_drives": False}):
            with pytest.raises(ConfinementError) as ei:
                confine_index_path(f, pol)
            assert ei.value.reason == "network_drive_denied"

    @pytest.mark.skipif(os.name != "nt", reason="GetDriveTypeW is Windows-only")
    def test_allowed_by_default_even_on_a_network_drive(
            self, home_env, tmp_path, monkeypatch):
        """The default (allow_network_drives unset -> True): a mapped drive
        indexes normally."""
        import ctypes
        monkeypatch.setattr(ctypes.windll.kernel32, "GetDriveTypeW",
                            lambda root: 4)
        f = tmp_path / "shared" / "a.txt"
        f.parent.mkdir(parents=True)
        f.write_text("x", encoding="utf-8")
        assert confine_index_path(f, _wl(f.parent)) == f.resolve()
        assert confine_index_path(f, None) == f.resolve()

    def test_ordinary_local_drive_never_rejected_even_when_disallowed(
            self, home_env, tmp_path):
        """Control: this box's real tmp_path drive is NOT a network share, so
        turning allow_network_drives off does not reject it. The guard is keyed
        on GetDriveTypeW's answer, not on the config flag alone."""
        f = tmp_path / "a.txt"
        f.write_text("x", encoding="utf-8")
        pol = {**_wl(tmp_path), "allow_network_drives": False}
        assert confine_index_path(f, pol) == f.resolve()

    @pytest.mark.skipif(os.name != "nt", reason="GetDriveTypeW is Windows-only")
    def test_policy_none_reads_config_fresh(self, home_env, tmp_path, monkeypatch):
        """policy=None (the CLI / settings_schema.py's own save-time
        validation) has no indexing_policy() dict to read the flag off, so it
        falls back to a fresh config read rather than skipping the check."""
        import ctypes
        import localm.config as cfg
        monkeypatch.setattr(ctypes.windll.kernel32, "GetDriveTypeW",
                            lambda root: 4)
        monkeypatch.setattr(cfg, "load_config",
                            lambda: {"allow_network_drives": False})
        f = tmp_path / "shared" / "a.txt"
        f.parent.mkdir(parents=True)
        f.write_text("x", encoding="utf-8")
        with pytest.raises(ConfinementError) as ei:
            confine_index_path(f, None)
        assert ei.value.reason == "network_drive_denied"

    @pytest.mark.skipif(os.name != "nt", reason="GetDriveTypeW is Windows-only")
    def test_key_scoped_policy_still_respects_the_global_toggle(
            self, home_env, tmp_path, monkeypatch):
        """indexing_policy(key_roots=...) replaces the whitelist SET for a
        per-key-scoped caller, but allow_network_drives is a WHOLE-MACHINE
        preference, not a per-key one: a scoped key cannot reach a network
        drive the owner disallowed."""
        import ctypes
        import localm.config as cfg
        monkeypatch.setattr(ctypes.windll.kernel32, "GetDriveTypeW",
                            lambda root: 4)
        monkeypatch.setattr(cfg, "load_config",
                            lambda: {"allow_network_drives": False})
        shared = tmp_path / "shared"
        f = shared / "a.txt"
        shared.mkdir(parents=True)
        f.write_text("x", encoding="utf-8")
        pol = indexing_policy(key_roots=[str(shared)])
        assert pol["key_scoped"] is True
        assert pol["allow_network_drives"] is False
        with pytest.raises(ConfinementError) as ei:
            confine_index_path(f, pol)
        assert ei.value.reason == "network_drive_denied"

    def test_key_scoped_policy_defaults_to_allowed(self, home_env, tmp_path):
        """Control: with no config override, a key-scoped policy still
        defaults to allowed, matching the global policy's own default."""
        shared = tmp_path / "shared"
        shared.mkdir()
        assert indexing_policy(key_roots=[str(shared)])["allow_network_drives"] is True


# --------------------------------------------------------------------------- #
#  Per-key rag_roots: indexing_policy(key_roots=...) and confine_index_path() #
#  key_scoped branch. A key-scoped policy REPLACES the whitelist set entirely: #
#  home/cwd/the global rag_allowed_roots are NOT implied on top of it, unlike #
#  the default (global) policy above. The hard floor (credential dirs, secret #
#  files, UNC/device paths) still applies underneath either policy shape.     #
# --------------------------------------------------------------------------- #

class TestIndexingPolicyKeyScoped:
    def test_no_key_roots_is_unaffected(self, home_env):
        # None and [] both fall through to the ordinary global policy.
        assert indexing_policy(key_roots=None).get("key_scoped") is not True
        assert indexing_policy(key_roots=[]).get("key_scoped") is not True

    def test_key_roots_forces_a_scoped_whitelist(self, home_env, tmp_path):
        a = tmp_path / "a"
        a.mkdir()
        pol = indexing_policy(key_roots=[str(a)])
        assert pol["mode"] == "whitelist"
        assert pol["key_scoped"] is True
        assert pol["allowed"] == [a.resolve()]
        assert pol["denied"] == []

    def test_key_roots_ignores_config_entirely(self, home_env, tmp_path, monkeypatch):
        # A non-empty key_roots overrides the global blacklist mode and its own
        # allow/deny lists: a per-key allowlist does not inherit the global policy.
        a = tmp_path / "a"
        cfg_allowed = tmp_path / "cfg_allowed"
        a.mkdir()
        cfg_allowed.mkdir()
        import localm.config as cfg
        monkeypatch.setattr(cfg, "load_config", lambda: {
            "rag_indexing_mode": "blacklist",
            "rag_allowed_roots": [str(cfg_allowed)],
            "rag_denied_roots": []})
        pol = indexing_policy(key_roots=[str(a)])
        assert pol["mode"] == "whitelist"
        assert pol["allowed"] == [a.resolve()]
        assert cfg_allowed.resolve() not in pol["allowed"]

    def test_unresolvable_key_root_is_dropped_not_crashed(self, home_env):
        # Same fail-closed shape as the global _resolve() helper: an entry that
        # cannot be resolved is dropped, not raised.
        pol = indexing_policy(key_roots=["\x00bad\x00path"])
        assert pol["allowed"] == []


class TestConfineIndexPathKeyScoped:
    def _ks(self, *allowed):
        """A key-scoped whitelist policy allowing ONLY *allowed* - no implicit
        home/cwd, unlike _wl() above."""
        return {"mode": "whitelist", "key_scoped": True,
                "allowed": [Path(a) for a in allowed], "denied": []}

    def test_path_inside_granted_root_ok(self, home_env, tmp_path):
        granted = tmp_path / "granted"
        granted.mkdir()
        f = granted / "a.txt"
        f.write_text("hi", encoding="utf-8")
        assert confine_index_path(f, self._ks(granted)) == f.resolve()

    def test_home_not_implicitly_allowed(self, home_env):
        # Home is NOT granted here, unlike the default global policy.
        home, _ = home_env
        f = home / "docs" / "a.txt"
        f.write_text("hi", encoding="utf-8")
        other_root = home.parent / "elsewhere"
        other_root.mkdir()
        with pytest.raises(ConfinementError) as ei:
            confine_index_path(f, self._ks(other_root))
        assert ei.value.reason == "outside_allowed"

    def test_cwd_not_implicitly_allowed(self, home_env, tmp_path, monkeypatch):
        cwd = tmp_path / "workdir"
        cwd.mkdir()
        f = cwd / "a.txt"
        f.write_text("hi", encoding="utf-8")
        monkeypatch.chdir(cwd)
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        with pytest.raises(ConfinementError) as ei:
            confine_index_path(f, self._ks(elsewhere))
        assert ei.value.reason == "outside_allowed"

    def test_outside_granted_root_rejected(self, home_env, tmp_path):
        granted = tmp_path / "granted"
        granted.mkdir()
        outside = tmp_path / "elsewhere" / "x.txt"
        outside.parent.mkdir(parents=True)
        outside.write_text("secret", encoding="utf-8")
        with pytest.raises(ConfinementError) as ei:
            confine_index_path(outside, self._ks(granted))
        assert ei.value.reason == "outside_allowed"

    def test_hard_floor_still_applies_inside_granted_root(self, home_env, tmp_path):
        # A credential folder is refused even INSIDE a key's own granted root -
        # the hard floor is mode-independent and policy-independent.
        granted = tmp_path / "granted"
        ssh = granted / ".ssh"
        ssh.mkdir(parents=True)
        key = ssh / "id_rsa"
        key.write_text("PRIVATE KEY", encoding="utf-8")
        with pytest.raises(ConfinementError) as ei:
            confine_index_path(key, self._ks(granted))
        assert ei.value.reason == "credential"

    def test_secret_file_still_refused_inside_granted_root(self, home_env, tmp_path):
        granted = tmp_path / "granted"
        granted.mkdir()
        pem = granted / "deploy.pem"
        pem.write_text(_PEM, encoding="utf-8")
        with pytest.raises(ConfinementError) as ei:
            confine_index_path(pem, self._ks(granted))
        assert ei.value.reason == "secret_file"


# --------------------------------------------------------------------------- #
#  add_paths(policy=...) (unit)                                               #
# --------------------------------------------------------------------------- #

class TestAddPathsConfinement:
    def test_in_home_indexes(self, home_env, tmp_path):
        home, _ = home_env
        (home / "docs" / "a.txt").write_text(
            "rocm gfx1030 runtime dll", encoding="utf-8")
        c = Collection("kb", base=tmp_path / "rag").create()
        res = c.add_paths([home / "docs"], policy=_wl())
        assert res["added"] == 1

    def test_out_of_root_raises(self, home_env, tmp_path):
        home, _ = home_env
        outside = tmp_path / "out"
        outside.mkdir()
        (outside / "x.txt").write_text("secret", encoding="utf-8")
        c = Collection("kb", base=tmp_path / "rag").create()
        with pytest.raises(ValueError):
            c.add_paths([outside], policy=_wl())

    def test_nested_secret_is_skipped_but_data_dir_is_not(self, home_env, tmp_path):
        # Third-party credential folders (.ssh) are still skipped by a folder
        # walk. The localm data dir is NOT: registry.json indexes like any
        # other file under an allowed root - it is not special-cased.
        home, localm = home_env
        (home / "docs" / "good.txt").write_text(
            "ordinary indexable document", encoding="utf-8")
        ssh = home / ".ssh"
        ssh.mkdir()
        (ssh / "id_rsa.txt").write_text("PRIVATE KEY MATERIAL", encoding="utf-8")
        (localm / "registry.json").write_text(
            '{"secret_model": true}', encoding="utf-8")
        c = Collection("kb", base=tmp_path / "rag").create()
        c.add_paths([home], policy=_wl())
        sources = " ".join(d["path"] for d in c.docs())
        assert "good.txt" in sources
        assert "id_rsa" not in sources
        assert "registry.json" in sources


# --------------------------------------------------------------------------- #
#  HTTP route /api/rag/collections/{name}/add                                 #
# --------------------------------------------------------------------------- #

@pytest.fixture
def rag_route_app(tmp_path, monkeypatch):
    """Minimal GUI app with the builtin rag plugin enabled; Path.home == tmp_path
    so docs placed under tmp_path are inside the whitelist. Open mode -> the
    caller is the owner."""
    from localm.plugins.engine import PluginManager
    from localm.plugins.gui.web import attach_gui
    home = tmp_path
    localm = home / ".localm"
    monkeypatch.setenv("LOCALM_HOME", str(localm))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", localm)
    monkeypatch.setattr(cfg, "MODELS_DIR", localm / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", localm / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", localm / "registry.json")

    app = FastAPI()
    PluginManager(app, external_root=tmp_path / "noplugins").install("rag")

    async def switch_model(name):
        pass
    attach_gui(app, self_url="http://127.0.0.1:9/v1",
               switch_model=switch_model, active_model=lambda: "model-a")
    return app, home


class TestRagAddRoute:
    def test_in_home_accepted(self, rag_route_app):
        app, home = rag_route_app
        docs = home / "kdocs"
        docs.mkdir()
        (docs / "a.md").write_text("rocm gfx1030 dll", encoding="utf-8")
        with TestClient(app) as client:
            client.post("/api/rag/collections", json={"name": "kb"})
            r = client.post("/api/rag/collections/kb/add",
                            json={"paths": [str(docs)], "embed": False})
            assert r.status_code == 200

    def test_out_of_whitelist_offers_consent_to_owner(self, rag_route_app,
                                                      tmp_path_factory):
        # An out-of-whitelist path is OFFERED to the owner (409 needs_consent)
        # rather than hard-blocked, so they can add it and continue.
        app, _ = rag_route_app
        outside = tmp_path_factory.mktemp("outside_home")
        secret = outside / "secret.txt"
        secret.write_text("TOPSECRET", encoding="utf-8")
        with TestClient(app) as client:
            client.post("/api/rag/collections", json={"name": "kb"})
            r = client.post("/api/rag/collections/kb/add",
                            json={"paths": [str(secret)], "embed": False})
            assert r.status_code == 409, r.text
            body = r.json()
            assert body["needs_consent"] is True
            assert any("secret.txt" in a for a in body["addable"])

    def test_data_dir_is_not_hard_blocked(self, rag_route_app):
        # The localm data directory is not special-cased: it indexes like any
        # other file under an allowed root (here, the home folder).
        app, home = rag_route_app
        localm = home / ".localm"
        localm.mkdir(exist_ok=True)
        keyfile = localm / "keys.json"
        keyfile.write_text("{}", encoding="utf-8")
        with TestClient(app) as client:
            client.post("/api/rag/collections", json={"name": "kb"})
            r = client.post("/api/rag/collections/kb/add",
                            json={"paths": [str(keyfile)], "embed": False})
            assert r.status_code == 200, r.text

    def test_data_dir_outside_defaults_gets_same_consent_flow_as_any_folder(
            self, rag_route_app, tmp_path_factory, monkeypatch):
        # The data directory is treated EXACTLY like any other folder outside the
        # default allowed roots (home + cwd): a miss offers 'add and continue'
        # (409), never a special hard block and never a silent auto-allow.
        app, home = rag_route_app
        localm_elsewhere = tmp_path_factory.mktemp("localm_home_elsewhere")
        monkeypatch.setenv("LOCALM_HOME", str(localm_elsewhere))
        import localm.config as cfg
        monkeypatch.setattr(cfg, "HOME_DIR", localm_elsewhere)
        keyfile = localm_elsewhere / "keys.json"
        keyfile.write_text("{}", encoding="utf-8")
        with TestClient(app) as client:
            client.post("/api/rag/collections", json={"name": "kb"})
            r = client.post("/api/rag/collections/kb/add",
                            json={"paths": [str(keyfile)], "embed": False})
            assert r.status_code == 409, r.text
            assert r.json()["needs_consent"] is True

    def test_non_owner_gets_403_not_409_on_whitelist_miss(
            self, tmp_path, monkeypatch, tmp_path_factory):
        # A scoped rag key (NOT the owner) is hard-refused (403) on a whitelist
        # miss rather than offered 409: only the owner can widen the list.
        from localm.plugins.engine import PluginManager
        from localm.plugins.gui.web import attach_gui
        home = tmp_path
        localm = home / ".localm"
        localm.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("LOCALM_HOME", str(localm))
        monkeypatch.setenv("LOCALM_API_KEY", "owner-key-xyz")   # owner configured
        monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
        import localm.config as cfg
        monkeypatch.setattr(cfg, "HOME_DIR", localm)
        monkeypatch.setattr(cfg, "MODELS_DIR", localm / "models")
        monkeypatch.setattr(cfg, "CONFIG_FILE", localm / "config.json")
        monkeypatch.setattr(cfg, "REGISTRY_FILE", localm / "registry.json")
        from localm import auth
        scoped = auth.create_key("dev", ["rag"])["key"]         # rag scope, NOT admin

        app = FastAPI()
        PluginManager(app, external_root=tmp_path / "noplugins").install("rag")

        async def switch_model(name):
            pass
        attach_gui(app, self_url="http://127.0.0.1:9/v1",
                   switch_model=switch_model, active_model=lambda: "model-a")

        outside = tmp_path_factory.mktemp("outside_home2")
        (outside / "x.txt").write_text("secret", encoding="utf-8")
        with TestClient(app) as client:
            sc = {"Authorization": f"Bearer {scoped}"}
            client.post("/api/rag/collections", json={"name": "kb"}, headers=sc)
            r = client.post("/api/rag/collections/kb/add",
                            json={"paths": [str(outside / "x.txt")], "embed": False},
                            headers=sc)
            assert r.status_code == 403, r.text
            assert "owner" in r.text.lower()

    @pytest.mark.parametrize("bad", [_UNC, _UNC_FWD, _DEVICE])
    def test_unc_and_device_path_rejected_without_touching_the_filesystem(
            self, rag_route_app, monkeypatch, bad):
        """The HTTP surface for the store.py-level guard above: POST
        /api/rag/collections/{name}/add's `paths` field reaches
        confine_index_path(), up to 50 entries per request, synchronously
        inside an async handler."""
        app, _ = rag_route_app
        real_resolve = Path.resolve
        seen: list = []

        def spy(self, *a, **kw):
            s = str(self)
            seen.append(s)
            if _is_unc_or_device(s):
                raise AssertionError(
                    f"Path.resolve() reached the filesystem with a UNC/device "
                    f"string: {s!r}")
            return real_resolve(self, *a, **kw)

        monkeypatch.setattr(Path, "resolve", spy)
        with TestClient(app) as client:
            client.post("/api/rag/collections", json={"name": "kb"})
            r = client.post("/api/rag/collections/kb/add",
                            json={"paths": [bad], "embed": False})
        assert r.status_code == 400, r.text
        assert not any(_is_unc_or_device(s) for s in seen), (
            "the UNC/device string reached Path.resolve() via the HTTP route")

    def test_blacklist_allows_outside_but_denies_listed(self, rag_route_app,
                                                        tmp_path_factory):
        app, _ = rag_route_app
        denied = tmp_path_factory.mktemp("denied_zone")
        (denied / "s.txt").write_text("x", encoding="utf-8")
        free = tmp_path_factory.mktemp("free_zone")
        (free / "ok.md").write_text("rocm gfx1030 dll", encoding="utf-8")
        from localm.config import load_config, save_config
        cfg = load_config()
        cfg["rag_indexing_mode"] = "blacklist"
        cfg["rag_denied_roots"] = [str(denied)]
        save_config(cfg)
        with TestClient(app) as client:
            client.post("/api/rag/collections", json={"name": "kb"})
            # outside home, not denied -> indexed (blacklist allows).
            r1 = client.post("/api/rag/collections/kb/add",
                             json={"paths": [str(free)], "embed": False})
            assert r1.status_code == 200, r1.text
            # a denied folder -> hard 400 (an explicit deny is not offered).
            r2 = client.post("/api/rag/collections/kb/add",
                             json={"paths": [str(denied / "s.txt")], "embed": False})
            assert r2.status_code == 400, r2.text


# --------------------------------------------------------------------------- #
#  Per-key rag_roots at the HTTP route: a key minted with an explicit         #
#  rag_roots allowlist (auth.create_key(rag_roots=[...])) is confined to      #
#  exactly those folders through the real /add route, and home is NOT         #
#  implicitly reachable for it. An ordinary scoped key with no rag_roots set  #
#  keeps its home+cwd+global-allowed reach.                                   #
# --------------------------------------------------------------------------- #

def _scoped_rag_app(tmp_path, monkeypatch, *, rag_roots=None):
    """An owner-configured app (like test_non_owner_gets_403_not_409_on_whitelist_
    miss above) plus a non-owner 'rag'-scoped key, optionally minted with a
    per-key rag_roots allowlist. Returns (app, home, scoped_key_headers)."""
    from localm.plugins.engine import PluginManager
    from localm.plugins.gui.web import attach_gui
    home = tmp_path
    localm = home / ".localm"
    localm.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LOCALM_HOME", str(localm))
    monkeypatch.setenv("LOCALM_API_KEY", "owner-key-xyz")   # owner configured
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", localm)
    monkeypatch.setattr(cfg, "MODELS_DIR", localm / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", localm / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", localm / "registry.json")
    from localm import auth
    scoped = auth.create_key("dev", ["rag"], rag_roots=rag_roots)["key"]

    app = FastAPI()
    PluginManager(app, external_root=tmp_path / "noplugins").install("rag")

    async def switch_model(name):
        pass
    attach_gui(app, self_url="http://127.0.0.1:9/v1",
              switch_model=switch_model, active_model=lambda: "model-a")
    return app, home, {"Authorization": f"Bearer {scoped}"}


class TestRagAddRouteKeyScopedRoots:
    def test_path_inside_granted_root_indexes(self, tmp_path, monkeypatch):
        granted = tmp_path / "granted"
        granted.mkdir()
        (granted / "a.md").write_text("rocm gfx1030 dll", encoding="utf-8")
        app, _, hdr = _scoped_rag_app(tmp_path, monkeypatch,
                                      rag_roots=[str(granted)])
        with TestClient(app) as client:
            client.post("/api/rag/collections", json={"name": "kb"}, headers=hdr)
            r = client.post("/api/rag/collections/kb/add",
                            json={"paths": [str(granted)], "embed": False},
                            headers=hdr)
            assert r.status_code == 200, r.text

    def test_home_is_refused_when_key_has_its_own_roots(self, tmp_path, monkeypatch):
        # A key given its own explicit roots does not also inherit the
        # home-directory default every other caller gets.
        granted = tmp_path / "granted"
        granted.mkdir()
        app, home, hdr = _scoped_rag_app(tmp_path, monkeypatch,
                                         rag_roots=[str(granted)])
        (home / "docs").mkdir()
        (home / "docs" / "a.md").write_text("rocm gfx1030 dll", encoding="utf-8")
        with TestClient(app) as client:
            client.post("/api/rag/collections", json={"name": "kb"}, headers=hdr)
            r = client.post("/api/rag/collections/kb/add",
                            json={"paths": [str(home / "docs")], "embed": False},
                            headers=hdr)
            assert r.status_code == 403, r.text
            assert "owner" in r.text.lower()

    def test_owner_is_never_confined_by_a_key_scoped_field(self, tmp_path,
                                                            monkeypatch):
        # A key's own rag_roots field never applies to the OWNER caller: the
        # owner authenticates with the owner key, not the scoped key, which
        # exercises effective_rag_roots' ADMIN short-circuit end to end.
        granted = tmp_path / "granted"
        granted.mkdir()
        app, home, _hdr = _scoped_rag_app(tmp_path, monkeypatch,
                                          rag_roots=[str(granted)])
        (home / "docs").mkdir()
        (home / "docs" / "a.md").write_text("rocm gfx1030 dll", encoding="utf-8")
        owner_hdr = {"Authorization": "Bearer owner-key-xyz"}
        with TestClient(app) as client:
            client.post("/api/rag/collections", json={"name": "kb"},
                        headers=owner_hdr)
            r = client.post("/api/rag/collections/kb/add",
                            json={"paths": [str(home / "docs")], "embed": False},
                            headers=owner_hdr)
            assert r.status_code == 200, r.text

    def test_key_without_rag_roots_keeps_home_reach(self, tmp_path, monkeypatch):
        # A non-owner key that never had rag_roots set (the default, [] via
        # auth.create_key) still reaches home.
        app, home, hdr = _scoped_rag_app(tmp_path, monkeypatch, rag_roots=None)
        (home / "docs").mkdir()
        (home / "docs" / "a.md").write_text("rocm gfx1030 dll", encoding="utf-8")
        with TestClient(app) as client:
            client.post("/api/rag/collections", json={"name": "kb"}, headers=hdr)
            r = client.post("/api/rag/collections/kb/add",
                            json={"paths": [str(home / "docs")], "embed": False},
                            headers=hdr)
            assert r.status_code == 200, r.text

    def test_credential_dir_still_refused_inside_a_granted_root(self, tmp_path,
                                                                 monkeypatch):
        granted = tmp_path / "granted"
        ssh = granted / ".ssh"
        ssh.mkdir(parents=True)
        (ssh / "id_rsa").write_text("PRIVATE KEY", encoding="utf-8")
        app, _, hdr = _scoped_rag_app(tmp_path, monkeypatch,
                                      rag_roots=[str(granted)])
        with TestClient(app) as client:
            client.post("/api/rag/collections", json={"name": "kb"}, headers=hdr)
            r = client.post("/api/rag/collections/kb/add",
                            json={"paths": [str(ssh / "id_rsa")], "embed": False},
                            headers=hdr)
            assert r.status_code == 400, r.text


# --------------------------------------------------------------------------- #
#  Explicitly-named secret / binary files under a policy                      #
#                                                                             #
#  The folder WALK skips model weights and secret material                    #
#  (BLACKLISTED_SUFFIXES / SECRET_INDEX_NAMES). The same filter applies to an #
#  EXPLICITLY-named top-level file whenever a policy is present (the API      #
#  path). The CLI (policy=None) stays unconfined.                             #
# --------------------------------------------------------------------------- #

# Stand-in for key/cert material: plain text, so without the filter it would
# sniff as .txt and be indexed verbatim. Kept free of a real PEM header, since
# the block keys off the file's SUFFIX / NAME rather than its bytes. The unique
# token lets a test assert the content is never retrievable back out.
_PEM = ("PRIVATE-KEY-PLACEHOLDER-DO-NOT-INDEX\n"
        "body SUPERSECRETKEYMATERIAL0123456789 not-a-real-key\n")


class TestExplicitSecretFileUnderPolicy:
    def test_explicit_pem_blocked_by_policy(self, home_env):
        home, _ = home_env
        pem = home / "docs" / "deploy.pem"
        pem.write_text(_PEM, encoding="utf-8")
        with pytest.raises(ConfinementError) as ei:
            confine_index_path(pem, _wl())
        assert ei.value.reason == "secret_file"

    def test_explicit_key_suffix_blocked(self, home_env):
        home, _ = home_env
        key = home / "docs" / "tls.key"
        key.write_text("key placeholder", encoding="utf-8")
        with pytest.raises(ConfinementError) as ei:
            confine_index_path(key, _wl())
        assert ei.value.reason == "secret_file"

    def test_explicit_extensionless_secret_name_blocked(self, home_env):
        home, _ = home_env
        key = home / "docs" / "id_rsa"          # secret NAME, no suffix
        key.write_text("ssh key placeholder", encoding="utf-8")
        with pytest.raises(ConfinementError) as ei:
            confine_index_path(key, _wl())
        assert ei.value.reason == "secret_file"

    def test_explicit_dotenv_blocked(self, home_env):
        home, _ = home_env
        env = home / "docs" / ".env"
        env.write_text("AWS_SECRET_ACCESS_KEY=xyz", encoding="utf-8")
        with pytest.raises(ConfinementError) as ei:
            confine_index_path(env, _wl())
        assert ei.value.reason == "secret_file"

    def test_secret_blocked_in_blacklist_mode_too(self, home_env):
        # It is a hard, mode-independent refusal: a .pem is refused even where the
        # location itself is allowed (blacklist mode, not on any denied root).
        home, _ = home_env
        pem = home / "docs" / "server.pem"
        pem.write_text(_PEM, encoding="utf-8")
        with pytest.raises(ConfinementError) as ei:
            confine_index_path(pem, _bl())
        assert ei.value.reason == "secret_file"

    def test_secret_takes_precedence_over_outside_allowed(self, home_env, tmp_path):
        # A secret file OUTSIDE the whitelist must be refused as a secret, NEVER
        # offered back through the "add this folder and continue" consent flow.
        outside = tmp_path / "external"
        outside.mkdir()
        pem = outside / "id_ed25519"
        pem.write_text("ssh key", encoding="utf-8")
        with pytest.raises(ConfinementError) as ei:
            confine_index_path(pem, _wl())
        assert ei.value.reason == "secret_file"   # not "outside_allowed"

    @pytest.mark.parametrize("fname", [
        "deploy.ppk", "signing.p8", "key.pk8", "bundle.pkcs12",
        "chain.p7b", "chain.p7c", "client.ovpn",   # by suffix
        ".envrc",                                   # by secret name
    ])
    def test_additional_credential_formats_blocked(self, home_env, fname):
        # Denylist completeness: common private-key / credential formats are
        # refused for both the walk and an explicit API pick.
        home, _ = home_env
        f = home / "docs" / fname
        f.write_text("secret-ish placeholder body", encoding="utf-8")
        with pytest.raises(ConfinementError) as ei:
            confine_index_path(f, _wl())
        assert ei.value.reason == "secret_file"

    def test_cli_policy_none_still_honours_explicit_secret(self, home_env):
        # The CLI local operator (policy=None) stays unconfined for an explicit
        # single-file pick.
        home, _ = home_env
        pem = home / "docs" / "deploy.pem"
        pem.write_text(_PEM, encoding="utf-8")
        assert confine_index_path(pem, None) == pem.resolve()

    def test_directory_named_credentials_not_over_blocked(self, home_env):
        # is_secret_index_name("credentials") is True, but the secret filter is
        # for FILES only: a real folder named "credentials" is still walkable, its
        # non-secret contents index, and its secret contents are skipped by the
        # walk.
        home, _ = home_env
        d = home / "docs" / "credentials"       # a directory, not a file
        d.mkdir()
        assert confine_index_path(d, _wl()) == d.resolve()


class TestExpandAndAddPathsSecretFilter:
    def test_expand_drops_explicit_secret_under_policy(self, home_env):
        # _expand([pem], policy) does not return the pem.
        home, _ = home_env
        pem = home / "docs" / "deploy.pem"
        pem.write_text(_PEM, encoding="utf-8")
        assert Collection._expand([pem], _wl()) == []

    def test_expand_keeps_explicit_secret_without_policy(self, home_env):
        # CLI path (policy=None): explicit pick still honoured.
        home, _ = home_env
        pem = home / "docs" / "deploy.pem"
        pem.write_text(_PEM, encoding="utf-8")
        assert Collection._expand([pem]) == [pem.resolve()]

    def test_add_paths_explicit_secret_raises_under_policy(self, home_env, tmp_path):
        # add_paths([pem], policy) does not add the pem.
        home, _ = home_env
        pem = home / "docs" / "deploy.pem"
        pem.write_text(_PEM, encoding="utf-8")
        c = Collection("kb", base=tmp_path / "rag").create()
        with pytest.raises(ConfinementError):
            c.add_paths([pem], policy=_wl())
        # And nothing leaked into the index.
        assert c.docs() == []

    def test_add_paths_folder_indexes_good_skips_secret(self, home_env, tmp_path):
        # A folder holding a good doc AND a secret: the doc indexes, the secret does
        # not (walk filter), even though the folder itself is an allowed location.
        home, _ = home_env
        (home / "docs" / "good.txt").write_text(
            "ordinary indexable document", encoding="utf-8")
        (home / "docs" / "deploy.pem").write_text(_PEM, encoding="utf-8")
        c = Collection("kb2", base=tmp_path / "rag").create()
        res = c.add_paths([home / "docs"], policy=_wl())
        assert res["added"] == 1
        sources = " ".join(d["path"] for d in c.docs())
        assert "good.txt" in sources
        assert "deploy.pem" not in sources

    def test_cli_add_paths_can_index_explicit_secret(self, home_env, tmp_path):
        # The CLI (policy=None) still indexes an explicitly-picked secret file:
        # the local operator is unconfined.
        home, _ = home_env
        pem = home / "docs" / "deploy.pem"
        pem.write_text(_PEM, encoding="utf-8")
        c = Collection("kb3", base=tmp_path / "rag").create()
        res = c.add_paths([pem])                 # no policy -> CLI contract
        assert res["added"] == 1


class TestRagAddRouteSecretFile:
    def test_owner_explicit_secret_file_hard_blocked(self, rag_route_app):
        app, home = rag_route_app
        docs = home / "kdocs"
        docs.mkdir()
        pem = docs / "deploy.pem"               # inside the whitelist location...
        pem.write_text(_PEM, encoding="utf-8")  # ...but a credential file
        with TestClient(app) as client:
            client.post("/api/rag/collections", json={"name": "kb"})
            r = client.post("/api/rag/collections/kb/add",
                            json={"paths": [str(pem)], "embed": False})
            assert r.status_code == 400, r.text   # hard block, never offered (409)
            assert "deploy.pem" in r.text

    def test_secret_content_not_retrievable_after_block(self, rag_route_app):
        # End to end: a loopback/API caller cannot read a credential back out.
        app, home = rag_route_app
        docs = home / "kdocs"
        docs.mkdir()
        (docs / "notes.md").write_text("rocm gfx1030 dll", encoding="utf-8")
        (docs / "secret.pem").write_text(_PEM, encoding="utf-8")
        with TestClient(app) as client:
            client.post("/api/rag/collections", json={"name": "kb"})
            # The good doc indexes fine.
            assert client.post("/api/rag/collections/kb/add",
                               json={"paths": [str(docs / "notes.md")],
                                     "embed": False}).status_code == 200
            # The credential is refused up front.
            assert client.post("/api/rag/collections/kb/add",
                               json={"paths": [str(docs / "secret.pem")],
                                     "embed": False}).status_code == 400
            # ...and its contents are not retrievable.
            q = client.post("/api/rag/collections/kb/query",
                            json={"query": "SUPERSECRETKEYMATERIAL", "k": 5})
            assert q.status_code == 200, q.text
            blob = " ".join(h.get("text", "") for h in q.json()["hits"])
            assert "SUPERSECRETKEYMATERIAL" not in blob

    def test_scoped_key_also_blocked_from_secret_file(
            self, tmp_path, monkeypatch, tmp_path_factory):
        # A non-owner scoped rag key is blocked from indexing a secret file too
        # (400), refused as a secret rather than offered the owner-only 409/403
        # whitelist-widening path.
        from localm.plugins.engine import PluginManager
        from localm.plugins.gui.web import attach_gui
        home = tmp_path
        localm = home / ".localm"
        localm.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("LOCALM_HOME", str(localm))
        monkeypatch.setenv("LOCALM_API_KEY", "owner-key-xyz")
        monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
        import localm.config as cfg
        monkeypatch.setattr(cfg, "HOME_DIR", localm)
        monkeypatch.setattr(cfg, "MODELS_DIR", localm / "models")
        monkeypatch.setattr(cfg, "CONFIG_FILE", localm / "config.json")
        monkeypatch.setattr(cfg, "REGISTRY_FILE", localm / "registry.json")
        from localm import auth
        scoped = auth.create_key("dev", ["rag"])["key"]

        app = FastAPI()
        PluginManager(app, external_root=tmp_path / "noplugins").install("rag")

        async def switch_model(name):
            pass
        attach_gui(app, self_url="http://127.0.0.1:9/v1",
                   switch_model=switch_model, active_model=lambda: "model-a")

        pem = home / "deploy.pem"               # under home -> whitelist location
        pem.write_text(_PEM, encoding="utf-8")
        with TestClient(app) as client:
            sc = {"Authorization": f"Bearer {scoped}"}
            client.post("/api/rag/collections", json={"name": "kb"}, headers=sc)
            r = client.post("/api/rag/collections/kb/add",
                            json={"paths": [str(pem)], "embed": False}, headers=sc)
            assert r.status_code == 400, r.text


# --------------------------------------------------------------------------- #
#  Collection.confined_to: whether every host-filesystem document a          #
#  collection holds resolves under a set of per-key rag_roots, independent   #
#  of the HTTP layer.                                                        #
# --------------------------------------------------------------------------- #

class TestCollectionConfinedToKeyRoots:
    def test_no_key_roots_is_always_confined(self, tmp_path):
        Collection("kb", base=tmp_path / "rag").create()
        assert Collection.confined_to("kb", [], base=tmp_path / "rag") is True

    def test_doc_inside_granted_root_is_confined(self, tmp_path):
        granted = tmp_path / "granted"
        granted.mkdir()
        f = granted / "a.txt"
        f.write_text("hi", encoding="utf-8")
        c = Collection("kb", base=tmp_path / "rag").create()
        c.add_paths([f])
        assert Collection.confined_to(
            "kb", [str(granted)], base=tmp_path / "rag") is True

    def test_doc_outside_granted_root_is_not_confined(self, tmp_path):
        granted = tmp_path / "granted"
        granted.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        f = outside / "a.txt"
        f.write_text("hi", encoding="utf-8")
        c = Collection("kb", base=tmp_path / "rag").create()
        c.add_paths([f])
        assert Collection.confined_to(
            "kb", [str(granted)], base=tmp_path / "rag") is False

    def test_one_doc_outside_makes_the_whole_collection_not_confined(self, tmp_path):
        granted = tmp_path / "granted"
        granted.mkdir()
        inside = granted / "in.txt"
        inside.write_text("hi", encoding="utf-8")
        outside_dir = tmp_path / "outside"
        outside_dir.mkdir()
        outside = outside_dir / "out.txt"
        outside.write_text("hi", encoding="utf-8")
        c = Collection("kb", base=tmp_path / "rag").create()
        c.add_paths([inside, outside])
        assert Collection.confined_to(
            "kb", [str(granted)], base=tmp_path / "rag") is False

    def test_upload_doc_is_exempt_from_the_check(self, tmp_path):
        granted = tmp_path / "granted"
        granted.mkdir()
        c = Collection("kb", base=tmp_path / "rag").create()
        c.add_uploads([{"filename": "notes.txt", "data": b"hello world"}])
        assert Collection.confined_to(
            "kb", [str(granted)], base=tmp_path / "rag") is True

    def test_unreadable_collection_returns_none(self, tmp_path):
        assert Collection.confined_to(
            "nosuch", [str(tmp_path)], base=tmp_path / "rag") is None

    def test_every_key_root_unresolvable_fails_closed(self, tmp_path):
        granted = tmp_path / "granted"
        granted.mkdir()
        f = granted / "a.txt"
        f.write_text("hi", encoding="utf-8")
        c = Collection("kb", base=tmp_path / "rag").create()
        c.add_paths([f])
        assert Collection.confined_to(
            "kb", ["bad\x00root"], base=tmp_path / "rag") is False


# --------------------------------------------------------------------------- #
#  HTTP routes /api/rag/collections/{name}/query and GET /api/rag/collections #
#  under a per-key rag_roots allowlist: a collection holding any document     #
#  indexed from outside the key's granted roots - even one added by a        #
#  different, less-restricted caller into the SAME collection - is refused   #
#  on query and left out of the listing.                                     #
# --------------------------------------------------------------------------- #

def _await_job(app, job_id, timeout=30.0):
    jobs = app.state.jobs
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = jobs.get(job_id)
        assert job is not None, f"job {job_id} vanished from the registry"
        if job.status != "running":
            return job
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} still running after {timeout}s")


class TestRagQueryRouteKeyScopedRoots:
    def test_query_inside_granted_root_allowed(self, tmp_path, monkeypatch):
        granted = tmp_path / "granted"
        granted.mkdir()
        (granted / "a.md").write_text("rocm gfx1030 dll", encoding="utf-8")
        app, _, hdr = _scoped_rag_app(tmp_path, monkeypatch,
                                      rag_roots=[str(granted)])
        with TestClient(app) as client:
            client.post("/api/rag/collections", json={"name": "kb"}, headers=hdr)
            r_add = client.post("/api/rag/collections/kb/add",
                                json={"paths": [str(granted)], "embed": False},
                                headers=hdr)
            assert r_add.status_code == 200, r_add.text
            _await_job(app, r_add.json()["job_id"])
            r = client.post("/api/rag/collections/kb/query",
                            json={"query": "rocm gfx1030", "k": 5}, headers=hdr)
            assert r.status_code == 200, r.text
            assert r.json()["hits"]

    def test_query_refused_when_collection_holds_content_from_outside(
            self, tmp_path, monkeypatch):
        # The owner (unconfined) indexes a folder the scoped key was never
        # granted into the SAME collection name the scoped key will query.
        granted = tmp_path / "granted"
        granted.mkdir()
        app, home, hdr = _scoped_rag_app(tmp_path, monkeypatch,
                                         rag_roots=[str(granted)])
        owner_hdr = {"Authorization": "Bearer owner-key-xyz"}
        outside_dir = home / "elsewhere"
        outside_dir.mkdir()
        (outside_dir / "secret.md").write_text(
            "TOPSECRET rocm gfx1030 dll", encoding="utf-8")
        with TestClient(app) as client:
            client.post("/api/rag/collections", json={"name": "kb"},
                        headers=owner_hdr)
            r_add = client.post("/api/rag/collections/kb/add",
                                json={"paths": [str(outside_dir)], "embed": False},
                                headers=owner_hdr)
            assert r_add.status_code == 200, r_add.text
            _await_job(app, r_add.json()["job_id"])
            r = client.post("/api/rag/collections/kb/query",
                            json={"query": "TOPSECRET", "k": 5}, headers=hdr)
            assert r.status_code == 403, r.text
            assert "TOPSECRET" not in r.text

    def test_owner_query_unaffected_by_a_scoped_keys_roots(
            self, tmp_path, monkeypatch):
        granted = tmp_path / "granted"
        granted.mkdir()
        app, home, _hdr = _scoped_rag_app(tmp_path, monkeypatch,
                                          rag_roots=[str(granted)])
        owner_hdr = {"Authorization": "Bearer owner-key-xyz"}
        (home / "docs").mkdir()
        (home / "docs" / "a.md").write_text("rocm gfx1030 dll", encoding="utf-8")
        with TestClient(app) as client:
            client.post("/api/rag/collections", json={"name": "kb"},
                        headers=owner_hdr)
            r_add = client.post("/api/rag/collections/kb/add",
                                json={"paths": [str(home / "docs")], "embed": False},
                                headers=owner_hdr)
            assert r_add.status_code == 200, r_add.text
            _await_job(app, r_add.json()["job_id"])
            r = client.post("/api/rag/collections/kb/query",
                            json={"query": "rocm gfx1030", "k": 5},
                            headers=owner_hdr)
            assert r.status_code == 200, r.text
            assert r.json()["hits"]

    def test_upload_only_collection_queryable_by_scoped_key(
            self, tmp_path, monkeypatch):
        granted = tmp_path / "granted"
        granted.mkdir()
        app, _, hdr = _scoped_rag_app(tmp_path, monkeypatch,
                                      rag_roots=[str(granted)])
        content = base64.b64encode(b"rocm gfx1030 dll uploaded notes").decode()
        with TestClient(app) as client:
            client.post("/api/rag/collections", json={"name": "kb"}, headers=hdr)
            r_up = client.post(
                "/api/rag/collections/kb/upload",
                json={"files": [{"filename": "notes.txt",
                                 "content_b64": content}], "embed": False},
                headers=hdr)
            assert r_up.status_code == 200, r_up.text
            _await_job(app, r_up.json()["job_id"])
            r = client.post("/api/rag/collections/kb/query",
                            json={"query": "rocm gfx1030", "k": 5}, headers=hdr)
            assert r.status_code == 200, r.text
            assert r.json()["hits"]


class TestRagListRouteKeyScopedRoots:
    def test_list_omits_collection_with_content_outside_granted_root(
            self, tmp_path, monkeypatch):
        granted = tmp_path / "granted"
        granted.mkdir()
        (granted / "in.md").write_text("rocm gfx1030 dll", encoding="utf-8")
        app, home, hdr = _scoped_rag_app(tmp_path, monkeypatch,
                                         rag_roots=[str(granted)])
        owner_hdr = {"Authorization": "Bearer owner-key-xyz"}
        outside_dir = home / "elsewhere"
        outside_dir.mkdir()
        (outside_dir / "out.md").write_text("rocm gfx1030 dll", encoding="utf-8")
        with TestClient(app) as client:
            client.post("/api/rag/collections", json={"name": "allowed"},
                        headers=owner_hdr)
            r1 = client.post("/api/rag/collections/allowed/add",
                             json={"paths": [str(granted)], "embed": False},
                             headers=owner_hdr)
            _await_job(app, r1.json()["job_id"])

            client.post("/api/rag/collections", json={"name": "forbidden"},
                        headers=owner_hdr)
            r2 = client.post("/api/rag/collections/forbidden/add",
                             json={"paths": [str(outside_dir)], "embed": False},
                             headers=owner_hdr)
            _await_job(app, r2.json()["job_id"])

            r_scoped = client.get("/api/rag/collections", headers=hdr)
            assert r_scoped.status_code == 200, r_scoped.text
            names = {c["name"] for c in r_scoped.json()["collections"]}
            assert names == {"allowed"}

            r_owner = client.get("/api/rag/collections", headers=owner_hdr)
            assert r_owner.status_code == 200, r_owner.text
            owner_names = {c["name"] for c in r_owner.json()["collections"]}
            assert owner_names == {"allowed", "forbidden"}

# SPDX-License-Identifier: AGPL-3.0-or-later
"""Admin filesystem routes: UNC/device rejection, allowlist-before-syscall, and
the cross-origin refusal for the host browser (CodeQL 3 and 10).

WHY THE ASSERTIONS LOOK LIKE THIS. The defect is not the HTTP response, it is the
SYSCALL that happens on the way to it. ``Path.resolve`` on Windows calls
``_getfinalpathname`` (CreateFileW) plus a stat, so a UNC path aimed at an
attacker's SMB server makes an outbound connection that Windows AUTO-AUTHENTICATES,
surrendering the host's net-NTLMv2 credential - and that completes before any
allowlist check downstream can return its 403. Both branches returning the same
403 is exactly why an earlier reading called this harmless; the exfiltration
channel is the connection, not the response body.

It also blocks: measured on a Windows box, a UNC path to a non-routable RFC5737
address stalled 271.33 SECONDS before WinError 64, and an unresolvable UNC
hostname 9.88s - inside an ``async def``, i.e. the whole event loop, per request.

So the tests assert on the CODE PATH (no filesystem call is ever made on an
unsanitized UNC string), not on wall clock alone. Wall clock is Windows-specific
and a Linux CI run cannot reproduce the stall, so a timing-only test would
silently prove nothing there.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from localm import scopes as S
from localm.plugins.gui.web import attach_gui

# A UNC path at a non-routable RFC5737 documentation address (TEST-NET-1). It is
# guaranteed never to route anywhere, so even a total failure of the fix cannot
# reach a real host from a developer's machine or from CI.
UNC = r"\\192.0.2.1\share"
UNC_FWD = "//192.0.2.1/share"
DEVICE = r"\\.\PhysicalDrive0"

# Upper bound for "the request returned without dialling the UNC host". A
# BACKSTOP for the _unc_calls(fs_spy) assertions, not a duplicate of them: the
# spy only sees calls through the surfaces it patches, so a dial down some
# unpatched path would show up here as elapsed time and nowhere else.
#
# The value has to sit between two other numbers, and both matter:
#   ABOVE runner and shared-box jitter. 192.0.2.1 is TEST-NET-1, guaranteed
#   unroutable, so a passing run does no network work at all and the measured
#   time is pure scheduling noise. This was 1.0 and went red on master at 1.04s
#   (CI run for cb8bc936) with both security assertions already passed - the
#   tightest wall-clock bound in the repo, where every other one is 2.0s+.
#   FAR BELOW a real SMB connect timeout, which on Windows is tens of seconds.
# 5.0 keeps the full detection power (a genuine dial cannot finish under it)
# while leaving room for a noisy hosted runner.
#
# MUST STAY WALL-CLOCK. Do NOT "fix" flakiness here by switching to
# time.process_time() the way tests/test_redos_bounds.py correctly does: those
# tests detect a CPU-bound scan, this one detects a BLOCKING NETWORK WAIT, which
# burns no CPU at all. Measured: a 1.2s block registers 0.000s of process_time.
# The conversion would leave a green test that can never fire again.
_NO_DIAL_SECONDS = 5.0


# Every Path method here reaches the filesystem, and on Windows each one dials
# SMB for a UNC path. resolve() is the one CodeQL flagged, but it is NOT the only
# door: is_dir() alone reproduces the full stall.
_FS_METHODS = ("resolve", "is_dir", "is_file", "exists", "stat", "iterdir")


def _is_unc(s: str) -> bool:
    """The spy's forbidden-prefix rule, kept IDENTICAL to the product contract in
    pathsafe.reject_unsafe_path_string. ``\\\\`` is UNC/device syntax everywhere;
    ``//`` is UNC on Windows only, because on POSIX a leading ``//`` is a legal
    path prefix equivalent to ``/``. If the spy used the stricter rule on POSIX it
    would fail the suite on Linux CI for a path the product correctly allows."""
    return s.startswith("\\\\") or (os.name == "nt" and s.startswith("//"))


@pytest.fixture
def fs_spy(monkeypatch):
    """Record every path string that reaches a filesystem call, and hard-fail the
    call when the path is UNC/device syntax.

    Guarding ALL of _FS_METHODS rather than just resolve() is what makes this
    test suite safe to run. MEASURED: with the fix reverted and only resolve()
    guarded, this file ran past a 10-MINUTE timeout - because the unguarded
    is_dir() went on to make the real SMB dial to the RFC5737 address, exactly
    the 271-second stall the unit is about. A regression must fail in
    milliseconds with a clear message, never hang CI.

    Recording AND raising is deliberate: the handlers wrap their bodies in
    ``except Exception -> HTTPException(500)``, so a raise alone could be
    swallowed into a 500 that a status assertion might tolerate. The recorded
    list is checked independently and cannot be swallowed."""
    seen: list = []
    reals = {m: getattr(Path, m) for m in _FS_METHODS}

    def make(method_name):
        real = reals[method_name]

        def spy(self, *a, **kw):
            s = str(self)
            seen.append(s)
            if _is_unc(s):
                raise AssertionError(
                    f"Path.{method_name}() reached the filesystem with a UNC "
                    f"string: {s!r} - this is the SMB dial (and the net-NTLMv2 "
                    f"leak), which happens before any status code is chosen")
            return real(self, *a, **kw)

        return spy

    for m in _FS_METHODS:
        monkeypatch.setattr(Path, m, make(m))
    return seen


def _unc_calls(seen) -> list:
    return [s for s in seen if _is_unc(s)]


@pytest.fixture
def fs_app(tmp_path, monkeypatch):
    """GUI stack on a throwaway home. Mirrors tests/test_fs_picker.py so the two
    files exercise the same wiring."""
    home = tmp_path / ".localm"
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    import localm.config as _cfg
    monkeypatch.setattr(_cfg, "HOME_DIR", home)
    monkeypatch.setattr(_cfg, "MODELS_DIR", home / "models")
    monkeypatch.setattr(_cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(_cfg, "REGISTRY_FILE", home / "registry.json")
    from localm.plugins.engine import attach_engine
    app = FastAPI()
    attach_engine(app)
    attach_gui(app, self_url="http://127.0.0.1:9/v1",
               switch_model=lambda name: None,
               active_model=lambda: "model-a")
    return app


def _hdr(key):
    return {"Authorization": f"Bearer {key}"}


def _host_key():
    from localm import auth
    return auth.create_key("h", [S.CONFIG_READ], fs_access="host")["key"]


def _cfgwrite_key():
    """config:write is a PRIVILEGED scope, so minting it needs the owner-equivalent
    allow_privileged flag (same convention as tests/test_auth.py)."""
    from localm import auth
    return auth.create_key("w", [S.CONFIG_WRITE], allow_privileged=True,
                           fs_access="host")["key"]


# --------------------------------------------------------------------------- #
#  GET /api/fs/dirs - the host folder picker (CodeQL 10)                       #
# --------------------------------------------------------------------------- #

class TestFsDirsRejectsUncBeforeAnySyscall:

    @pytest.mark.parametrize("bad", [UNC, DEVICE, r"\\192.0.2.1\share\sub"])
    def test_unc_refused_without_touching_the_filesystem(
            self, fs_app, fs_spy, bad):
        with TestClient(fs_app) as c:
            t0 = time.perf_counter()
            r = c.get("/api/fs/dirs", params={"path": bad},
                      headers=_hdr(_host_key()))
            elapsed = time.perf_counter() - t0

        assert r.status_code == 400, r.text
        assert _unc_calls(fs_spy) == [], (
            "the UNC string reached a filesystem call - the SMB dial (and the "
            "net-NTLMv2 leak) happens there, before any status code is chosen")
        assert elapsed < _NO_DIAL_SECONDS, f"took {elapsed:.2f}s"

    def test_forward_slash_unc_refused_on_windows(self, fs_app, fs_spy):
        """``//host/share`` is an equivalent UNC spelling on Windows. On POSIX a
        leading ``//`` is a legal path prefix equivalent to ``/``, so it is NOT
        rejected there and the picker keeps working - the assertion is
        platform-split on purpose, not skipped."""
        with TestClient(fs_app) as c:
            r = c.get("/api/fs/dirs", params={"path": UNC_FWD},
                      headers=_hdr(_host_key()))
        if os.name == "nt":
            assert r.status_code == 400, r.text
            assert _unc_calls(fs_spy) == []
        else:
            assert r.status_code == 404, r.text   # a plain missing local path

    def test_ordinary_directory_still_lists(self, fs_app, tmp_path):
        """The picker's whole purpose is browsing the host disk unconstrained -
        only the UNC/device form is refused. A regression that over-blocks would
        break the folder picker outright."""
        d = tmp_path / "data"
        d.mkdir()
        (d / "sub").mkdir()
        (d / "a.txt").write_text("hi", encoding="utf-8")
        with TestClient(fs_app) as c:
            r = c.get("/api/fs/dirs", params={"path": str(d)},
                      headers=_hdr(_host_key()))
        assert r.status_code == 200, r.text
        assert "sub" in r.json()["dirs"]

    def test_empty_path_still_lists_roots(self, fs_app, tmp_path, monkeypatch):
        """An empty path lists the roots - drive letters on Windows, the
        filesystem root on POSIX.

        The POSIX branch is pointed at a DISPOSABLE fixture root, never the
        machine's real "/". Listing "/" stats every child (/opt, /etc, /usr),
        which is a real system-path touch and is exactly what the guard in
        tests/conftest.py fails a test for. An earlier revision of this test did
        exactly that: it passed on Windows, where the roots are drive letters,
        and reddened master on the ubuntu CI leg - a platform blind spot rather
        than a logic error, and the reason single-platform local green is a
        weaker gate than it looks."""
        from localm.plugins.gui.routes import admin as admin_routes
        fake_root = tmp_path / "fakeroot"
        (fake_root / "alpha").mkdir(parents=True)
        (fake_root / "beta").mkdir()
        monkeypatch.setattr(admin_routes, "_POSIX_FS_ROOT", str(fake_root))

        with TestClient(fs_app) as c:
            r = c.get("/api/fs/dirs", headers=_hdr(_host_key()))

        assert r.status_code == 200, r.text
        dirs = r.json()["dirs"]
        assert dirs, "empty path returned no roots"
        if os.name != "nt":
            # POSIX went through the injected root, so we know exactly what to
            # expect - and nothing outside tmp_path was read.
            assert sorted(dirs) == ["alpha", "beta"]


# --------------------------------------------------------------------------- #
#  POST /api/comfyui/create-launcher (CodeQL 3)                                #
# --------------------------------------------------------------------------- #

class TestCreateLauncherAllowlistsBeforeResolving:

    @staticmethod
    def _configure(workdir):
        from localm.config import load_config, save_config
        cfg = load_config()
        cfg["comfy_workdir"] = str(workdir)
        save_config(cfg)

    @pytest.mark.parametrize("bad", [UNC, DEVICE])
    def test_unc_refused_without_touching_the_filesystem(
            self, fs_app, tmp_path, fs_spy, bad):
        wd = tmp_path / "comfy"
        wd.mkdir()
        self._configure(wd)
        with TestClient(fs_app) as c:
            t0 = time.perf_counter()
            r = c.post("/api/comfyui/create-launcher", params={"workdir": bad},
                       headers=_hdr(_cfgwrite_key()))
            elapsed = time.perf_counter() - t0

        assert r.status_code in (400, 403), r.text
        assert _unc_calls(fs_spy) == [], (
            "the UNC string reached a filesystem call BEFORE the allowlist - which "
            "is the whole finding: the 403 is returned after the SMB dial")
        # Nothing was written anywhere on the way to the refusal. Ordered BEFORE
        # the timing backstop deliberately: pytest stops at the first failure, so
        # with the timing assert above it, a slow runner would mask this one and
        # a real stray write would be reported as a timing flake.
        assert not list(wd.iterdir())
        assert elapsed < _NO_DIAL_SECONDS, f"took {elapsed:.2f}s"

    def test_unlisted_local_dir_is_refused_without_resolving_it(
            self, fs_app, tmp_path, fs_spy):
        """The core of the fix: the RAW string is matched against the configured
        allowlist first, so no syscall ever touches an unlisted path. Before the
        fix this path was resolved (two filesystem opens) and only then refused."""
        configured = tmp_path / "comfy"
        configured.mkdir()
        self._configure(configured)
        outsider = tmp_path / "not-configured"
        outsider.mkdir()

        with TestClient(fs_app) as c:
            r = c.post("/api/comfyui/create-launcher",
                       params={"workdir": str(outsider)},
                       headers=_hdr(_cfgwrite_key()))

        assert r.status_code == 403, r.text
        assert str(outsider) not in fs_spy, (
            "an unlisted workdir reached a filesystem call before being refused")
        assert not (outsider / "launch-comfyui.bat").exists()
        assert not (outsider / "launch-comfyui.sh").exists()

    def test_relative_workdir_is_refused_without_resolving_it(
            self, fs_app, tmp_path, fs_spy):
        """A relative workdir would resolve against the SERVER's cwd, which is
        not a path the caller can be said to have named. The raw-string allowlist
        refuses it before any syscall, so it never reaches resolve()."""
        wd = tmp_path / "comfy"
        wd.mkdir()
        self._configure(wd)
        with TestClient(fs_app) as c:
            r = c.post("/api/comfyui/create-launcher",
                       params={"workdir": "comfy"},
                       headers=_hdr(_cfgwrite_key()))
        assert r.status_code == 403, r.text
        assert "comfy" not in fs_spy

    def test_configured_unc_workdir_is_not_broken_by_the_fix(
            self, fs_app, tmp_path, monkeypatch):
        """A ComfyUI workdir on a network share is a REAL setup, so the fix must
        not refuse a UNC path the user configured themselves. Dialling a path the
        user chose is not the attack; dialling an UNLISTED one is. Asserted at
        the gate rather than end-to-end - there is no SMB server here, so the
        check is that the request gets PAST the allowlist (it then fails on the
        directory not existing, not on a refusal)."""
        # Inject the config directly rather than through save_config: the point
        # of this test is the ALLOWLIST's verdict, so it must not depend on
        # whatever validation the config writer may or may not apply to a UNC
        # string. admin.py imports load_config inside the handler, so patching
        # the module attribute is what the handler will see.
        import localm.config as _cfg
        monkeypatch.setattr(_cfg, "load_config", lambda: {"comfy_workdir": UNC})

        # Neutralise BOTH syscalls the handler would make on the UNC path. Only
        # patching resolve() is not enough: is_dir() dials SMB too, and on
        # Windows that is the 271-second stall this whole unit is about - the
        # test itself would hang. Narrow patches (UNC only) so the rest of the
        # request keeps using the real filesystem.
        real_resolve, real_is_dir = Path.resolve, Path.is_dir

        def no_dial_resolve(self, *a, **kw):
            return self if str(self).startswith("\\\\") else real_resolve(self, *a, **kw)

        def no_dial_is_dir(self, *a, **kw):
            return False if str(self).startswith("\\\\") else real_is_dir(self, *a, **kw)

        monkeypatch.setattr(Path, "resolve", no_dial_resolve)
        monkeypatch.setattr(Path, "is_dir", no_dial_is_dir)

        with TestClient(fs_app) as c:
            r = c.post("/api/comfyui/create-launcher", params={"workdir": UNC},
                       headers=_hdr(_cfgwrite_key()))

        assert r.status_code != 403, (
            "a user-configured UNC workdir was refused by the allowlist - the "
            "fix broke a legitimate network-share setup")
        # It got past the gate and failed on the (stubbed-absent) directory,
        # which is the correct downstream behavior, not a refusal.
        assert r.status_code == 400 and "not a valid directory" in r.text

    def test_configured_workdir_still_writes_the_launcher(self, fs_app, tmp_path):
        """The happy path, byte for byte: the allowlist prefilter must not break
        the legitimate caller, and the script content must be unchanged."""
        wd = tmp_path / "comfy"
        wd.mkdir()
        self._configure(wd)
        with TestClient(fs_app) as c:
            r = c.post("/api/comfyui/create-launcher",
                       params={"workdir": str(wd)},
                       headers=_hdr(_cfgwrite_key()))

        assert r.status_code == 200, r.text
        assert r.json() == {"status": "ok"}
        if os.name == "nt":
            script = wd / "launch-comfyui.bat"
            assert script.is_file()
            body = script.read_text(encoding="utf-8")
            assert body.startswith("@echo off")
            assert "python main.py" in body
        else:
            script = wd / "launch-comfyui.sh"
            assert script.is_file()
            body = script.read_text(encoding="utf-8")
            assert body.startswith("#!/bin/bash")
            assert "python3 main.py" in body

    def test_equivalent_spelling_of_the_configured_workdir_is_accepted(
            self, fs_app, tmp_path):
        """The prefilter is normcase+normpath, so a trailing separator or a "."
        segment - both of which a GUI can legitimately send - still match. If
        this breaks, the prefilter is too strict and the feature is broken."""
        wd = tmp_path / "comfy"
        wd.mkdir()
        self._configure(wd)
        spelled = str(wd) + "/./"
        with TestClient(fs_app) as c:
            r = c.post("/api/comfyui/create-launcher",
                       params={"workdir": spelled},
                       headers=_hdr(_cfgwrite_key()))
        assert r.status_code == 200, r.text


# --------------------------------------------------------------------------- #
#  POST /api/logs/export - same class, same file                               #
# --------------------------------------------------------------------------- #

def test_logs_export_rejects_unc_destination(fs_app, fs_spy):
    """export_logs stats a caller-supplied destination and then runs a whole
    shutil.copy2 loop. Same UNC dial, same event-loop stall."""
    with TestClient(fs_app) as c:
        t0 = time.perf_counter()
        r = c.post("/api/logs/export", json={"dest": UNC},
                   headers=_hdr(_cfgwrite_key()))
        elapsed = time.perf_counter() - t0
    assert r.status_code == 400, r.text
    assert _unc_calls(fs_spy) == []
    assert elapsed < _NO_DIAL_SECONDS, f"took {elapsed:.2f}s"


# --------------------------------------------------------------------------- #
#  POST /api/fs/mkdir - create a new folder from the picker                    #
# --------------------------------------------------------------------------- #

class TestFsMkdir:

    @pytest.fixture
    def browse_dir(self, tmp_path):
        """A dedicated subdirectory to browse/create into, kept separate from
        tmp_path's root: fs_app points LOCALM_HOME at tmp_path/".localm", so
        minting a key (every test here needs one for auth) side-effect-creates
        that folder directly under tmp_path the first time it is touched. A
        test that lists/compares tmp_path's own contents would see that
        unrelated directory and misattribute it to the mkdir under test."""
        d = tmp_path / "browse"
        d.mkdir()
        return d

    def test_creates_folder_and_navigable_result(self, fs_app, browse_dir):
        with TestClient(fs_app) as c:
            r = c.post("/api/fs/mkdir",
                       json={"path": str(browse_dir), "name": "New Folder"},
                       headers=_hdr(_cfgwrite_key()))
        assert r.status_code == 200, r.text
        created = browse_dir / "New Folder"
        assert created.is_dir()
        body = r.json()
        assert body["name"] == "New Folder"
        assert Path(body["path"]) == created.resolve()

    @pytest.mark.parametrize("bad", [UNC, DEVICE, r"\\192.0.2.1\share\sub"])
    def test_unc_parent_refused_without_touching_the_filesystem(
            self, fs_app, fs_spy, bad):
        with TestClient(fs_app) as c:
            t0 = time.perf_counter()
            r = c.post("/api/fs/mkdir", json={"path": bad, "name": "x"},
                       headers=_hdr(_cfgwrite_key()))
            elapsed = time.perf_counter() - t0
        assert r.status_code == 400, r.text
        assert _unc_calls(fs_spy) == [], (
            "the UNC parent reached a filesystem call - the SMB dial (and the "
            "net-NTLMv2 leak) happens there, before any status code is chosen")
        assert elapsed < _NO_DIAL_SECONDS, f"took {elapsed:.2f}s"

    @pytest.mark.parametrize("bad_name", ["../evil", "a/b", "..", ".", ""])
    def test_unsafe_name_rejected_and_nothing_created(
            self, fs_app, browse_dir, bad_name):
        before = set(browse_dir.iterdir())
        with TestClient(fs_app) as c:
            r = c.post("/api/fs/mkdir",
                       json={"path": str(browse_dir), "name": bad_name},
                       headers=_hdr(_cfgwrite_key()))
        assert r.status_code == 400, r.text
        assert set(browse_dir.iterdir()) == before, (
            "an unsafe name still created something inside the parent")

    def test_reserved_character_rejected(self, fs_app, browse_dir):
        with TestClient(fs_app) as c:
            r = c.post("/api/fs/mkdir",
                       json={"path": str(browse_dir), "name": "evil:stream"},
                       headers=_hdr(_cfgwrite_key()))
        assert r.status_code == 400, r.text
        assert not list(browse_dir.iterdir())

    def test_never_overwrites_an_existing_folder(self, fs_app, browse_dir):
        existing = browse_dir / "already-here"
        existing.mkdir()
        (existing / "keep.txt").write_text("do not touch", encoding="utf-8")
        with TestClient(fs_app) as c:
            r = c.post("/api/fs/mkdir",
                       json={"path": str(browse_dir), "name": "already-here"},
                       headers=_hdr(_cfgwrite_key()))
        assert r.status_code == 409, r.text
        assert (existing / "keep.txt").read_text(encoding="utf-8") == "do not touch"

    def test_never_overwrites_an_existing_file(self, fs_app, browse_dir):
        existing = browse_dir / "already-here"
        existing.write_text("do not touch", encoding="utf-8")
        with TestClient(fs_app) as c:
            r = c.post("/api/fs/mkdir",
                       json={"path": str(browse_dir), "name": "already-here"},
                       headers=_hdr(_cfgwrite_key()))
        assert r.status_code == 409, r.text
        assert existing.is_file()
        assert existing.read_text(encoding="utf-8") == "do not touch"

    def test_missing_parent_directory_is_404(self, fs_app, browse_dir):
        missing = browse_dir / "does-not-exist"
        with TestClient(fs_app) as c:
            r = c.post("/api/fs/mkdir", json={"path": str(missing), "name": "x"},
                       headers=_hdr(_cfgwrite_key()))
        assert r.status_code == 404, r.text

    def test_permission_denied_is_reported_not_swallowed(
            self, fs_app, browse_dir, monkeypatch):
        # Mint the key BEFORE sabotaging Path.mkdir: creating a key
        # side-effect-creates LOCALM_HOME via the same Path.mkdir the patch
        # below hijacks, so minting it afterwards would fail key creation
        # itself rather than exercise the route's own PermissionError handling.
        key = _cfgwrite_key()

        def flaky_mkdir(self, *a, **kw):
            raise PermissionError(13, "Permission denied")
        monkeypatch.setattr(Path, "mkdir", flaky_mkdir)
        with TestClient(fs_app) as c:
            r = c.post("/api/fs/mkdir", json={"path": str(browse_dir), "name": "x"},
                       headers=_hdr(key))
        assert r.status_code == 403, r.text
        assert "permission" in r.text.lower()

    def test_config_read_only_key_is_refused(self, fs_app, browse_dir):
        """A key that can browse but not write must not be able to mkdir - the
        same CONFIG_WRITE gate as export_logs/create_comfy_launcher."""
        with TestClient(fs_app) as c:
            r = c.post("/api/fs/mkdir", json={"path": str(browse_dir), "name": "x"},
                       headers=_hdr(_host_key()))
        assert r.status_code == 403, r.text
        assert not (browse_dir / "x").exists()

    def test_config_write_without_host_fs_access_is_refused(self, fs_app, browse_dir):
        """CONFIG_WRITE alone is not enough - the target is an arbitrary host
        path, same as fs_dirs, so require_fs_host applies too."""
        from localm import auth
        key = auth.create_key("w-mkdir", [S.CONFIG_WRITE], allow_privileged=True,
                              fs_access="none")["key"]
        with TestClient(fs_app) as c:
            r = c.post("/api/fs/mkdir", json={"path": str(browse_dir), "name": "x"},
                       headers=_hdr(key))
        assert r.status_code == 403, r.text
        assert not (browse_dir / "x").exists()


# --------------------------------------------------------------------------- #
#  POST /api/fs/rename - rename a file or folder in place                      #
# --------------------------------------------------------------------------- #

class TestFsRename:

    def test_renames_a_file(self, fs_app, tmp_path):
        f = tmp_path / "old.txt"
        f.write_text("content", encoding="utf-8")
        with TestClient(fs_app) as c:
            r = c.post("/api/fs/rename",
                       json={"path": str(f), "new_name": "new.txt"},
                       headers=_hdr(_cfgwrite_key()))
        assert r.status_code == 200, r.text
        assert not f.exists()
        new_f = tmp_path / "new.txt"
        assert new_f.is_file()
        assert new_f.read_text(encoding="utf-8") == "content"
        assert r.json()["name"] == "new.txt"

    def test_renames_a_folder_preserving_contents(self, fs_app, tmp_path):
        d = tmp_path / "old-dir"
        d.mkdir()
        (d / "inside.txt").write_text("hi", encoding="utf-8")
        with TestClient(fs_app) as c:
            r = c.post("/api/fs/rename",
                       json={"path": str(d), "new_name": "new-dir"},
                       headers=_hdr(_cfgwrite_key()))
        assert r.status_code == 200, r.text
        assert not d.exists()
        new_d = tmp_path / "new-dir"
        assert new_d.is_dir()
        assert (new_d / "inside.txt").read_text(encoding="utf-8") == "hi"

    @pytest.mark.parametrize("bad", [UNC, DEVICE, r"\\192.0.2.1\share\sub"])
    def test_unc_target_refused_without_touching_the_filesystem(
            self, fs_app, fs_spy, bad):
        with TestClient(fs_app) as c:
            t0 = time.perf_counter()
            r = c.post("/api/fs/rename", json={"path": bad, "new_name": "x"},
                       headers=_hdr(_cfgwrite_key()))
            elapsed = time.perf_counter() - t0
        assert r.status_code == 400, r.text
        assert _unc_calls(fs_spy) == [], (
            "the UNC target reached a filesystem call - the SMB dial (and the "
            "net-NTLMv2 leak) happens there, before any status code is chosen")
        assert elapsed < _NO_DIAL_SECONDS, f"took {elapsed:.2f}s"

    @pytest.mark.parametrize("bad_name", ["../evil", "a/b", "..", ".", ""])
    def test_unsafe_new_name_rejected_and_nothing_renamed(
            self, fs_app, tmp_path, bad_name):
        f = tmp_path / "keep.txt"
        f.write_text("content", encoding="utf-8")
        with TestClient(fs_app) as c:
            r = c.post("/api/fs/rename",
                       json={"path": str(f), "new_name": bad_name},
                       headers=_hdr(_cfgwrite_key()))
        assert r.status_code == 400, r.text
        assert f.is_file()
        assert f.read_text(encoding="utf-8") == "content"

    def test_never_overwrites_an_existing_file(self, fs_app, tmp_path):
        src = tmp_path / "source.txt"
        src.write_text("SOURCE", encoding="utf-8")
        dst = tmp_path / "target.txt"
        dst.write_text("TARGET - must survive", encoding="utf-8")
        with TestClient(fs_app) as c:
            r = c.post("/api/fs/rename",
                       json={"path": str(src), "new_name": "target.txt"},
                       headers=_hdr(_cfgwrite_key()))
        assert r.status_code == 409, r.text
        assert src.is_file(), "the source must be untouched on refusal"
        assert src.read_text(encoding="utf-8") == "SOURCE"
        assert dst.read_text(encoding="utf-8") == "TARGET - must survive"

    def test_never_overwrites_an_existing_folder(self, fs_app, tmp_path):
        src = tmp_path / "source-dir"
        src.mkdir()
        dst = tmp_path / "target-dir"
        dst.mkdir()
        (dst / "keep.txt").write_text("do not touch", encoding="utf-8")
        with TestClient(fs_app) as c:
            r = c.post("/api/fs/rename",
                       json={"path": str(src), "new_name": "target-dir"},
                       headers=_hdr(_cfgwrite_key()))
        assert r.status_code == 409, r.text
        assert src.is_dir(), "the source must be untouched on refusal"
        assert (dst / "keep.txt").read_text(encoding="utf-8") == "do not touch"

    def test_never_overwrites_even_if_the_os_rename_call_would_silently_succeed(
            self, fs_app, tmp_path, monkeypatch):
        """POSIX rename(2) SILENTLY REPLACES an existing FILE target instead of
        raising - that is documented rename(2) behavior, unlike Windows, which
        refuses on its own. The two test_never_overwrites_* tests above pass on
        THIS platform (Windows) regardless of whether fs_rename's explicit
        `new_path.exists()` pre-check is present, because Path.rename() itself
        already refuses here - so they cannot tell the pre-check apart from
        incidental OS behavior (measured: removing the pre-check left them
        green). This test makes Path.rename behave like POSIX's rename(2) on
        EVERY platform, so only the explicit pre-check - never the OS call - can
        be what refuses the overwrite."""
        src = tmp_path / "source.txt"
        src.write_text("SOURCE", encoding="utf-8")
        dst = tmp_path / "target.txt"
        dst.write_text("TARGET - must survive", encoding="utf-8")
        key = _cfgwrite_key()

        def posix_style_rename(self, target):
            Path(target).write_bytes(self.read_bytes())
            self.unlink()
        monkeypatch.setattr(Path, "rename", posix_style_rename)

        with TestClient(fs_app) as c:
            r = c.post("/api/fs/rename",
                       json={"path": str(src), "new_name": "target.txt"},
                       headers=_hdr(key))
        assert r.status_code == 409, r.text
        assert dst.read_text(encoding="utf-8") == "TARGET - must survive"

    def test_missing_source_is_404(self, fs_app, tmp_path):
        with TestClient(fs_app) as c:
            r = c.post("/api/fs/rename",
                       json={"path": str(tmp_path / "nope"), "new_name": "x"},
                       headers=_hdr(_cfgwrite_key()))
        assert r.status_code == 404, r.text

    def test_same_name_is_refused_as_a_no_op(self, fs_app, tmp_path):
        f = tmp_path / "same.txt"
        f.write_text("x", encoding="utf-8")
        with TestClient(fs_app) as c:
            r = c.post("/api/fs/rename",
                       json={"path": str(f), "new_name": "same.txt"},
                       headers=_hdr(_cfgwrite_key()))
        assert r.status_code == 400, r.text
        assert f.is_file()

    def test_permission_denied_is_reported_not_swallowed(
            self, fs_app, tmp_path, monkeypatch):
        f = tmp_path / "old.txt"
        f.write_text("x", encoding="utf-8")
        # Mint the key BEFORE sabotaging Path.rename - see the identical note
        # in TestFsMkdir.test_permission_denied_is_reported_not_swallowed.
        key = _cfgwrite_key()

        def flaky_rename(self, target):
            raise PermissionError(13, "Permission denied")
        monkeypatch.setattr(Path, "rename", flaky_rename)
        with TestClient(fs_app) as c:
            r = c.post("/api/fs/rename",
                       json={"path": str(f), "new_name": "new.txt"},
                       headers=_hdr(key))
        assert r.status_code == 403, r.text
        assert "permission" in r.text.lower()

    def test_config_read_only_key_is_refused(self, fs_app, tmp_path):
        f = tmp_path / "old.txt"
        f.write_text("x", encoding="utf-8")
        with TestClient(fs_app) as c:
            r = c.post("/api/fs/rename",
                       json={"path": str(f), "new_name": "new.txt"},
                       headers=_hdr(_host_key()))
        assert r.status_code == 403, r.text
        assert f.is_file()

    def test_config_write_without_host_fs_access_is_refused(self, fs_app, tmp_path):
        from localm import auth
        key = auth.create_key("w-rename", [S.CONFIG_WRITE], allow_privileged=True,
                              fs_access="none")["key"]
        f = tmp_path / "old.txt"
        f.write_text("x", encoding="utf-8")
        with TestClient(fs_app) as c:
            r = c.post("/api/fs/rename",
                       json={"path": str(f), "new_name": "new.txt"},
                       headers=_hdr(key))
        assert r.status_code == 403, r.text
        assert f.is_file()


# --------------------------------------------------------------------------- #
#  Cross-origin refusal for /api/fs/ in EVERY mode                             #
# --------------------------------------------------------------------------- #

class TestFsBrowserRefusedCrossOrigin:
    """/api/fs/* enumerates the user's disk and is a GET, so it is exempt from
    the CSRF gate (unsafe methods only) and from the open-mode shell-token gate.
    The generic /api,/v1 metadata-GET gate does cover it, but only inside the
    ``not any_key_configured()`` branch - so in PROTECTED mode a
    cookie-authenticated cross-origin GET would execute. SameSite=strict does not
    help: "site" ignores port, so any other loopback-port page is same-site."""

    @staticmethod
    def _app(tmp_path, monkeypatch, *, with_key: bool):
        import localm.config as cfg
        home = tmp_path / ".localm"
        home.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("LOCALM_HOME", str(home))
        monkeypatch.delenv("LOCALM_API_KEY", raising=False)
        monkeypatch.setattr(cfg, "HOME_DIR", home)
        monkeypatch.setattr(cfg, "MODELS_DIR", home / "models")
        monkeypatch.setattr(cfg, "CONFIG_FILE", home / "config.json")
        monkeypatch.setattr(cfg, "REGISTRY_FILE", home / "registry.json")
        if with_key:
            from localm.auth import set_api_key
            set_api_key("k" * 32)
        # create_app installs the _origin_guard middleware; attach_gui mounts the
        # real /api/fs/* routes onto it. Both are needed: the guard runs before
        # routing, so testing it against an unmounted path would pass even if the
        # picker lived somewhere else entirely.
        from localm.inference.http_server import create_app
        from localm.plugins.engine import attach_engine
        app = create_app(None)
        attach_engine(app)
        attach_gui(app, self_url="http://127.0.0.1:9/v1",
                   switch_model=lambda name: None,
                   active_model=lambda: "model-a")
        assert any(getattr(r, "path", None) == "/api/fs/dirs"
                   for r in app.router.routes), "the picker route did not mount"
        return app

    @pytest.mark.parametrize("with_key", [False, True], ids=["open", "protected"])
    @pytest.mark.parametrize("path", ["/api/fs/dirs", "/api/fs/places"])
    def test_refused_cross_origin_in_every_mode(
            self, tmp_path, monkeypatch, with_key, path):
        app = self._app(tmp_path, monkeypatch, with_key=with_key)
        c = TestClient(app)
        r = c.get(path, headers={"Origin": "http://127.0.0.1:3000"})
        assert r.status_code == 403, r.text
        assert "cross-origin" in r.json()["detail"].lower()

    def test_same_origin_not_refused_by_the_origin_guard(
            self, tmp_path, monkeypatch):
        """The legitimate loopback GUI shell must still reach the picker. It can
        still be refused by AUTH (that is a different gate and a different
        message); what must not happen is a cross-origin refusal."""
        app = self._app(tmp_path, monkeypatch, with_key=False)
        c = TestClient(app)
        r = c.get("/api/fs/dirs",
                  headers={"Origin": "http://testserver", "Host": "testserver"})
        detail = r.json().get("detail", "") if r.headers.get(
            "content-type", "").startswith("application/json") else ""
        assert "cross-origin" not in str(detail).lower()

    def test_ordinary_api_get_unaffected(self, tmp_path, monkeypatch):
        """Only /api/fs/* is added to the all-mode refusal; the prefix must not
        leak onto neighbouring routes."""
        app = self._app(tmp_path, monkeypatch, with_key=True)
        c = TestClient(app)
        r = c.get("/health", headers={"Origin": "http://127.0.0.1:3000"})
        assert r.status_code in (200, 503)

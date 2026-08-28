# SPDX-License-Identifier: AGPL-3.0-or-later
"""The update apply ENGINE (localm/_apply_update): verify, extract, and the
swap/backup/rollback file primitives on a fake install tree. The live re-exec
orchestration is integration-level and verified on a real install; these pin the
correctness + safety of the file ops (deletions applied, never-touch preserved,
rollback restores the exact pre-apply state)."""

import shutil
import subprocess
import sys
import zipfile

import pytest

from localm import _apply_update as au


def _make_zip(path, files, wrap=None):
    """Write a zip at *path* with {relpath: content}; optionally under a wrapper dir."""
    with zipfile.ZipFile(path, "w") as z:
        for rel, content in files.items():
            arc = f"{wrap}/{rel}" if wrap else rel
            z.writestr(arc, content)


# ------------------------------ verify ----------------------------------

def test_verify_accepts_flat_build(tmp_path):
    zp = tmp_path / "b.zip"
    _make_zip(zp, {"VERSION": "0.2.0", "pyproject.toml": "[project]", "localm/__init__.py": ""})
    au.verify_zip(zp)  # no raise


def test_verify_accepts_wrapped_build(tmp_path):
    zp = tmp_path / "b.zip"
    _make_zip(zp, {"VERSION": "0.2.0", "pyproject.toml": "[project]"}, wrap="localm-0.2.0")
    au.verify_zip(zp)  # no raise


def test_verify_rejects_non_zip(tmp_path):
    p = tmp_path / "x.zip"
    p.write_text("not a zip", encoding="utf-8")
    with pytest.raises(ValueError):
        au.verify_zip(p)


def test_verify_rejects_zip_without_version(tmp_path):
    zp = tmp_path / "b.zip"
    _make_zip(zp, {"readme.md": "hi", "localm/__init__.py": ""})
    with pytest.raises(ValueError):
        au.verify_zip(zp)


def test_verify_rejects_missing_file(tmp_path):
    with pytest.raises(ValueError):
        au.verify_zip(tmp_path / "nope.zip")


# ------------------------------ extract ---------------------------------

def test_extract_flat(tmp_path):
    zp = tmp_path / "b.zip"
    _make_zip(zp, {"VERSION": "0.2.0", "pyproject.toml": "x", "localm/a.py": "1"})
    root = au.extract(zp, tmp_path / "stg")
    assert (root / "VERSION").read_text().strip() == "0.2.0"
    assert (root / "localm" / "a.py").exists()


def test_extract_descends_into_wrapper(tmp_path):
    zp = tmp_path / "b.zip"
    _make_zip(zp, {"VERSION": "0.2.0", "pyproject.toml": "x"}, wrap="localm-0.2.0")
    root = au.extract(zp, tmp_path / "stg")
    assert root.name == "localm-0.2.0"
    assert (root / "VERSION").exists()


# --------------------------- swap entries -------------------------------

def test_swap_entries_excludes_never_touch(tmp_path):
    staged = tmp_path / "s"
    (staged / "localm").mkdir(parents=True)
    (staged / ".venv").mkdir()
    (staged / "home").mkdir()
    (staged / "VERSION").write_text("0.2.0")
    names = au.swap_entries(staged)
    assert "localm" in names and "VERSION" in names
    assert ".venv" not in names and "home" not in names


# --------------------- swap + backup + rollback -------------------------

def _fake_install(root):
    (root / "localm").mkdir(parents=True)
    (root / "localm" / "__init__.py").write_text("old", encoding="utf-8")
    (root / "localm" / "gone.py").write_text("removed upstream", encoding="utf-8")
    (root / "VERSION").write_text("0.1.0", encoding="utf-8")
    (root / ".venv").mkdir()
    (root / ".venv" / "marker").write_text("keep", encoding="utf-8")
    (root / "home").mkdir()
    (root / "home" / "config.json").write_text("user data", encoding="utf-8")


def _staged_build(root):
    (root / "localm").mkdir(parents=True)
    (root / "localm" / "__init__.py").write_text("new", encoding="utf-8")
    (root / "localm" / "added.py").write_text("added", encoding="utf-8")
    (root / "VERSION").write_text("0.2.0", encoding="utf-8")
    (root / "pyproject.toml").write_text("[project]", encoding="utf-8")


def test_swap_applies_changes_deletions_and_preserves_never_touch(tmp_path):
    inst, staged, bdir = tmp_path / "inst", tmp_path / "staged", tmp_path / "bak"
    inst.mkdir(); staged.mkdir()
    _fake_install(inst)
    _staged_build(staged)

    names = au.swap_with_backup(staged, inst, bdir)

    # New content swapped in.
    assert (inst / "localm" / "__init__.py").read_text() == "new"
    assert (inst / "localm" / "added.py").exists()
    assert (inst / "VERSION").read_text().strip() == "0.2.0"
    assert (inst / "pyproject.toml").exists()
    # Upstream deletion applied (whole-tree replace of localm/).
    assert not (inst / "localm" / "gone.py").exists()
    # Never-touch trees untouched.
    assert (inst / ".venv" / "marker").read_text() == "keep"
    assert (inst / "home" / "config.json").read_text() == "user data"
    assert "VERSION" in names and ".venv" not in names


def test_rollback_restores_exact_pre_apply_state(tmp_path):
    inst, staged, bdir = tmp_path / "inst", tmp_path / "staged", tmp_path / "bak"
    inst.mkdir(); staged.mkdir()
    _fake_install(inst)
    _staged_build(staged)

    names = au.swap_with_backup(staged, inst, bdir)
    au.rollback(bdir, inst, names)

    assert (inst / "localm" / "__init__.py").read_text() == "old"
    assert (inst / "localm" / "gone.py").read_text() == "removed upstream"
    assert not (inst / "localm" / "added.py").exists()
    assert (inst / "VERSION").read_text().strip() == "0.1.0"
    # pyproject was NEW (absent before) -> removed by rollback, not restored.
    assert not (inst / "pyproject.toml").exists()
    # Never-touch survived throughout.
    assert (inst / ".venv" / "marker").read_text() == "keep"
    assert (inst / "home" / "config.json").read_text() == "user data"


# --------- preserve provisioned native binaries across a swap -----------
# The runtime wheel's lib/ holds the .dll/.so that `localm setup-llama` provisions.
# They are gitignored, so a build.zip carries only the empty scaffold.
# PRESERVE_WITHIN keeps them across a whole-tree replace of runtime/.

_LIB = "runtime/localm_llama_runtime/lib"


def _install_with_runtime(root):
    (root / "localm").mkdir(parents=True)
    (root / "localm" / "__init__.py").write_text("old", encoding="utf-8")
    (root / "VERSION").write_text("0.1.0", encoding="utf-8")
    rt = root / "runtime" / "localm_llama_runtime"
    (rt).mkdir(parents=True)
    (rt / "__init__.py").write_text("SCAFFOLD_V1", encoding="utf-8")
    (root / "runtime" / "README.md").write_text("rt v1", encoding="utf-8")
    lib = rt / "lib"
    lib.mkdir()
    (lib / ".gitkeep").write_text("", encoding="utf-8")
    (lib / "llama.dll").write_bytes(b"PROVISIONED-NATIVE-BINARY")   # setup-llama output
    (lib / "ggml-base.dll").write_bytes(b"GGML")


def _staged_with_runtime(root):
    (root / "localm").mkdir(parents=True)
    (root / "localm" / "__init__.py").write_text("new", encoding="utf-8")
    (root / "VERSION").write_text("0.2.0", encoding="utf-8")
    (root / "pyproject.toml").write_text("[project]", encoding="utf-8")
    rt = root / "runtime" / "localm_llama_runtime"
    (rt).mkdir(parents=True)
    (rt / "__init__.py").write_text("SCAFFOLD_V2", encoding="utf-8")   # scaffold CHANGED
    (root / "runtime" / "README.md").write_text("rt v2", encoding="utf-8")
    lib = rt / "lib"
    lib.mkdir()
    (lib / ".gitkeep").write_text("", encoding="utf-8")                 # scaffold only, NO binary


def test_preserve_within_helpers():
    assert au._within_preserved(_LIB) is True
    assert au._within_preserved(_LIB + "/llama.dll") is True
    assert au._within_preserved("runtime/localm_llama_runtime/__init__.py") is False
    assert au._name_has_preserved("runtime") is True
    assert au._name_has_preserved("localm") is False


def test_swap_preserves_provisioned_binaries_and_updates_scaffold(tmp_path):
    inst, staged, bdir = tmp_path / "i", tmp_path / "s", tmp_path / "b"
    inst.mkdir(); staged.mkdir()
    _install_with_runtime(inst)
    _staged_with_runtime(staged)

    au.swap_with_backup(staged, inst, bdir)

    # Provisioned binaries SURVIVE.
    assert (inst / _LIB / "llama.dll").read_bytes() == b"PROVISIONED-NATIVE-BINARY"
    assert (inst / _LIB / "ggml-base.dll").exists()
    # ... while the rest of runtime/ (scaffold source) still updates.
    assert (inst / "runtime/localm_llama_runtime/__init__.py").read_text() == "SCAFFOLD_V2"
    assert (inst / "runtime" / "README.md").read_text() == "rt v2"
    # backup skips the preserved binaries (nothing to restore; avoids copying them).
    assert not (bdir / _LIB / "llama.dll").exists()


def test_rollback_keeps_provisioned_binaries_and_restores_scaffold(tmp_path):
    inst, staged, bdir = tmp_path / "i", tmp_path / "s", tmp_path / "b"
    inst.mkdir(); staged.mkdir()
    _install_with_runtime(inst)
    _staged_with_runtime(staged)

    names = au.swap_with_backup(staged, inst, bdir)
    au.rollback(bdir, inst, names)

    # Binaries never left; scaffold restored to its pre-apply content.
    assert (inst / _LIB / "llama.dll").read_bytes() == b"PROVISIONED-NATIVE-BINARY"
    assert (inst / "runtime/localm_llama_runtime/__init__.py").read_text() == "SCAFFOLD_V1"
    assert (inst / "runtime" / "README.md").read_text() == "rt v1"


def test_swap_does_not_choke_on_a_held_open_binary(tmp_path):
    """While localm runs, a loaded DLL is file-locked on Windows, and a swap that
    deletes it raises WinError 32 and can strand a half-applied tree. The preserve
    keeps the swap off the binary entirely, so it completes with the binary intact.
    On POSIX an open file is deletable, so this also pins that the binary is
    preserved rather than replaced by the empty scaffold."""
    inst, staged, bdir = tmp_path / "i", tmp_path / "s", tmp_path / "b"
    inst.mkdir(); staged.mkdir()
    _install_with_runtime(inst)
    _staged_with_runtime(staged)

    with open(inst / _LIB / "llama.dll", "rb") as _held:   # simulate the running process
        au.swap_with_backup(staged, inst, bdir)            # must not raise
    assert (inst / _LIB / "llama.dll").read_bytes() == b"PROVISIONED-NATIVE-BINARY"
    assert (inst / "runtime/localm_llama_runtime/__init__.py").read_text() == "SCAFFOLD_V2"


# --------------------------- post-swap cmd ------------------------------

def test_post_swap_command_per_class():
    assert au.post_swap_command("reboot") is None
    assert au.post_swap_command("deps")[:3] == ["uv", "pip", "install"]
    assert au.post_swap_command("setup") is None


def test_post_swap_command_runtime_uses_absolute_interpreter_not_bare_localm():
    """A bare "localm" argv[0] resolves back to the calling exe itself when running as
    the default native LocaLM.exe launcher (see the live repro below), which then
    mis-invokes and rolls back every "runtime"-class update. Must go through
    sys.executable + "-m localm" instead, same as every other self-invocation site
    (setup_llama.py, applaunch.py, http_server.py, ...)."""
    cmd = au.post_swap_command("runtime", backend="cuda")
    assert cmd[0] == sys.executable
    assert cmd[1:3] == ["-m", "localm"]
    assert cmd[3:] == ["setup-llama", "--backend", "cuda", "--force", "--yes"]


def test_post_swap_command_runtime_forces_and_never_prompts():
    """NEW-UPDATE-RUNTIME-CLASS-IS-A-NO-OP: without --force, setup-llama's own
    "already provisioned" guard short-circuits on any already-present library, so
    a "runtime"-class update re-provisioned nothing and still reported success.
    --yes must ALSO be present: this argv is run with no one watching its stdin,
    and --force alone would newly reach _cuda_setup_dialogue's click.confirm(),
    which is gated on assume_yes only (not on isatty) and would hang forever."""
    cmd = au.post_swap_command("runtime", backend="vulkan")
    assert "--force" in cmd
    assert "--yes" in cmd


def test_bare_localm_argv_on_the_launcher_exe_fails_with_exit_2(tmp_path):
    """The native launcher build (LocaLM.exe) is a renamed copy of the
    interpreter (see applaunch.py's
    `os.path.basename(sys.executable).lower() == "localm.exe"` check), and a bare
    "localm" argv[0] resolves back to that same exe on the default install
    (Windows favors a same-directory match over a PATH-installed console-script
    shim). A copy of the interpreter invoked with a bare-name post-swap argv
    therefore fails to do what was asked; the absolute sys.executable argv the
    code uses does not depend on that name resolution.

    The EXACT failure signature depends on what sys.executable is in the process
    running this test: a standalone interpreter treats "setup-llama" as a script
    path it cannot open and exits 2; a venv's own python.exe refuses to run away
    from its venv at all ("No pyvenv.cfg file", a distinct nonzero code); a
    POSIX venv's ``bin/python`` is typically a symlink to the base interpreter,
    so copy2() (which follows symlinks) copies the base binary itself, which
    then cannot find its own relocated stdlib and fails to bootstrap at all
    ("Could not find platform independent libraries" / "No module named
    'encodings'"). All three are the bare-copy approach breaking, which is all
    this test asserts - it does not check that a fully-loaded backend started."""
    launcher = tmp_path / ("localm.exe" if sys.platform == "win32" else "localm")
    shutil.copy2(sys.executable, launcher)
    if sys.platform != "win32":
        launcher.chmod(0o755)

    old_style_cmd = [str(launcher)] + au.post_swap_command("runtime", backend="vulkan")[3:]
    result = subprocess.run(old_style_cmd, capture_output=True, text=True)

    assert result.returncode != 0, result.stdout
    assert ("setup-llama" in result.stderr or "pyvenv.cfg" in result.stderr
           or "Could not find platform independent libraries" in result.stderr), \
        result.stderr


# ---------------------- safe extraction (zip slip) ----------------------

def test_extract_rejects_traversal_member(tmp_path):
    zp = tmp_path / "b.zip"
    _make_zip(zp, {"VERSION": "0.2.0", "pyproject.toml": "x", "../evil.txt": "pwn"})
    with pytest.raises(ValueError):
        au.extract(zp, tmp_path / "stg")
    # Nothing escaped the staging dir.
    assert not (tmp_path / "evil.txt").exists()


def test_extract_rejects_absolute_member(tmp_path):
    zp = tmp_path / "b.zip"
    _make_zip(zp, {"VERSION": "0.2.0", "pyproject.toml": "x", "/abs/evil.txt": "pwn"})
    with pytest.raises(ValueError):
        au.extract(zp, tmp_path / "stg")


def test_unsafe_member_detection():
    assert au._unsafe_member("../evil") is True
    assert au._unsafe_member("a/../../evil") is True
    assert au._unsafe_member("/etc/passwd") is True
    assert au._unsafe_member("Z:/windows") is True
    assert au._unsafe_member("localm/foo.py") is False
    assert au._unsafe_member("VERSION") is False
    assert au._unsafe_member("a..b/c") is False   # '..' only as a path component counts


# ----------------- rollback surfaces restore failures -------------------

class TestRollbackRefusesPoisonedManifestNames:
    """rollback() names come from <home>/updates/applied_names.json, an ordinary
    file in the data dir, and each name reaches `installed / name` and then
    shutil.rmtree/unlink.

    update_watchdog.py calls main(['--yes']) automatically after a failed health
    probe with stdin=DEVNULL, so the isatty() confirmation is skipped and this can
    fire with no human present.

    Every vector asserts the same two things: the target OUTSIDE the install
    survives, and the refusal is REPORTED (RuntimeError), never silently skipped.
    """

    @staticmethod
    def _install_and_backup(tmp_path):
        inst, bdir = tmp_path / "inst", tmp_path / "bak"
        inst.mkdir(); bdir.mkdir()
        _fake_install(inst)
        return inst, bdir

    # EVERY vector resolves INSIDE tmp_path: the absolute and drive-qualified
    # vectors are BUILT from tmp_path rather than naming a real location, so
    # they exercise the identical code path (an absolute component REPLACES the
    # base under pathlib) with the blast radius inside the fixture.
    ABS = "<ABS_OUTSIDE>"          # -> tmp_path/victim, absolute + drive-qualified
    UNC_HOST = "<UNC>"             # -> a UNC form; never dialed, rejected lexically

    @pytest.mark.parametrize("vector", [
        "../../victim",            # traversal out of the install
        "../victim",
        ABS,                       # absolute + drive-qualified (replaces the base)
        UNC_HOST,                  # UNC
        "\\\\198.51.100.7\\share",  # UNC backslash spelling, RFC5737 TEST-NET-2
    ])
    def test_escaping_name_is_refused_and_reported(self, tmp_path, vector):
        inst, bdir = self._install_and_backup(tmp_path)
        victim = tmp_path / "victim"
        victim.mkdir()
        (victim / "keep.txt").write_text("do not delete me", encoding="utf-8")
        name = {self.ABS: str(victim),
                self.UNC_HOST: "//198.51.100.7/share/x"}.get(vector, vector)

        with pytest.raises(RuntimeError) as ei:
            au.rollback(bdir, inst, [name])

        assert "unsafe" in str(ei.value).lower()
        assert victim.is_dir() and (victim / "keep.txt").exists()

    @pytest.mark.parametrize("name", ["", ".", "   ", "./"])
    def test_collapsing_name_does_not_delete_the_install(self, tmp_path, name):
        """`Path(install) / ""` and `/ "."` both collapse to the install dir
        itself, so rmtree on the result would delete the WHOLE installation.
        These do not escape the root, so an escape-only check (_unsafe_member)
        returns False for them - this is the case that check cannot see."""
        inst, bdir = self._install_and_backup(tmp_path)

        with pytest.raises(RuntimeError) as ei:
            au.rollback(bdir, inst, [name])

        assert "unsafe" in str(ei.value).lower()
        assert inst.is_dir()
        assert (inst / "VERSION").exists(), "the install was deleted"
        assert (inst / "localm" / "__init__.py").exists()

    @pytest.mark.parametrize("name", ["home", ".venv", ".git", "issues", "qa"])
    def test_never_touch_name_is_refused(self, tmp_path, name):
        """swap_entries EXCLUDES NEVER_TOUCH, so a manifest naming one is poisoned
        by construction. It matters most on a portable install, where `home` sits
        INSIDE the install root: rolling back an entry named `home` would delete
        the user's models, chat history and auth keys, and the backup does not
        hold them to restore."""
        inst, bdir = self._install_and_backup(tmp_path)

        with pytest.raises(RuntimeError) as ei:
            au.rollback(bdir, inst, [name])

        assert "unsafe" in str(ei.value).lower()
        assert (inst / "home" / "config.json").read_text() == "user data"
        assert (inst / ".venv" / "marker").read_text() == "keep"

    def test_safe_names_alongside_a_poisoned_one_still_roll_back(self, tmp_path):
        """Best-effort, matching rollback()'s existing contract: it attempts every
        name, then reports the collected failures. A poisoned entry must not
        abort the legitimate half of the rollback."""
        inst, bdir = self._install_and_backup(tmp_path)
        (bdir / "VERSION").write_text("0.1.0", encoding="utf-8")
        (inst / "VERSION").write_text("0.2.0", encoding="utf-8")

        with pytest.raises(RuntimeError):
            au.rollback(bdir, inst, ["VERSION", "../../victim"])

        assert (inst / "VERSION").read_text().strip() == "0.1.0"   # restored anyway

    def test_unsafe_swap_name_detection(self):
        """Unit-level truth table, so a future edit to the helper cannot quietly
        widen it. Includes the two shapes _unsafe_member alone passes.

        These are PURE string checks - _unsafe_swap_name touches no filesystem -
        so naming a drive here deletes nothing. Contrast the parametrized cases
        above, which drive the real rollback and must stay inside tmp_path."""
        for bad in ["", ".", "..", "   ", "../x", "Z:/x", "/x", "//h/s",
                    "a/b", "home", ".venv", "__pycache__", None, 3]:
            assert au._unsafe_swap_name(bad), f"should be unsafe: {bad!r}"
        for ok in ["localm", "VERSION", "pyproject.toml", "runtime", "docs"]:
            assert not au._unsafe_swap_name(ok), f"should be safe: {ok!r}"
        # Names an escape-only check accepts.
        assert not au._unsafe_member("")
        assert not au._unsafe_member(".")
        assert not au._unsafe_member("home")


def test_rollback_raises_when_a_restore_fails(tmp_path, monkeypatch):
    inst, bdir = tmp_path / "inst", tmp_path / "bak"
    inst.mkdir(); bdir.mkdir()
    (bdir / "localm").mkdir()
    (bdir / "localm" / "x.py").write_text("orig", encoding="utf-8")

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(au.shutil, "copytree", boom)   # restore fails
    with pytest.raises(RuntimeError, match="rollback incomplete"):
        au.rollback(bdir, inst, ["localm"])


def test_swap_with_backup_surfaces_double_failure(tmp_path, monkeypatch):
    staged, inst, bdir = tmp_path / "s", tmp_path / "i", tmp_path / "b"
    staged.mkdir(); inst.mkdir()
    (staged / "VERSION").write_text("0.2.0", encoding="utf-8")

    def raise_os(*a, **k):
        raise OSError("boom")

    monkeypatch.setattr(au, "apply_files", raise_os)   # swap fails
    monkeypatch.setattr(au, "rollback", raise_os)      # AND rollback fails
    with pytest.raises(RuntimeError, match="rollback also failed"):
        au.swap_with_backup(staged, inst, bdir)


# --------- _prune surfaces a file it cannot remove (do not hide) ---------
# _prune RETURNS the file-removal failures so the caller surfaces them, while a
# benign already-empty-dir rmdir failure stays swallowed.



def test_prune_returns_error_for_unremovable_file(tmp_path, monkeypatch):
    """A FILE _prune cannot unlink is REPORTED (not silently left behind); files it can
    remove are still removed for real, and the return is [] when everything succeeds."""
    good = tmp_path / "good"
    (good / "a").mkdir(parents=True)
    (good / "a" / "f.py").write_text("stale", encoding="utf-8")
    assert au._prune(good, "good") == []                     # happy path: removed, no errors
    assert not (good / "a" / "f.py").exists()

    dst = tmp_path / "runtime"
    (dst / "sub").mkdir(parents=True)
    (dst / "sub" / "locked.py").write_text("stale", encoding="utf-8")
    (dst / "sub" / "ok.py").write_text("stale", encoding="utf-8")

    real_unlink = au.Path.unlink

    def flaky_unlink(self, *a, **k):
        if self.name == "locked.py":
            raise OSError("locked by AV")
        return real_unlink(self, *a, **k)

    monkeypatch.setattr(au.Path, "unlink", flaky_unlink)
    errs = au._prune(dst, "runtime")                         # rollback-style: remove all
    assert any("locked.py" in e for e in errs)
    assert not (dst / "sub" / "ok.py").exists()              # removable file WAS removed
    assert (dst / "sub" / "locked.py").exists()              # locked one stays behind (stale)


def test_rollback_surfaces_prune_unlink_failure(tmp_path, monkeypatch):
    """A scaffold file _prune cannot strip while unwinding a PRESERVE_WITHIN name during
    rollback must reach rollback()'s errors and RAISE - never a stale file left under a
    silent rolled_back:True (do-not-hide-problems)."""
    inst, staged, bdir = tmp_path / "i", tmp_path / "s", tmp_path / "b"
    inst.mkdir(); staged.mkdir()
    _install_with_runtime(inst)
    _staged_with_runtime(staged)
    names = au.swap_with_backup(staged, inst, bdir)          # inst/runtime now scaffold V2

    real_unlink = au.Path.unlink

    def flaky_unlink(self, *a, **k):
        if self.name == "__init__.py" and "localm_llama_runtime" in self.as_posix():
            raise OSError("locked by AV")            # the scaffold file rollback must strip
        return real_unlink(self, *a, **k)

    monkeypatch.setattr(au.Path, "unlink", flaky_unlink)
    with pytest.raises(RuntimeError, match="rollback incomplete"):
        au.rollback(bdir, inst, names)


def test_apply_files_logs_prune_failure_without_raising(tmp_path, monkeypatch, caplog):
    """On the FORWARD apply path a stale file that cannot be pruned is LOGGED, not raised:
    a leftover file must not brick an otherwise-good update, but must stay visible."""
    import logging
    inst, staged = tmp_path / "i", tmp_path / "s"
    inst.mkdir(); staged.mkdir()
    _install_with_runtime(inst)
    _staged_with_runtime(staged)
    # A file the new build DROPPED (present in the install runtime, absent in staged) so
    # apply_files' _prune(keep_src=...) actually tries to remove it.
    dropped = inst / "runtime" / "localm_llama_runtime" / "old_helper.py"
    dropped.write_text("dropped by the new build", encoding="utf-8")
    names = au.swap_entries(staged)

    real_unlink = au.Path.unlink

    def flaky_unlink(self, *a, **k):
        if self.name == "old_helper.py":
            raise OSError("locked by AV")
        return real_unlink(self, *a, **k)

    monkeypatch.setattr(au.Path, "unlink", flaky_unlink)
    with caplog.at_level(logging.WARNING, logger="localm"):
        au.apply_files(staged, inst, names)                  # must NOT raise
    assert "could not remove" in caplog.text
    assert dropped.exists()                                  # the un-prunable file stayed


def test_prune_swallows_empty_dir_rmdir_failure(tmp_path, monkeypatch):
    """The BENIGN half of the fix: an already-empty dir whose rmdir fails (lock/perms) is
    SWALLOWED, never surfaced - a leftover empty dir strands nothing. Only FILE-removal
    failures reach the returned errors (guards against a regression that would spuriously
    fail rollback() on a harmless leftover dir)."""
    dst = tmp_path / "runtime"
    (dst / "emptydir").mkdir(parents=True)   # an empty subdir _prune will try to rmdir

    real_rmdir = au.Path.rmdir

    def flaky_rmdir(self, *a, **k):
        if self.name == "emptydir":
            raise OSError("dir locked")
        return real_rmdir(self, *a, **k)

    monkeypatch.setattr(au.Path, "rmdir", flaky_rmdir)
    errs = au._prune(dst, "runtime")                          # rollback-style: remove all
    assert errs == []                                        # empty-dir rmdir failure swallowed
    assert (dst / "emptydir").exists()                       # ... dir harmlessly left behind

# SPDX-License-Identifier: AGPL-3.0-or-later
"""The sibling credential writers must CREATE their temp file already private.

``localm/sessions.py`` (session records carrying ``key_hash``, the same sha256
digest the keystore stores), ``localm/instances.py`` (the per-instance attach
token, written by both ``register_instance`` and ``set_mode``) and
``localm/gpu_registry.py`` (the coordination token) all do the atomic
temp+replace dance. Each restricts the TEMP file before the rename and retries
the destination on failure. Creating that temp with ``tmp.write_text(...)``
would leave the payload at the umask default (commonly 0644) on POSIX between
the create and the chmod that follows it, so it is created private instead.

WHERE THE OBSERVABLE WINDOW IS, because it decides which assertion can fail
here. With a ``write_text``-created temp the POSIX sequence is:

    write_text -> tmp is 0644        <- THE WINDOW
    restrict   -> tmp is 0600
    replace    -> destination is 0600

so at rename time the temp is ALREADY 0600, and a rename-time fingerprint (what
``tests/test_auth_secret_write_perms.py`` measures) reports the same value
either way. The only instrument that can see the window reads the temp file's
mode AT THE MOMENT THE RESTRICTION IS ASKED FOR, which is what
``_install_spies`` captures below.

Consequences:

* ``test_the_temp_file_is_created_already_private`` is the regression guard for
  this property and is POSIX-ONLY by construction. On Windows ``os.open``'s mode
  argument writes no ACL at all, so the temp carries the inherited ACL until
  ``restrict_file_perms`` runs icacls.
* ``test_every_sibling_writer_restricts_its_temp_before_the_rename`` runs on
  both platforms and pins the ordering and the end state. It guards a future
  reordering, which is a different defect.

The POSIX test pins the umask to 0o022 itself rather than inheriting the
environment's: under umask 0077 a ``write_text``-created file is ALREADY 0600,
so the guard would silently collapse on a machine where nothing is wrong.
"""

import contextlib
import json
import os
import subprocess

import pytest


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A throwaway data dir, with sessions.py's module cache reset around the
    test (state that would otherwise carry between tests)."""
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path)
    import localm.sessions as sessions
    monkeypatch.setattr(sessions, "_CACHE", {"mtime": None, "records": None})
    yield tmp_path


def _perm_fingerprint(path):
    """A comparable description of *path*'s permissions on this platform.

    Same shape as tests/test_auth_secret_write_perms.py::_perm_fingerprint and
    tests/test_auth_kdf.py::_acl_fingerprint. On Windows this reads icacls, the
    same mechanism ``restrict_file_perms`` itself uses; ``os.stat`` mode bits
    are meaningless there.
    """
    if os.name == "posix":
        return oct(os.stat(path).st_mode & 0o777)
    out = subprocess.run(["icacls", str(path)], capture_output=True,
                         check=False).stdout.decode("utf-8", "replace")
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    return sorted(ln.replace(str(path), "").strip()
                  for ln in lines if "Successfully processed" not in ln)


@contextlib.contextmanager
def _umask(value):
    """Pin the process umask for the duration. POSIX only; the control must not
    be inherited from the environment."""
    old = os.umask(value)
    try:
        yield
    finally:
        os.umask(old)


def _install_spies(monkeypatch):
    """Record every ``restrict_file_perms`` call and every ``os.replace``,
    capturing the SUBJECT file's permissions and content at each instant.

    The restrict spy's capture is the one that can see the create-window: it
    reads the file BEFORE delegating, so it observes whatever the creation left
    behind. The replace spy's capture is after the restriction has run, and
    afterwards the temp file no longer exists to be inspected at all.

    Both spies delegate to the real implementation and assert nothing
    themselves: ``instances.set_mode`` and ``gpu_registry.write_entry`` wrap
    this region in ``except OSError``, so an OSError-shaped complaint raised
    here would be swallowed by the code under test.
    """
    import localm.config as cfg
    events = []
    real_restrict = cfg.restrict_file_perms
    real_replace = os.replace

    def _capture(path):
        try:
            return (_perm_fingerprint(path),
                    open(path, "r", encoding="utf-8-sig").read())
        except OSError:                      # not one of ours; stay transparent
            return (None, None)

    def spy_restrict(path, **kwargs):
        events.append(("restrict", str(path), *_capture(path)))
        return real_restrict(path, **kwargs)

    def spy_replace(src, dst, *a, **kw):
        events.append(("replace", str(src), *_capture(src), str(dst)))
        return real_replace(src, dst, *a, **kw)

    monkeypatch.setattr(cfg, "restrict_file_perms", spy_restrict)
    monkeypatch.setattr(os, "replace", spy_replace)
    return events


SESSION_MARKER = "deadbeef" * 8
ATTACH_TOKEN = "attach-token-h3b"
COORD_TOKEN = "coordination-token-h3b"


def _drive_every_sibling_writer(home):
    """Run all four sibling writers once. Returns the temp path each used,
    paired with a marker that must appear in that temp file's content - without
    which a test could pass by fingerprinting an empty or unrelated file.

    ``register_instance`` and ``set_mode`` write the SAME destination, so
    callers must not assume one rename per destination.
    """
    import localm.gpu_registry as gpu_registry
    import localm.instances as instances
    import localm.sessions as sessions

    sessions.create(scopes=["chat"], key_hash=SESSION_MARKER)
    reg = instances.register_instance(
        home, instance_id="iid-h3b", port=1234, host="127.0.0.1",
        root_dir=str(home), mode="api", token=ATTACH_TOKEN)
    assert instances.set_mode(home, "iid-h3b", "full") is True
    gpath = gpu_registry.write_entry(
        home / "gpu-reg", instance_id="iid-h3b", pid=os.getpid(), port=1234,
        host="127.0.0.1", scheme="http", model=None,
        vram_estimate_bytes=None, gpu_index=0,
        coordination_token=COORD_TOKEN)
    assert gpath is not None, "gpu_registry.write_entry reported a failure"

    return {
        str(sessions.sessions_file()) + ".tmp": SESSION_MARKER,
        str(reg) + ".tmp": ATTACH_TOKEN,          # register_instance AND set_mode
        str(gpath) + ".tmp": COORD_TOKEN,
    }


def _temp_events(events, kind, expected):
    """The recorded *kind* events for the temp paths in *expected*, checked for
    count so a writer that silently stopped writing cannot pass by absence."""
    hits = [e for e in events if e[0] == kind and e[1] in expected]
    assert len(hits) == 4, (
        f"expected 4 {kind} events across the four sibling writers, saw "
        f"{len(hits)}: {[e[1] for e in events if e[0] == kind]}")
    return hits


@pytest.mark.skipif(os.name != "posix",
                    reason="POSIX-only by construction: on Windows os.open's "
                           "mode argument writes no ACL, so the temp carries "
                           "the inherited one until icacls runs, before AND "
                           "after this fix. Windows was never affected.")
def test_the_temp_file_is_created_already_private(home, monkeypatch):
    """Every sibling writer's temp file is already 0600 when the restriction is
    asked for, so it never existed at the umask default."""
    with _umask(0o022):
        # The instrument's own control, inside the same umask: a plainly-created
        # file must NOT look restricted, so a umask that makes every fresh file
        # 0600 cannot hide an unfixed writer.
        loose = home / "created-the-old-way.txt"
        loose.write_text("x", encoding="utf-8")
        assert _perm_fingerprint(loose) == "0o644", (
            "the pinned umask did not take, so this test cannot distinguish a "
            "create-restricted file from a write_text one")

        events = _install_spies(monkeypatch)
        expected = _drive_every_sibling_writer(home)

    for _kind, path, fingerprint, content in _temp_events(
            events, "restrict", expected):
        assert expected[path] in content, (
            f"{path}: fingerprinted a file that does not hold the payload - "
            f"saw {content[:120]!r}")
        assert fingerprint == "0o600", (
            f"{path}: the temp file was {fingerprint} when the restriction was "
            f"asked for, so the whole payload existed at the umask default "
            f"between the create and the chmod")


def test_every_sibling_writer_restricts_its_temp_before_the_rename(
        home, monkeypatch):
    """The ordering and the end state, on both platforms.

    Guards a future reordering (restricting the destination instead of the
    temp).
    """
    import localm.config as cfg

    # What "restricted" looks like in THIS directory on THIS box, produced by
    # the real helper before any spy is installed.
    reference = home / "reference.txt"
    reference.write_text("x", encoding="utf-8")
    assert cfg.restrict_file_perms(reference) is True
    restricted = _perm_fingerprint(reference)

    events = _install_spies(monkeypatch)
    expected = _drive_every_sibling_writer(home)

    renames = _temp_events(events, "replace", expected)
    for _kind, src, fingerprint, content, dst in renames:
        assert expected[src] in content, (src, content[:120])

        order = [e for e in events if e[1] == src and e[0] in ("restrict",
                                                               "replace")]
        assert order and order[0][0] == "restrict", (
            f"{src}: its temp file was renamed into place before anything "
            f"restricted it. Order seen: {[e[0] for e in order]}")

        assert fingerprint == restricted, (
            f"{src}: temp permissions at rename time {fingerprint} differ from "
            f"a genuinely restricted file {restricted}")

        # AND THE END STATE IS NOT WEAKENED. Restricting the temp instead of the
        # destination relies on os.replace carrying the source's ACL and mode
        # across; pinned here rather than assumed.
        assert _perm_fingerprint(dst) == restricted, (
            f"{dst}: the finished file is not restricted - os.replace did not "
            f"carry the temp file's permissions onto it")


def test_a_failed_restriction_still_persists_the_entry_and_never_raises(
        home, monkeypatch):
    """The contract stays BEST-EFFORT. ``restrict_file_perms`` returns False on
    a non-NTFS volume or when icacls is unavailable; that must not stop a
    session being minted or an instance registering, and it must fall back to
    retrying the destination rather than leaving the tightening to a call that
    already reported failure."""
    import localm.config as cfg
    calls = []

    def always_fails(path, **kwargs):
        calls.append(str(path))
        return False

    monkeypatch.setattr(cfg, "restrict_file_perms", always_fails)
    expected = _drive_every_sibling_writer(home)

    for tmp_path_str in expected:
        dest = tmp_path_str[:-len(".tmp")]
        assert os.path.isfile(dest), f"{dest} was not persisted"
        assert tmp_path_str in calls, f"{dest}: the temp file was never restricted"
        assert dest in calls, (
            f"{dest}: the destination was not retried after the temp attempt "
            f"reported failure - one failure would then be the single point of "
            f"failure")


def test_the_written_bytes_are_unchanged_by_the_permission_fix(tmp_path):
    """A permissions fix must not quietly rewrite every registry file.

    Adding ``os.O_BINARY`` to the ``os.open`` would switch every Windows
    install's session store, instance registry and GPU coordination entry from
    CRLF to LF on their next write. Without it, os.open is in TEXT mode and
    reproduces ``Path.write_text``'s output exactly.
    """
    from localm.config import atomic_write_private

    payloads = [json.dumps([{"id": "s1", "key_hash": "ab" * 32}], indent=2),
                json.dumps({"instance_id": "i", "token": "t"}, indent=2),
                "a-trailing-newline-payload\n"]
    for i, payload in enumerate(payloads):
        old = tmp_path / f"old{i}.json"
        new = tmp_path / f"new{i}.json"
        old.write_text(payload, encoding="utf-8")     # the replaced idiom
        atomic_write_private(new, payload)
        assert new.read_bytes() == old.read_bytes(), payload
        assert not (tmp_path / f"new{i}.json.tmp").exists()


def test_a_partial_write_is_resumed_rather_than_truncated(tmp_path, monkeypatch):
    """The write is looped: ``os.write`` is one syscall and may consume less
    than the whole buffer, so an unlooped raw write would allow a silently
    truncated file."""
    from localm.config import atomic_write_private

    payload = json.dumps({"coordination_token": COORD_TOKEN}, indent=2)
    real_write = os.write

    def one_byte_at_a_time(fd, data):
        # Only OUR payload is throttled. os.write is also how pytest's own
        # capture machinery moves bytes, so an unconditional stub would break
        # the runner rather than the code under test.
        if COORD_TOKEN.encode() in data or len(data) < len(payload):
            return real_write(fd, data[:1])
        return real_write(fd, data)

    monkeypatch.setattr(os, "write", one_byte_at_a_time)
    dest = tmp_path / "partial.json"
    atomic_write_private(dest, payload)
    monkeypatch.undo()

    assert dest.read_text(encoding="utf-8-sig") == payload


def test_a_zero_byte_write_raises_and_leaves_the_destination_intact(
        tmp_path, monkeypatch):
    """The other half of the loop: a write that accepts NOTHING must raise,
    never spin and never rename a truncated file over a good one."""
    from localm.config import atomic_write_private

    dest = tmp_path / "existing.json"
    good = json.dumps({"keep": "me"}, indent=2)
    dest.write_text(good, encoding="utf-8")

    payload = json.dumps({"coordination_token": COORD_TOKEN}, indent=2)
    real_write = os.write

    def accepts_nothing(fd, data):
        if COORD_TOKEN.encode() in data:
            return 0
        return real_write(fd, data)

    monkeypatch.setattr(os, "write", accepts_nothing)
    with pytest.raises(OSError) as excinfo:
        atomic_write_private(dest, payload)
    monkeypatch.undo()

    assert "short write" in str(excinfo.value)
    assert dest.read_text(encoding="utf-8-sig") == good, (
        "a refused write replaced the previous file anyway")

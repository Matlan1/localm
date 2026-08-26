# SPDX-License-Identifier: AGPL-3.0-or-later
"""Instance discovery registry: localm/instances.py + the GET /whoami endpoint.

Covers root-dir resolution, atomic register/unregister, stale reaping (dead PID
removed, live + self kept, corrupt removed), the identity payload (no token/pid
leak), and the advertise() context manager lifecycle.
"""

import json
import os
import re
import subprocess
import sys
import time

import pytest

from localm import instances


# ------------------------------------------------------------------ #
#  root_dir resolution                                               #
# ------------------------------------------------------------------ #

def test_resolve_root_dir_walks_to_marker(tmp_path):
    proj = tmp_path / "proj"
    (proj / ".git").mkdir(parents=True)
    sub = proj / "a" / "b"
    sub.mkdir(parents=True)
    assert instances.resolve_root_dir(start=str(sub)) == str(proj.resolve())


def test_resolve_root_dir_localcoder_marker(tmp_path):
    proj = tmp_path / "proj"
    (proj / ".localcoder").mkdir(parents=True)
    assert instances.resolve_root_dir(start=str(proj)) == str(proj.resolve())


def test_resolve_root_dir_falls_back_to_start(tmp_path):
    plain = tmp_path / "no-marker"
    plain.mkdir()
    assert instances.resolve_root_dir(start=str(plain)) == str(plain.resolve())


def test_resolve_root_dir_override_wins(tmp_path):
    other = tmp_path / "explicit"
    other.mkdir()
    assert instances.resolve_root_dir(start=str(tmp_path),
                                      override=str(other)) == str(other.resolve())


# ------------------------------------------------------------------ #
#  register / list / unregister                                      #
# ------------------------------------------------------------------ #

def test_register_writes_full_schema(tmp_path):
    iid = instances.new_instance_id()
    path = instances.register_instance(
        tmp_path, instance_id=iid, port=8642, host="0.0.0.0",
        root_dir=str(tmp_path), mode="full", token="tok-secret")
    assert path.exists()
    entry = json.loads(path.read_text(encoding="utf-8"))
    for key in ("instance_id", "pid", "port", "host", "root_dir", "mode",
                "version", "started", "token"):
        assert key in entry, f"missing {key}"
    assert entry["instance_id"] == iid
    assert entry["pid"] == os.getpid()
    assert entry["mode"] == "full"
    assert entry["token"] == "tok-secret"


# ------------------------------------------------------------------ #
#  _version() reflects the LIVE build, not stale installed dist-info  #
# ------------------------------------------------------------------ #

def test_version_prefers_live_version_file(monkeypatch):
    monkeypatch.setattr("localm._version.read_version", lambda: "9.9.9")
    assert instances._version() == "9.9.9"


def test_version_falls_back_to_dist_info_when_live_unknown(monkeypatch):
    monkeypatch.setattr("localm._version.read_version", lambda: "unknown")
    monkeypatch.setattr("importlib.metadata.version", lambda name: "1.2.3-installed")
    assert instances._version() == "1.2.3-installed"


def test_version_falls_back_to_dist_info_when_live_raises(monkeypatch):
    def boom():
        raise OSError("no VERSION file")
    monkeypatch.setattr("localm._version.read_version", boom)
    monkeypatch.setattr("importlib.metadata.version", lambda name: "1.2.3-installed")
    assert instances._version() == "1.2.3-installed"


def test_version_reports_unknown_not_a_fabricated_literal(monkeypatch):
    """When BOTH the live VERSION file and dist-info are unreadable, _version()
    must return the honest "unknown" sentinel, never a hardcoded version literal:
    /whoami's version gates the post-update health watchdog by EQUALITY, so a
    stale literal that ever equalled the expected version would false-PASS a build
    whose version machinery is actually broken. "unknown" can never equal a real
    target -> fails safe."""
    def boom(*a, **k):
        raise OSError("no version anywhere")
    monkeypatch.setattr("localm._version.read_version", boom)
    monkeypatch.setattr("importlib.metadata.version", boom)
    v = instances._version()
    assert v == "unknown"
    # It must not resemble a real semver a watchdog could match by equality.
    assert not v[:1].isdigit()


def test_list_entries_and_unregister(tmp_path):
    iid = instances.new_instance_id()
    path = instances.register_instance(
        tmp_path, instance_id=iid, port=9000, host="127.0.0.1",
        root_dir=str(tmp_path), mode="api", token="t")
    entries = instances.list_entries(tmp_path)
    assert len(entries) == 1 and entries[0]["instance_id"] == iid
    instances.unregister_instance(path)
    assert not path.exists()
    assert instances.list_entries(tmp_path) == []


def test_list_entries_skips_corrupt(tmp_path):
    instances.run_dir(tmp_path).mkdir(parents=True)
    (instances.run_dir(tmp_path) / "bad.json").write_text("{not json", encoding="utf-8")
    assert instances.list_entries(tmp_path) == []


# ------------------------------------------------------------------ #
#  read_entry diagnostics: size/mtime on an unreadable entry          #
# ------------------------------------------------------------------ #

def test_read_entry_empty_file_is_rejected_and_reports_size_zero(tmp_path, caplog):
    instances.run_dir(tmp_path).mkdir(parents=True)
    p = instances.run_dir(tmp_path) / "empty.json"
    p.write_bytes(b"")
    with caplog.at_level("WARNING"):
        assert instances.read_entry(p) is None
    assert "size=0" in caplog.text
    assert "char 0" in caplog.text   # json's own message for an empty string


def test_read_entry_truncated_file_reports_nonzero_size(tmp_path, caplog):
    """A file with SOME content that is still invalid JSON - the shape a
    mid-write truncation would leave - must report a NONZERO size, the exact
    signal that distinguishes it from the empty-file case above."""
    instances.run_dir(tmp_path).mkdir(parents=True)
    p = instances.run_dir(tmp_path) / "truncated.json"
    p.write_text('{"instance_id": "abc", "pid":', encoding="utf-8")   # cut off
    with caplog.at_level("WARNING"):
        assert instances.read_entry(p) is None
    assert "size=0" not in caplog.text
    m = re.search(r"size=(\d+)", caplog.text)
    assert m and int(m.group(1)) > 0, caplog.text


def test_read_entry_missing_file_does_not_log(tmp_path, caplog):
    """The normal, expected case (no entry yet, or already reaped) must stay
    silent - only a genuinely unreadable EXISTING file is diagnostic-worthy."""
    p = instances.run_dir(tmp_path) / "does-not-exist.json"
    with caplog.at_level("WARNING"):
        assert instances.read_entry(p) is None
    assert caplog.text == ""


def test_reap_removes_the_empty_entry_read_entry_flags(tmp_path):
    """The existing reap behaviour (already covered by test_reap_removes_corrupt
    for a non-empty corrupt file) must hold for the empty-file case too - the
    instrumentation is diagnostic-only and must not change what gets cleaned up."""
    instances.run_dir(tmp_path).mkdir(parents=True)
    bad = instances.run_dir(tmp_path) / "empty.json"
    bad.write_bytes(b"")
    instances.reap_stale(tmp_path, is_alive=lambda e: True)
    assert not bad.exists()


@pytest.mark.skipif(__import__("sys").platform == "win32", reason="POSIX modes only")
def test_registry_file_is_owner_only_on_posix(tmp_path):
    import stat
    iid = instances.new_instance_id()
    path = instances.register_instance(
        tmp_path, instance_id=iid, port=1, host="127.0.0.1",
        root_dir=str(tmp_path), mode="api", token="t")
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600


# ------------------------------------------------------------------ #
#  reaping                                                            #
# ------------------------------------------------------------------ #

def _raw(tmp_path, iid, pid):
    d = instances.run_dir(tmp_path)
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{iid}.json"
    p.write_text(json.dumps({"instance_id": iid, "pid": pid, "port": 1,
                             "host": "127.0.0.1", "root_dir": str(tmp_path),
                             "mode": "api", "version": "0", "started": "0",
                             "token": "t"}), encoding="utf-8")
    return p


def test_reap_removes_dead_keeps_live_and_self(tmp_path):
    dead = _raw(tmp_path, "dead0001", 111)
    live = _raw(tmp_path, "live0001", 222)
    mine = _raw(tmp_path, "self0001", 333)
    alive_pids = {222, 333}
    removed = instances.reap_stale(
        tmp_path, self_id="self0001",
        is_alive=lambda e: int(e["pid"]) in alive_pids)
    assert "dead0001" in removed
    assert not dead.exists()
    assert live.exists(), "a live instance must never be reaped"
    assert mine.exists(), "self must never be reaped"


def test_reap_removes_corrupt(tmp_path):
    instances.run_dir(tmp_path).mkdir(parents=True)
    bad = instances.run_dir(tmp_path) / "corrupt.json"
    bad.write_text("xxx", encoding="utf-8")
    instances.reap_stale(tmp_path, is_alive=lambda e: True)
    assert not bad.exists()


def test_reap_keeps_all_on_probe_error(tmp_path):
    e = _raw(tmp_path, "keep0001", 444)

    def boom(_entry):
        raise RuntimeError("probe failed")

    instances.reap_stale(tmp_path, is_alive=boom)
    assert e.exists(), "an errored probe must not delete the entry"


def test_pid_alive_self_and_invalid():
    assert instances.pid_alive(os.getpid()) is True
    assert instances.pid_alive(-1) is False
    assert instances.pid_alive(0) is False


# ------------------------------------------------------------------ #
#  kill_pid (the `localm stop` direct-kill fallback)                 #
# ------------------------------------------------------------------ #

def _wait_until_reads_dead(pid: int, timeout: float = 10.0) -> bool:
    """Wait for *pid* to stop reading as alive, and report whether it did.

    Popen.wait() returns as soon as the process handle is signalled, which is
    process TERMINATION, not the end of process rundown. On Windows the exited
    process stays in the OS process enumeration for a short window afterwards,
    and psutil.pid_exists() (what pid_alive uses there) is enumeration-based,
    not exit-code-based: OpenProcess + GetExitCodeProcess already report the
    real exit code while psutil still answers True. Sampling pid_alive ONCE
    right after wait() therefore samples inside that window and is inherently
    racy.

    The contract under test is that a pid which has exited stops reading as
    alive, not that the OS process table updates synchronously with exit, so
    this polls for it. The timeout is orders of magnitude larger than the
    enumeration window, so a genuine regression (a pid that never stops reading
    as alive) still fails hard rather than being waited away.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if instances.pid_alive(pid) is False:
            return True
        time.sleep(0.001)
    return False


def test_kill_pid_invalid_pid_is_noop():
    assert instances.kill_pid(-1) is True
    assert instances.kill_pid(0) is True


def test_kill_pid_already_dead_returns_true():
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait(timeout=10)
    assert _wait_until_reads_dead(proc.pid), (
        "an exited pid never stopped reading as alive")
    assert instances.kill_pid(proc.pid) is True


def test_kill_pid_terminates_a_real_live_process():
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        assert instances.pid_alive(proc.pid) is True
        assert instances.kill_pid(proc.pid, timeout=10) is True
        assert instances.pid_alive(proc.pid) is False
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)


# ------------------------------------------------------------------ #
#  whoami payload + advertise lifecycle                              #
# ------------------------------------------------------------------ #

def test_whoami_payload_has_identity_no_secrets():
    p = instances.whoami_payload("abc123", "/home/x/proj", "full")
    assert p["app"] == "localm"
    assert p["instance_id"] == "abc123"
    assert p["root_dir"] == "/home/x/proj"
    assert p["mode"] == "full"
    assert "version" in p
    assert "token" not in p and "pid" not in p


class _FakeApp:
    class _State:
        pass

    def __init__(self):
        self.state = _FakeApp._State()


def test_advertise_registers_sets_state_and_cleans_up(tmp_path):
    app = _FakeApp()
    assert instances.list_entries(tmp_path) == []
    with instances.advertise(app, tmp_path, host="0.0.0.0", port=8642, mode="full") as info:
        entries = instances.list_entries(tmp_path)
        assert len(entries) == 1
        assert entries[0]["instance_id"] == info["instance_id"]
        # app.state is wired for /whoami
        assert app.state.instance_id == info["instance_id"]
        assert app.state.instance_mode == "full"
        assert app.state.root_dir == info["root_dir"]
        assert app.state.instance_token  # token set, not exposed via whoami
    # cleaned up on exit (only our file)
    assert instances.list_entries(tmp_path) == []


def test_advertise_project_override(tmp_path):
    app = _FakeApp()
    proj = tmp_path / "explicit-root"
    proj.mkdir()
    with instances.advertise(app, tmp_path, host="127.0.0.1", port=1,
                             mode="api", project=str(proj)) as info:
        assert info["root_dir"] == str(proj.resolve())


def test_advertise_sets_bind_coordinates_on_state(tmp_path):
    """The instance records its own port + scheme on app.state so it can build
    its loopback /v1 self-url when it mounts a surface on demand."""
    app = _FakeApp()
    with instances.advertise(app, tmp_path, host="0.0.0.0", port=8651,
                             mode="api", scheme="https"):
        assert app.state.instance_port == 8651
        assert app.state.instance_scheme == "https"


def test_set_mode_rewrites_entry_in_place(tmp_path):
    """An on-demand GUI mount flips this instance's registry mode."""
    iid = instances.new_instance_id()
    instances.register_instance(
        tmp_path, instance_id=iid, port=8642, host="127.0.0.1",
        root_dir=str(tmp_path), mode="api", token="tok", scheme="http")
    assert instances.set_mode(tmp_path, iid, "full") is True
    entry = instances.read_entry(instances.registry_path(tmp_path, iid))
    assert entry["mode"] == "full"
    # The token + other fields survive the rewrite.
    assert entry["token"] == "tok"
    assert entry["port"] == 8642


def test_set_mode_missing_entry_is_false(tmp_path):
    assert instances.set_mode(tmp_path, "nonexistent-id", "full") is False
    assert instances.set_mode(tmp_path, "", "full") is False


# ------------------------------------------------------------------ #
#  GET /whoami endpoint                                              #
# ------------------------------------------------------------------ #

def _mock_engine():
    from unittest.mock import MagicMock
    e = MagicMock()
    e.display_name = "test-model"
    type(e).loaded = property(lambda self: True)
    return e


def test_whoami_endpoint_reports_state_unauthenticated(tmp_path):
    from fastapi.testclient import TestClient
    from localm.inference.http_server import create_app

    app = create_app(_mock_engine())
    app.state.instance_id = "iid-xyz"
    app.state.root_dir = str(tmp_path)
    app.state.instance_mode = "full"

    client = TestClient(app)
    r = client.get("/whoami")          # no Authorization header
    assert r.status_code == 200
    body = r.json()
    assert body == {
        "app": "localm",
        "version": body["version"],
        "instance_id": "iid-xyz",
        "root_dir": str(tmp_path),
        "mode": "full",
    }
    assert "token" not in body and "pid" not in body


def test_whoami_endpoint_before_wiring_returns_nulls():
    from fastapi.testclient import TestClient
    from localm.inference.http_server import create_app

    client = TestClient(create_app(_mock_engine()))
    body = client.get("/whoami").json()
    assert body["app"] == "localm"
    assert body["instance_id"] is None and body["mode"] is None


def test_whoami_omits_root_dir_on_a_network_bind(tmp_path):
    """root_dir is an absolute host path (it can carry the OS username); it must
    not leak to LAN clients. It is disclosed only on a loopback bind - discovery
    matches root_dir from the registry file, not /whoami, so identity still
    works."""
    from fastapi.testclient import TestClient
    from localm.inference.http_server import create_app

    app = create_app(_mock_engine())
    app.state.instance_id = "iid-net"
    app.state.root_dir = str(tmp_path)
    app.state.instance_mode = "api"
    app.state.bind_host = "0.0.0.0"          # network bind
    body = TestClient(app).get("/whoami").json()
    assert body["root_dir"] is None           # omitted over the network
    assert body["instance_id"] == "iid-net"   # identity handshake still works
    assert body["mode"] == "api"


def test_whoami_keeps_root_dir_on_a_loopback_bind(tmp_path):
    from fastapi.testclient import TestClient
    from localm.inference.http_server import create_app

    app = create_app(_mock_engine())
    app.state.root_dir = str(tmp_path)
    app.state.bind_host = "127.0.0.1"
    body = TestClient(app).get("/whoami").json()
    assert body["root_dir"] == str(tmp_path)


# ------------------------------------------------------------------ #
#  Phase 4: scheme + attach-or-spawn discovery                       #
# ------------------------------------------------------------------ #

def test_register_records_scheme(tmp_path):
    p = instances.register_instance(
        tmp_path, instance_id="i", port=1, host="0.0.0.0",
        root_dir=str(tmp_path), mode="full", token="t", scheme="https")
    assert json.loads(p.read_text(encoding="utf-8"))["scheme"] == "https"


def test_attach_url():
    assert instances.attach_url({"scheme": "https", "port": 8651}) == "https://127.0.0.1:8651/"
    assert instances.attach_url({"port": 8642}) == "http://127.0.0.1:8642/"   # default http


def test_find_attachable_returns_live_same_dir(tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    instances.register_instance(
        tmp_path, instance_id="live01", port=1, host="127.0.0.1",
        root_dir=str(proj), mode="full", token="t", scheme="http")
    got = instances.find_attachable(tmp_path, str(proj), probe=lambda e: True)
    assert got is not None and got["instance_id"] == "live01"


def test_find_attachable_reaps_confident_dead_same_dir(tmp_path, monkeypatch):
    # Dead process + failing handshake -> confident dead -> reaped. Liveness is
    # injected (a hardcoded "probably dead" PID is flaky on hosts with a high
    # pid_max, e.g. CI Linux runners).
    p = _raw(tmp_path, "dead01", 4242)
    monkeypatch.setattr(instances, "pid_alive", lambda pid: False)
    assert instances.find_attachable(tmp_path, str(tmp_path), probe=lambda e: False) is None
    assert not p.exists(), "a confident-dead same-dir entry must be reaped"


def test_find_attachable_keeps_live_pid_on_probe_miss(tmp_path, monkeypatch):
    # Live process but the handshake misses (slow /whoami, or an impostor): never
    # attach, and never reap a live entry.
    p = _raw(tmp_path, "slow01", 4242)
    monkeypatch.setattr(instances, "pid_alive", lambda pid: True)
    assert instances.find_attachable(tmp_path, str(tmp_path), probe=lambda e: False) is None
    assert p.exists(), "a live-PID entry must NOT be reaped on a transient probe miss"


def test_find_attachable_ignores_and_keeps_other_dir(tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    mine = tmp_path / "mine"
    mine.mkdir()
    p = instances.register_instance(
        tmp_path, instance_id="o01", port=1, host="127.0.0.1",
        root_dir=str(other), mode="full", token="t")
    # Even a probe that would say "alive" must not match a different dir, and must
    # not reap another dir's entry.
    assert instances.find_attachable(tmp_path, str(mine), probe=lambda e: True) is None
    assert p.exists()


def test_find_attachable_case_insensitive_on_windows(tmp_path):
    import os
    proj = tmp_path / "Proj"
    proj.mkdir()
    instances.register_instance(
        tmp_path, instance_id="ci01", port=1, host="127.0.0.1",
        root_dir=str(proj), mode="full", token="t")
    query = str(proj).upper() if os.name == "nt" else str(proj)
    got = instances.find_attachable(tmp_path, query, probe=lambda e: True)
    assert got is not None, "Windows paths are case-insensitive; root_dir must match"


def test_default_probe_uses_recorded_scheme(monkeypatch):
    calls = []
    monkeypatch.setattr(instances, "_try_whoami",
                        lambda s, p, i, t, h=None: (calls.append(s), s == "https")[1])
    assert instances.default_probe({"port": 1, "instance_id": "i", "scheme": "https"}) is True
    assert calls == ["https"], "a recorded scheme must be probed first/only"


def test_default_probe_missing_scheme_tries_both(monkeypatch):
    calls = []
    monkeypatch.setattr(instances, "_try_whoami",
                        lambda s, p, i, t, h=None: (calls.append(s), False)[1])
    assert instances.default_probe({"port": 1, "instance_id": "i"}) is False
    assert calls == ["http", "https"]


def test_default_probe_forwards_the_recorded_bind_host(monkeypatch):
    """The entry's bind host must reach _try_whoami, or an IPv6-bound server is
    probed on an IPv4 loopback it is not listening on and reported DEAD while
    it is serving."""
    seen = []
    monkeypatch.setattr(instances, "_try_whoami",
                        lambda s, p, i, t, h=None: (seen.append(h), True)[1])
    assert instances.default_probe(
        {"port": 1, "instance_id": "i", "scheme": "http", "host": "::1"}) is True
    assert seen == ["::1"]


def test_attach_target_for_running_instance(tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    instances.register_instance(
        tmp_path, instance_id="t01", port=8651, host="0.0.0.0",
        root_dir=str(proj), mode="full", token="the-token", scheme="https")
    tgt = instances.attach_target(tmp_path, str(proj), probe=lambda e: True)
    assert tgt == {
        "base_url": "https://127.0.0.1:8651/v1",
        "token": "the-token",
        "port": 8651,
        "scheme": "https",
        "mode": "full",
    }


def test_attach_target_none_when_no_instance(tmp_path):
    assert instances.attach_target(tmp_path, str(tmp_path), probe=lambda e: True) is None


def test_advertise_isolated_is_invisible(tmp_path):
    app = _FakeApp()
    with instances.advertise(app, tmp_path, host="127.0.0.1", port=1,
                             mode="full", isolated=True) as info:
        # /whoami still works (state set) but nothing is in the registry
        assert app.state.instance_id
        assert instances.list_entries(tmp_path) == []
        assert info["path"] is None
    assert instances.list_entries(tmp_path) == []

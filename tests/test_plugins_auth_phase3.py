# SPDX-License-Identifier: AGPL-3.0-or-later
"""Security regressions for the plugin engine and the auth keystore.

- PluginManager.uninstall() must honour the protected-plugin guard, so a
  protected plugin (e.g. chat) cannot be deleted. It is the ONE plugin-removal
  path, so it is the one this guards.
- The auth keystore is locked: two concurrent create_key calls must not
  read-modify-write the shared record list and lose one write.
"""

import threading

import pytest
from fastapi import FastAPI


# --------------------------------------------------------------------------- #
#  uninstall() refuses a protected plugin                                     #
# --------------------------------------------------------------------------- #

def _write_plugin(root, name, *, protected=None):
    """Create an installed engine-contract plugin dir. *protected* None omits
    the key entirely; True/False writes it explicitly."""
    d = root / name
    d.mkdir(parents=True)
    lines = ["[plugin]", f'name = "{name}"', 'register = "plug"']
    if protected is not None:
        lines.append(f"protected = {'true' if protected else 'false'}")
    (d / "plugin.toml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (d / "plug.py").write_text(
        "def register(host):\n    pass\n\n\ndef unregister():\n    pass\n",
        encoding="utf-8")
    return d


def _manager(tmp_path):
    from localm.plugins.engine import PluginManager
    installed = tmp_path / "installed"
    installed.mkdir(parents=True, exist_ok=True)
    return PluginManager(FastAPI(), external_root=installed, builtin_root=None)


def test_uninstall_refuses_catalog_protected_chat(tmp_path):
    """The catalog marks chat as protected; uninstall() must refuse it and
    leave its directory on disk."""
    mgr = _manager(tmp_path)
    _write_plugin(mgr._installed_root, "chat")
    with pytest.raises(ValueError, match="protected"):
        mgr.uninstall("chat")
    assert (mgr._installed_root / "chat").is_dir()   # not deleted


def test_uninstall_refuses_manifest_protected_plugin(tmp_path):
    """A plugin not in the catalog but whose installed manifest sets
    protected = true must also be refused (manifest is the engine's source)."""
    mgr = _manager(tmp_path)
    _write_plugin(mgr._installed_root, "guarded", protected=True)
    with pytest.raises(ValueError, match="protected"):
        mgr.uninstall("guarded")
    assert (mgr._installed_root / "guarded").is_dir()


def test_uninstall_allows_unprotected_plugin(tmp_path):
    """Control: an ordinary plugin (no protected flag) is still removable."""
    mgr = _manager(tmp_path)
    _write_plugin(mgr._installed_root, "demo", protected=False)
    assert mgr.uninstall("demo") is True
    assert not (mgr._installed_root / "demo").exists()


def test_uninstall_allows_plugin_without_protected_key(tmp_path):
    """A manifest that omits the protected key entirely defaults to removable."""
    mgr = _manager(tmp_path)
    _write_plugin(mgr._installed_root, "plain", protected=None)
    assert mgr.uninstall("plain") is True
    assert not (mgr._installed_root / "plain").exists()


def test_uninstall_missing_raises_keyerror(tmp_path):
    """An absent plugin raises KeyError (not a protected error)."""
    mgr = _manager(tmp_path)
    with pytest.raises(KeyError):
        mgr.uninstall("ghost")


# --------------------------------------------------------------------------- #
#  Concurrent create_key calls all persist (no lost write)                    #
# --------------------------------------------------------------------------- #

@pytest.fixture
def auth(tmp_path, monkeypatch):
    """localm.auth pointed at a throwaway data dir, clean auth env."""
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path)
    monkeypatch.setattr(cfg, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", tmp_path / "registry.json")
    import localm.auth as a
    return a


def test_concurrent_create_key_no_lost_write(auth):
    """Many threads minting keys at once must all persist. With an unserialized
    read-modify-write the slow JSON round-trip lets writers clobber each other,
    so fewer than N records survive; the lock guarantees all N."""
    from localm import scopes as S

    n = 40
    barrier = threading.Barrier(n)
    errors = []
    made = []
    made_lock = threading.Lock()

    def worker(i):
        try:
            barrier.wait()          # maximise overlap of the read-modify-write
            rec = auth.create_key(f"key{i}", [S.CHAT])
            with made_lock:
                made.append(rec)
        except Exception as e:      # pragma: no cover - surfaced via assert
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"create_key raised under concurrency: {errors}"
    listed = auth.list_keys()
    assert len(listed) == n, f"lost write: expected {n} keys, found {len(listed)}"
    # ids are unique and every minted key actually verifies.
    ids = {r["id"] for r in listed}
    assert len(ids) == n
    for rec in made:
        assert auth.verify(rec["key"]) == {S.CHAT}


def test_concurrent_create_and_revoke_consistent(auth):
    """Interleaved create and revoke calls must not corrupt the store or lose a
    surviving key. Pre-seed keys, then concurrently revoke half while minting a
    fresh batch; the final count must be exactly (kept + newly created)."""
    from localm import scopes as S

    seed = [auth.create_key(f"seed{i}", [S.CHAT]) for i in range(10)]
    to_revoke = seed[:5]
    n_new = 10
    barrier = threading.Barrier(len(to_revoke) + n_new)
    errors = []

    def revoker(rec):
        try:
            barrier.wait()
            auth.revoke_key(rec["id"])
        except Exception as e:      # pragma: no cover
            errors.append(e)

    def creator(i):
        try:
            barrier.wait()
            auth.create_key(f"new{i}", [S.CHAT])
        except Exception as e:      # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=revoker, args=(r,)) for r in to_revoke]
    threads += [threading.Thread(target=creator, args=(i,)) for i in range(n_new)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"raised under concurrency: {errors}"
    listed = auth.list_keys()
    # 10 seeded - 5 revoked + 10 created = 15
    assert len(listed) == 15, f"expected 15 keys, found {len(listed)}"
    surviving_ids = {r["id"] for r in listed}
    for rec in to_revoke:
        assert rec["id"] not in surviving_ids   # revoked keys gone
    for rec in seed[5:]:
        assert rec["id"] in surviving_ids       # kept seeds survived

# SPDX-License-Identifier: AGPL-3.0-or-later
"""M2 Phase 3 security regressions for the legacy plugin loader and the auth
keystore.

SEC-6: loader.remove_plugin (and DELETE /v1/plugins/{name} that calls it) must
       honour the engine's protected-plugin guard, so a protected plugin (e.g.
       chat) cannot be deleted via the legacy path.
SEC-10: the auth keystore had no lock; two concurrent create_key calls could
        read-modify-write the shared record list and lose one write.
"""

import threading

import pytest


# --------------------------------------------------------------------------- #
#  SEC-6: remove_plugin refuses a protected plugin                            #
# --------------------------------------------------------------------------- #

def _write_plugin(root, name, *, protected=None, register=False):
    """Create an installed-plugin dir with a manifest. *protected* None omits the
    key entirely; True/False writes it explicitly. *register* makes it an
    engine-contract plugin instead of a legacy one."""
    d = root / name
    d.mkdir(parents=True)
    lines = ["[plugin]", f'name = "{name}"']
    if register:
        lines.append('register = "plug"')
    else:
        lines.append('entry = "x:main"')
    if protected is not None:
        lines.append(f"protected = {'true' if protected else 'false'}")
    (d / "plugin.toml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return d


@pytest.fixture
def loader(tmp_path, monkeypatch):
    """localm.plugins.loader with plugins_dir() redirected to a throwaway dir."""
    root = tmp_path / "plugins"
    root.mkdir()
    import localm.plugins.loader as ldr
    monkeypatch.setattr(ldr, "plugins_dir", lambda: root)
    ldr._root = root  # convenience handle for tests
    return ldr


def test_remove_refuses_catalog_protected_chat(loader):
    """The catalog marks chat as protected; remove_plugin must refuse it and
    leave its directory on disk (the bug deleted it)."""
    root = loader._root
    # chat ships as an engine-contract plugin (register=, protected=true).
    _write_plugin(root, "chat", protected=True, register=True)
    with pytest.raises(loader.PluginError, match="protected"):
        loader.remove_plugin("chat")
    assert (root / "chat").is_dir()   # not deleted


def test_remove_refuses_manifest_protected_plugin(loader):
    """A plugin not in the catalog but whose installed manifest sets
    protected = true must also be refused (manifest is the engine's source)."""
    root = loader._root
    _write_plugin(root, "guarded", protected=True)
    with pytest.raises(loader.PluginError, match="protected"):
        loader.remove_plugin("guarded")
    assert (root / "guarded").is_dir()


def test_remove_allows_unprotected_plugin(loader):
    """Control: an ordinary plugin (no protected flag) is still removable."""
    root = loader._root
    _write_plugin(root, "demo", protected=False)
    assert loader.remove_plugin("demo") is True
    assert not (root / "demo").exists()


def test_remove_allows_plugin_without_protected_key(loader):
    """A manifest that omits the protected key entirely defaults to removable."""
    root = loader._root
    _write_plugin(root, "plain", protected=None)
    assert loader.remove_plugin("plain") is True
    assert not (root / "plain").exists()


def test_remove_missing_still_returns_false(loader):
    """An absent plugin returns False (not a protected error)."""
    assert loader.remove_plugin("ghost") is False


# --------------------------------------------------------------------------- #
#  SEC-10: concurrent create_key calls all persist (no lost write)            #
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

# SPDX-License-Identifier: AGPL-3.0-or-later
"""GET /api/models `missing`/`last_path` plus POST /api/models/relocate - the GUI
form of `localm relocate MODEL NEW_PATH`, and the GUI's view of the 'missing'
condition `localm list` flags.

SECURITY: new_path always names a location on the server's own disk (unlike a
pull spec, which may be a remote HuggingFace repo id), so relocate is gated on
host filesystem access exactly like /api/models/scan - see
test_scan_workdir_route.py, whose fixture shape this file mirrors."""

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from localm import scopes as S


def _gguf(p: Path) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"GGUF" + b"\x00" * 2048)      # valid GGUF magic + past the size floor
    return p


@pytest.fixture
def relocate_app(tmp_path, monkeypatch):
    """Full stack (engine + GUI) on a throwaway home. No owner key by default,
    so a minted key's own scopes/fs_access govern what it can reach - same
    shape as test_scan_workdir_route.py's scan_app fixture."""
    home = tmp_path / ".localm"
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    import localm.config as _cfg
    monkeypatch.setattr(_cfg, "HOME_DIR", home)
    monkeypatch.setattr(_cfg, "MODELS_DIR", home / "models")
    monkeypatch.setattr(_cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(_cfg, "REGISTRY_FILE", home / "registry.json")
    import localm.model_manager as mm
    monkeypatch.setattr(mm, "MODELS_DIR", home / "models", raising=False)
    monkeypatch.setattr(mm, "REGISTRY_FILE", home / "registry.json", raising=False)
    from localm.plugins.engine import attach_engine
    from localm.plugins.gui.web import attach_gui
    app = FastAPI()
    attach_engine(app)
    attach_gui(app, self_url="http://127.0.0.1:9/v1",
               switch_model=lambda name: None,
               active_model=lambda: "")
    return app, tmp_path


def _hdr(key):
    return {"Authorization": f"Bearer {key}"}


def _writer_key():
    """models:write only, default fs_access ('none') - must not reach relocate."""
    from localm import auth
    return auth.create_key("writer", [S.MODELS_WRITE])["key"]


def _host_writer_key():
    """models:write AND host filesystem access - the level relocate requires."""
    from localm import auth
    return auth.create_key("host-writer", [S.MODELS_WRITE], fs_access="host")["key"]


def _no_models_key():
    from localm import auth
    return auth.create_key("narrow", [S.MCP])["key"]


def _host_read_write_key():
    """models:read AND models:write AND host filesystem access - WRITE alone
    does not imply READ (only ADMIN/coder:full carry an implication), so a
    test that both lists and relocates through one key needs both."""
    from localm import auth
    return auth.create_key("host-rw", [S.MODELS_READ, S.MODELS_WRITE],
                           fs_access="host")["key"]


class TestMissingSurfacedOnModelsList:
    def test_a_model_whose_file_is_gone_is_flagged_missing(self, relocate_app):
        import localm.model_manager as mm
        app, tmp = relocate_app
        mm.save_registry({"gone": {"path": str(tmp / "nowhere" / "x.gguf"), "source": "local"}})
        with TestClient(app) as c:
            r = c.get("/api/models")
            assert r.status_code == 200, r.text
            row = next(m for m in r.json()["models"] if m["name"] == "gone")
            assert row["missing"] is True
            assert row["last_path"] == str(tmp / "nowhere" / "x.gguf")

    def test_a_healthy_model_carries_neither_field(self, relocate_app):
        import localm.model_manager as mm
        app, tmp = relocate_app
        good = _gguf(tmp / "models" / "here.gguf")
        mm.save_registry({"here": {"path": str(good), "source": "local"}})
        with TestClient(app) as c:
            r = c.get("/api/models")
            assert r.status_code == 200, r.text
            row = next(m for m in r.json()["models"] if m["name"] == "here")
            assert "missing" not in row
            assert "last_path" not in row


class TestRelocateRequiresHostFsAccess:
    def test_403s_for_a_models_write_only_key(self, relocate_app):
        import localm.model_manager as mm
        app, tmp = relocate_app
        old = _gguf(tmp / "ext" / "m.gguf")
        mm.save_registry({"m": {"path": str(old), "source": "local"}})
        new = _gguf(tmp / "moved" / "m.gguf")
        with TestClient(app) as c:
            r = c.post("/api/models/relocate", headers=_hdr(_writer_key()),
                       json={"model": "m", "new_path": str(new)})
            assert r.status_code == 403, r.text

    def test_the_scope_gate_applies_before_the_fs_gate(self, relocate_app):
        import localm.model_manager as mm
        app, tmp = relocate_app
        old = _gguf(tmp / "ext" / "m.gguf")
        mm.save_registry({"m": {"path": str(old), "source": "local"}})
        with TestClient(app) as c:
            r = c.post("/api/models/relocate", headers=_hdr(_no_models_key()),
                       json={"model": "m", "new_path": str(old)})
            assert r.status_code == 403, r.text

    def test_succeeds_for_a_host_fs_access_key(self, relocate_app):
        import localm.model_manager as mm
        app, tmp = relocate_app
        old = _gguf(tmp / "ext" / "m.gguf")
        mm.save_registry({"m": {"path": str(old), "source": "local", "missing": True}})
        new = _gguf(tmp / "moved" / "m.gguf")
        with TestClient(app) as c:
            r = c.post("/api/models/relocate", headers=_hdr(_host_writer_key()),
                       json={"model": "m", "new_path": str(new)})
            assert r.status_code == 200, r.text
            assert r.json()["path"] == str(new.resolve())
        reg = mm.load_registry()
        assert Path(reg["m"]["path"]) == new.resolve()
        assert "missing" not in reg["m"]

    def test_succeeds_in_open_mode(self, relocate_app):
        """No key configured -> loopback owner -> host access implied."""
        import localm.model_manager as mm
        app, tmp = relocate_app
        old = _gguf(tmp / "ext" / "m.gguf")
        mm.save_registry({"m": {"path": str(old), "source": "local"}})
        new = _gguf(tmp / "moved" / "m.gguf")
        with TestClient(app) as c:
            r = c.post("/api/models/relocate", json={"model": "m", "new_path": str(new)})
            assert r.status_code == 200, r.text


class TestRelocateValidation:
    def test_404_for_an_unregistered_model(self, relocate_app):
        app, tmp = relocate_app
        new = _gguf(tmp / "moved" / "m.gguf")
        with TestClient(app) as c:
            r = c.post("/api/models/relocate", headers=_hdr(_host_writer_key()),
                       json={"model": "nope", "new_path": str(new)})
            assert r.status_code == 404, r.text

    def test_400_when_the_new_path_does_not_exist(self, relocate_app):
        import localm.model_manager as mm
        app, tmp = relocate_app
        old = _gguf(tmp / "ext" / "m.gguf")
        mm.save_registry({"m": {"path": str(old), "source": "local"}})
        with TestClient(app) as c:
            r = c.post("/api/models/relocate", headers=_hdr(_host_writer_key()),
                       json={"model": "m", "new_path": str(tmp / "nope.gguf")})
            assert r.status_code == 400, r.text
            assert "does not exist" in r.json()["detail"]
        # The bad request must not have touched the registry.
        assert Path(mm.load_registry()["m"]["path"]) == old

    def test_400_for_a_file_with_the_wrong_magic(self, relocate_app):
        import localm.model_manager as mm
        app, tmp = relocate_app
        old = _gguf(tmp / "ext" / "m.gguf")
        mm.save_registry({"m": {"path": str(old), "source": "local"}})
        bad = tmp / "not-really.gguf"
        bad.write_bytes(b"NOPE not a gguf")
        with TestClient(app) as c:
            r = c.post("/api/models/relocate", headers=_hdr(_host_writer_key()),
                       json={"model": "m", "new_path": str(bad)})
            assert r.status_code == 400, r.text
            assert "GGUF" in r.json()["detail"]

    def test_relocate_clears_missing_from_the_models_list(self, relocate_app):
        """End-to-end: a row that reads missing=True stops reading that way
        after a successful relocate, through the same /api/models route.

        Mints its key BEFORE the first GET: any_key_configured() flips the
        server out of open/loopback mode the instant a key exists anywhere, so
        an unauthenticated GET made AFTER minting one would 401 - not a second
        auth boundary this test needs to prove, so it uses the one key
        throughout instead of relying on open mode for the earlier reads."""
        import localm.model_manager as mm
        app, tmp = relocate_app
        old = tmp / "ext" / "m.gguf"          # never created - starts missing
        mm.save_registry({"m": {"path": str(old), "source": "local"}})
        new = _gguf(tmp / "moved" / "m.gguf")
        key = _host_read_write_key()
        with TestClient(app) as c:
            before = c.get("/api/models", headers=_hdr(key)).json()
            assert next(m for m in before["models"] if m["name"] == "m")["missing"] is True
            r = c.post("/api/models/relocate", headers=_hdr(key),
                       json={"model": "m", "new_path": str(new)})
            assert r.status_code == 200, r.text
            after = c.get("/api/models", headers=_hdr(key)).json()
            row = next(m for m in after["models"] if m["name"] == "m")
            assert "missing" not in row

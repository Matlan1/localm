# SPDX-License-Identifier: AGPL-3.0-or-later
"""Two security properties:

  - setting a media backend launch_cmd/api_url requires admin
  - `localm key recover` mints a fresh owner key (the lockout escape)
"""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path)
    monkeypatch.setattr(cfg, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", tmp_path / "registry.json")
    return tmp_path


# --------------------------------------------------------------------------- #
#  Media launch_cmd requires admin                                             #
# --------------------------------------------------------------------------- #

def test_media_launch_cmd_requires_admin(env):
    from localm import auth, scopes
    from localm.inference.http_server import create_app

    auth.set_api_key("owner-secret-key-123")             # protected mode
    # config:write is a privileged scope (owner-mintable); the owner mints a
    # companion key that carries it - which must still NOT be able to set launch_cmd.
    writer = auth.create_key("cfgwriter", [scopes.CONFIG_WRITE],
                             allow_privileged=True)["key"]

    client = TestClient(create_app(None))
    owner_hdr = {"Authorization": "Bearer owner-secret-key-123"}
    writer_hdr = {"Authorization": f"Bearer {writer}"}

    # A config:write key may set an ORDINARY media field ...
    ok = client.post("/v1/media/config/image", json={"steps": 20}, headers=writer_hdr)
    assert ok.status_code != 403, ok.text

    # ... but NOT launch_cmd (shell) or api_url (network target) - admin only.
    denied = client.post("/v1/media/config/image",
                         json={"launch_cmd": "evil.bat"}, headers=writer_hdr)
    assert denied.status_code == 403, denied.text

    denied_url = client.post("/v1/media/config/image",
                             json={"api_url": "http://x"}, headers=writer_hdr)
    assert denied_url.status_code == 403

    # The owner (ADMIN) key may.
    allowed = client.post("/v1/media/config/image",
                          json={"launch_cmd": "start.bat"}, headers=owner_hdr)
    assert allowed.status_code != 403, allowed.text


def test_media_launch_cmd_allowed_in_open_mode(env):
    # Open mode = the trusted local owner (already gated by the origin/shell-token
    # guard). caller_scopes is None, so the admin check does not block.
    from localm.inference.http_server import create_app
    app = create_app(None)
    client = TestClient(app)
    r = client.post("/v1/media/config/image", json={"launch_cmd": "start.bat"},
                    headers={"Authorization": f"Bearer {app.state.shell_token}"})
    assert r.status_code != 403, r.text


# --------------------------------------------------------------------------- #
#  Owner key recovery                                                          #
# --------------------------------------------------------------------------- #

def test_key_recover_mints_fresh_owner_key(env):
    from click.testing import CliRunner
    from localm import auth
    from localm.cli import keys as keyscli

    assert auth.get_api_key() is None
    runner = CliRunner()
    r = runner.invoke(keyscli.key_group, ["recover"])
    assert r.exit_code == 0, r.output
    first = auth.get_api_key()
    assert first, "recover must persist a new owner key"
    assert "recover" in r.output.lower()

    # Running it again rotates the owner key (a fresh recovery).
    r2 = runner.invoke(keyscli.key_group, ["recover"])
    assert r2.exit_code == 0
    assert auth.get_api_key() and auth.get_api_key() != first

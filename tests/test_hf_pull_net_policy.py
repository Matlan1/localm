# SPDX-License-Identifier: AGPL-3.0-or-later
"""A Hugging Face pull is checked against the network policy: the hub endpoint
before any metadata or download call, and every redirect hop of the size
check before the download, over the pinned transport. The hub's metadata and
download calls are replaced by recorders; the policy and the pull are the real
code."""

from __future__ import annotations

from pathlib import Path

import pytest

from localm import model_manager as mm
from localm import netpolicy
from localm.model_manager import pull as pull_mod
from tests._net_config import set_config
from tests._web_retrieval_fixtures import FakeResponse, Transport

SPEC = "owner/repo:model.gguf"


class _Stop(Exception):
    pass


@pytest.fixture()
def hub(tmp_path, monkeypatch):
    """Config ``net_mode=ask`` (tests add keys to the returned dict), public
    DNS, and recorders for every hub call the pull makes. The digest lookup
    records and raises ``_Stop`` unless ``state['digests']`` is set."""
    cfg = {"net_mode": "ask"}
    monkeypatch.delenv("LOCALM_NET_MODE", raising=False)
    set_config(monkeypatch, lambda: cfg)
    monkeypatch.setattr("socket.getaddrinfo",
                        lambda *a, **k: [(2, 1, 6, "", ("93.184.216.34", 0))])
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    monkeypatch.setattr(mm, "MODELS_DIR", models_dir)
    monkeypatch.setattr(mm, "ensure_dirs", lambda: None)
    monkeypatch.setattr(mm, "_check_disk_space", lambda *a, **k: True)
    monkeypatch.setattr(mm, "_hf_repo_mmproj_filename", lambda *a, **k: None)
    calls = {"sha": [], "download": [], "pipeline": []}
    state = {"cfg": cfg, "calls": calls, "stop_at_sha": True}

    def fake_sha(repo, filename, token=None):
        calls["sha"].append((repo, filename))
        if state["stop_at_sha"]:
            raise _Stop()
        return None

    def fake_download(repo_id, filename, local_dir, **kw):
        calls["download"].append(filename)
        p = Path(local_dir) / filename
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"GGUF")
        return str(p)

    import huggingface_hub
    monkeypatch.setattr(mm, "_hf_file_sha256", fake_sha)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    monkeypatch.setattr(pull_mod, "_hf_pipeline_tag_to_type",
                        lambda repo: calls["pipeline"].append(repo) or "llm")
    return state


class TestEndpointGate:
    def test_deny_listed_hub_refuses_before_any_hub_call(self, hub, capsys):
        hub["cfg"]["net_deny"] = ["huggingface.co"]
        assert mm.pull_model(SPEC) is False
        assert hub["calls"] == {"sha": [], "download": [], "pipeline": []}
        assert "Network policy refused the Hugging Face pull" in capsys.readouterr().out

    def test_hub_missing_from_allow_list_refuses(self, hub):
        hub["cfg"]["net_allow"] = ["docs.python.org"]
        assert mm.pull_model("owner/repo") is False
        assert hub["calls"]["pipeline"] == [] and hub["calls"]["sha"] == []

    def test_hub_resolving_to_a_private_address_refuses(self, hub, monkeypatch):
        monkeypatch.setattr("socket.getaddrinfo",
                            lambda *a, **k: [(2, 1, 6, "", ("10.0.0.5", 0))])
        assert mm.pull_model(SPEC) is False
        assert hub["calls"]["sha"] == []

    def test_allowed_hub_reaches_the_metadata_call(self, hub):
        with pytest.raises(_Stop):
            mm.pull_model(SPEC)
        assert hub["calls"]["sha"] == [("owner/repo", "model.gguf")]

    def test_downloads_exemption_still_needs_the_lists(self, hub):
        hub["cfg"].update(net_mode="off", net_allow_model_downloads=True,
                          net_deny=["hf.co", "huggingface.co"])
        assert mm.pull_model(SPEC) is False
        assert hub["calls"]["sha"] == []

    def test_downloads_exemption_lets_an_allowed_hub_through(self, hub):
        hub["cfg"].update(net_mode="off", net_allow_model_downloads=True)
        with pytest.raises(_Stop):
            mm.pull_model(SPEC)
        assert hub["calls"]["sha"] == [("owner/repo", "model.gguf")]

    def test_backfill_skips_a_deny_listed_hub(self, hub, monkeypatch, tmp_path):
        hub["cfg"]["net_deny"] = ["huggingface.co"]
        fetched = []
        monkeypatch.setattr(pull_mod, "mmproj_backfill_candidate", lambda e, p: True)
        monkeypatch.setattr(pull_mod, "_maybe_fetch_repo_mmproj",
                            lambda *a, **k: fetched.append(a))
        entry = {"source": "hf:owner/repo", "type": "llm"}
        assert pull_mod.backfill_mmproj_for_entry(entry, tmp_path / "m.gguf") is None
        assert fetched == []
        hub["cfg"]["net_deny"] = []
        pull_mod.backfill_mmproj_for_entry(entry, tmp_path / "m.gguf")
        assert len(fetched) == 1


HUB_FILE = "https://huggingface.co/owner/repo/resolve/main/model.gguf"
CDN_FILE = "https://cas-bridge.xethub.hf.co/xet/abc?sig=1"


class TestPolicyHead:
    @pytest.fixture()
    def net(self, monkeypatch):
        cfg = {"net_mode": "ask"}
        monkeypatch.delenv("LOCALM_NET_MODE", raising=False)
        set_config(monkeypatch, lambda: cfg)
        monkeypatch.setattr("socket.getaddrinfo",
                            lambda *a, **k: [(2, 1, 6, "", ("93.184.216.34", 0))])
        transport = Transport().install(monkeypatch)
        return cfg, transport

    def test_follows_redirects_and_keeps_the_first_hop_etag(self, net):
        cfg, transport = net
        transport.route("HEAD", HUB_FILE, FakeResponse(
            status=302, redirect=CDN_FILE,
            headers={"X-Linked-Etag": '"abc123"', "X-Linked-Size": "4"}))
        transport.route("HEAD", CDN_FILE, FakeResponse(
            headers={"Content-Length": "4"}))
        resp = pull_mod._policy_head(HUB_FILE)
        assert resp.headers["Content-Length"] == "4"
        assert pull_mod._etag_from_head(resp) == "abc123"
        assert transport.urls() == [HUB_FILE, CDN_FILE]
        assert all(kw["allow_redirects"] is False for _, _, kw in transport.calls)
        assert [kw["headers"]["Host"] for _, _, kw in transport.calls] == [
            "huggingface.co", "cas-bridge.xethub.hf.co"]

    def test_redirect_to_a_private_address_is_refused(self, net):
        cfg, transport = net
        transport.route("HEAD", HUB_FILE, FakeResponse(
            status=302, redirect="http://169.254.169.254/latest/meta-data/"))
        with pytest.raises(netpolicy.NetworkPolicyError, match="non-public"):
            pull_mod._policy_head(HUB_FILE)
        assert transport.urls() == [HUB_FILE]

    def test_redirect_to_a_deny_listed_host_is_refused(self, net):
        cfg, transport = net
        cfg["net_deny"] = ["xethub.hf.co"]
        transport.route("HEAD", HUB_FILE, FakeResponse(status=302, redirect=CDN_FILE))
        with pytest.raises(netpolicy.NetworkPolicyError, match="deny list"):
            pull_mod._policy_head(HUB_FILE)
        assert transport.urls() == [HUB_FILE]

    def test_a_refused_hop_stops_the_pull_before_the_download(self, hub, monkeypatch, capsys):
        hub["stop_at_sha"] = False

        def refuse(url):
            raise netpolicy.NetworkPolicyError("'cdn.example' is on the deny list (net_deny). ")
        monkeypatch.setattr(pull_mod, "_policy_head", refuse)
        assert mm.pull_model(SPEC) is False
        assert hub["calls"]["download"] == []
        assert "Network policy refused the download" in capsys.readouterr().out

    def test_an_allowed_preflight_downloads(self, hub, monkeypatch):
        hub["stop_at_sha"] = False
        seen, downloads = [], []

        def head(url):
            seen.append(url)
            return FakeResponse(headers={"content-length": "4"})

        def download(repo_id, filename, local_dir, **kw):
            downloads.append(filename)
            raise RuntimeError("stop after the download call")
        import huggingface_hub
        monkeypatch.setattr(pull_mod, "_policy_head", head)
        monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
        assert mm.pull_model(SPEC) is False
        assert downloads == ["model.gguf"]
        assert len(seen) == 1
        assert seen[0].startswith("https://huggingface.co/owner/repo/resolve/")

# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for localm.model_manager.sources - the HFSource/CivitAISource seam.
All CivitAI/HF calls are mocked; no real network."""

import json as _json
import urllib.parse

import pytest

from localm.model_manager import sources


@pytest.fixture(autouse=True)
def _online(monkeypatch):
    monkeypatch.setenv("LOCALM_NET_MODE", "ask")
    # HFSource/CivitAISource resolve a token from model_source_credentials
    # when the caller does not pass one explicitly, which falls back to these
    # env vars - strip them so a real HF_TOKEN on the machine running the
    # tests can never leak into an assertion here.
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("CIVITAI_API_KEY", raising=False)


def _mock_civitai(monkeypatch, by_path: dict, seen: "list | None" = None):
    """Patch netpolicy.safe_fetch_bytes to answer with *by_path[path]* for a
    request whose URL path matches (query string ignored). *seen*, when
    given, collects each request's path in call order."""

    def fake(url, **kw):
        parsed = urllib.parse.urlparse(url)
        if seen is not None:
            seen.append(parsed.path)
        if parsed.path not in by_path:
            raise AssertionError(f"unexpected CivitAI request: {url}")
        return url, "application/json", _json.dumps(by_path[parsed.path]).encode("utf-8")

    monkeypatch.setattr("localm.netpolicy.safe_fetch_bytes", fake)


def _model_item(**over):
    base = {
        "id": 122359, "name": "Detail Tweaker XL", "type": "LORA",
        "minor": False, "nsfw": False, "poi": False,
        "stats": {"downloadCount": 448464},
        "modelVersions": [{"id": 135867}],
    }
    base.update(over)
    return base


def _version_detail(**over):
    base = {
        "id": 135867, "modelId": 122359,
        "model": {"name": "Detail Tweaker XL", "type": "LORA", "nsfw": False},
        "files": [{
            "id": 99264, "sizeKB": 223097.99, "name": "add-detail-xl.safetensors",
            "primary": True, "metadata": {"format": "SafeTensor"},
            "hashes": {"SHA256": "0D9BD1B873A7863E128B4672E3E245838858F71469A3CEC58123C16C06F83BD7"},
            "downloadUrl": "https://civitai.com/api/download/models/135867?fileId=99264",
        }],
    }
    base.update(over)
    return base


# ------------------------------------------------------------------ #
#  civitai_search                                                     #
# ------------------------------------------------------------------ #

class TestCivitaiSearch:
    def test_defaults_to_nsfw_false(self, monkeypatch):
        seen_params = {}

        def fake(url, **kw):
            parsed = urllib.parse.urlparse(url)
            seen_params.update(urllib.parse.parse_qsl(parsed.query))
            return url, "application/json", b'{"items": [], "metadata": {}}'

        monkeypatch.setattr("localm.netpolicy.safe_fetch_bytes", fake)
        sources.civitai_search("detail")
        assert seen_params["nsfw"] == "false"

    def test_excludes_unmapped_types_even_if_returned(self, monkeypatch):
        _mock_civitai(monkeypatch, {
            "/api/v1/models": {
                "items": [_model_item(), _model_item(id=1, type="Poses", name="a pose set")],
                "metadata": {},
            }})
        result = sources.civitai_search("x")
        assert [it["id"] for it in result["items"]] == [122359]

    def test_hard_excludes_minor_flagged_content_regardless_of_nsfw(self, monkeypatch):
        _mock_civitai(monkeypatch, {
            "/api/v1/models": {
                "items": [_model_item(minor=True)],
                "metadata": {},
            }})
        result = sources.civitai_search("x", nsfw=True)
        assert result["items"] == []

    def test_next_cursor_from_metadata(self, monkeypatch):
        _mock_civitai(monkeypatch, {
            "/api/v1/models": {"items": [], "metadata": {"nextCursor": "abc123"}},
        })
        result = sources.civitai_search("x")
        assert result["next_cursor"] == "abc123"

    def test_types_narrowed_to_known_map_raises_when_none_valid(self, monkeypatch):
        with pytest.raises(sources.ModelSourceError):
            sources.civitai_search("x", types=["Poses", "Wildcards"])


# ------------------------------------------------------------------ #
#  civitai_list_files                                                 #
# ------------------------------------------------------------------ #

class TestCivitaiListFiles:
    def _files_payload(self):
        return {
            "/api/v1/model-versions/135867": {
                "id": 135867, "modelId": 122359,
                "files": [
                    {"id": 1, "name": "safe.safetensors",
                     "metadata": {"format": "SafeTensor"}},
                    {"id": 2, "name": "legacy.ckpt",
                     "metadata": {"format": "PickleTensor"}},
                    {"id": 3, "name": "no-format-field.bin", "metadata": {}},
                    {"id": 4, "name": "quantized.gguf",
                     "metadata": {"format": "GGUF"}},
                ],
            }
        }

    def test_default_excludes_legacy_and_unknown_formats(self, monkeypatch):
        _mock_civitai(monkeypatch, self._files_payload())
        files = sources.civitai_list_files(135867)
        assert {f["id"] for f in files} == {1, 4}

    def test_include_legacy_formats_returns_everything(self, monkeypatch):
        _mock_civitai(monkeypatch, self._files_payload())
        files = sources.civitai_list_files(135867, include_legacy_formats=True)
        assert {f["id"] for f in files} == {1, 2, 3, 4}


# ------------------------------------------------------------------ #
#  CivitAISource.resolve_download                                     #
# ------------------------------------------------------------------ #

class TestResolveDownload:
    def _wire(self, monkeypatch, version=None, model=None):
        _mock_civitai(monkeypatch, {
            "/api/v1/model-versions/135867": version or _version_detail(),
            "/api/v1/models/122359": model or _model_item(),
        })

    @pytest.mark.parametrize("civitai_type,expected_model_type,expected_subfolder", [
        ("Checkpoint", "diffusion-unet", "checkpoints"),
        ("LORA", "lora", "loras"),
        ("LoCon", "lora", "loras"),
        ("DoRA", "lora", "loras"),
        ("TextualInversion", "embedding", "embeddings"),
        ("VAE", "vae", "vae"),
        ("Controlnet", "unknown", "controlnet"),
        ("Upscaler", "unknown", "upscale_models"),
    ])
    def test_type_map_placement(self, monkeypatch, civitai_type, expected_model_type,
                                 expected_subfolder):
        version = _version_detail()
        version["model"]["type"] = civitai_type
        self._wire(monkeypatch, version=version)
        resolved = sources.CivitAISource().resolve_download(135867, None)
        assert resolved.model_type == expected_model_type
        assert resolved.comfy_subfolder == expected_subfolder
        assert resolved.source_tag == "civitai:135867"
        assert resolved.url == "https://civitai.com/api/download/models/135867?fileId=99264"
        assert resolved.filename == "add-detail-xl.safetensors"
        assert resolved.size_bytes == int(223097.99 * 1024)
        assert resolved.sha256 == "0d9bd1b873a7863e128b4672e3e245838858f71469a3cec58123c16c06f83bd7"

    def test_refuses_an_unmapped_type(self, monkeypatch):
        version = _version_detail()
        version["model"]["type"] = "Poses"
        self._wire(monkeypatch, version=version)
        with pytest.raises(sources.ModelSourceError, match="no known ComfyUI placement"):
            sources.CivitAISource().resolve_download(135867, None)

    def test_refuses_a_minor_flagged_model(self, monkeypatch):
        self._wire(monkeypatch, model=_model_item(minor=True))
        with pytest.raises(sources.ModelSourceError, match="minor"):
            sources.CivitAISource().resolve_download(135867, None)

    def test_auto_pick_skips_a_legacy_primary_for_a_safe_sibling(self, monkeypatch):
        version = _version_detail()
        version["files"] = [
            {"id": 1, "name": "primary-legacy.ckpt", "primary": True,
             "metadata": {"format": "PickleTensor"},
             "hashes": {}, "downloadUrl": "https://civitai.com/api/download/models/135867?fileId=1"},
            {"id": 2, "name": "sibling-safe.safetensors", "primary": False,
             "metadata": {"format": "SafeTensor"},
             "hashes": {}, "downloadUrl": "https://civitai.com/api/download/models/135867?fileId=2"},
        ]
        self._wire(monkeypatch, version=version)
        resolved = sources.CivitAISource().resolve_download(135867, None)
        assert resolved.filename == "sibling-safe.safetensors"

    def test_explicit_file_id_on_legacy_format_refused_without_the_flag(self, monkeypatch):
        version = _version_detail()
        version["files"] = [{
            "id": 1, "name": "legacy.ckpt", "primary": True,
            "metadata": {"format": "PickleTensor"}, "hashes": {},
            "downloadUrl": "https://civitai.com/api/download/models/135867?fileId=1",
        }]
        self._wire(monkeypatch, version=version)
        with pytest.raises(sources.ModelSourceError, match="legacy"):
            sources.CivitAISource().resolve_download(135867, 1)

    def test_explicit_file_id_on_legacy_format_allowed_with_the_flag(self, monkeypatch):
        version = _version_detail()
        version["files"] = [{
            "id": 1, "name": "legacy.ckpt", "primary": True,
            "metadata": {"format": "PickleTensor"}, "hashes": {},
            "downloadUrl": "https://civitai.com/api/download/models/135867?fileId=1",
        }]
        self._wire(monkeypatch, version=version)
        resolved = sources.CivitAISource().resolve_download(
            135867, 1, include_legacy_formats=True)
        assert resolved.filename == "legacy.ckpt"

    def test_never_follows_the_download_redirect_itself(self, monkeypatch):
        """resolve_download must only ever call safe_fetch_bytes against the
        CivitAI metadata API, never pinned_request/HEAD against the download
        endpoint - the actual redirect is time-boxed and resolved once, at
        download time, in pull.py."""
        calls = []
        self._wire(monkeypatch)

        def _forbidden(*a, **kw):
            calls.append((a, kw))
            raise AssertionError("resolve_download must not follow the download redirect")

        monkeypatch.setattr("localm.netpolicy.pinned_request", _forbidden)
        sources.CivitAISource().resolve_download(135867, None)
        assert calls == []


# ------------------------------------------------------------------ #
#  HFSource - delegates to the existing discover.py machinery          #
# ------------------------------------------------------------------ #

class TestHFSource:
    def test_search_delegates_to_discover_hf_search(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            "localm.discover.hf_search",
            lambda query, limit=20, **kw: calls.append((query, limit, kw)) or [{"id": "x/y"}])
        result = sources.HFSource().search("qwen", limit=5)
        # No HF_TOKEN configured (see the _online fixture): token resolves to
        # None, so this remains the plain anonymous call it always was.
        assert calls == [("qwen", 5, {"token": None})]
        assert result == {"items": [{"id": "x/y"}]}

    def test_list_files_delegates_to_discover_hf_gguf_files(self, monkeypatch):
        monkeypatch.setattr(
            "localm.discover.hf_gguf_files",
            lambda repo, token=None: [{"file": "a.gguf"}] if repo == "org/repo" else [])
        assert sources.HFSource().list_files("org/repo") == [{"file": "a.gguf"}]

    def test_resolve_download_is_descriptive_only(self, monkeypatch):
        monkeypatch.setattr(
            "huggingface_hub.hf_hub_url",
            lambda repo_id, filename, endpoint=None: f"{endpoint}/{repo_id}/resolve/main/{filename}")
        monkeypatch.setattr(
            "localm.model_manager.pull._hf_file_sha256",
            lambda repo_id, filename, token=None: "abc123")
        resolved = sources.HFSource().resolve_download(
            "org/repo", {"file": "model.gguf", "size_bytes": 42}, model_type="llm")
        assert resolved.source_tag == "hf:org/repo"
        assert resolved.comfy_subfolder is None
        assert resolved.filename == "model.gguf"
        assert resolved.size_bytes == 42
        assert resolved.sha256 == "abc123"

    def test_search_uses_the_configured_hf_token(self, monkeypatch):
        """The credential store (ADR-0015) is the default source of the token
        when a caller does not pass one explicitly."""
        from localm.model_source_credentials import set_credentials
        set_credentials({"hf_token": "hf_configured_token"})
        calls = []
        monkeypatch.setattr(
            "localm.discover.hf_search",
            lambda query, limit=20, **kw: calls.append(kw) or [])
        sources.HFSource().search("qwen")
        assert calls == [{"token": "hf_configured_token"}]

    def test_search_explicit_token_overrides_the_stored_one(self, monkeypatch):
        from localm.model_source_credentials import set_credentials
        set_credentials({"hf_token": "hf_stored"})
        calls = []
        monkeypatch.setattr(
            "localm.discover.hf_search",
            lambda query, limit=20, **kw: calls.append(kw) or [])
        sources.HFSource().search("qwen", token="hf_explicit")
        assert calls == [{"token": "hf_explicit"}]

    def test_list_files_and_resolve_download_use_the_configured_token(self, monkeypatch):
        from localm.model_source_credentials import set_credentials
        set_credentials({"hf_token": "hf_configured_token"})
        seen = {}
        monkeypatch.setattr(
            "localm.discover.hf_gguf_files",
            lambda repo, token=None: seen.setdefault("list_files_token", token) or [])
        sources.HFSource().list_files("org/repo")
        assert seen["list_files_token"] == "hf_configured_token"

        monkeypatch.setattr(
            "huggingface_hub.hf_hub_url",
            lambda repo_id, filename, endpoint=None: "https://example/resolved")
        monkeypatch.setattr(
            "localm.model_manager.pull._hf_file_sha256",
            lambda repo_id, filename, token=None: seen.setdefault("resolve_token", token))
        sources.HFSource().resolve_download("org/repo", {"file": "m.gguf"})
        assert seen["resolve_token"] == "hf_configured_token"


# ------------------------------------------------------------------ #
#  A configured token/key actually reaches the real HTTP layer         #
# ------------------------------------------------------------------ #

class TestAuthHeaderReachesTheRealFetch:
    """Everything above mocks discover.py/pull.py's own functions. These prove
    the header survives all the way to netpolicy.safe_fetch_bytes - the actual
    call that would leave this machine - for both providers, present when
    configured and absent when not."""

    def test_civitai_search_sends_no_auth_header_when_unconfigured(self, monkeypatch):
        captured = {}

        def fake(url, **kw):
            captured.update(kw)
            return url, "application/json", b'{"items": [], "metadata": {}}'

        monkeypatch.setattr("localm.netpolicy.safe_fetch_bytes", fake)
        sources.CivitAISource().search("x")
        assert captured.get("extra_headers") is None

    def test_civitai_search_sends_the_configured_key_as_a_bearer_header(self, monkeypatch):
        from localm.model_source_credentials import set_credentials
        set_credentials({"civitai_api_key": "civ_configured_key"})
        captured = {}

        def fake(url, **kw):
            captured.update(kw)
            return url, "application/json", b'{"items": [], "metadata": {}}'

        monkeypatch.setattr("localm.netpolicy.safe_fetch_bytes", fake)
        sources.CivitAISource().search("x")
        assert captured.get("extra_headers") == {"Authorization": "Bearer civ_configured_key"}

    def test_civitai_resolve_download_sends_the_key_on_both_lookups(self, monkeypatch):
        from localm.model_source_credentials import set_credentials
        set_credentials({"civitai_api_key": "civ_configured_key"})
        calls = []

        def fake(url, **kw):
            calls.append(kw.get("extra_headers"))
            parsed = urllib.parse.urlparse(url)
            payload = ({"id": 135867, "modelId": 122359,
                        "model": {"type": "LORA"}, "files": [{
                            "id": 99264, "primary": True,
                            "metadata": {"format": "SafeTensor"},
                            "downloadUrl": "https://civitai.com/dl/99264",
                        }]}
                       if "model-versions" in parsed.path else _model_item())
            return url, "application/json", _json.dumps(payload).encode("utf-8")

        monkeypatch.setattr("localm.netpolicy.safe_fetch_bytes", fake)
        sources.CivitAISource().resolve_download(135867, None)
        assert len(calls) == 2, "both the version and the model lookup must run"
        assert all(c == {"Authorization": "Bearer civ_configured_key"} for c in calls)

    def test_hf_search_sends_no_auth_header_when_unconfigured(self, monkeypatch):
        captured = {}

        def fake(url, **kw):
            captured.update(kw)
            return url, "application/json", b"[]"

        monkeypatch.setattr("localm.netpolicy.safe_fetch_bytes", fake)
        sources.HFSource().search("qwen")
        assert captured.get("extra_headers") is None

    def test_hf_search_sends_the_configured_token_as_a_bearer_header(self, monkeypatch):
        from localm.model_source_credentials import set_credentials
        set_credentials({"hf_token": "hf_configured_token"})
        captured = {}

        def fake(url, **kw):
            captured.update(kw)
            return url, "application/json", b"[]"

        monkeypatch.setattr("localm.netpolicy.safe_fetch_bytes", fake)
        sources.HFSource().search("qwen")
        assert captured.get("extra_headers") == {"Authorization": "Bearer hf_configured_token"}


def test_get_source_returns_the_registered_sources():
    assert isinstance(sources.get_source("hf"), sources.HFSource)
    assert isinstance(sources.get_source("civitai"), sources.CivitAISource)
    with pytest.raises(sources.ModelSourceError):
        sources.get_source("nope")

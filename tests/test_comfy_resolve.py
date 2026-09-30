# SPDX-License-Identifier: AGPL-3.0-or-later
"""comfy_resolve: where localm can download a missing ComfyUI workflow model
file from. The curated table answers first; otherwise HuggingFace is searched
for a repository file with exactly that name. Every HuggingFace request goes
through discover._get, which these tests replace with a recorder serving
realistic /api/models payloads."""

import pytest

from localm import discover
from localm.model_manager import comfy_resolve as cr


def _repo(repo_id, files, downloads=0, **flags):
    return {"id": repo_id, "downloads": downloads,
            "siblings": [{"rfilename": f} for f in files], **flags}


class FakeHF:
    """Records every discover._get call and answers from its tables."""

    def __init__(self, org=(), searches=None, trees=None, error=None):
        self.org = list(org)
        self.searches = searches or {}
        self.trees = trees or {}
        self.error = error
        self.calls = []

    def __call__(self, url, params=None, *, token=None):
        self.calls.append((url, dict(params or {})))
        if self.error is not None:
            raise self.error
        if url.endswith("/api/models"):
            if (params or {}).get("author") == cr.COMFY_ORG:
                return self.org
            return self.searches.get(params["search"], [])
        return self.trees.get(url, [])

    @property
    def searched(self):
        return [p["search"] for _u, p in self.calls if "search" in p]


@pytest.fixture
def hf(monkeypatch):
    cr.clear_lookup_cache()
    fake = FakeHF()
    monkeypatch.setattr(discover, "_get", fake)
    monkeypatch.setattr(discover, "_ensure_online", lambda: None)
    import localm.model_source_credentials as creds
    monkeypatch.setattr(creds, "get_hf_token", lambda: None)
    yield fake
    cr.clear_lookup_cache()


def _tree_url(repo, folder=""):
    url = f"{discover.HF_API}/api/models/{repo}/tree/main"
    return url + ("/" + folder if folder else "")


class TestSlotFolders:
    @pytest.mark.parametrize("class_type,input_name,folder", [
        ("CheckpointLoaderSimple", "ckpt_name", "checkpoints"),
        ("UNETLoader", "unet_name", "unet"),
        ("UnetLoaderGGUF", "unet_name", "unet"),
        ("CLIPLoader", "clip_name", "clip"),
        ("DualCLIPLoader", "clip_name2", "clip"),
        ("CLIPVisionLoader", "clip_name", "clip_vision"),
        ("VAELoader", "vae_name", "vae"),
        ("LoraLoader", "lora_name", "loras"),
        ("ControlNetLoader", "control_net_name", "controlnet"),
        ("UpscaleModelLoader", "model_name", "upscale_models"),
        ("StyleModelLoader", "style_model_name", "style_models"),
        ("SomeCustomNode", "model_name", None),
        ("SomeCustomNode", "weird_input", None),
    ])
    def test_slot_to_folder(self, class_type, input_name, folder):
        assert cr.comfy_slot_folder(class_type, input_name) == folder

    def test_every_folder_is_one_the_managed_comfyui_reads(self):
        from localm.media.managed_comfy import _MODEL_FOLDER_TYPES
        folders = {cr.comfy_slot_folder(c, i) for c, i in [
            ("CheckpointLoaderSimple", "ckpt_name"), ("UNETLoader", "unet_name"),
            ("CLIPLoader", "clip_name"), ("CLIPVisionLoader", "clip_name"),
            ("VAELoader", "vae_name"), ("LoraLoader", "lora_name"),
            ("ControlNetLoader", "control_net_name"),
            ("UpscaleModelLoader", "model_name"), ("StyleModelLoader", "style_model_name"),
        ]}
        assert folders <= set(_MODEL_FOLDER_TYPES)


class TestNames:
    @pytest.mark.parametrize("name", [
        "model.safetensors", "diffusion_pytorch_model.safetensors",
        "pytorch_model.safetensors", "vae.safetensors", "ae.gguf", "1234567.safetensors",
    ])
    def test_generic_names_are_not_specific(self, name):
        assert cr.is_specific_model_name(name) is False

    @pytest.mark.parametrize("name", [
        "ace_step_v1_3.5b.safetensors", "4x-UltraSharp.safetensors", "flux1-schnell-Q4_K_S.gguf",
    ])
    def test_model_names_are_specific(self, name):
        assert cr.is_specific_model_name(name) is True

    @pytest.mark.parametrize("name", [
        "wan2.1_t2v_1.3B_fp16.safetensors", "4x-UltraSharp.safetensors",
        "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "a_b_c_d_e_f_g_h_i_j.safetensors",
        "flux1-schnell-Q4_K_S.gguf",
    ])
    def test_queries_start_with_the_stem_and_stay_bounded(self, name):
        queries = cr._search_queries(name)
        assert queries[0] == cr._stem(name)
        assert len(queries) <= cr.MAX_HF_REQUESTS - 2
        assert len({q.lower() for q in queries}) == len(queries)


class TestRefusedWithoutNetwork:
    @pytest.mark.parametrize("filename,class_type,input_name,reason", [
        ("RealESRGAN_x4plus.pth", "UpscaleModelLoader", "model_name", cr.REASON_FORMAT),
        ("custom_thing_v2.ckpt", "CheckpointLoaderSimple", "ckpt_name", cr.REASON_FORMAT),
        ("custom_thing_v2.safetensors", "SomeCustomNode", "weird_input", cr.REASON_FOLDER),
        ("model.safetensors", "CheckpointLoaderSimple", "ckpt_name", cr.REASON_NAME),
        ("sub/dir.safetensors", "CheckpointLoaderSimple", "ckpt_name", cr.REASON_NAME),
        ("..\\x_model_v2.safetensors", "CheckpointLoaderSimple", "ckpt_name", cr.REASON_NAME),
    ])
    def test_unsupported(self, hf, filename, class_type, input_name, reason):
        lookup = cr.lookup_comfy_download(filename, class_type, input_name)
        assert hf.calls == [], "an unsupported request reached HuggingFace"
        assert (lookup.status, lookup.reason) == (cr.LOOKUP_UNSUPPORTED, reason)
        assert lookup.download is None
        assert cr.search_refusal(filename, class_type, input_name).reason == reason

    def test_the_curated_table_answers_without_network(self, hf):
        lookup = cr.lookup_comfy_download(
            "ace_step_v1_3.5b.safetensors", "CheckpointLoaderSimple", "ckpt_name")
        assert hf.calls == []
        assert lookup.status == cr.LOOKUP_FOUND
        assert lookup.download.origin == cr.ORIGIN_CURATED
        assert lookup.download.path == "all_in_one/ace_step_v1_3.5b.safetensors"
        assert lookup.download.comfy_subfolder == "checkpoints"


WAN = "wan2.1_t2v_1.3B_fp16.safetensors"


class TestHuggingFaceSearch:
    def test_a_comfy_org_file_wins_over_a_more_downloaded_mirror(self, hf):
        hf.org = [_repo("Comfy-Org/Wan_2.1_ComfyUI_repackaged",
                        [f"split_files/diffusion_models/{WAN}"], downloads=10)]
        hf.searches = {cr._stem(WAN): [_repo("someone/mirror", [WAN], downloads=10**9)]}
        hf.trees = {_tree_url("Comfy-Org/Wan_2.1_ComfyUI_repackaged",
                              "split_files/diffusion_models"): [
            {"path": f"split_files/diffusion_models/{WAN}", "size": 99,
             "lfs": {"size": 2838303560}}]}

        lookup = cr.lookup_comfy_download(WAN, "UNETLoader", "unet_name")

        assert hf.searched == [], "a Comfy-Org match needs no search"
        assert lookup.status == cr.LOOKUP_FOUND
        d = lookup.download
        assert (d.repo, d.path) == ("Comfy-Org/Wan_2.1_ComfyUI_repackaged",
                                    f"split_files/diffusion_models/{WAN}")
        assert (d.comfy_subfolder, d.model_type, d.origin) == (
            "unet", "diffusion-unet", cr.ORIGIN_HUGGINGFACE)
        assert d.size_bytes == 2838303560

    def test_without_a_comfy_org_file_the_most_downloaded_repo_wins(self, hf):
        hf.org = [_repo("Comfy-Org/other", ["unrelated.safetensors"])]
        stem = cr._stem(WAN)
        hf.searches = {
            stem: [_repo("small/copy", [WAN], downloads=5),
                   _repo("big/official", [f"weights/{WAN}"], downloads=5000)],
            "wan2.1": [_repo("gated/one", [WAN], downloads=10**6, gated="auto"),
                       _repo("private/one", [WAN], downloads=10**6, private=True),
                       _repo("disabled/one", [WAN], downloads=10**6, disabled=True)],
        }
        lookup = cr.lookup_comfy_download(WAN, "UNETLoader", "unet_name")
        assert (lookup.download.repo, lookup.download.path) == ("big/official", f"weights/{WAN}")
        assert lookup.download.size_bytes is None, "no tree entry means an unknown size"

    def test_only_an_exact_file_name_matches(self, hf):
        hf.searches = {cr._stem(WAN): [
            _repo("a/b", [WAN.upper(), "x" + WAN, WAN + ".bak", f"dir/{WAN}.part"],
                  downloads=100)]}
        lookup = cr.lookup_comfy_download(WAN, "UNETLoader", "unet_name")
        assert lookup.status == cr.LOOKUP_NOT_FOUND

    @pytest.mark.parametrize("path", [
        f"../{WAN}", f"a/../{WAN}", f"./{WAN}", f"a//{WAN}", f".hidden/{WAN}",
    ])
    def test_a_repo_path_that_could_escape_is_skipped(self, hf, path):
        hf.searches = {cr._stem(WAN): [_repo("a/b", [path], downloads=100)]}
        lookup = cr.lookup_comfy_download(WAN, "UNETLoader", "unet_name")
        assert lookup.status == cr.LOOKUP_NOT_FOUND

    def test_an_odd_repo_id_is_skipped(self, hf):
        hf.searches = {cr._stem(WAN): [_repo("../evil", [WAN], downloads=100),
                                        _repo("a/b/c", [WAN], downloads=100),
                                        _repo("a..b/c", [WAN], downloads=100),
                                        _repo("-a/b", [WAN], downloads=100)]}
        lookup = cr.lookup_comfy_download(WAN, "UNETLoader", "unet_name")
        assert lookup.status == cr.LOOKUP_NOT_FOUND


class TestOutcomesAndCache:
    def test_not_found_is_cached(self, hf):
        first = cr.lookup_comfy_download(WAN, "UNETLoader", "unet_name")
        n = len(hf.calls)
        second = cr.lookup_comfy_download(WAN, "UNETLoader", "unet_name")
        assert first.status == second.status == cr.LOOKUP_NOT_FOUND
        assert n > 0 and len(hf.calls) == n, "a cached miss searched again"

    def test_a_found_file_is_cached_and_offered_without_network(self, hf):
        hf.org = [_repo("Comfy-Org/x", [WAN])]
        assert cr.cached_comfy_download(WAN, "UNETLoader", "unet_name") is None
        found = cr.lookup_comfy_download(WAN, "UNETLoader", "unet_name").download
        n = len(hf.calls)
        assert cr.cached_comfy_download(WAN, "UNETLoader", "unet_name") == found
        assert cr.cached_comfy_download(WAN, "CheckpointLoaderSimple", "ckpt_name") is None
        assert len(hf.calls) == n

    def test_the_comfy_org_listing_is_fetched_once_for_many_lookups(self, hf):
        cr.lookup_comfy_download(WAN, "UNETLoader", "unet_name")
        cr.lookup_comfy_download("other_model_v2.safetensors", "VAELoader", "vae_name")
        org_calls = [p for _u, p in hf.calls if p.get("author") == cr.COMFY_ORG]
        assert len(org_calls) == 1

    def test_network_off_is_offline_and_not_cached(self, hf, monkeypatch):
        def _off():
            raise discover.DiscoverError("Network access is disabled", off=True)
        monkeypatch.setattr(discover, "_ensure_online", _off)
        lookup = cr.lookup_comfy_download(WAN, "UNETLoader", "unet_name")
        assert (lookup.status, lookup.download) == (cr.LOOKUP_OFFLINE, None)
        monkeypatch.setattr(discover, "_ensure_online", lambda: None)
        hf.org = [_repo("Comfy-Org/x", [WAN])]
        assert cr.lookup_comfy_download(WAN, "UNETLoader", "unet_name").status == cr.LOOKUP_FOUND

    def test_a_failed_request_is_failed_not_not_found(self, hf):
        hf.error = discover.DiscoverError("HuggingFace request failed: timed out")
        lookup = cr.lookup_comfy_download(WAN, "UNETLoader", "unet_name")
        assert lookup.status == cr.LOOKUP_FAILED
        assert "timed out" in lookup.detail


def test_lookup_timeout_covers_every_request_timing_out():
    from localm.plugins.gui.routes.models import acquisition
    assert acquisition._LOOKUP_TIMEOUT > cr.MAX_HF_REQUESTS * discover._TIMEOUT

# SPDX-License-Identifier: AGPL-3.0-or-later
"""Role-based ComfyUI node resolution.

Injection resolves nodes by class_type plus graph edges, never by hardcoded ids,
so a user's own exported graph with arbitrary node ids is driven correctly. The
end-to-end test runs generate_video against a fully RENUMBERED Wan graph and
checks the prompt, seed, latent and fps still land in the right places.
"""

import json
from unittest.mock import MagicMock, patch

from localm.image_gen import comfy as shared
from localm.media import comfy_client


# A Wan-shaped graph with non-template ids (a mix of numeric and named).
RENUMBERED_WAN = {
    "100": {"inputs": {"unet_name": "wan2.2_ti2v_5B_fp16.safetensors",
                       "weight_dtype": "default"}, "class_type": "UNETLoader"},
    "200": {"inputs": {"clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
                       "type": "wan", "device": "default"}, "class_type": "CLIPLoader"},
    "300": {"inputs": {"vae_name": "wan2.2_vae.safetensors"}, "class_type": "VAELoader"},
    "pos": {"inputs": {"text": "", "clip": ["200", 0]}, "class_type": "CLIPTextEncode"},
    "neg": {"inputs": {"text": "template-neg", "clip": ["200", 0]},
            "class_type": "CLIPTextEncode"},
    "lat": {"inputs": {"width": 1280, "height": 704, "length": 121, "batch_size": 1,
                       "vae": ["300", 0]}, "class_type": "Wan22ImageToVideoLatent"},
    "msd": {"inputs": {"model": ["100", 0], "shift": 8.0},
            "class_type": "ModelSamplingSD3"},
    "ks": {"inputs": {"seed": 0, "steps": 30, "cfg": 5.0, "sampler_name": "uni_pc",
                      "scheduler": "simple", "denoise": 1.0, "model": ["msd", 0],
                      "positive": ["pos", 0], "negative": ["neg", 0],
                      "latent_image": ["lat", 0]}, "class_type": "KSampler"},
    "dec": {"inputs": {"samples": ["ks", 0], "vae": ["300", 0]},
            "class_type": "VAEDecode"},
    "cv": {"inputs": {"images": ["dec", 0], "fps": 24}, "class_type": "CreateVideo"},
    "save": {"inputs": {"filename_prefix": "video/localm", "format": "mp4",
                        "codec": "h264", "video": ["cv", 0]}, "class_type": "SaveVideo"},
}


class TestResolver:
    def test_resolve_sampler_roles_follows_edges(self):
        roles = shared.resolve_sampler_roles(RENUMBERED_WAN)
        assert roles["sampler"][0] == "ks"
        assert roles["positive"][0] == "pos"
        assert roles["negative"][0] == "neg"
        assert roles["latent"][0] == "lat"

    def test_resolve_empty_when_no_sampler(self):
        roles = shared.resolve_sampler_roles(
            {"1": {"class_type": "VAELoader", "inputs": {}}})
        assert roles["sampler"] == (None, None)
        assert roles["positive"] == (None, None)

    def test_find_node_by_class(self):
        nid, node = shared.find_node_by_class(RENUMBERED_WAN, "CreateVideo")
        assert nid == "cv" and node["inputs"]["fps"] == 24
        assert shared.find_node_by_class(RENUMBERED_WAN, "NopeNode") == (None, None)

    def test_find_nodes_by_class_all(self):
        pairs = shared.find_nodes_by_class(RENUMBERED_WAN, "CLIPTextEncode")
        assert {nid for nid, _ in pairs} == {"pos", "neg"}

    def test_next_node_id_avoids_collision(self):
        # numeric max is 300; named ids are ignored for the numeric bump
        assert shared.next_node_id(RENUMBERED_WAN) == "301"
        assert shared.next_node_id({"a": {}, "b": {}}) == "localm_node_0"

    def test_set_seed_on_seed_field(self):
        node = {"inputs": {"seed": 0}}
        shared.set_seed_on(node, 42)
        assert node["inputs"]["seed"] == 42

    def test_set_seed_on_noise_seed_field(self):
        node = {"inputs": {"noise_seed": 0}}
        shared.set_seed_on(node, 7)
        assert node["inputs"]["noise_seed"] == 7

    def test_set_seed_on_no_seed_field_is_noop(self):
        node = {"inputs": {"foo": 1}}
        shared.set_seed_on(node, 9)
        assert node["inputs"] == {"foo": 1}      # never invents a field

    def test_is_link_distinguishes_links_from_values(self):
        assert shared._is_link(["7", 0]) is True
        assert shared._is_link([7, 1]) is True
        assert shared._is_link([True, 0]) is False     # bool is not a node id
        assert shared._is_link("text") is False
        assert shared._is_link([1, 2, 3]) is False

    def test_set_seed_on_all_covers_every_sampler(self):
        # A two-stage graph (Wan high/low-noise, SDXL refiner): the seed must land on
        # BOTH samplers, not just the first.
        wf = {
            "a": {"class_type": "KSampler", "inputs": {"seed": 0}},
            "b": {"class_type": "KSamplerAdvanced", "inputs": {"noise_seed": 0}},
            "c": {"class_type": "RandomNoise", "inputs": {"noise_seed": 0}},
            "d": {"class_type": "VAELoader", "inputs": {"vae_name": "x"}},
        }
        n = shared.set_seed_on_all(wf, 77)
        assert n == 3
        assert wf["a"]["inputs"]["seed"] == 77
        assert wf["b"]["inputs"]["noise_seed"] == 77
        assert wf["c"]["inputs"]["noise_seed"] == 77
        assert "seed" not in wf["d"]["inputs"]          # non-sampler untouched


class TestEndToEndRenumberedGraph:
    """generate_video drives a renumbered Wan graph correctly via role resolution."""

    def _fake_comfy(self, captured):
        outputs = {"save": {"images": [
            {"filename": "clip.mp4", "subfolder": "", "type": "output"}], "animated": [True]}}

        def fake_urlopen(req, timeout=None):
            url = req if isinstance(req, str) else req.full_url
            m = MagicMock()
            m.__enter__ = lambda s=m: s
            m.__exit__ = MagicMock(return_value=False)
            if "/object_info" in url:
                m.read.return_value = b"{}"        # preflight no-op
            elif "/prompt" in url:
                captured["workflow"] = json.loads(req.data.decode())["prompt"]
                m.read.return_value = b'{"prompt_id": "p1"}'
            elif "/history/" in url:
                m.read.return_value = json.dumps({"p1": {"outputs": outputs}}).encode()
            elif "/view" in url:
                m.read.return_value = b"FAKE-MP4"
            else:
                m.read.return_value = b"{}"
            return m
        return fake_urlopen

    def test_renumbered_graph_injection(self, tmp_path):
        from localm.video_gen import comfy as vcomfy
        wf_file = tmp_path / "wan_local.json"
        wf_file.write_text(json.dumps(RENUMBERED_WAN), encoding="utf-8")
        captured = {}
        with patch.object(vcomfy, "workflow_path", return_value=wf_file), \
             patch.object(vcomfy, "ensure_comfy", return_value=(True, "up")), \
             patch.object(vcomfy, "_localm_unload"), \
             patch.object(comfy_client, "_comfy_urlopen", self._fake_comfy(captured)), \
             patch.object(vcomfy.time, "sleep"):
            ok, msg = vcomfy.generate_video(
                "a red fox", tmp_path / "out.mp4", negative_prompt="blurry",
                seconds=5.0, fps=24, seed=99, steps=20, cfg=4.0)
        assert ok, msg
        wf = captured["workflow"]
        # Injection landed on the RENUMBERED nodes.
        assert wf["pos"]["inputs"]["text"] == "a red fox"
        assert wf["neg"]["inputs"]["text"] == "blurry"
        assert wf["ks"]["inputs"]["seed"] == 99
        assert wf["ks"]["inputs"]["steps"] == 20
        assert wf["ks"]["inputs"]["cfg"] == 4.0
        assert wf["lat"]["inputs"]["length"] == 121
        assert wf["cv"]["inputs"]["fps"] == 24


def _fake_video_urlopen(captured):
    outputs = {"save": {"images": [
        {"filename": "clip.mp4", "subfolder": "", "type": "output"}], "animated": [True]}}

    def fake_urlopen(req, timeout=None):
        url = req if isinstance(req, str) else req.full_url
        m = MagicMock()
        m.__enter__ = lambda s=m: s
        m.__exit__ = MagicMock(return_value=False)
        if "/object_info" in url:
            m.read.return_value = b"{}"
        elif "/prompt" in url:
            captured["workflow"] = json.loads(req.data.decode())["prompt"]
            m.read.return_value = b'{"prompt_id": "p1"}'
        elif "/history/" in url:
            m.read.return_value = json.dumps({"p1": {"outputs": outputs}}).encode()
        elif "/view" in url:
            m.read.return_value = b"FAKE-MP4"
        else:
            m.read.return_value = b"{}"
        return m
    return fake_urlopen


class TestVideoRobustness:
    def test_two_sampler_graph_seeds_every_stage(self, tmp_path):
        """A two-stage graph must get the seed on BOTH samplers (regression: a fixed
        template seed on the second stage made 'random' output partly deterministic)."""
        from localm.video_gen import comfy as vcomfy
        graph = dict(RENUMBERED_WAN)
        # A second sampling stage that refines the first stage's latent.
        graph["ks2"] = {"inputs": {"seed": 0, "steps": 30, "cfg": 5.0,
                                   "sampler_name": "uni_pc", "scheduler": "simple",
                                   "denoise": 0.5, "model": ["msd", 0],
                                   "positive": ["pos", 0], "negative": ["neg", 0],
                                   "latent_image": ["ks", 0]},
                        "class_type": "KSampler"}
        wf_file = tmp_path / "two_sampler.json"
        wf_file.write_text(json.dumps(graph), encoding="utf-8")
        captured = {}
        with patch.object(vcomfy, "workflow_path", return_value=wf_file), \
             patch.object(vcomfy, "ensure_comfy", return_value=(True, "up")), \
             patch.object(vcomfy, "_localm_unload"), \
             patch.object(comfy_client, "_comfy_urlopen", _fake_video_urlopen(captured)), \
             patch.object(vcomfy.time, "sleep"):
            ok, msg = vcomfy.generate_video("a fox", tmp_path / "out.mp4", seed=99)
        assert ok, msg
        wf = captured["workflow"]
        assert wf["ks"]["inputs"]["seed"] == 99
        assert wf["ks2"]["inputs"]["seed"] == 99      # the second stage too

    def test_malformed_latent_node_returns_clean_error_not_crash(self, tmp_path):
        """A user graph whose latent node has no inputs dict must return a clean
        (False, msg), not raise KeyError - and must not unload the chat model."""
        from localm.video_gen import comfy as vcomfy
        bad = {
            "lat": {"class_type": "Wan22ImageToVideoLatent"},   # no "inputs" key
            "pos": {"class_type": "CLIPTextEncode", "inputs": {"text": ""}},
            "neg": {"class_type": "CLIPTextEncode", "inputs": {"text": ""}},
            "ks": {"class_type": "KSampler",
                   "inputs": {"seed": 0, "positive": ["pos", 0],
                              "negative": ["neg", 0], "latent_image": ["lat", 0]}},
        }
        wf_file = tmp_path / "bad.json"
        wf_file.write_text(json.dumps(bad), encoding="utf-8")
        unload = MagicMock()
        with patch.object(vcomfy, "workflow_path", return_value=wf_file), \
             patch.object(vcomfy, "ensure_comfy", return_value=(True, "up")), \
             patch.object(vcomfy, "_localm_unload", unload):
            ok, msg = vcomfy.generate_video("a fox", tmp_path / "out.mp4")
        assert ok is False
        assert "inputs" in msg.lower()
        unload.assert_not_called()

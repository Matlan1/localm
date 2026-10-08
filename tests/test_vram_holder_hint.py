# SPDX-License-Identifier: AGPL-3.0-or-later
"""GgufBackend._vram_holder_hint(): the low-VRAM warning's "who is holding
this VRAM" attribution.

Regression coverage for the self-attribution bug: a lookup that does not tell
THIS SAME PROCESS apart from a sibling names it as "another localm instance", so
a server on port 8642 emits "Low VRAM ... Likely cause: another localm instance
(port 8642) is running 'gemma...' - POST /v1/models/unload on port 8642 to free
it" while IT is port 8642, telling the user to unload the model they are talking
to.

The peers come from ``gpu_registry.list_gpu_peers`` (which never returns this
process; see test_gpu_peer_detection.py) and this server's own model from
``gpu_registry.own_status``; only those two seams and the hardware-detection seam
(resolve_main_gpu_index) are faked here.
"""

import pytest

from localm import gpu_registry
from localm.inference.backends.gguf import GgufBackend


def _backend(tmp_path):
    # A tiny real file: the constructor only resolves the path, and no header is
    # read for _vram_holder_hint().
    f = tmp_path / "model.gguf"
    f.write_bytes(b"\0" * 4096)
    return GgufBackend(str(f), n_gpu_layers=99, n_gpu_layers_auto=False, n_ctx=4096)


@pytest.fixture
def seams(monkeypatch):
    """Set the peers and this server's own status; GPU 0 is the device sized for."""
    state = {"peers": [], "own": None}
    monkeypatch.setattr(gpu_registry, "list_gpu_peers", lambda **kw: list(state["peers"]))
    monkeypatch.setattr(gpu_registry, "own_status", lambda: state["own"])
    monkeypatch.setattr("localm.discover.resolve_main_gpu_index",
                        lambda configured, **k: 0)
    return state


def _peer(port, model, gpu_index=0):
    return {"instance_id": f"peer-{port}", "pid": 1, "port": port,
            "host": "127.0.0.1", "scheme": "http", "model": model,
            "gpu_index": gpu_index}


class TestVramHolderHint:
    def test_own_model_is_named_as_this_server_never_as_another_instance(
            self, tmp_path, seams):
        """No sibling holds the GPU, only this process's own model does. The hint
        must NOT claim "another localm instance" - that false attribution tells a
        user to unload the model they are talking to - and instead names this
        server's own model."""
        seams["own"] = {"instance_id": "self-iid", "model": "gemma-4-12b",
                        "gpu_index": 0}

        hint = _backend(tmp_path)._vram_holder_hint()

        assert "another localm instance" not in hint
        assert "gemma-4-12b" in hint

    def test_genuine_other_instance_is_still_named(self, tmp_path, seams):
        seams["peers"] = [_peer(9111, "peer-model")]

        hint = _backend(tmp_path)._vram_holder_hint()

        assert "another localm instance (port 9111)" in hint
        assert "peer-model" in hint
        assert "POST /v1/models/unload on port 9111 to free it." in hint

    def test_own_model_and_a_peer_both_present_names_the_peer(self, tmp_path, seams):
        """This server and a genuine peer both hold a model on this GPU: the peer
        wins the attribution (it is the one a POST-unload can actually reach
        usefully)."""
        seams["own"] = {"instance_id": "aaa-self", "model": "my-own-model",
                        "gpu_index": 0}
        seams["peers"] = [_peer(9222, "peer-model")]

        hint = _backend(tmp_path)._vram_holder_hint()

        assert "another localm instance (port 9222)" in hint
        assert "peer-model" in hint

    def test_no_peers_and_no_own_model_falls_back_to_generic(self, tmp_path, seams):
        hint = _backend(tmp_path)._vram_holder_hint()

        assert hint == ("another GPU app is holding memory "
                        "(ComfyUI, a browser, another model).")

    def test_own_model_on_a_different_gpu_index_is_not_blamed(self, tmp_path, seams):
        """This server holds a model, but on a different GPU device than the one
        being sized for - it must not be offered as the holder of THIS device's
        VRAM."""
        seams["own"] = {"instance_id": "self-iid", "model": "gemma-4-12b",
                        "gpu_index": 1}

        hint = _backend(tmp_path)._vram_holder_hint()

        assert "gemma-4-12b" not in hint
        assert hint == ("another GPU app is holding memory "
                        "(ComfyUI, a browser, another model).")

    def test_a_peer_on_a_different_gpu_index_is_not_blamed(self, tmp_path, seams):
        seams["peers"] = [_peer(9333, "elsewhere", gpu_index=1)]

        hint = _backend(tmp_path)._vram_holder_hint()

        assert "9333" not in hint
        assert hint == ("another GPU app is holding memory "
                        "(ComfyUI, a browser, another model).")

    def test_a_failing_lookup_falls_back_to_generic(self, tmp_path, monkeypatch, seams):
        def boom(**kw):
            raise RuntimeError("lookup exploded")
        monkeypatch.setattr(gpu_registry, "list_gpu_peers", boom)

        hint = _backend(tmp_path)._vram_holder_hint()

        assert hint == ("another GPU app is holding memory "
                        "(ComfyUI, a browser, another model).")

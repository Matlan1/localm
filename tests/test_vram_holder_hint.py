# SPDX-License-Identifier: AGPL-3.0-or-later
"""GgufBackend._vram_holder_hint(): the low-VRAM warning's "who is holding
this VRAM" attribution.

Regression coverage for the self-attribution bug: a registry lookup that does
not check whether an entry is THIS SAME PROCESS's OWN names it as "another
localm instance", so a server on port 8642 emits "Low VRAM ... Likely cause:
another localm instance (port 8642) is running 'gemma...' - POST
/v1/models/unload on port 8642 to free it" while IT is port 8642, telling the
user to unload the model they are talking to.

These tests exercise the real gpu_registry.list_gpu_peers() / own_entry()
logic end to end (only the process-boundary and hardware-detection seams -
pid_alive, _try_whoami, resolve_main_gpu_index - are mocked), so the logic
inside gpu_registry.py is what is under test, not a re-statement of it.
"""

import os

from localm import gpu_registry
from localm.inference.backends.gguf import GgufBackend


def _backend(tmp_path):
    # A tiny real file: the constructor only resolves the path, and no header is
    # read for _vram_holder_hint().
    f = tmp_path / "model.gguf"
    f.write_bytes(b"\0" * 4096)
    return GgufBackend(str(f), n_gpu_layers=99, n_gpu_layers_auto=False, n_ctx=4096)


def _stub_registry(monkeypatch, tmp_path, *, verified=True, alive=True):
    d = tmp_path / "reg"
    monkeypatch.setattr(gpu_registry, "registry_dir", lambda: d)
    monkeypatch.setattr(gpu_registry, "pid_alive", lambda pid: alive)
    monkeypatch.setattr(gpu_registry, "_try_whoami",
                        lambda scheme, port, iid, timeout: verified)
    monkeypatch.setattr("localm.discover.resolve_main_gpu_index",
                        lambda configured, **k: 0)
    return d


class TestVramHolderHint:
    def test_self_is_never_blamed_as_another_instance(self, tmp_path, monkeypatch):
        """The ONLY registry entry on this GPU is this same process's own
        (pid == os.getpid()). The hint must NOT claim "another localm
        instance" - that false attribution tells a user to unload the model
        they are talking to - and instead names this server's own model."""
        d = _stub_registry(monkeypatch, tmp_path)
        gpu_registry.write_entry(
            d, instance_id="self-iid", pid=os.getpid(), port=8642,
            host="127.0.0.1", scheme="http", model="gemma-4-12b",
            vram_estimate_bytes=None, gpu_index=0, coordination_token="t")

        hint = _backend(tmp_path)._vram_holder_hint()

        assert "another localm instance" not in hint
        assert "gemma-4-12b" in hint

    def test_genuine_other_instance_is_still_named(self, tmp_path, monkeypatch):
        """A live, identity-verified peer on a DIFFERENT pid must still be
        named - the self-exclusion must not blind the hint to real siblings,
        only to itself."""
        d = _stub_registry(monkeypatch, tmp_path)
        gpu_registry.write_entry(
            d, instance_id="peer-iid", pid=os.getpid() + 1, port=9111,
            host="127.0.0.1", scheme="http", model="peer-model",
            vram_estimate_bytes=None, gpu_index=0, coordination_token="t")

        hint = _backend(tmp_path)._vram_holder_hint()

        assert "another localm instance (port 9111)" in hint
        assert "peer-model" in hint
        assert "POST /v1/models/unload on port 9111 to free it." in hint

    def test_self_and_peer_both_present_names_the_peer(self, tmp_path, monkeypatch):
        """Self plus a genuine peer both hold entries on this GPU: the peer
        must win the attribution (it is the one a POST-unload can actually
        reach usefully). instance_id is chosen so the SELF entry sorts
        FIRST alphabetically (list_entries() reads entries in sorted
        filename order) - without pid-based exclusion, self would be the
        first match and would wrongly win the attribution instead of the
        real peer, so this is not a coincidental pass either way."""
        d = _stub_registry(monkeypatch, tmp_path)
        gpu_registry.write_entry(
            d, instance_id="aaa-self", pid=os.getpid(), port=8642,
            host="127.0.0.1", scheme="http", model="my-own-model",
            vram_estimate_bytes=None, gpu_index=0, coordination_token="t")
        gpu_registry.write_entry(
            d, instance_id="zzz-peer", pid=os.getpid() + 1, port=9222,
            host="127.0.0.1", scheme="http", model="peer-model",
            vram_estimate_bytes=None, gpu_index=0, coordination_token="t")

        hint = _backend(tmp_path)._vram_holder_hint()

        assert "another localm instance (port 9222)" in hint
        assert "peer-model" in hint

    def test_no_registry_entries_falls_back_to_generic(self, tmp_path, monkeypatch):
        _stub_registry(monkeypatch, tmp_path)

        hint = _backend(tmp_path)._vram_holder_hint()

        assert hint == ("another GPU app is holding memory "
                        "(ComfyUI, a browser, another model).")

    def test_self_entry_on_a_different_gpu_index_is_not_blamed(self, tmp_path, monkeypatch):
        """A self entry exists but on a different GPU device than the one
        being sized for - it must not be offered as the holder of THIS
        device's VRAM."""
        d = _stub_registry(monkeypatch, tmp_path)
        gpu_registry.write_entry(
            d, instance_id="self-iid", pid=os.getpid(), port=8642,
            host="127.0.0.1", scheme="http", model="gemma-4-12b",
            vram_estimate_bytes=None, gpu_index=1, coordination_token="t")

        hint = _backend(tmp_path)._vram_holder_hint()

        assert "gemma-4-12b" not in hint
        assert hint == ("another GPU app is holding memory "
                        "(ComfyUI, a browser, another model).")

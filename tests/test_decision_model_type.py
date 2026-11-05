# SPDX-License-Identifier: AGPL-3.0-or-later
"""A decision-model GGUF is registered as type 'decision', not as a chat model.

Decision models (Perplexity pplx-decider, Bespoke Nimble, JEV) keep a chat
architecture in ``general.architecture`` (``qwen35``) and are told apart only by
their own metadata keys: ``pplx_decider.*`` or ``<architecture>.decision.*``.
The key names below are the ones read from the published GGUF headers.
"""

import pytest

from localm import model_manager as mm
from localm.model_manager import gguf as G
from localm.model_manager import registry as R
from tests._gguf_builder import STRING, UINT32, build_gguf

PPLX_KEYS = [
    ("pplx_decider.protocol_version", UINT32, 3),
    ("pplx_decider.template", STRING, "joint-question-inline-readout-v1"),
    ("pplx_decider.marker", STRING, "<|fim_suffix|>"),
    ("pplx_decider.base_model", STRING, "Qwen/Qwen3.5-4B"),
]
NIMBLE_KEYS = [("qwen35.decision.type", STRING, "nimble")]
JEV_KEYS = [
    ("qwen35.decision.type", STRING, "jev"),
    ("qwen35.decision.temperature.noul", STRING, "1.0143134751376188"),
    ("qwen35.decision.verbalizer_ids.noul", STRING, "[3721, 1802]"),
]
MOE_KEYS = [("qwen35moe.decision.type", STRING, "jev")]


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    import localm.config as cfg
    home = tmp_path / ".localm"
    (home / "models").mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.setattr(cfg, "HOME_DIR", home)
    monkeypatch.setattr(cfg, "MODELS_DIR", home / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", home / "registry.json")
    monkeypatch.setattr(mm, "MODELS_DIR", home / "models")
    monkeypatch.setattr(mm, "REGISTRY_FILE", home / "registry.json")
    return home


def _write(tmp_path, name, arch, kvs):
    path = tmp_path / name
    path.write_bytes(build_gguf(arch, kvs))
    return path


CASES = [
    ("pplx", "qwen35", PPLX_KEYS),
    ("nimble", "qwen35", NIMBLE_KEYS),
    ("jev", "qwen35", JEV_KEYS),
    ("moe", "qwen35moe", MOE_KEYS),
]


@pytest.mark.parametrize("name,arch,kvs", CASES, ids=[c[0] for c in CASES])
class TestDecisionHeaders:
    def test_the_probe_sees_the_decision_keys(self, tmp_path, name, arch, kvs):
        f = _write(tmp_path, f"{name}.gguf", arch, kvs)
        assert G._gguf_metadata_probe(f)["has_decision_keys"] is True
        assert G.gguf_decision_signal(f) is True

    def test_the_file_is_typed_decision_not_llm(self, tmp_path, name, arch, kvs):
        f = _write(tmp_path, f"{name}.gguf", arch, kvs)
        detected, meta = R._detect_local_model_type(f, is_gguf=True, is_hf=False)
        assert detected == "decision"
        assert meta["architecture"] == arch

    def test_adding_the_file_registers_type_decision(self, isolated_home, tmp_path, name, arch, kvs):
        f = _write(tmp_path, f"{name}.gguf", arch, kvs)
        assert mm.add_local(str(f)) is True
        assert mm.load_registry()[name]["model_type"] == "decision"

    def test_the_keys_after_the_tokenizer_keys_still_count(self, tmp_path, name, arch, kvs):
        late = [("tokenizer.ggml.model", STRING, "gpt2"), *kvs]
        f = _write(tmp_path, f"{name}-late.gguf", arch, late)
        assert G.gguf_decision_signal(f) is True


class TestNotADecisionModel:
    @pytest.mark.parametrize("arch", ["qwen35", "qwen35moe", "llama"])
    def test_a_plain_chat_header_stays_llm(self, tmp_path, arch):
        f = _write(tmp_path, "plain.gguf", arch, [(f"{arch}.context_length", UINT32, 4096)])
        assert G.gguf_decision_signal(f) is False
        assert R._detect_local_model_type(f, is_gguf=True, is_hf=False)[0] == "llm"

    def test_the_word_in_a_value_or_a_near_miss_key_is_not_a_signal(self, tmp_path):
        kvs = [("general.name", STRING, "pplx_decider.decision"),
               ("qwen35.decisions", UINT32, 1),
               ("qwen35.decision", UINT32, 1),
               ("tokenizer.chat_template", STRING, "{{ decision }}"),
               ("decision.type", STRING, "x"),
               ("qwen35.my.decision.type", STRING, "x")]
        f = _write(tmp_path, "near.gguf", "qwen35", kvs)
        assert G.gguf_decision_signal(f) is False
        assert R._detect_local_model_type(f, is_gguf=True, is_hf=False)[0] == "llm"

    def test_an_embedding_header_stays_embedding(self, tmp_path):
        f = _write(tmp_path, "emb.gguf", "bert", [("bert.pooling_type", UINT32, 1)])
        assert R._detect_local_model_type(f, is_gguf=True, is_hf=False)[0] == "embedding"


class TestDecisionModelIsNotAChatModel:
    ENTRY = {"path": "models/pplx.gguf", "model_type": "decision"}

    def test_it_is_a_registry_type(self):
        assert "decision" in R.MODEL_TYPES

    def test_it_is_never_auto_picked_for_chat(self):
        reg = {"a-decider": dict(self.ENTRY), "b-chat": {"path": "b.gguf", "model_type": "llm"}}
        assert R.is_auto_chat_eligible(self.ENTRY) is False
        assert R.auto_chat_candidates(reg) == ["b-chat"]

    def test_it_is_not_a_chat_llm_for_pickers_and_routing(self):
        assert R.is_llm(self.ENTRY) is False

    def test_a_chat_request_naming_it_is_refused_with_a_plain_message(self):
        msg = R.chat_load_refusal("pplx", {"pplx": dict(self.ENTRY)})
        assert msg == ("'pplx' is a decision model, not a model that can be loaded or chatted "
                       "with. It answers typed questions with option probabilities and cannot "
                       "write replies.")

    def test_v1_models_leaves_it_out(self):
        assert R.is_unlisted_entry(self.ENTRY) is True

    def test_set_type_can_turn_it_into_a_chat_model_anyway(self, isolated_home, tmp_path):
        f = _write(tmp_path, "pplx.gguf", "qwen35", PPLX_KEYS)
        assert mm.add_local(str(f)) is True
        assert mm.set_model_type("pplx", "llm") is True
        assert mm.load_registry()["pplx"]["model_type"] == "llm"


from tests.test_pull_mmproj_autofetch import (  # noqa: E402,F401
    _wire_download, _wire_repo_listing, fake_registry)


@pytest.fixture
def pull_registry(request):
    return request.getfixturevalue("fake_registry")


class TestPullRegistersADecisionModelAsDecision:
    def test_a_fresh_download_is_typed_decision(self, pull_registry, monkeypatch, capsys):
        store, _models_dir = pull_registry
        _wire_repo_listing(monkeypatch, ["dec.gguf"])
        _wire_download(monkeypatch, {"dec.gguf": build_gguf("qwen35", PPLX_KEYS)})
        assert mm._pull_gguf_file("o/r:dec.gguf", None, type_is_auto=True) is True
        assert store["dec"]["model_type"] == "decision"
        assert "decision model" in capsys.readouterr().out

    def test_an_already_downloaded_file_is_typed_decision(self, pull_registry, monkeypatch):
        store, models_dir = pull_registry
        (models_dir / "dec.gguf").write_bytes(build_gguf("qwen35", NIMBLE_KEYS))
        _wire_repo_listing(monkeypatch, ["dec.gguf"])
        _wire_download(monkeypatch, {})
        assert mm._pull_gguf_file("o/r:dec.gguf", None, type_is_auto=True) is True
        assert store["dec"]["model_type"] == "decision"

    def test_a_plain_chat_download_is_still_llm(self, pull_registry, monkeypatch):
        store, _models_dir = pull_registry
        _wire_repo_listing(monkeypatch, ["chat.gguf"])
        _wire_download(monkeypatch, {"chat.gguf": build_gguf("qwen35", [])})
        assert mm._pull_gguf_file("o/r:chat.gguf", None, type_is_auto=True) is True
        assert store["chat"]["model_type"] == "llm"

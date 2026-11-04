# SPDX-License-Identifier: AGPL-3.0-or-later
"""A model folder that ships its own code, with 'Allow model-bundled custom code' off: it runs on
transformers' own classes when there are any, its code never runs, and a failure says the code did
not run, as a possible cause. Real worker processes on the CPU."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from localm.inference.backends import hf as hf_backend
from localm.inference.backends.hf import BUNDLED_CODE_NOTE, HFBackend, with_bundled_code_note

torch = pytest.importorskip("torch")
tr = pytest.importorskip("transformers")

TEMPLATE = "{% for m in messages %}{{ m['content'] }}\n{% endfor %}"
BOOBY_TRAP = "raise AssertionError('the code of the model folder ran')\n"


def _model_folder(folder: Path, trap: bool = True) -> Path:
    config = tr.LlamaConfig(hidden_size=32, num_attention_heads=2, num_hidden_layers=1, intermediate_size=32,
                            vocab_size=64, bos_token_id=1, eos_token_id=2)
    torch.manual_seed(0)
    tr.LlamaForCausalLM(config).save_pretrained(folder)
    vocab = {"<unk>": 0, "<s>": 1, "</s>": 2, **{chr(97 + i): 3 + i for i in range(26)}, **{str(i): 30 + i for i in range(10)}}
    from tokenizers import Tokenizer, models, pre_tokenizers
    backend = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Split("", "isolated")
    tr.PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>", bos_token="<s>", eos_token="</s>",
                               chat_template=TEMPLATE).save_pretrained(folder)
    if trap:
        data = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        data["auto_map"] = {"AutoConfig": "modeling_trap.TrapConfig", "AutoModelForCausalLM": "modeling_trap.TrapModel"}
        (folder / "config.json").write_text(json.dumps(data), encoding="utf-8")
        (folder / "modeling_trap.py").write_text(BOOBY_TRAP, encoding="utf-8")
    return folder


@pytest.fixture
def setting(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    import localm.config as cfg
    monkeypatch.setattr(cfg, "CONFIG_FILE", tmp_path / "home" / "config.json", raising=False)

    def write(on: bool) -> None:
        (tmp_path / "home" / "config.json").write_text(json.dumps({"hf_trust_remote_code": on}), encoding="utf-8")
        monkeypatch.setattr(hf_backend, "_trust_remote_code_enabled", lambda: on)

    write(False)
    return write


def _reply(backend: HFBackend) -> str:
    return "".join(backend.chat_stream([{"role": "user", "content": "abc"}], max_tokens=4, temperature=0.0))


def test_the_model_runs_on_transformers_own_classes_and_its_code_never_runs(tmp_path, setting, capsys):
    folder = _model_folder(tmp_path / "m")
    assert hf_backend.runs_without_its_own_code(str(folder)) is True
    backend = HFBackend(str(folder), device="cpu")
    try:
        backend.load()
        assert "this model's own code did not run" in capsys.readouterr().out.replace("\n", " ")
        assert isinstance(_reply(backend), str)
    finally:
        backend.unload()


def test_weights_that_do_not_match_fail_the_load_and_name_the_code_that_did_not_run(tmp_path, setting):
    from safetensors.torch import load_file, save_file
    folder = _model_folder(tmp_path / "m")
    weights = folder / "model.safetensors"
    state = load_file(str(weights))
    dropped = next(k for k in state if "self_attn.q_proj" in k)
    del state[dropped]
    save_file(state, str(weights), metadata={"format": "pt"})
    backend = HFBackend(str(folder), device="cpu")
    with pytest.raises(RuntimeError) as raised:
        backend.load()
    assert "weights are missing" in str(raised.value)
    assert BUNDLED_CODE_NOTE in str(raised.value)


def test_a_failure_while_generating_names_the_code_that_did_not_run(tmp_path, setting):
    folder = _model_folder(tmp_path / "m")
    data = json.loads((folder / "tokenizer_config.json").read_text(encoding="utf-8"))
    data["chat_template"] = "{{ raise_exception('this template stops') }}"
    (folder / "tokenizer_config.json").write_text(json.dumps(data), encoding="utf-8")
    template = folder / "chat_template.jinja"
    if template.exists():
        template.write_text(data["chat_template"], encoding="utf-8")
    backend = HFBackend(str(folder), device="cpu")
    try:
        backend.load()
        with pytest.raises(RuntimeError) as raised:
            _reply(backend)
        assert "this template stops" in str(raised.value)
        assert BUNDLED_CODE_NOTE in str(raised.value)
    finally:
        backend.unload()


def test_with_the_setting_on_a_failure_says_nothing_about_it_and_the_folders_code_is_what_runs(tmp_path, setting):
    setting(True)
    folder = _model_folder(tmp_path / "m")
    assert hf_backend.runs_without_its_own_code(str(folder)) is False
    backend = HFBackend(str(folder), device="cpu")
    with pytest.raises(RuntimeError) as raised:
        backend.load()
    assert "the code of the model folder ran" in str(raised.value)
    assert BUNDLED_CODE_NOTE not in str(raised.value)


def test_a_folder_that_ships_no_code_fails_without_the_note(tmp_path, setting):
    folder = _model_folder(tmp_path / "m", trap=False)
    (folder / "model.safetensors").write_bytes(b"\0" * 8)
    backend = HFBackend(str(folder), device="cpu")
    with pytest.raises(RuntimeError) as raised:
        backend.load()
    assert BUNDLED_CODE_NOTE not in str(raised.value)


def test_a_failure_for_lack_of_memory_has_no_note(tmp_path, setting):
    folder = _model_folder(tmp_path / "m")
    err = RuntimeError("CUDA out of memory. Tried to allocate 2 GiB")
    assert with_bundled_code_note(str(folder), err) is err


def test_the_note_is_added_once(tmp_path, setting):
    folder = _model_folder(tmp_path / "m")
    once = with_bundled_code_note(str(folder), RuntimeError("it broke"))
    assert str(once).startswith("it broke ") and BUNDLED_CODE_NOTE in str(once)
    assert with_bundled_code_note(str(folder), once) is once


def test_a_model_type_transformers_does_not_ship_is_still_refused(tmp_path, setting):
    folder = tmp_path / "m"
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps(
        {"model_type": "localm_no_such_type", "auto_map": {"AutoModelForCausalLM": "modeling_trap.TrapModel"}}),
        encoding="utf-8")
    assert hf_backend.runs_without_its_own_code(str(folder)) is False
    with pytest.raises(RuntimeError, match="hf_trust_remote_code"):
        hf_backend._check_custom_code_allowed(str(folder))


@pytest.mark.parametrize("model_type", [None, 3, "", "../../evil", "no such type"])
def test_a_model_type_that_names_no_folder_of_transformers_has_no_support(tmp_path, setting, model_type):
    folder = tmp_path / "m"
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps({"model_type": model_type}), encoding="utf-8")
    assert hf_backend._has_transformers_support(str(folder)) is False


def _kimi_folder(tmp_path) -> Path:
    folder = tmp_path / "kimi"
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps(
        {"model_type": "kimi_linear", "auto_map": {"AutoModelForCausalLM": "modeling_kimi.KimiLinearForCausalLM"}}),
        encoding="utf-8")
    (folder / "modeling_kimi.py").write_text(
        "try:\n    from fla.ops.kda import chunk_kda\nexcept ImportError:\n    raise ImportError('pip install fla-core')\n",
        encoding="utf-8")
    return folder


def test_kimi_linear_uses_transformers_classes_with_the_setting_on_too_and_needs_no_package(tmp_path, setting):
    from localm.model_manager import missing_packages as mp
    from localm.model_manager import model_packages as mpk
    folder = _kimi_folder(tmp_path)
    assert hf_backend.uses_transformers_classes(str(folder)) is True
    setting(True)
    assert hf_backend.uses_transformers_classes(str(folder)) is True
    assert hf_backend.runs_without_its_own_code(str(folder)) is False
    assert mp.folder_refusal(folder) is None
    state = mpk.folder_state(folder)
    assert state.trusted is True and state.missing == []
    assert with_bundled_code_note(str(folder), RuntimeError("x")).args == ("x",)


def test_another_model_type_keeps_its_own_code_with_the_setting_on(tmp_path, setting):
    folder = _model_folder(tmp_path / "m")
    setting(True)
    assert hf_backend.uses_transformers_classes(str(folder)) is False


def test_the_note_the_gui_translates_is_the_one_the_server_sends():
    catalog = (Path(__file__).resolve().parents[1] / "localm/plugins/gui/static/app/i18n-en.js").read_text(encoding="utf-8")
    assert f'"models.bundledCodeNote": {json.dumps(BUNDLED_CODE_NOTE)}' in catalog

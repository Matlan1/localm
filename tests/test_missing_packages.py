# SPDX-License-Identifier: AGPL-3.0-or-later
"""A model folder whose bundled code imports a package this install lacks is
refused early with a localm sentence, and transformers' own pip-flavoured
message is re-worded."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from localm.inference.backends import hf as hf_backend
from localm.inference.backends.base import ModelRefusedError
from localm.model_manager import missing_packages as mp
from localm.model_manager.unsupported import hf_folder_refusal

ABSENT = "localm_no_such_package_zz"

KIMI_TEXT = (
    "503: Failed to load 'Kimi-Linear-48B-A3B-Instruct': ImportError: This modeling file "
    "requires the following packages that were not found in your environment: tiktoken. "
    "Run `pip install tiktoken`")


def _model(tmp_path: Path, modeling: str, auto_map: bool = True) -> Path:
    folder = tmp_path / "model"
    folder.mkdir()
    config = {"architectures": ["XForCausalLM"], "model_type": "x"}
    if auto_map:
        config["auto_map"] = {"AutoModelForCausalLM": "modeling_x.XForCausalLM"}
    (folder / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (folder / "modeling_x.py").write_text(modeling, encoding="utf-8")
    return folder


@pytest.fixture
def trusted(monkeypatch):
    monkeypatch.setattr(hf_backend, "_trust_remote_code_enabled", lambda: True)


def test_imports_skip_a_guarded_import_and_follow_relative_ones():
    source = (
        "import os\nimport einops\nfrom .helpers import thing\nfrom . import more\n"
        "try:\n    import fla\nexcept ImportError:\n    fla = None\n"
        "def f():\n    import tiktoken\n")
    absolute, relative = mp.imports_of(source)
    assert absolute == {"os", "einops", "tiktoken"}
    assert relative == {"helpers"}


def test_entry_modules_reads_every_auto_map_shape():
    configs = [{"auto_map": {"AutoConfig": "configuration_x.C",
                             "AutoTokenizer": ["tokenization_x.T", None],
                             "AutoModel": "other/repo--modeling_y.M"}}, {}]
    assert mp.entry_modules(configs) == ["configuration_x", "tokenization_x"]


def test_imports_behind_an_availability_check_are_not_read():
    source = ("if is_flash_attn_2_available():\n    import flash_attn\n"
              "if is_torch_available():\n    import not_read_either\n"
              "if some_other_check():\n    import read_here\n")
    absolute, _ = mp.imports_of(source, frozenset({"is_torch_available"}))
    assert absolute == {"read_here"}


def test_a_source_too_deep_to_walk_yields_no_imports():
    assert mp.imports_of("x = " + "+".join(["1"] * 3000) + "\nimport einops\n") == (set(), set())


def test_a_processor_only_auto_map_is_not_scanned(tmp_path, trusted):
    folder = tmp_path / "model"
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps({"model_type": "x"}), encoding="utf-8")
    (folder / "preprocessor_config.json").write_text(
        json.dumps({"auto_map": {"AutoProcessor": "processing_x.P"}}), encoding="utf-8")
    (folder / "processing_x.py").write_text(f"import {ABSENT}\n", encoding="utf-8")
    assert mp.folder_refusal(folder) is None


def test_unparsable_source_yields_no_imports():
    assert mp.imports_of("def (:") == (set(), set())


def test_a_missing_package_in_a_followed_module_is_found(tmp_path, trusted):
    folder = _model(tmp_path, "from .helpers import thing\n")
    (folder / "helpers.py").write_text(f"import {ABSENT}\n", encoding="utf-8")
    assert mp.missing_modules(folder) == [ABSENT]


def test_folder_refusal_names_the_package_and_gives_no_pip_advice(tmp_path, trusted):
    folder = _model(tmp_path, f"import torch\nimport {ABSENT}\n")
    reason = mp.folder_refusal(folder)
    assert reason is not None
    assert ABSENT.replace("_", "-") in reason
    assert "pip" not in reason.lower()
    assert "cannot run this model" in reason
    assert hf_folder_refusal(folder) == reason


def test_a_package_localm_ships_but_this_install_lacks_says_to_update(tmp_path, trusted, monkeypatch):
    monkeypatch.setattr(mp, "SHIPPED_MODULES", frozenset({ABSENT}))
    reason = mp.folder_refusal(_model(tmp_path, f"import {ABSENT}\n"))
    assert "normally includes" in reason and "pip" not in reason.lower()


def test_installed_packages_and_stdlib_are_not_missing(tmp_path, trusted):
    folder = _model(tmp_path, "import json\nimport pytest\n")
    assert mp.folder_refusal(folder) is None


def test_no_refusal_without_bundled_code(tmp_path, trusted):
    assert mp.folder_refusal(_model(tmp_path, f"import {ABSENT}\n", auto_map=False)) is None


def test_no_package_refusal_while_custom_code_is_not_trusted(tmp_path, monkeypatch):
    monkeypatch.setattr(hf_backend, "_trust_remote_code_enabled", lambda: False)
    assert mp.folder_refusal(_model(tmp_path, f"import {ABSENT}\n")) is None


def test_transformers_import_error_is_reworded_without_pip():
    sentence = mp.rewrite_import_error("Kimi-Linear-48B-A3B-Instruct", KIMI_TEXT)
    assert sentence is not None
    assert "tiktoken" in sentence and "Kimi-Linear-48B-A3B-Instruct" in sentence
    assert "pip" not in sentence.lower()


@pytest.mark.parametrize("text,module", [
    ("ModuleNotFoundError: No module named 'einops.layers'", "einops"),
    ("This tokenizer requires the SentencePiece library but it was not found", "sentencepiece"),
    ("requires the protobuf library", "protobuf"),
])
def test_other_missing_package_messages_are_reworded(text, module):
    sentence = mp.rewrite_import_error("m", text)
    assert sentence is not None and module in sentence


@pytest.mark.parametrize("text,label", [
    ("This model requires the scikit-learn library but it was not found", "scikit-learn"),
    ("requires the PyTorch library but it was not found in your environment", "torch"),
    ("requires the \U0001f917 Datasets library. You can install it with pip", "datasets"),
])
def test_library_display_names_map_to_the_right_package(text, label):
    sentence = mp.rewrite_import_error("m", text)
    assert sentence is not None and label in sentence and "pip" not in sentence.lower()


def test_a_library_version_complaint_is_not_a_missing_package():
    assert mp.rewrite_import_error("m", "requires the tokenizers library version 0.5") is None


def test_a_dynamic_module_import_failure_is_not_a_missing_package():
    assert mp.rewrite_import_error("m", "No module named 'transformers_modules.x.y'") is None


def test_text_that_still_mentions_pip_is_replaced_by_a_neutral_sentence():
    out = mp.strip_pip_instructions("You can install it with pip: see the docs")
    assert "pip" not in out.lower() and "needs a Python package this install does not have" in out


def test_unrelated_failures_are_not_reworded():
    assert mp.rewrite_import_error("m", "CUDA out of memory") is None


def test_strip_pip_instructions_keeps_the_rest():
    out = mp.strip_pip_instructions("The tokenizer failed. Run `pip install sentencepiece`")
    assert out == "The tokenizer failed." and "pip" not in out


def test_without_pip_advice_rewrites_the_remote_code_error_as_a_refusal():
    out = hf_backend._without_pip_advice("/m/Kimi", RuntimeError(KIMI_TEXT))
    assert isinstance(out, ModelRefusedError) and "pip" not in str(out).lower()


def test_without_pip_advice_leaves_other_errors_alone():
    err = RuntimeError("CUDA out of memory")
    assert hf_backend._without_pip_advice("/m/x", err) is err


def test_load_refuses_a_folder_whose_code_needs_a_missing_package_before_any_child(
        tmp_path, trusted, monkeypatch):
    class _Runner:
        def __init__(self):
            raise AssertionError("a child was started")

    monkeypatch.setattr(hf_backend, "HFRunner", _Runner)
    folder = _model(tmp_path, f"import {ABSENT}\n")
    with pytest.raises(Exception) as raised:
        hf_backend.HFBackend(str(folder)).load()
    assert ABSENT.replace("_", "-") in str(raised.value) and "pip" not in str(raised.value).lower()
    assert not isinstance(raised.value, AssertionError)


def test_load_reports_a_remote_code_import_error_without_pip(tmp_path, monkeypatch):
    class _Runner:
        def spawn_and_load(self, params, timeout):
            raise RuntimeError(KIMI_TEXT)

    monkeypatch.setattr(hf_backend, "HFRunner", _Runner)
    plain = tmp_path / "plain-model"
    plain.mkdir()
    (plain / "config.json").write_text(json.dumps({"model_type": "x"}), encoding="utf-8")
    with pytest.raises(ModelRefusedError) as raised:
        hf_backend.HFBackend(str(plain)).load()
    assert "tiktoken" in str(raised.value) and "pip" not in str(raised.value).lower()


REPO_ROOT = Path(__file__).resolve().parents[1]
CENSUS_PACKAGES = {"tiktoken": "tiktoken", "sentencepiece": "sentencepiece",
                   "protobuf": "google", "einops": "einops", "peft": "peft",
                   "librosa": "librosa", "scipy": "scipy",
                   "scikit-learn": "sklearn", "opencv-python-headless": "cv2",
                   "datasets": "datasets", "matplotlib": "matplotlib"}


def _census_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "hf_remote_code_census", REPO_ROOT / "scripts" / "hf_remote_code_census.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_hf_and_gpu_extras_ship_the_packages_transformers_repos_import():
    import tomllib
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras = data["project"]["optional-dependencies"]
    for extra in ("hf", "gpu"):
        names = {re.split(r"[<>=;\[ ]", r, maxsplit=1)[0].replace("_", "-").lower()
                 for r in extras[extra]}
        missing = set(CENSUS_PACKAGES) - names
        assert not missing, f"[{extra}] must declare {sorted(missing)}"


def test_torch_coupled_packages_stay_out_of_hf():
    import tomllib
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    names = {re.split(r"[<>=;\[ ]", r, maxsplit=1)[0].replace("_", "-").lower()
             for r in data["project"]["optional-dependencies"]["hf"]}
    assert not names & {"timm", "open-clip-torch", "torchaudio", "torchvision"}


def test_the_shipped_packages_are_locked_and_the_census_agrees():
    census = _census_module()
    shipped = census.shipped_distributions(REPO_ROOT / "uv.lock")
    for dist, module in CENSUS_PACKAGES.items():
        assert dist in shipped, f"uv.lock does not resolve {dist} from the hf/gpu extras"
        assert census.module_is_shipped(module, shipped)
        assert module in mp.SHIPPED_MODULES


def test_every_module_localm_calls_shipped_is_shipped_by_the_lock():
    census = _census_module()
    shipped = census.shipped_distributions(REPO_ROOT / "uv.lock")
    unshipped = sorted(m for m in mp.SHIPPED_MODULES if not census.module_is_shipped(m, shipped))
    assert unshipped == []


def test_the_census_flags_a_package_nothing_ships_or_refuses():
    census = _census_module()
    shipped = census.shipped_distributions(REPO_ROOT / "uv.lock")
    assert not census.module_is_shipped("localm_no_such_package_zz", shipped)
    assert "localm_no_such_package_zz" not in census.ACCEPTED_REFUSALS
    assert not set(census.ACCEPTED_REFUSALS) & set(mp.SHIPPED_MODULES)

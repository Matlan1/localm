# SPDX-License-Identifier: AGPL-3.0-or-later
"""A model folder whose bundled code imports a package this install lacks is
refused early with a localm sentence, and transformers' own pip-flavoured
message is re-worded."""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest

from localm.inference.backends import hf as hf_backend
from localm.inference.backends.base import ModelRefusedError
from localm.inference.backends.llamacpp._runner import load_error_text
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


def _on(monkeypatch, plat: str, machine: str) -> None:
    monkeypatch.setattr(mp.sys, "platform", plat)
    monkeypatch.setattr(mp.platform, "machine", lambda: machine)


@pytest.mark.parametrize("plat, machine, label", [
    ("linux", "aarch64", "Linux ARM64"),
    ("win32", "ARM64", "Windows ARM64"),
    ("darwin", "arm64", "macOS ARM64"),
    ("darwin", "x86_64", "macOS x64"),
])
def test_a_package_localm_does_not_ship_for_this_platform_is_named_as_such(
        tmp_path, trusted, monkeypatch, plat, machine, label):
    _on(monkeypatch, plat, machine)
    monkeypatch.setattr(mp, "is_importable", lambda module: module != "decord")
    reason = mp.folder_refusal(_model(tmp_path, "import torch\nimport decord\n"))
    assert reason == (
        f"model cannot be loaded: its code needs the Python package decord, which is "
        f"not available for this platform ({label}). localm cannot run this model here.")


@pytest.mark.parametrize("plat, machine", [("linux", "x86_64"), ("win32", "AMD64")])
def test_decord_missing_where_localm_ships_it_still_says_to_update(monkeypatch, plat, machine):
    _on(monkeypatch, plat, machine)
    reason = mp.refusal_for("m", ["decord"])
    assert "normally includes" in reason and "not available for this platform" not in reason


def test_the_platform_gap_wins_over_a_package_the_install_lost(monkeypatch):
    _on(monkeypatch, "win32", "ARM64")
    reason = mp.refusal_for("m", ["tiktoken", "cv2", "einops"])
    assert "packages tiktoken, opencv, which are not available for this platform (Windows ARM64)" \
        in reason
    assert "normally includes" not in reason and "einops" not in reason


def test_a_text_import_error_for_a_platform_gap_is_reworded_the_same_way(monkeypatch):
    _on(monkeypatch, "darwin", "x86_64")
    reason = mp.rewrite_import_error("m", "ModuleNotFoundError: No module named 'monai'")
    assert reason is not None and "not available for this platform (macOS x64)" in reason
    assert "pip" not in reason.lower()


_ENVIRONMENTS = [("linux", "x86_64"), ("linux", "aarch64"), ("win32", "AMD64"),
                 ("win32", "ARM64"), ("darwin", "arm64"), ("darwin", "x86_64")]
_DIST_MODULES = {"opencv-python-headless": "cv2", "scikit-learn": "sklearn",
                 "protobuf": "google"}


def _extra_install_map(extra: str) -> dict[str, dict[tuple[str, str], bool]]:
    """For each distribution of *extra*, whether its requirements install it on each
    environment of _ENVIRONMENTS, as packaging evaluates their markers."""
    import tomllib

    from packaging.requirements import Requirement
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    out: dict[str, dict[tuple[str, str], bool]] = {}
    for text in data["project"]["optional-dependencies"][extra]:
        req = Requirement(text)
        name = req.name.lower().replace("_", "-")
        row = out.setdefault(name, {env: False for env in _ENVIRONMENTS})
        for plat, machine in _ENVIRONMENTS:
            env = {"sys_platform": plat, "platform_machine": machine, "extra": extra}
            if req.marker is None or req.marker.evaluate(env):
                row[(plat, machine)] = True
    return out


def test_platform_gaps_match_the_extras_markers():
    hf = _extra_install_map("hf")
    for dist, row in hf.items():
        module = _DIST_MODULES.get(dist, dist.replace("-", "_"))
        assert module in mp.SHIPPED_MODULES, f"[hf] installs {dist}, SHIPPED_MODULES lacks {module}"
        for (plat, machine), installed in row.items():
            assert mp.shipped_here(module, plat, machine) is installed, (dist, plat, machine)
    gpu = _extra_install_map("gpu")
    for dist in hf:
        assert gpu[dist] == hf[dist], f"[gpu] and [hf] install {dist} on different platforms"


def test_every_platform_gap_is_a_shipped_module_with_a_marker():
    hf = _extra_install_map("hf")
    gated = {_DIST_MODULES.get(d, d.replace("-", "_")) for d, row in hf.items()
             if not all(row.values())}
    assert set(mp.PLATFORM_GAPS) == gated


def test_decord_is_left_out_everywhere_its_wheels_are_missing():
    hf = _extra_install_map("hf")
    assert {env for env, on in hf["decord"].items() if on} == {
        ("linux", "x86_64"), ("win32", "AMD64")}


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


PIP_NAME = ABSENT.replace("_", "-")
MODEL_STOPS = (
    f"try:\n    import {ABSENT}\nexcept ImportError:\n"
    f"    raise ImportError(\"Plese run `pip install -U {PIP_NAME}`\")\n")
MODEL_FLAGS = (
    f"try:\n    import {ABSENT}\n    HAS_IT = True\nexcept ImportError:\n    HAS_IT = False\n"
    f"if not HAS_IT:\n    raise ImportError(\"Plese run `pip install -U {PIP_NAME}`\")\n")


def _real_import_error(folder: Path) -> ImportError:
    spec = importlib.util.spec_from_file_location("modeling_x_probe", folder / "modeling_x.py")
    module = importlib.util.module_from_spec(spec)
    with pytest.raises(ImportError) as raised:
        spec.loader.exec_module(module)
    return raised.value


def _refusing_runner(monkeypatch, text: str) -> None:
    class _Runner:
        def spawn_and_load(self, params, timeout):
            raise RuntimeError(text)

    monkeypatch.setattr(hf_backend, "HFRunner", _Runner)


def test_a_try_whose_import_handler_raises_marks_its_imports_required():
    source = (
        "try:\n    import needed_a\nexcept ImportError:\n    raise ImportError('install it')\n"
        "try:\n    from needed_b import x\nexcept (ValueError, ModuleNotFoundError):\n    raise\n"
        "try:\n    import optional_a\nexcept ImportError:\n    optional_a = None\n"
        "try:\n    import optional_b\nexcept ValueError:\n    raise\n"
        "try:\n    import optional_c\nexcept ImportError:\n    def later():\n        raise X\n"
        "try:\n    import optional_d\nexcept ImportError:\n    import fallback_d\n")
    absolute, _ = mp.imports_of(source)
    assert absolute == {"needed_a", "needed_b"}
    assert mp.imports_of(source, guarded=True)[0] == {
        "needed_a", "needed_b", "optional_a", "optional_b", "optional_c", "optional_d"}


def test_a_model_that_stops_when_a_guarded_import_fails_is_refused_before_the_load(
        tmp_path, trusted, monkeypatch):
    class _Runner:
        def __init__(self):
            raise AssertionError("a child was started")

    monkeypatch.setattr(hf_backend, "HFRunner", _Runner)
    folder = _model(tmp_path, MODEL_STOPS)
    assert mp.missing_modules(folder) == [ABSENT]
    added = hf_folder_refusal(folder)
    assert added is not None and PIP_NAME in added and "pip install" not in added
    with pytest.raises(ModelRefusedError) as raised:
        hf_backend.HFBackend(str(folder)).load()
    assert str(raised.value) == added


def test_a_real_import_error_from_the_models_own_code_is_a_refusal_at_load(
        tmp_path, trusted, monkeypatch):
    folder = _model(tmp_path, MODEL_FLAGS)
    assert mp.missing_modules(folder) == []
    failure = _real_import_error(folder)
    text = load_error_text(failure)
    assert text.startswith("ImportError: ") and "pip install" in text
    _refusing_runner(monkeypatch, text)
    with pytest.raises(ModelRefusedError) as raised:
        hf_backend.HFBackend(str(folder)).load()
    message = str(raised.value)
    assert PIP_NAME in message and "pip" not in message.lower()


def test_a_guarded_import_the_error_does_not_name_is_not_blamed(tmp_path, trusted, monkeypatch):
    folder = _model(tmp_path, MODEL_FLAGS)
    _refusing_runner(monkeypatch, "ImportError: cannot import name 'thing' from 'somewhere'")
    with pytest.raises(RuntimeError) as raised:
        hf_backend.HFBackend(str(folder)).load()
    assert not isinstance(raised.value, ModelRefusedError)
    assert ABSENT not in str(raised.value) and PIP_NAME not in str(raised.value)


def test_an_import_error_telling_to_run_pip_names_the_package_without_a_folder_scan():
    text = f"ImportError: Plese run `pip install -U {PIP_NAME}`"
    sentence = mp.rewrite_import_error("m", text)
    assert sentence is not None and PIP_NAME in sentence and "pip" not in sentence.lower()
    assert mp.rewrite_import_error("m", "ImportError: Plese run `pip install -U numpy`") is None


def test_an_import_error_that_names_no_package_is_still_a_refusal_without_pip():
    out = hf_backend._without_pip_advice(
        "/m/x", RuntimeError("ImportError: this needs more. Please run pip install"))
    assert isinstance(out, ModelRefusedError) and "pip" not in str(out).lower()


REPO_ROOT =Path(__file__).resolve().parents[1]
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

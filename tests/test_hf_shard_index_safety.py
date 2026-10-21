# SPDX-License-Identifier: AGPL-3.0-or-later
"""A shard index may only name files inside its own model directory.

``transformers`` joins ``weight_map`` values onto the model directory and opens
the result without validating it, so a ``..`` component escapes and an absolute
or drive-qualified value replaces the directory outright.
"""

import json
import os
from pathlib import Path
from unittest import mock

import pytest

from localm.inference import hf_shard_index_safety as safety


def _model_dir(tmp_path: Path, weight_map, name: str = "model.safetensors.index.json") -> Path:
    d = tmp_path / "models" / "victim"
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(
        json.dumps({"metadata": {"total_size": 1}, "weight_map": weight_map}),
        encoding="utf-8")
    return d


def _secret(tmp_path: Path) -> Path:
    out = tmp_path / "outside"
    out.mkdir(parents=True, exist_ok=True)
    f = out / "SECRET.safetensors"
    f.write_text("TOP SECRET", encoding="utf-8")
    return f


# ------------------------------------------------------------------ #
#  Legitimate models keep loading                                      #
# ------------------------------------------------------------------ #

def test_flat_shard_names_pass(tmp_path):
    d = _model_dir(tmp_path, {
        "a.weight": "model-00001-of-00002.safetensors",
        "b.weight": "model-00002-of-00002.safetensors",
    })
    safety.validate_shard_index(str(d))


def test_nested_shard_names_pass(tmp_path):
    d = _model_dir(tmp_path, {"a.weight": "sub/dir/shard.safetensors"})
    safety.validate_shard_index(str(d))


def test_no_index_file_is_a_noop(tmp_path):
    d = tmp_path / "models" / "plain"
    d.mkdir(parents=True)
    safety.validate_shard_index(str(d))


def test_unparseable_index_is_a_noop(tmp_path):
    d = tmp_path / "models" / "broken"
    d.mkdir(parents=True)
    (d / "model.safetensors.index.json").write_text("{not json", encoding="utf-8")
    safety.validate_shard_index(str(d))


def test_index_without_weight_map_is_a_noop(tmp_path):
    d = tmp_path / "models" / "nomap"
    d.mkdir(parents=True)
    (d / "model.safetensors.index.json").write_text('{"metadata": {}}', encoding="utf-8")
    safety.validate_shard_index(str(d))


def test_missing_model_dir_is_a_noop(tmp_path):
    safety.validate_shard_index(str(tmp_path / "does" / "not" / "exist"))


# ------------------------------------------------------------------ #
#  Hostile indexes are refused - and the outside file stays unopened   #
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("shard", [
    "../../outside/SECRET.safetensors",
    "../outside/SECRET.safetensors",
    "sub/../../outside/SECRET.safetensors",
])
def test_relative_traversal_is_refused(tmp_path, shard):
    secret = _secret(tmp_path)
    d = _model_dir(tmp_path, {"a.weight": shard})

    opened = []
    real_open = Path.open

    def spy(self, *a, **kw):
        opened.append(str(self))
        return real_open(self, *a, **kw)

    exc = None
    with mock.patch.object(Path, "open", spy):
        try:
            safety.validate_shard_index(str(d))
        except RuntimeError as e:
            exc = e

    # Assert on the WORLD before the exception: a refusal that still read the
    # file has not protected anything.
    assert str(secret) not in opened, f"the outside file was opened: {opened}"
    assert exc is not None, f"traversal {shard!r} was NOT refused"
    assert "outside the model directory" in str(exc)


def test_absolute_path_is_refused(tmp_path):
    secret = _secret(tmp_path)
    d = _model_dir(tmp_path, {"a.weight": str(secret)})
    exc = None
    try:
        safety.validate_shard_index(str(d))
    except RuntimeError as e:
        exc = e
    assert secret.read_text(encoding="utf-8") == "TOP SECRET"
    assert exc is not None, "an absolute shard path was NOT refused"


def test_unc_path_is_refused(tmp_path):
    d = _model_dir(tmp_path, {"a.weight": "//server/share/x.safetensors"})
    with pytest.raises(RuntimeError):
        safety.validate_shard_index(str(d))


def test_empty_shard_name_is_refused(tmp_path):
    d = _model_dir(tmp_path, {"a.weight": ""})
    with pytest.raises(RuntimeError):
        safety.validate_shard_index(str(d))


def test_non_string_shard_is_refused(tmp_path):
    d = _model_dir(tmp_path, {"a.weight": ["x.safetensors"]})
    with pytest.raises(RuntimeError, match="non-string"):
        safety.validate_shard_index(str(d))


def test_pytorch_bin_index_is_checked_too(tmp_path):
    d = _model_dir(tmp_path, {"a.weight": "../../outside/SECRET.safetensors"},
                   name="pytorch_model.bin.index.json")
    with pytest.raises(RuntimeError):
        safety.validate_shard_index(str(d))


def test_index_one_level_down_is_checked(tmp_path):
    d = tmp_path / "models" / "nested"
    (d / "sub").mkdir(parents=True)
    (d / "sub" / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": {"a": "../../../outside/SECRET.safetensors"}}),
        encoding="utf-8")
    with pytest.raises(RuntimeError):
        safety.validate_shard_index(str(d))


# ------------------------------------------------------------------ #
#  The gate runs in HFBackend.load(), before any child is spawned      #
# ------------------------------------------------------------------ #

def test_load_refuses_before_spawning_a_child(tmp_path):
    from localm.inference.backends import hf as hf_backend

    _secret(tmp_path)
    d = _model_dir(tmp_path, {"a.weight": "../../outside/SECRET.safetensors"})

    backend = hf_backend.HFBackend.__new__(hf_backend.HFBackend)
    backend.model_path = str(d)
    backend._device = "cpu"

    runner = mock.MagicMock()
    exc = None
    with mock.patch.object(hf_backend, "_check_custom_code_allowed", lambda *_a, **_k: None), \
         mock.patch("localm.inference.hf_tokenizer_safety.validate_tokenizer_json",
                    lambda *_a, **_k: None), \
         mock.patch.object(hf_backend, "HFRunner", runner):
        try:
            backend.load()
        except RuntimeError as e:
            exc = e

    # The world first: a refusal that still spawned the child would have handed
    # the hostile directory to transformers anyway.
    assert runner.call_count == 0, "a child was spawned for a hostile shard index"
    assert exc is not None, "HFBackend.load() did NOT refuse the hostile index"
    assert "outside the model directory" in str(exc)


def _hub_sharded(tmp_path: Path, blobs_repo: str = "models--org--m") -> Path:
    snap = tmp_path / "hub" / "models--org--m" / "snapshots" / "rev1"
    snap.mkdir(parents=True)
    blobs = tmp_path / "hub" / blobs_repo / "blobs"
    blobs.mkdir(parents=True, exist_ok=True)
    (blobs / "h1").write_text(json.dumps(
        {"weight_map": {"a.weight": "model-00001-of-00001.safetensors"}}), encoding="utf-8")
    (blobs / "h2").write_bytes(b"x")
    try:
        for name, blob in (("model.safetensors.index.json", "h1"),
                           ("model-00001-of-00001.safetensors", "h2")):
            (snap / name).symlink_to(os.path.relpath(blobs / blob, snap))
    except (OSError, NotImplementedError) as e:
        pytest.skip(f"symlink creation not permitted here: {e}")
    return snap


def test_hub_cache_snapshot_with_shards_linked_into_its_blobs_passes(tmp_path):
    safety.validate_shard_index(str(_hub_sharded(tmp_path)))


def test_hub_cache_shard_linked_into_another_repos_blobs_is_refused(tmp_path):
    with pytest.raises(RuntimeError, match="outside the model directory"):
        safety.validate_shard_index(str(_hub_sharded(tmp_path, "models--other--repo")))


@pytest.mark.parametrize("shard", [
    "\\\\attacker.invalid\\share\\x.safetensors", "//attacker.invalid/share/x.safetensors",
    "\\\\?\\UNC\\attacker.invalid\\share\\x.safetensors",
])
def test_weight_files_refuses_a_network_shard_without_touching_it(tmp_path, monkeypatch, shard):
    from localm.inference.backends import _hf_fp8
    touched = []
    for name in ("resolve", "is_file", "stat", "is_symlink"):
        real = getattr(Path, name)

        def spy(self, *a, _real=real, **k):
            if "attacker.invalid" in str(self):
                touched.append(str(self))
                raise OSError("network path refused by the test")
            return _real(self, *a, **k)
        monkeypatch.setattr(Path, name, spy)
    d = _model_dir(tmp_path, {"a.weight": shard})
    assert _hf_fp8.weight_files(str(d)) == []
    assert touched == []


def test_weight_files_names_the_shards_of_a_hub_cache_snapshot(tmp_path):
    from localm.inference.backends import _hf_fp8
    snap = _hub_sharded(tmp_path)
    files = _hf_fp8.weight_files(str(snap))
    assert [f.name for f in files] == ["model-00001-of-00001.safetensors"]
    assert files[0].read_bytes() == b"x"


def test_hub_cache_shard_linked_outside_any_blobs_is_refused(tmp_path):
    snap = _hub_sharded(tmp_path)
    secret = _secret(tmp_path)
    shard = snap / "model-00001-of-00001.safetensors"
    shard.unlink()
    shard.symlink_to(secret)
    with pytest.raises(RuntimeError, match="outside the model directory"):
        safety.validate_shard_index(str(snap))

# SPDX-License-Identifier: AGPL-3.0-or-later
"""``_model_file_size()`` (localm/inference/http_server.py) - the size feeding
``_gpu_registry_sync``'s ``vram_estimate_bytes``, a field OTHER localm
instances read as "how much VRAM does this peer hold".

The one property under test: a directory resolve that finds NO files (an
empty directory, or one whose real weights sit somewhere ``rglob`` does not
reach) must report None ("not measured"), never a 0 ("measured as zero
bytes") - the two mean different things to a reader of the registry.

Narrower than residency.model_footprint_bytes, which always returns an int
because its caller needs a numeric decision input. The two are NOT expected to
agree on the empty-directory case.
"""

from localm.inference import http_server as hs


def _patch_get_model_info(monkeypatch, path):
    monkeypatch.setattr(
        "localm.model_manager.get_model_info",
        lambda name: (str(path), None) if path is not None else None)


def test_single_file_returns_its_real_size(tmp_path, monkeypatch):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"x" * 4096)
    _patch_get_model_info(monkeypatch, model)
    assert hs._model_file_size("m") == 4096


def test_single_file_that_is_genuinely_empty_returns_zero_not_none(tmp_path, monkeypatch):
    """A single FILE that stat() confirms is 0 bytes is a real answer (unlike
    the directory case below, where 0 would mean "found nothing to sum") and
    must stay 0, never become None."""
    model = tmp_path / "model.gguf"
    model.write_bytes(b"")
    _patch_get_model_info(monkeypatch, model)
    assert hs._model_file_size("m") == 0


def test_directory_sums_real_shard_sizes(tmp_path, monkeypatch):
    d = tmp_path / "model-dir"
    d.mkdir()
    (d / "shard1.gguf").write_bytes(b"x" * 1000)
    (d / "shard2.gguf").write_bytes(b"x" * 2000)
    _patch_get_model_info(monkeypatch, d)
    assert hs._model_file_size("m") == 3000


def test_empty_directory_returns_none_not_a_suspicious_zero(tmp_path, monkeypatch):
    """An empty directory (or one whose weights rglob never reaches) must
    report "not measured" (None), not a zero that a reader could mistake for
    "this peer legitimately holds 0 bytes of model on GPU"."""
    d = tmp_path / "empty-model-dir"
    d.mkdir()
    _patch_get_model_info(monkeypatch, d)
    assert hs._model_file_size("m") is None


def test_directory_with_only_subdirectories_returns_none(tmp_path, monkeypatch):
    """rglob('*') matches the subdirectory entries themselves, but the
    is_file() filter drops every one of them - same "nothing measured" shape
    as a flat empty directory, reached via a different path through the
    generator."""
    d = tmp_path / "model-dir"
    (d / "nested" / "deeper").mkdir(parents=True)
    _patch_get_model_info(monkeypatch, d)
    assert hs._model_file_size("m") is None


def test_unregistered_model_returns_none(monkeypatch):
    monkeypatch.setattr("localm.model_manager.get_model_info", lambda name: None)
    assert hs._model_file_size("does-not-exist") is None


def test_resolved_path_that_no_longer_exists_returns_none(tmp_path, monkeypatch):
    missing = tmp_path / "gone.gguf"
    _patch_get_model_info(monkeypatch, missing)
    assert hs._model_file_size("m") is None

# SPDX-License-Identifier: AGPL-3.0-or-later
"""A re-provision downloads and verifies the new build in a staging directory
before it touches the installed runtime, so a failed download (offline, a bad
archive) leaves the working runtime in place."""

import pytest

import localm.setup_llama as sl
from localm.setup_llama.download import ArtifactError


def _installed(target):
    target.mkdir(parents=True)
    (target / sl._lib_name()).write_bytes(b"working runtime")
    (target / "ggml-base.dll").write_bytes(b"old ggml")
    (target / ".localm-backend").write_text("cpu b11757", encoding="utf-8")
    (target / ".localm-backend-choice").write_text('{"requested": "vulkan"}', encoding="utf-8")
    (target / ".gitkeep").write_text("", encoding="utf-8")
    return {p.name: p.read_bytes() for p in target.iterdir()}


def _offline(monkeypatch):
    def no_network(url, dest, *a, **k):
        raise ArtifactError(f"could not download {url}: network is unreachable")
    monkeypatch.setattr(sl, "_download", no_network)
    monkeypatch.setattr(sl, "_resolve_backend_asset",
                        lambda backend, cuda_line=None, tag=None:
                        (f"https://example.invalid/{backend}.zip", None, "b11928"))
    monkeypatch.setattr(sl, "_install_runtime_wheel", lambda pkg: True)
    monkeypatch.setattr(sl.sys.stdin, "isatty", lambda: False)


def test_failed_download_keeps_the_runtime(tmp_path, monkeypatch):
    target = tmp_path / "lib"
    before = _installed(target)
    _offline(monkeypatch)
    with pytest.raises(SystemExit):
        sl._provision_with_fallback("cpu", target, None, False, assume_yes=True)
    assert {p.name: p.read_bytes() for p in target.iterdir()} == before
    assert not sl._staging_path(target).exists()


def test_failed_url_download_keeps_the_runtime(tmp_path, monkeypatch):
    target = tmp_path / "lib"
    before = _installed(target)
    _offline(monkeypatch)
    with pytest.raises(SystemExit):
        sl.cli._provision_from_url("https://example.invalid/x.zip", target,
                                   sl._lib_name(), None)
    assert {p.name: p.read_bytes() for p in target.iterdir()} == before


def test_a_staged_build_replaces_the_runtime_and_keeps_the_choice_record(tmp_path):
    target = tmp_path / "lib"
    _installed(target)
    (target / "rocblas").mkdir()
    (target / "rocblas" / "old.dat").write_bytes(b"old")

    def fill(staging):
        (staging / sl._lib_name()).write_bytes(b"new runtime")
        (staging / "rocblas").mkdir()
        (staging / "rocblas" / "new.dat").write_bytes(b"new")
        return "b11928"

    assert sl._replace_target(target, fill) == "b11928"
    names = sorted(p.name for p in target.iterdir())
    assert names == sorted([sl._lib_name(), ".localm-backend-choice", ".gitkeep", "rocblas"])
    assert (target / sl._lib_name()).read_bytes() == b"new runtime"
    assert [p.name for p in (target / "rocblas").iterdir()] == ["new.dat"]
    assert not sl._staging_path(target).exists()


def test_a_fill_that_raises_changes_nothing(tmp_path):
    target = tmp_path / "lib"
    before = _installed(target)

    def fill(staging):
        (staging / sl._lib_name()).write_bytes(b"half")
        raise ArtifactError("sha256 mismatch")

    with pytest.raises(ArtifactError):
        sl._replace_target(target, fill)
    assert {p.name: p.read_bytes() for p in target.iterdir()} == before
    assert not sl._staging_path(target).exists()


def _snapshot(target):
    return {p.name: p.read_bytes() for p in target.iterdir() if p.is_file()}


def _fill_with(content):
    def fill(staging):
        (staging / sl._lib_name()).write_bytes(content)
        (staging / ".localm-backend").write_text("vulkan b99999", encoding="utf-8")
        return "b99999"
    return fill


def test_a_build_that_does_not_load_is_replaced_by_the_one_before(tmp_path):
    target = tmp_path / "lib"
    before = _installed(target)
    failures = []
    with pytest.raises(SystemExit) as exc:
        with sl.keep_previous_runtime(target, on_failure=lambda: failures.append(1)):
            sl._replace_target(target, _fill_with(b"new build that does not load"))
            sl._replace_target(target, _fill_with(b"fallback that does not load either"))
            sl.sys.exit(1)
    assert exc.value.code == 1
    assert _snapshot(target) == before
    assert failures == [1]
    assert not sl._backup_path(target).exists()


def test_a_build_that_loads_is_kept_and_the_old_one_deleted(tmp_path):
    target = tmp_path / "lib"
    _installed(target)
    failures = []
    with sl.keep_previous_runtime(target, on_failure=lambda: failures.append(1)):
        sl._replace_target(target, _fill_with(b"new working build"))
    assert (target / sl._lib_name()).read_bytes() == b"new working build"
    assert (target / ".localm-backend").read_text(encoding="utf-8") == "vulkan b99999"
    assert not (target / "ggml-base.dll").exists()
    assert failures == []
    assert not sl._backup_path(target).exists()


def test_an_exception_restores_the_runtime_too(tmp_path):
    target = tmp_path / "lib"
    before = _installed(target)
    with pytest.raises(RuntimeError):
        with sl.keep_previous_runtime(target):
            sl._replace_target(target, _fill_with(b"new"))
            raise RuntimeError("load test crashed")
    assert _snapshot(target) == before


def test_a_failure_before_any_replace_still_calls_on_failure(tmp_path):
    target = tmp_path / "lib"
    before = _installed(target)
    failures = []
    with pytest.raises(SystemExit):
        with sl.keep_previous_runtime(target, on_failure=lambda: failures.append(1)):
            sl.sys.exit(1)
    assert _snapshot(target) == before
    assert failures == [1]


def test_a_failed_provision_puts_the_pin_back(monkeypatch):
    import localm.config as config
    store = {"llama_runtime_pin": "b11000"}
    monkeypatch.setattr(config, "load_config", lambda *a, **k: dict(store))

    def _update(fn, *a, **k):
        fn(store)
    monkeypatch.setattr(config, "update_config", _update)
    before = sl.cli._stored_pin()
    store["llama_runtime_pin"] = "b12345"
    sl.cli._restore_pin(before)
    assert store["llama_runtime_pin"] == "b11000"
    store.pop("llama_runtime_pin")
    sl.cli._restore_pin(None)
    assert "llama_runtime_pin" not in store

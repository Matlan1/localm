# SPDX-License-Identifier: AGPL-3.0-or-later
"""#957: a GUI/MCP pull of a vision GGUF had no way to pass --mmproj, so the
projector was never fetched and the model downloaded silently unable to see
an image. _pull_gguf_file now checks the HF repo's OWN file listing for an
mmproj sibling at pull time (free metadata call) and, when found, fetches and
records it on the registry entry - no CLI flag required. An explicit --mmproj
(mmproj_spec) still wins when given (never silently override a user choice).
"""

import struct
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from localm import model_manager as mm


# ---------------------------------------------------------------------------
# Minimal-but-structurally-valid GGUF bytes, same layout _gguf_metadata_probe
# parses (mirrors tests/test_model_type_detection.py::_build_gguf_bytes).
# ---------------------------------------------------------------------------

def _gguf_bytes(architecture: str) -> bytes:
    def _kv_string(key: str, value: str) -> bytes:
        kb, vb = key.encode("utf-8"), value.encode("utf-8")
        return (struct.pack("<Q", len(kb)) + kb
                + struct.pack("<I", 8)          # GGUF_TYPE_STRING
                + struct.pack("<Q", len(vb)) + vb)

    buf = bytearray()
    buf += b"GGUF"
    buf += struct.pack("<I", 3)                 # version 3
    buf += struct.pack("<Q", 0)                 # tensor_count
    buf += struct.pack("<Q", 1)                 # kv_count
    buf += _kv_string("general.architecture", architecture)
    buf += b"\x00" * 64                          # padding, never parsed
    return bytes(buf)


_LLM_BYTES = _gguf_bytes("llama")
_CLIP_BYTES = _gguf_bytes("clip")


@pytest.fixture()
def fake_registry(tmp_path, monkeypatch):
    """In-memory registry + temp MODELS_DIR wired into model_manager (mirrors
    test_model_manager_phase3.py's fixture of the same name)."""
    store: dict = {}
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    monkeypatch.setattr(mm, "MODELS_DIR", models_dir)
    monkeypatch.setattr(mm, "ensure_dirs", lambda: None)
    monkeypatch.setattr(mm, "_check_disk_space", lambda *a, **k: True)
    monkeypatch.setattr(mm, "load_registry", lambda: dict(store))

    def _save(reg):
        store.clear()
        store.update(reg)

    monkeypatch.setattr(mm, "save_registry", _save)

    def _update(mutator):
        reg = dict(store)
        mutator(reg)
        store.clear()
        store.update(reg)
        return dict(store)

    monkeypatch.setattr(mm, "update_registry", _update)
    monkeypatch.setattr(mm, "_hf_file_sha256", lambda repo, fn: None)
    monkeypatch.setattr("requests.head", lambda *a, **k: MagicMock(
        headers={"content-length": "16"}))
    return store, models_dir


def _wire_repo_listing(monkeypatch, files):
    class _FakeHfApi:
        def list_repo_files(self, repo_id):
            return files

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "HfApi", _FakeHfApi)


def _wire_download(monkeypatch, bytes_by_filename: dict, default: bytes = _LLM_BYTES):
    downloaded = []

    def _fake_download(repo_id, filename, local_dir, **kw):
        downloaded.append(filename)
        p = Path(local_dir) / filename
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(bytes_by_filename.get(filename, default))
        return str(p)

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _fake_download)
    return downloaded


class TestAutoDetectSingleCandidate:
    def test_single_repo_mmproj_auto_fetched_and_registered(
            self, fake_registry, monkeypatch):
        store, models_dir = fake_registry
        _wire_repo_listing(monkeypatch, ["main.gguf", "mmproj-main-f16.gguf"])
        _wire_download(monkeypatch, {"mmproj-main-f16.gguf": _CLIP_BYTES})

        ok = mm._pull_gguf_file("o/r:main.gguf", None)

        assert ok is True
        assert store["main"]["model_type"] == "llm"
        assert store["main"]["mmproj"] == str((models_dir / "mmproj-main-f16.gguf").resolve())
        assert (models_dir / "mmproj-main-f16.gguf").is_file()

    def test_already_downloaded_branch_also_attaches_mmproj(
            self, fake_registry, monkeypatch):
        """The fast 'Already downloaded' path must get the same treatment as a
        fresh download - otherwise a re-pull of an already-present vision GGUF
        never picks up its projector."""
        store, models_dir = fake_registry
        (models_dir / "main.gguf").write_bytes(_LLM_BYTES)
        _wire_repo_listing(monkeypatch, ["main.gguf", "mmproj-main-f16.gguf"])
        _wire_download(monkeypatch, {"mmproj-main-f16.gguf": _CLIP_BYTES})

        ok = mm._pull_gguf_file("o/r:main.gguf", None)

        assert ok is True
        assert store["main"]["mmproj"] == str((models_dir / "mmproj-main-f16.gguf").resolve())


class TestNoCandidate:
    def test_vision_looking_name_with_no_projector_warns(
            self, fake_registry, monkeypatch, capsys):
        store, _ = fake_registry
        _wire_repo_listing(monkeypatch, ["Qwen3-VL-8B.gguf"])
        _wire_download(monkeypatch, {})

        ok = mm._pull_gguf_file("mrader/Qwen3-VL-8B-GGUF:Qwen3-VL-8B.gguf", None)

        assert ok is True
        assert "mmproj" not in store["Qwen3-VL-8B"]
        out = capsys.readouterr().out.lower()
        assert "vision" in out and "projector" in out

    def test_listing_api_failure_is_silent_not_a_false_negative(
            self, fake_registry, monkeypatch, capsys):
        """A repo-listing FAILURE (network hiccup, HF API error) must never be
        read as 'this repo has no projector' - that would print a false 'no
        vision projector found' note on an unmeasured premise. Distinguishing
        the two matters even though the functional outcome (no auto-fetch) is
        the same either way."""
        store, _ = fake_registry

        class _FailingHfApi:
            def list_repo_files(self, repo_id):
                raise RuntimeError("simulated HF API outage")

        import huggingface_hub
        monkeypatch.setattr(huggingface_hub, "HfApi", _FailingHfApi)
        _wire_download(monkeypatch, {})

        ok = mm._pull_gguf_file("mrader/Qwen3-VL-8B-GGUF:Qwen3-VL-8B.gguf", None)

        assert ok is True
        assert "mmproj" not in store["Qwen3-VL-8B"]
        out = capsys.readouterr().out.lower()
        assert "projector" not in out

    def test_non_vision_name_with_no_projector_is_silent(
            self, fake_registry, monkeypatch, capsys):
        store, _ = fake_registry
        _wire_repo_listing(monkeypatch, ["plain-chat-model.gguf"])
        _wire_download(monkeypatch, {})

        ok = mm._pull_gguf_file("o/r:plain-chat-model.gguf", None)

        assert ok is True
        assert "mmproj" not in store["plain-chat-model"]
        out = capsys.readouterr().out.lower()
        assert "projector" not in out


class TestAmbiguousMultipleCandidates:
    def test_stem_match_disambiguates(self, fake_registry, monkeypatch):
        store, models_dir = fake_registry
        _wire_repo_listing(monkeypatch, [
            "modelA.gguf", "mmproj-modelA-f16.gguf", "mmproj-modelB-f16.gguf"])
        _wire_download(monkeypatch, {
            "mmproj-modelA-f16.gguf": _CLIP_BYTES,
            "mmproj-modelB-f16.gguf": _CLIP_BYTES,
        })

        ok = mm._pull_gguf_file("o/r:modelA.gguf", None)

        assert ok is True
        assert store["modelA"]["mmproj"] == str((models_dir / "mmproj-modelA-f16.gguf").resolve())

    def test_unresolvable_ambiguity_falls_back_to_f16(self, fake_registry, monkeypatch):
        """Both candidates share the model's own leading token (same-repo
        quant variants of the SAME projector), so the stem heuristic alone
        can't narrow it - unlike a cross-model directory glob, every
        candidate here is already scoped to the one repo being pulled, so we
        pick deterministically (prefer f16) instead of giving up."""
        store, models_dir = fake_registry
        _wire_repo_listing(monkeypatch, [
            "main.gguf", "mmproj-main-f16.gguf", "mmproj-main-q8_0.gguf"])
        _wire_download(monkeypatch, {
            "mmproj-main-f16.gguf": _CLIP_BYTES,
            "mmproj-main-q8_0.gguf": _CLIP_BYTES,
        })

        ok = mm._pull_gguf_file("o/r:main.gguf", None)

        assert ok is True
        assert store["main"]["mmproj"] == str((models_dir / "mmproj-main-f16.gguf").resolve())


class TestVerificationRejectsBadCandidate:
    def test_non_clip_candidate_is_not_attached(self, fake_registry, monkeypatch, capsys):
        """The filename match ('mmproj' substring) is only a heuristic - the
        downloaded bytes must pass the real GGUF-metadata check
        (gguf_is_mmproj) before being attached, or a bad file could silently
        become the model's projector."""
        store, models_dir = fake_registry
        _wire_repo_listing(monkeypatch, ["main.gguf", "mmproj-fake.gguf"])
        _wire_download(monkeypatch, {"mmproj-fake.gguf": _LLM_BYTES})  # NOT clip

        ok = mm._pull_gguf_file("o/r:main.gguf", None)

        assert ok is True
        assert "mmproj" not in store["main"]
        out = capsys.readouterr().out.lower()
        assert "does not look like a valid vision" in out


class TestSkippedScopes:
    def test_pulling_a_projector_itself_never_recurses(self, fake_registry, monkeypatch):
        """A file whose own name contains 'mmproj' must not trigger a search
        for ITS OWN companion projector."""
        store, _ = fake_registry
        list_spy = MagicMock(side_effect=AssertionError("must not be called"))

        class _FakeHfApi:
            list_repo_files = list_spy

        import huggingface_hub
        monkeypatch.setattr(huggingface_hub, "HfApi", _FakeHfApi)
        _wire_download(monkeypatch, {})

        ok = mm._pull_gguf_file("o/r:mmproj-standalone.gguf", None)

        assert ok is True
        list_spy.assert_not_called()

    def test_dest_dir_skips_autodetect(self, fake_registry, tmp_path, monkeypatch):
        """A ComfyUI-style dest_dir pull is not one of localm's own chat
        models - never spend a network call looking for a projector."""
        store, _ = fake_registry
        list_spy = MagicMock(side_effect=AssertionError("must not be called"))

        class _FakeHfApi:
            list_repo_files = list_spy

        import huggingface_hub
        monkeypatch.setattr(huggingface_hub, "HfApi", _FakeHfApi)
        _wire_download(monkeypatch, {})
        dest_dir = tmp_path / "comfy" / "unet"

        ok = mm._pull_gguf_file("o/r:main.gguf", None, dest_dir=dest_dir, register=True)

        assert ok is True
        list_spy.assert_not_called()

    def test_non_llm_registration_skips_autodetect(self, fake_registry, monkeypatch):
        """An embedding/mmproj/etc. pull cannot itself host a projector."""
        store, _ = fake_registry
        list_spy = MagicMock(side_effect=AssertionError("must not be called"))

        class _FakeHfApi:
            list_repo_files = list_spy

        import huggingface_hub
        monkeypatch.setattr(huggingface_hub, "HfApi", _FakeHfApi)
        _wire_download(monkeypatch, {})

        ok = mm._pull_gguf_file("o/r:embed.gguf", None, model_type="embedding")

        assert ok is True
        list_spy.assert_not_called()


class TestExplicitMmprojWins:
    def test_explicit_mmproj_overrides_autodetected_sibling(
            self, fake_registry, monkeypatch):
        """The repo being pulled DOES ship its own mmproj sibling, but the
        user explicitly named a different one via --mmproj - the explicit
        choice must win, never be silently replaced by auto-detection."""
        store, models_dir = fake_registry
        _wire_repo_listing(monkeypatch, ["main.gguf", "mmproj-main-f16.gguf"])
        _wire_download(monkeypatch, {
            "mmproj-main-f16.gguf": _CLIP_BYTES,
            "custom-mmproj.gguf": _CLIP_BYTES,
        })

        ok = mm.pull_model("o/r:main.gguf", mmproj_spec="other/repo:custom-mmproj.gguf")

        assert ok is True
        assert store["main"]["mmproj"] == str((models_dir / "custom-mmproj.gguf").resolve())

    def test_explicit_mmproj_bad_spec_is_refused(self, fake_registry, monkeypatch, capsys):
        store, _ = fake_registry
        _wire_repo_listing(monkeypatch, ["main.gguf"])
        _wire_download(monkeypatch, {})

        ok = mm._pull_gguf_file("o/r:main.gguf", None, mmproj_spec="not-a-file-spec")

        assert ok is True   # the main model still pulls fine
        assert "mmproj" not in store["main"]
        out = capsys.readouterr().out.lower()
        assert "mmproj spec must be a specific file" in out

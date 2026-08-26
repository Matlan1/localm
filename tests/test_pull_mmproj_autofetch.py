# SPDX-License-Identifier: AGPL-3.0-or-later
"""A GUI/MCP pull of a vision GGUF has no way to pass --mmproj, so the projector
would never be fetched and the model would download silently unable to see an
image. _pull_gguf_file checks the HF repo's OWN file listing for an mmproj
sibling at pull time (free metadata call) and, when found, fetches and records it
on the registry entry - no CLI flag required. An explicit --mmproj (mmproj_spec)
still wins when given.
"""

import struct
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from localm import model_manager as mm


# ---------------------------------------------------------------------------
# Minimal-but-structurally-valid GGUF bytes, the same layout
# _gguf_metadata_probe parses.
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
        def __init__(self, *a, **kw):
            pass

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
            def __init__(self, *a, **kw):
                pass

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


class TestTraversalGuardOnRepoListing:
    """The repo file listing is REMOTE, untrusted input (a malicious or
    compromised repo could list anything). A candidate must be confined the same
    way an explicit --mmproj filename already is - filtered out before it is even
    considered, not merely rejected after being picked."""

    def test_unsafe_candidate_is_filtered_before_picking(
            self, fake_registry, monkeypatch):
        store, models_dir = fake_registry
        evil = "..\\..\\evil-mmproj.gguf"
        _wire_repo_listing(monkeypatch, ["main.gguf", evil, "mmproj-main-f16.gguf"])
        downloaded = _wire_download(monkeypatch, {"mmproj-main-f16.gguf": _CLIP_BYTES})

        ok = mm._pull_gguf_file("o/r:main.gguf", None)

        assert ok is True
        # The legit candidate still wins - an unsafe sibling in the listing
        # must not poison the whole auto-detect.
        assert store["main"]["mmproj"] == str((models_dir / "mmproj-main-f16.gguf").resolve())
        assert evil not in downloaded
        assert not (models_dir.parent / "evil-mmproj.gguf").exists()

    def test_only_unsafe_candidate_yields_no_projector_not_a_crash(
            self, fake_registry, monkeypatch, capsys):
        store, models_dir = fake_registry
        evil = "..\\..\\evil-mmproj.gguf"
        _wire_repo_listing(monkeypatch, ["main.gguf", evil])
        downloaded = _wire_download(monkeypatch, {})

        ok = mm._pull_gguf_file("o/r:main.gguf", None)

        assert ok is True
        assert "mmproj" not in store["main"]
        assert evil not in downloaded    # only the main model's own file downloads
        assert not (models_dir.parent / "evil-mmproj.gguf").exists()


class TestBareExplicitMmprojSpecRejectedCleanly:
    """A --mmproj value with no 'owner/repo' at all (just a bare filename) must
    be refused with the clean message, never an IndexError from the
    rsplit('/', 1) parse (that value has no '/' to split on)."""

    def test_bare_filename_mmproj_spec_does_not_crash(
            self, fake_registry, monkeypatch, capsys):
        store, _ = fake_registry
        _wire_repo_listing(monkeypatch, ["main.gguf"])
        _wire_download(monkeypatch, {})

        ok = mm._pull_gguf_file("o/r:main.gguf", None, mmproj_spec="justafile.gguf")

        assert ok is True
        assert "mmproj" not in store["main"]
        out = capsys.readouterr().out.lower()
        assert "mmproj spec must be a specific file" in out


class TestAmbiguityNeverCrossAttaches:
    """_pick_best_of_same_repo_mmprojs's 'same repo, near-certainly the same
    projector' trust assumption only holds once a candidate is already known
    to be about THIS model. Candidates that share no relation to the model's
    own name at all must never be guessed among."""

    def test_completely_unrelated_candidates_stay_unattached(
            self, fake_registry, monkeypatch):
        """Neither candidate's name has anything to do with 'main' - a repo
        listing two unrelated models' projectors must not resolve to either."""
        store, _ = fake_registry
        _wire_repo_listing(monkeypatch, [
            "main.gguf", "mmproj-alpha-f16.gguf", "mmproj-beta-f16.gguf"])
        _wire_download(monkeypatch, {
            "mmproj-alpha-f16.gguf": _CLIP_BYTES,
            "mmproj-beta-f16.gguf": _CLIP_BYTES,
        })

        ok = mm._pull_gguf_file("o/r:main.gguf", None)

        assert ok is True
        assert "mmproj" not in store["main"]


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
            def __init__(self, *a, **kw):
                pass
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
            def __init__(self, *a, **kw):
                pass
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
            def __init__(self, *a, **kw):
                pass
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


class TestSyncModelsDirBackfillsExistingEntry:
    """An entry pulled BEFORE the auto-attach existed already sits in the
    registry with no mmproj key, and a re-pull is not an acceptable fix: an
    already-pulled vision model must work just as a freshly pulled one.
    sync_models_dir notices and backfills it on its own, using the source the
    entry already recorded - a case a test that starts from an EMPTY registry
    and calls _pull_gguf_file cannot reach, since this one starts from an
    ALREADY-REGISTERED entry and calls sync_models_dir with no pull at all."""

    def _preexisting_entry(self, store, models_dir, name="main", source="hf:o/r"):
        (models_dir / f"{name}.gguf").write_bytes(_LLM_BYTES)
        store[name] = {
            "path": str((models_dir / f"{name}.gguf").resolve()),
            "source": source, "model_type": "llm",
        }

    def test_preexisting_entry_gets_mmproj_backfilled_with_no_repull(
            self, fake_registry, monkeypatch):
        store, models_dir = fake_registry
        self._preexisting_entry(store, models_dir)
        _wire_repo_listing(monkeypatch, ["main.gguf", "mmproj-main-f16.gguf"])
        _wire_download(monkeypatch, {"mmproj-main-f16.gguf": _CLIP_BYTES})

        result = mm.sync_models_dir()

        assert store["main"]["mmproj"] == str(
            (models_dir / "mmproj-main-f16.gguf").resolve())
        assert (models_dir / "mmproj-main-f16.gguf").is_file()
        assert result.mmproj_backfilled == 1
        assert result.changed is True

    def test_entry_already_carrying_mmproj_is_never_re_fetched(
            self, fake_registry, monkeypatch):
        """An entry with mmproj already recorded (even resolved-empty in some
        odd past state) must never be re-queried - matches the architecture/
        expert_count backfill's own 'key present, even falsy, means resolved'
        rule two blocks up."""
        store, models_dir = fake_registry
        self._preexisting_entry(store, models_dir)
        store["main"]["mmproj"] = str(models_dir / "already-set.gguf")
        called = []

        class _SpyHfApi:
            def __init__(self, *a, **kw):
                pass

            def list_repo_files(self, repo_id):
                called.append(repo_id)
                return ["main.gguf", "mmproj-main-f16.gguf"]

        import huggingface_hub
        monkeypatch.setattr(huggingface_hub, "HfApi", _SpyHfApi)

        result = mm.sync_models_dir()

        assert called == [], "an entry with mmproj already set must not be re-queried"
        assert store["main"]["mmproj"] == str(models_dir / "already-set.gguf")
        assert result.mmproj_backfilled == 0

    def test_repo_genuinely_has_no_projector_is_not_counted_as_backfilled(
            self, fake_registry, monkeypatch):
        """A repo that was actually checked (net_mode allowed it) but has no
        mmproj sibling at all is the legitimate silent case - distinct from
        net_mode blocking the check. `mmproj_backfilled` counts SUCCESSFUL
        attaches only: this attempt found nothing, wrote nothing to the
        registry, so it must not be counted, and `changed` must stay False -
        a no-op reconciliation pass must not report itself as having
        changed anything."""
        store, models_dir = fake_registry
        self._preexisting_entry(store, models_dir)
        # Pre-set architecture/expert_count so the unrelated architecture and
        # expert-count backfill does not also fire on this entry and set
        # `backfilled`, which would make `changed` True for a reason unrelated
        # to mmproj.
        store["main"]["architecture"] = "llama"
        store["main"]["expert_count"] = 0
        _wire_repo_listing(monkeypatch, ["main.gguf"])  # no mmproj sibling
        _wire_download(monkeypatch, {})

        result = mm.sync_models_dir()

        assert "mmproj" not in store["main"]
        assert result.mmproj_backfilled == 0
        assert result.changed is False

    def test_out_of_directory_result_is_refused_not_attached(
            self, fake_registry, monkeypatch, tmp_path):
        """Defense in depth: backfill_mmproj_for_entry's own result already
        passed a traversal guard in pull.py (_safe_models_filename, several
        call-frames away), but sync_models_dir re-verifies locally, at the
        exact point an HF-repo-derived path is written into the registry,
        rather than trusting a distant caller unconditionally. A result
        outside the model's own directory (simulating a bug or a
        compromised/unexpected return) must be refused, not attached."""
        store, models_dir = fake_registry
        self._preexisting_entry(store, models_dir)
        outside = tmp_path / "elsewhere" / "evil.gguf"
        outside.parent.mkdir(parents=True)
        outside.write_bytes(_CLIP_BYTES)

        import localm.model_manager.pull as pull_mod
        monkeypatch.setattr(pull_mod, "backfill_mmproj_for_entry",
                            lambda entry, path: outside)

        result = mm.sync_models_dir()

        assert "mmproj" not in store["main"]
        assert result.mmproj_backfilled == 0

    def test_non_hf_source_is_never_a_candidate(self, fake_registry, monkeypatch):
        """A locally-added model (source='local' or similar) has no repo to
        even check - must not crash or attempt a listing."""
        store, models_dir = fake_registry
        self._preexisting_entry(store, models_dir, source="local")
        called = []

        class _SpyHfApi:
            def __init__(self, *a, **kw):
                pass

            def list_repo_files(self, repo_id):
                called.append(repo_id)
                return []

        import huggingface_hub
        monkeypatch.setattr(huggingface_hub, "HfApi", _SpyHfApi)

        result = mm.sync_models_dir()

        assert called == []
        assert "mmproj" not in store["main"]
        assert result.mmproj_backfilled == 0

    def test_net_mode_off_blocks_the_fetch_and_names_the_model_precisely(
            self, fake_registry, monkeypatch):
        """The ONE case that must be loud: net_mode=off blocks the check, and
        the resulting note names the model and net_mode by name - never
        collapsed with the silent 'looked and found nothing' outcome."""
        store, models_dir = fake_registry
        self._preexisting_entry(store, models_dir)
        called = []

        class _SpyHfApi:
            def __init__(self, *a, **kw):
                pass

            def list_repo_files(self, repo_id):
                called.append(repo_id)
                return []

        import huggingface_hub
        monkeypatch.setattr(huggingface_hub, "HfApi", _SpyHfApi)
        monkeypatch.setattr("localm.netpolicy.network_mode", lambda: "off")

        result = mm.sync_models_dir()

        assert called == [], "net_mode=off must block the check itself, not just the download"
        assert "mmproj" not in store["main"]
        assert result.mmproj_backfilled == 0
        assert "main" in result.note
        assert "net_mode" in result.note

    def test_net_allow_model_downloads_bypasses_off_for_the_backfill(
            self, fake_registry, monkeypatch):
        """net_allow_model_downloads exempts the retroactive backfill from the
        off floor too - the same override an explicit pull respects. Asserted
        on the real fetch happening (not just a non-empty note), the same
        discipline as the sibling test above."""
        store, models_dir = fake_registry
        self._preexisting_entry(store, models_dir)
        _wire_repo_listing(monkeypatch, ["main.gguf", "mmproj-main-f16.gguf"])
        _wire_download(monkeypatch, {"mmproj-main-f16.gguf": _CLIP_BYTES})
        monkeypatch.setattr("localm.netpolicy.network_mode", lambda: "off")
        monkeypatch.setattr("localm.netpolicy.downloads_allowed_when_off", lambda: True)

        result = mm.sync_models_dir()

        assert store["main"]["mmproj"] == str(
            (models_dir / "mmproj-main-f16.gguf").resolve())
        assert result.mmproj_backfilled == 1
        assert not result.note, "a successful backfill has nothing to warn about"

    def test_net_mode_ask_still_backfills_no_half_measure_behind_a_setting(
            self, fake_registry, monkeypatch):
        """The backfill must hold under the DEFAULT net_mode ("ask"), not only for
        installs that separately opted into net_mode=allow - matching
        _pull_gguf_file's own net_mode gate for this identical operation on an
        explicit pull."""
        store, models_dir = fake_registry
        self._preexisting_entry(store, models_dir)
        _wire_repo_listing(monkeypatch, ["main.gguf", "mmproj-main-f16.gguf"])
        _wire_download(monkeypatch, {"mmproj-main-f16.gguf": _CLIP_BYTES})
        monkeypatch.setattr("localm.netpolicy.network_mode", lambda: "ask")

        result = mm.sync_models_dir()

        assert store["main"]["mmproj"] == str(
            (models_dir / "mmproj-main-f16.gguf").resolve())
        assert result.mmproj_backfilled == 1

    def test_backfill_is_capped_per_call(self, fake_registry, monkeypatch):
        store, models_dir = fake_registry
        for i in range(5):
            self._preexisting_entry(store, models_dir, name=f"m{i}", source=f"hf:o/r{i}")
        _wire_repo_listing(monkeypatch, ["main.gguf", "mmproj-main-f16.gguf"])
        _wire_download(monkeypatch, {"mmproj-main-f16.gguf": _CLIP_BYTES})

        result = mm.sync_models_dir()

        assert result.mmproj_backfilled == 3, "capped at _MMPROJ_BACKFILL_CAP, not all 5 at once"

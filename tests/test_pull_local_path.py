# SPDX-License-Identifier: AGPL-3.0-or-later
"""`localm pull <local path>` must register the file in place instead of
mis-parsing a Windows drive-colon as owner/repo:file or rejecting "Unknown
spec".
"""

from pathlib import Path

import pytest

import localm.model_manager as mm
from localm.config import load_registry
from localm.model_manager import add_local, pull_model


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    # load_registry/save_registry read the module-level REGISTRY_FILE frozen at
    # import, so the autouse LOCALM_HOME env alone does not isolate them.
    # Redirect the config paths to a throwaway dir.
    import localm.config as cfg
    home = tmp_path / ".localm"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.setattr(cfg, "HOME_DIR", home)
    monkeypatch.setattr(cfg, "MODELS_DIR", home / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", home / "registry.json")
    return home


def _gguf(tmp_path, name="m.gguf"):
    # Unique content per name -> unique sha256, so two fixtures never collide on
    # the content-dedup path.
    p = tmp_path / name
    p.write_bytes(b"GGUF\x00\x00\x00\x00" + name.encode())
    return p


class TestAddLocalReturnsBool:
    def test_gguf_returns_true_and_registers(self, tmp_path, isolated_home):
        assert add_local(str(_gguf(tmp_path))) is True
        assert "m" in load_registry()

    def test_missing_returns_false(self, tmp_path, isolated_home):
        assert add_local(str(tmp_path / "nope.gguf")) is False

    def test_non_model_returns_false(self, tmp_path, isolated_home):
        bad = tmp_path / "notes.txt"
        bad.write_text("hello")
        assert add_local(str(bad)) is False


class TestPullByLocalPath:
    def test_pull_registers_local_gguf(self, tmp_path, isolated_home):
        p = _gguf(tmp_path, "mymodel.gguf")
        assert pull_model(str(p)) is True
        assert "mymodel" in load_registry()

    def test_pull_non_model_path_returns_false(self, tmp_path, isolated_home):
        bad = tmp_path / "data.txt"
        bad.write_text("x")
        assert pull_model(str(bad)) is False

    def test_pull_hf_spec_not_treated_as_local(self, monkeypatch):
        # a bare owner/repo must still route to HF, never to add_local.
        called = {"add_local": False, "hf": False}
        monkeypatch.setattr(mm, "add_local",
                            lambda *a, **k: called.__setitem__("add_local", True) or True)
        monkeypatch.setattr(mm, "_pull_hf_snapshot",
                            lambda *a, **k: called.__setitem__("hf", True) or True)
        pull_model("owner/repo")
        assert called["hf"] is True
        assert called["add_local"] is False


class TestPullLocalSha256:
    """A user-supplied --sha256 on a LOCAL pull is a safety assertion: it must
    be verified against the real bytes, never silently ignored while reporting
    success."""

    def _sha(self, p):
        import hashlib
        return hashlib.sha256(p.read_bytes()).hexdigest()

    def test_matching_hash_verifies_and_registers(self, tmp_path, isolated_home):
        p = _gguf(tmp_path, "okmodel.gguf")
        assert pull_model(str(p), expected_sha256=self._sha(p)) is True
        assert "okmodel" in load_registry()

    def test_wrong_hash_refuses_and_does_not_register(self, tmp_path, isolated_home):
        p = _gguf(tmp_path, "badmodel.gguf")
        assert pull_model(str(p), expected_sha256="de" * 32) is False
        # The safety assertion failed -> the model must NOT be registered.
        assert "badmodel" not in load_registry()

    def test_uppercase_hash_matches(self, tmp_path, isolated_home):
        p = _gguf(tmp_path, "casemodel.gguf")
        assert pull_model(str(p), expected_sha256=self._sha(p).upper()) is True

    def test_directory_with_sha256_refused(self, tmp_path, isolated_home):
        # An HF model dir has no single digest; --sha256 on it is refused, not
        # silently ignored.
        d = tmp_path / "hfmodel"
        d.mkdir()
        (d / "config.json").write_text('{"architectures":["LlamaForCausalLM"]}')
        (d / "tokenizer.json").write_text("{}")
        assert pull_model(str(d), expected_sha256="de" * 32) is False


class TestSafeFilenameFailsClosed:
    """The download-destination guard must fail CLOSED (return None) on unsafe
    input, never raise: a NUL byte must not escape the OSError-only guard as an
    uncaught ValueError."""

    def test_nul_byte_returns_none(self, isolated_home):
        assert mm._safe_models_filename("evil\x00.gguf") is None

    def test_nul_in_pull_spec_no_traceback(self, tmp_path, isolated_home, monkeypatch):
        # A NUL in the gguf spec must be rejected cleanly (False), never crash.
        monkeypatch.setattr("localm.netpolicy.network_mode", lambda: "ask")
        assert pull_model("owner/repo:evil\x00.gguf") is False

    @pytest.mark.parametrize("good", ["model.gguf", "a-b_c.1.gguf"])
    def test_normal_names_still_pass(self, good, isolated_home):
        assert mm._safe_models_filename(good) == good


class TestSafeFilenameRejectsWindowsHazardClass:
    """A filename that passes the single-component and no-traversal checks can
    still be a Windows filename-CONFUSION hazard rather than a directory
    escape: it stays confined to base_dir, but names something other than what
    it appears to.

    - 'somefile.exe:mmproj.gguf' opens an NTFS Alternate Data Stream. The write
      succeeds, `os.listdir` shows only a 0-byte 'somefile.exe', and
      gguf_is_mmproj's hard-verify (an ordinary open()) transparently reads the
      hidden stream too - so a crafted CLIP-architecture payload placed there
      passes BOTH the filename guard and the metadata hard-verify.
    - 'evil.gguf.' / 'evil.gguf ' - Windows silently strips a trailing dot or
      space when resolving a path, so these name the SAME file as 'evil.gguf':
      a download using one of these forms would overwrite an existing model,
      with the collision invisible in any directory listing, which only ever
      shows the stripped form."""

    @pytest.mark.parametrize("hazard", [
        "somefile.exe:mmproj.gguf",     # NTFS ADS
        "evil.gguf:hidden",             # ADS on a plausible model name
        "evil.gguf.",                   # trailing dot - silent collision
        "evil.gguf ",                   # trailing space - silent collision
        "CON",                          # bare reserved device name
        "CON.gguf",                     # reserved device name with extension
        "con.mmproj.gguf",              # reserved stem before the FIRST dot
        "NUL.gguf",
        "COM1.gguf",
        "LPT1.gguf",
        "evil<file.gguf",
        "evil>file.gguf",
        "evil|file.gguf",
        "evil?file.gguf",
        "evil*file.gguf",
        'evil"file.gguf',
    ])
    def test_hazard_rejected(self, hazard, isolated_home):
        assert mm._safe_models_filename(hazard) is None

    @pytest.mark.parametrize("legit", [
        "Qwen3-VL-8B-Instruct-abliterated.Q2_K.gguf",
        "mmproj-model-f16.gguf",
        "model-00001-of-00005.gguf",
        "a.b.c.gguf",
        "CONTROL.gguf",   # starts with "CON" but its stem is "CONTROL", not "CON"
        "CONSOLE.gguf",
    ])
    def test_legitimate_name_still_passes(self, legit, isolated_home):
        assert mm._safe_models_filename(legit) == legit

    def test_ads_candidate_never_enters_the_real_pick_pool(self, tmp_path, isolated_home):
        """End-to-end: a malicious HF repo listing offering an ADS-named
        'mmproj' sibling must never be picked as a candidate by the real
        caller, not just rejected by the validator in isolation."""
        from localm.model_manager.pull import _pick_mmproj_from_listing
        base = tmp_path / "models"
        base.mkdir()
        files = ["model.gguf", "somefile.exe:mmproj.gguf"]
        assert _pick_mmproj_from_listing(files, "model.gguf", base) is None

    def test_alias_resolving_to_a_different_existing_file_is_rejected(
            self, tmp_path, isolated_home, monkeypatch):
        """An OS-level alias - an NTFS 8.3 short name is the case here: on a
        volume with short-name generation enabled, 'LONGMO~1.GGU' resolves to a
        pre-existing 'LongModelNameThatIsVeryLong.gguf', passing every earlier
        check while naming something other than what it appears to. A
        production caller's own dest.exists() "already downloaded" convention
        (pull.py) would then treat the victim's unrelated content as the
        requested download.

        8dot3 generation is volume- and config-dependent, so this simulates the
        alias condition directly via Path.resolve rather than depending on the
        OS having generated one."""
        base = tmp_path / "models"
        base.mkdir()
        victim = base / "LongModelNameThatIsVeryLong.gguf"
        victim.write_bytes(b"VICTIM-CONTENT")

        real_resolve = Path.resolve

        def fake_resolve(self, *a, **k):
            if self.name == "LONGMO~1.GGU":
                # What real 8.3 resolution does: the short-name path
                # resolves to the long-named file's real, on-disk path.
                return victim.resolve()
            return real_resolve(self, *a, **k)

        monkeypatch.setattr(Path, "resolve", fake_resolve)
        assert mm._safe_models_filename("LONGMO~1.GGU", base) is None

    def test_same_name_repull_of_existing_file_still_accepted(
            self, tmp_path, isolated_home):
        """The alias check must not reject the ordinary, legitimate case of
        re-pulling a model that is already on disk under the exact name
        requested."""
        base = tmp_path / "models"
        base.mkdir()
        (base / "model.gguf").write_bytes(b"CONTENT")
        assert mm._safe_models_filename("model.gguf", base) == "model.gguf"

    def test_case_variant_of_existing_file_still_accepted(
            self, tmp_path, isolated_home):
        """Windows path resolution returns the ON-DISK casing regardless of the
        requested casing (resolving 'MODEL.GGUF' against an existing
        'model.gguf' yields dest.name == 'model.gguf'), so a benign
        case-variant request for an already-downloaded file must not be
        mistaken for an alias-substitution attack."""
        base = tmp_path / "models"
        base.mkdir()
        (base / "model.gguf").write_bytes(b"CONTENT")
        assert mm._safe_models_filename("MODEL.GGUF", base) == "MODEL.GGUF"


class TestGetModelInfoPathologicalName:
    """get_model_info must return None for a pathological or nonexistent name,
    never crash run/benchmark/serve. On Windows a dots-only name ('....') stats
    as an existing dir but iterdir() on the raw string raises
    FileNotFoundError."""

    @pytest.mark.parametrize("name", ["....", "...", "..", "a..b", "doesnotexist"])
    def test_pathological_names_resolve_to_none(self, name, isolated_home):
        assert mm.get_model_info(name) is None

# SPDX-License-Identifier: AGPL-3.0-or-later
"""A non-secret binary in an explicit add must not fail the WHOLE batch.

BLACKLISTED_SUFFIXES conflates two very different things: SECRET material (.pem,
.key, id_rsa, .env) and ordinary non-secret binaries (.mp4, .db, .sqlite, .7z,
.bmp, fonts, model weights). confine_index_path raising the same
ConfinementError for both means the rag_add route turns it into an
HTTPException(400) for the entire request, since it 400s any ConfinementError
that is not `outside_allowed`.

So POST /api/rag/collections/kb/add {"paths": ["notes.txt", "clip.mp4"]} would
index NOTHING and 400, and a user multi-selecting from a Downloads folder that
happens to contain one video indexes zero files.

The split: refusing a SECRET stays a loud, whole-request refusal (it is a
security boundary, and the caller must know). A non-secret binary is not a
security question at all - it is just "no text in it", so it is reported as an
individual per-file failure like any other unreadable file.

The perf guard that motivated the blacklist (a multi-GB .gguf must not be read
into RAM and sha256-hashed twice just to be rejected) is preserved: the per-file
refusal happens BEFORE stat/read_bytes.
"""

from __future__ import annotations

import pytest

from localm.rag.extract import (BLACKLISTED_SUFFIXES, SECRET_SUFFIXES,
                                UNINDEXABLE_SUFFIXES)
from localm.rag.store import Collection, ConfinementError, confine_index_path


class TestSuffixSetsAreDisjointAndComplete:
    def test_union_is_unchanged(self):
        """The folder walk filters on BLACKLISTED_SUFFIXES; splitting the set must
        not change WHAT a walk skips, only how an explicit pick is reported."""
        assert BLACKLISTED_SUFFIXES == SECRET_SUFFIXES | UNINDEXABLE_SUFFIXES

    def test_the_two_categories_do_not_overlap(self):
        assert SECRET_SUFFIXES & UNINDEXABLE_SUFFIXES == set()

    @pytest.mark.parametrize("suffix", [".pem", ".key", ".ovpn", ".kdbx", ".p12"])
    def test_key_material_is_classified_secret(self, suffix):
        assert suffix in SECRET_SUFFIXES

    @pytest.mark.parametrize("suffix", [".mp4", ".db", ".sqlite", ".7z", ".bmp",
                                        ".gguf", ".safetensors", ".ttf", ".exe"])
    def test_media_and_weights_are_classified_unindexable_not_secret(self, suffix):
        assert suffix in UNINDEXABLE_SUFFIXES
        assert suffix not in SECRET_SUFFIXES


class TestNonSecretBinaryIsNotAConfinementRefusal:
    @pytest.mark.parametrize("name", ["clip.mp4", "data.db", "app.sqlite",
                                      "bundle.7z", "icon.bmp", "model.gguf",
                                      "weights.safetensors"])
    def test_explicit_non_secret_binary_does_not_raise(self, tmp_path, name):
        f = tmp_path / name
        f.write_bytes(b"\x00\x01binary")
        policy = {"mode": "blacklist", "denied": [], "allowed": []}
        confine_index_path(f, policy)   # must NOT raise


class TestSecretRefusalIsUnchanged:
    """NEGATIVE CASE: the security boundary must survive the split. Without
    these, dropping the check entirely would pass everything above while
    re-opening the secret leak."""

    @pytest.mark.parametrize("name", ["deploy.pem", "tls.key", "vpn.ovpn",
                                      "id_rsa", ".env", "vault.kdbx"])
    def test_explicit_secret_still_refused(self, tmp_path, name):
        f = tmp_path / name
        f.write_text("secret placeholder", encoding="utf-8")
        policy = {"mode": "blacklist", "denied": [], "allowed": []}
        with pytest.raises(ConfinementError) as ei:
            confine_index_path(f, policy)
        assert ei.value.reason == "secret_file"

    def test_safe_env_template_is_not_refused(self, tmp_path):
        # The template is documentation, not a secret.
        f = tmp_path / ".env.example"
        f.write_text("API_KEY=your-key-here", encoding="utf-8")
        policy = {"mode": "blacklist", "denied": [], "allowed": []}
        confine_index_path(f, policy)   # must NOT raise


class TestMixedBatchIndexesTheGoodFiles:
    """The actual user-visible bug: one bad file must not zero out the batch."""

    def _coll(self, tmp_path):
        return Collection("kb", base=tmp_path / "rag").create()

    def test_mixed_batch_indexes_text_and_reports_binary_as_failed(self, tmp_path):
        good = tmp_path / "notes.txt"
        good.write_text("the quick brown fox jumps over the lazy dog",
                        encoding="utf-8")
        binary = tmp_path / "clip.mp4"
        binary.write_bytes(b"\x00\x01\x02not really a video")

        coll = self._coll(tmp_path)
        policy = {"mode": "blacklist", "denied": [], "allowed": []}
        result = coll.add_paths([good, binary], policy=policy)

        assert result["added"] == 1, "the good file must still index"
        failed_paths = [f["path"] for f in result["failed"]]
        assert str(binary.resolve()) in failed_paths, \
            "the binary must be reported, not silently dropped"
        assert result["chunks"] > 0

    def test_the_binary_failure_says_why(self, tmp_path):
        binary = tmp_path / "clip.mp4"
        binary.write_bytes(b"\x00\x01\x02")
        coll = self._coll(tmp_path)
        result = coll.add_paths(
            [binary], policy={"mode": "blacklist", "denied": [], "allowed": []})
        assert result["added"] == 0
        assert len(result["failed"]) == 1
        # A reason the user can act on, not a silent skip.
        assert "no extractable text" in result["failed"][0]["error"]

    def test_a_named_model_weight_is_refused_without_being_read(self, tmp_path):
        """The rag-blacklist-model-files perf guard: reject BEFORE read_bytes, so
        a multi-GB weight file is never pulled into RAM just to be rejected."""
        weights = tmp_path / "model.gguf"
        weights.write_bytes(b"GGUF" + b"\x00" * 64)
        read_calls = []

        real_read_bytes = type(weights).read_bytes

        def _tracked_read_bytes(self, *a, **kw):
            read_calls.append(str(self))
            return real_read_bytes(self, *a, **kw)

        import pathlib
        orig = pathlib.Path.read_bytes
        pathlib.Path.read_bytes = _tracked_read_bytes
        try:
            coll = self._coll(tmp_path)
            result = coll.add_paths(
                [weights],
                policy={"mode": "blacklist", "denied": [], "allowed": []})
        finally:
            pathlib.Path.read_bytes = orig

        assert result["added"] == 0
        assert len(result["failed"]) == 1
        assert str(weights.resolve()) not in read_calls, \
            "the weight file must be refused BEFORE its bytes are read"

    def test_secret_in_a_batch_still_refuses_the_whole_request(self, tmp_path):
        """Deliberate asymmetry: a SECRET is a security refusal the caller must
        notice, so it still raises out of add_paths rather than degrading to a
        per-file note."""
        good = tmp_path / "notes.txt"
        good.write_text("ordinary text", encoding="utf-8")
        secret = tmp_path / "id_rsa"
        secret.write_text("PRIVATE KEY", encoding="utf-8")
        coll = self._coll(tmp_path)
        with pytest.raises(ValueError):
            coll.add_paths([good, secret],
                           policy={"mode": "blacklist", "denied": [],
                                   "allowed": []})

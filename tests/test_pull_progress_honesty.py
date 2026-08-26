# SPDX-License-Identifier: AGPL-3.0-or-later
"""The download progress stream must not state anything the download does not
know.

Five defects, one function cluster, all in the pull progress path:

* a terminal 100% emitted from `finally`, so EVERY exit claims completion -
  including the `return False` a failed part takes from inside the `with`
  block, which unwinds cleanly and defeats exception detection.
* the opening event hardcoding `0` instead of calling the numerator the poll
  loop uses 0.7s later. True on a fresh pull, FALSE on every resume.
* `_pull_url` deriving its total from the resume offset, so a resumed chunked
  download runs to completion at a stuck, confident 100%.
* the single-file path emitting NOTHING when it cannot size the download,
  where its sibling streams an honest indeterminate stream.
* the in-flight byte scan hardcoded to MODELS_DIR and unfiltered, so it reads
  the wrong tree for a --comfy-dest-dir pull and adds a CONCURRENT pull's temp
  file to this job's numerator.

A NOTE ON THE ASSERTIONS, because the obvious ones are unsound here. Each
context manager starts its poll thread BEFORE emitting the opening event, and
the thread's first iteration runs immediately - so which of the two lands first
is a race. A test phrased as "the first event reports the measurement" can
therefore PASS against the unfixed code whenever the poll thread wins, which is
a test that cannot reliably fail. The seed tests below assert instead that NO
event reports zero while bytes are on disk. That is race-proof, and it fails
deterministically on a seed that always emits 0.
"""

import json
from unittest.mock import MagicMock

import pytest

from localm import model_manager as mm


def _events(capsys, phase=None):
    """Every progress payload emitted so far, in order.

    *phase* filters to one stage. It exists because the channel carries more
    than one: the pull path also emits a ``verify`` stage while it hashes the
    finished file. A test about what the DOWNLOAD knows must therefore say so,
    or it reads the verifier's events as the downloader's - and those are
    legitimately different, since a file on disk has a size even when the
    transfer that produced it never advertised one.

    A harness that identifies its subject by a property several producers share
    cannot tell them apart. Here the shared property is "is a progress event",
    and it stops being discriminating the moment a second phase exists."""
    out = capsys.readouterr().out
    evs = [json.loads(line.split(mm.PROGRESS_SENTINEL, 1)[1])
           for line in out.splitlines() if mm.PROGRESS_SENTINEL in line]
    return [e for e in evs if e.get("phase") == phase] if phase else evs


@pytest.fixture()
def gui(monkeypatch):
    monkeypatch.setenv("LOCALM_PROGRESS_JSON", "1")


# --------------------------------------------------------------- no false 100%

class TestTerminalHundredIsAClaimAboutTheOutcome:
    """100% asserts the download finished. Only the success path may say it."""

    def test_snapshot_failure_reports_the_measured_partial(self, gui, capsys):
        with mm._snapshot_progress(lambda: 30, 100):
            pass                      # never calls .ok() -> did not succeed
        last = _events(capsys)[-1]
        assert last["downloaded"] == 30, (
            f"a download that did not succeed reported {last['downloaded']} of "
            f"{last['total']}; the terminal event must report what actually landed")
        assert last["pct"] != 100.0

    def test_snapshot_success_does_report_the_total(self, gui, capsys):
        with mm._snapshot_progress(lambda: 30, 100) as prog:
            prog.ok()
        last = _events(capsys)[-1]
        assert last["downloaded"] == 100 and last["pct"] == 100.0, (
            "a successful download must still finish at 100%; this fix must not "
            "trade a false failure-100% for a missing success-100%")

    def test_an_early_return_from_inside_the_with_block_is_not_success(
            self, gui, capsys, tmp_path):
        """THE CASE EXCEPTION DETECTION MISSES, and the reason the outcome is
        explicit rather than inferred. _pull_gguf_file reports a failed part
        with `return False` from inside its `with`. That unwinds with no
        exception, so a context manager watching for one sees a clean exit."""
        part = tmp_path / "m.gguf"
        part.write_bytes(b"x" * 40)

        def _download_then_bail():
            with mm._download_progress([part], 100):
                return "failed early"       # no .ok(), exactly like the real path

        assert _download_then_bail() == "failed early"
        last = _events(capsys)[-1]
        assert last["downloaded"] == 40, (
            f"an early return announced {last['downloaded']} of {last['total']}; "
            "a clean unwind is not evidence of success")
        assert last["pct"] != 100.0

    def test_download_success_reports_the_total(self, gui, capsys, tmp_path):
        part = tmp_path / "m.gguf"
        part.write_bytes(b"x" * 40)
        with mm._download_progress([part], 100) as prog:
            prog.ok()
        assert _events(capsys)[-1]["pct"] == 100.0


# ------------------------------------------------- seed from the measurement

class TestTheOpeningEventComesFromTheMeasurement:
    """A hardcoded 0 is true on a fresh pull and false on every resume."""

    def test_a_resumed_download_never_reports_zero_bytes(self, gui, capsys, tmp_path):
        """3 of 12 bytes already on disk. Announcing 0 contradicts the file the
        code's own next poll is about to stat."""
        part = tmp_path / "m.gguf"
        part.write_bytes(b"xxx")
        with mm._download_progress([part], 12) as prog:
            prog.ok()
        evs = _events(capsys)
        assert evs, "no progress emitted at all"
        zeros = [e for e in evs if e["downloaded"] == 0]
        assert not zeros, (
            f"reported 0 bytes downloaded with 3 already on disk: {zeros}")

    def test_a_resumed_snapshot_never_reports_zero_bytes(self, gui, capsys):
        with mm._snapshot_progress(lambda: 4096, 8192) as prog:
            prog.ok()
        zeros = [e for e in _events(capsys) if e["downloaded"] == 0]
        assert not zeros, f"reported 0 bytes with 4096 on disk: {zeros}"

    def test_a_genuinely_fresh_download_still_starts_at_zero(self, gui, capsys, tmp_path):
        """The fix must not invent bytes either: nothing on disk really is 0."""
        part = tmp_path / "absent.gguf"
        with mm._download_progress([part], 12) as prog:
            prog.ok()
        assert _events(capsys)[0]["downloaded"] == 0


# ------------------------------------- indeterminate beats silence

class TestAnUnknownTotalStreamsRatherThanGoingSilent:
    def test_the_single_file_path_still_emits_with_no_total(self, gui, capsys, tmp_path):
        """One failed HEAD zeroes total_size. Gating the whole progress mechanism
        on a nonzero total makes it a no-op there, so a multi-GB pull goes
        completely silent."""
        part = tmp_path / "m.gguf"
        part.write_bytes(b"y" * 7)
        with mm._download_progress([part], 0) as prog:
            prog.ok()
        evs = _events(capsys)
        assert evs, "an unsized download emitted nothing at all"
        assert all(e["pct"] is None for e in evs), (
            "an unsized download must report pct null, never a number")

    def test_an_unknown_total_does_not_pin_the_byte_count_at_zero(
            self, gui, capsys, tmp_path):
        """The clamp is `min(measured, total)`, and min(x, 0) is 0 - so an
        unguarded clamp reports a permanently frozen 0 B for the whole
        download, which is the busy-bar equivalent of the bug above."""
        part = tmp_path / "m.gguf"
        part.write_bytes(b"y" * 7)
        with mm._download_progress([part], 0) as prog:
            prog.ok()
        assert any(e["downloaded"] == 7 for e in _events(capsys)), (
            "an unsized download reported no real byte count")


# ------------------------------------------------- the scan is scoped

class TestTheInFlightScanIsScopedToThisJob:
    def _incomplete(self, base, rel, prefix, size):
        d = base / ".cache" / "huggingface" / "download"
        d.mkdir(parents=True, exist_ok=True)
        f = d / f"{prefix}.deadbeef.incomplete"
        f.write_bytes(b"z" * size)
        return f

    def test_a_concurrent_pulls_temp_file_is_not_counted(self, gui, capsys, tmp_path):
        """An unfiltered rglob scan lets every other download in flight inflate
        this job's numerator."""
        from huggingface_hub._local_folder import _short_hash
        from huggingface_hub._local_folder import get_local_download_paths
        base = tmp_path / "models"
        base.mkdir()
        mine_rel = "mine.gguf"
        mine_prefix = _short_hash(get_local_download_paths(base, mine_rel).metadata_path.name)
        theirs_prefix = _short_hash(
            get_local_download_paths(base, "someone-elses.gguf").metadata_path.name)
        assert mine_prefix != theirs_prefix
        self._incomplete(base, mine_rel, mine_prefix, 100)
        self._incomplete(base, "someone-elses.gguf", theirs_prefix, 900)

        with mm._download_progress([base / mine_rel], 1000, base_dir=base,
                                   rel_parts=[mine_rel]) as prog:
            prog.ok()
        # ONE capture: capsys.readouterr() drains the buffer, so calling the
        # helper twice would leave the second list empty.
        seen = [e["downloaded"] for e in _events(capsys)]
        assert 100 in seen, f"never saw my own 100-byte temp file: {seen}"
        # The terminal event reports the total on success; every event before it
        # reflects the measurement.
        assert 1000 not in seen[:-1], (
            f"counted another pull's 900-byte temp file: {seen}")

    def test_the_destination_is_scanned_not_models_dir(self, gui, capsys,
                                                       tmp_path, monkeypatch):
        """A --comfy-dest-dir pull downloads into dest_dir, but the scan was
        hardcoded to MODELS_DIR - so it watched a tree the download never
        touched and progress only moved when a whole part landed."""
        from huggingface_hub._local_folder import _short_hash
        from huggingface_hub._local_folder import get_local_download_paths
        models = tmp_path / "models"
        models.mkdir()
        monkeypatch.setattr(mm, "MODELS_DIR", models)
        dest = tmp_path / "comfy"
        dest.mkdir()
        rel = "vae.safetensors"
        prefix = _short_hash(get_local_download_paths(dest, rel).metadata_path.name)
        self._incomplete(dest, rel, prefix, 250)

        with mm._download_progress([dest / rel], 1000, base_dir=dest,
                                   rel_parts=[rel]) as prog:
            prog.ok()
        assert any(e["downloaded"] == 250 for e in _events(capsys)), (
            "in-flight bytes in the destination tree were never seen")

    def test_an_unknown_hub_layout_degrades_to_coarse_not_to_wrong(
            self, gui, capsys, tmp_path, monkeypatch):
        """The prefix computation reaches into huggingface_hub internals. If a
        future version moves them, progress must get chunkier - never wrong,
        and never crash the download."""
        # Patch the DEFINING module, not the package. `mm._download_progress` IS
        # `pull._download_progress`, so it resolves _incomplete_prefixes from
        # pull's globals; the package re-exports the context managers but not
        # this helper.
        monkeypatch.setattr(
            "localm.model_manager.pull._incomplete_prefixes", lambda *a, **k: None)
        base = tmp_path / "models"
        base.mkdir()
        part = base / "m.gguf"
        part.write_bytes(b"q" * 11)
        with mm._download_progress([part], 100, base_dir=base,
                                   rel_parts=["m.gguf"]) as prog:
            prog.ok()
        assert any(e["downloaded"] == 11 for e in _events(capsys))


# ------------------------------------------- a total is not a resume offset

def _resp(status, body: bytes, content_length=None):
    r = MagicMock()
    r.status_code = status
    r.raise_for_status = MagicMock()
    cl = len(body) if content_length is None else content_length
    r.headers = {"content-length": str(cl)}

    def _iter(chunk_size):
        for i in range(0, len(body), chunk_size):
            yield body[i:i + chunk_size]
    r.iter_content = _iter
    return r


@pytest.fixture()
def url_env(tmp_path, monkeypatch):
    models = tmp_path / "models"
    models.mkdir()
    monkeypatch.setattr(mm, "MODELS_DIR", models)
    monkeypatch.setattr(mm, "ensure_dirs", lambda: None)
    monkeypatch.setattr(mm, "_check_disk_space", lambda *a, **k: True)
    monkeypatch.setattr(mm, "find_by_sha256", lambda *a, **k: [])
    monkeypatch.setattr(mm, "_register", MagicMock())
    monkeypatch.setattr(mm, "_register_with_dedup", MagicMock())
    monkeypatch.setattr(
        "socket.getaddrinfo",
        lambda *a, **k: [(2, 1, 6, "", ("93.184.216.34", 0))])
    return models


def _wire_http(monkeypatch, head_total: int, response):
    def fake_pinned_request(method, url, **kwargs):
        if method == "HEAD":
            h = MagicMock()
            h.status_code = 200
            h.headers = {"content-length": str(head_total)}
            return h
        return response
    monkeypatch.setattr("localm.netpolicy.pinned_request", fake_pinned_request)


class TestAResumeOffsetIsNotATotal:
    def test_a_resumed_chunked_download_never_claims_a_percentage(
            self, url_env, monkeypatch, capsys):
        """THE DEFECT: `total_display = (already_have + content_length) or None`.
        A chunked response carries no content-length, so on a RESUME the total
        becomes already_have - the first poll reads already_have of already_have,
        i.e. 100%, and the change-gate suppresses every later event. The whole
        real transfer then runs at a stuck, confident 100%.

        A FRESH download collapses to None correctly, so only the resume path is
        ever wrong - the one the .part machinery exists for. The fixture must
        therefore set BOTH a non-empty .part file and a zero content-length;
        either alone cannot reproduce it.
        """
        models = url_env
        (models / "model.gguf.part").write_bytes(b"01234")   # 5 already on disk
        monkeypatch.setenv("LOCALM_PROGRESS_JSON", "1")
        _wire_http(monkeypatch, 10, _resp(206, b"56789", content_length=0))

        mm._pull_url("http://example.com/model.gguf", "mymodel")

        evs = _events(capsys, phase="download")
        assert evs, "GUI mode emitted no download progress at all"
        lying = [e for e in evs if e["pct"] is not None]
        assert not lying, (
            "claimed a definite percentage for a download of unknown size "
            f"(total came from the resume offset): {lying}")

    def test_a_known_length_resume_still_reports_a_real_total(
            self, url_env, monkeypatch, capsys):
        """The fix must not throw away a total we genuinely have."""
        models = url_env
        (models / "model.gguf.part").write_bytes(b"01234")
        monkeypatch.setenv("LOCALM_PROGRESS_JSON", "1")
        _wire_http(monkeypatch, 10, _resp(206, b"56789", content_length=5))

        mm._pull_url("http://example.com/model.gguf", "mymodel")

        # Scoped to the download stage: the finished file is also 10 bytes, so the
        # verify stage reports total=10 too, and an unscoped assertion would be
        # satisfied by the verifier.
        totals = {e["total"] for e in _events(capsys, phase="download")}
        assert 10 in totals, (
            f"lost a real total on a resumed download: {totals}")

# SPDX-License-Identifier: AGPL-3.0-or-later
"""The model pull progress payload carries per-file info for a multi-file
(split GGUF) download, so the GUI can show "file 2 of 3: <name>" alongside the
percentage / bytes / rate / ETA. A single-file download omits the per-file
fields.
"""

import json

from localm import model_manager as mm


def _parse_emit(capsys):
    out = capsys.readouterr().out
    assert mm.PROGRESS_SENTINEL in out, "no progress line emitted"
    _, _, payload = out.partition(mm.PROGRESS_SENTINEL)
    return json.loads(payload.split("\n", 1)[0])


def test_emit_progress_basic_has_no_file_fields(capsys):
    mm._emit_progress(5, 10)
    d = _parse_emit(capsys)
    assert d["downloaded"] == 5 and d["total"] == 10 and d["pct"] == 50.0
    assert "count" not in d and "index" not in d and "name" not in d


def test_emit_progress_multifile_includes_file_fields(capsys):
    mm._emit_progress(5, 10, name="model-00002-of-00003.gguf", index=2, count=3)
    d = _parse_emit(capsys)
    assert d["count"] == 3 and d["index"] == 2
    assert d["name"] == "model-00002-of-00003.gguf"


def test_emit_progress_single_file_omits_file_fields(capsys):
    # count <= 1 -> no per-file clutter even if a name is passed.
    mm._emit_progress(1, 2, name="solo.gguf", index=1, count=1)
    d = _parse_emit(capsys)
    assert "count" not in d and "index" not in d and "name" not in d


def test_progress_file_info_tracks_current_part(tmp_path):
    parts = [tmp_path / f"m-0000{i}-of-00003.gguf" for i in range(1, 4)]
    # Nothing on disk yet -> file 1 of 3 is in flight.
    assert mm._progress_file_info(parts) == (parts[0].name, 1, 3)
    parts[0].write_bytes(b"x")                       # part 1 landed
    assert mm._progress_file_info(parts) == (parts[1].name, 2, 3)
    for p in parts:
        p.write_bytes(b"x")                          # all landed
    assert mm._progress_file_info(parts) == (parts[2].name, 3, 3)


def test_progress_file_info_single_file_is_noop(tmp_path):
    assert mm._progress_file_info([tmp_path / "solo.gguf"]) == (None, 0, 0)
    assert mm._progress_file_info([]) == (None, 0, 0)

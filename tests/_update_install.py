# SPDX-License-Identifier: AGPL-3.0-or-later
"""Install a test build into a test install through ``updater_handoff.prepare``, for tests
of what an update checks and keeps before anything moves: the download, the signature
and key set, the version, the bridges and the data copy of an older release.

Three steps stand in for parts these tests are not about: the ``check`` of the new
build (it answers "go" without running), the start of the applier (only
``updater_handoff.keep_before_start``, what ``hand_off`` keeps first, runs), and the move
of the files (the extracted release is copied over the install). The tests of
``localm.update_mechanism`` and ``tests/test_update_applier.py`` cover those for real."""

import json
import shutil
from pathlib import Path
from unittest import mock

from localm import updater_handoff
from localm.update_mechanism import mover


def _check_says_go(program, root, update_id, say) -> int:
    folder = mover.work_folder(Path(root), update_id)
    (folder / "check.json").write_text(json.dumps({"edited": [], "notes": []}),
                                       encoding="utf-8")
    return 0


def install(asset_id, *, signature, installed, download_opener, **kw):
    """:func:`install_build` called the way a download is described: the build comes from
    *download_opener* (``(url, headers, timeout, dest)``, as ``updater.download`` takes)."""
    import tempfile
    with tempfile.TemporaryDirectory(dir=Path(installed).parent) as scratch:
        dest = Path(scratch) / "build.zip"
        download_opener(None, {}, 0, dest)
        data = dest.read_bytes()
    return install_build(installed, data, signature, asset_id=asset_id, **kw)


def install_build(inst, data: bytes, signature, *, asset_id=5, **kw):
    """Prepare the build zip *data* (signature envelope *signature*) for the install *inst*
    as an update does, keep what the hand-off keeps, copy the extracted release over
    *inst* and return the ``Prepared``. *kw* goes to ``prepare`` (``target_version``,
    ``allow_unverified``, ``expect_sha256``). Raises what ``prepare`` raises, with nothing
    of *inst* changed."""
    inst = Path(inst)

    def opener(url, headers, timeout, dest):
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_bytes(data)

    with mock.patch.object(updater_handoff, "_run_check", _check_says_go):
        prepared = updater_handoff.prepare(asset_id, signature=signature, installed=inst,
                                           download_opener=opener, **kw)
    try:
        updater_handoff.keep_before_start(prepared)
        new = mover.work_folder(inst, prepared.id) / "new"
        for entry in new.iterdir():
            dst = inst / entry.name
            if entry.is_dir():
                shutil.copytree(entry, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(entry, dst)
    finally:
        prepared.discard()
    return prepared

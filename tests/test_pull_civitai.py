# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for _pull_civitai_file() and pull_model()'s civitai: spec dispatch:
dest_dir routing to the ComfyUI-subfoldered tree, eager registration with the
right source/model_type, the SSRF-guarded redirect treatment, and SHA256
verification. No real network - CivitAISource.resolve_download and the HTTP
layer are mocked."""

import contextlib
import hashlib
from pathlib import Path

import pytest

from localm import model_manager as mm
from localm.model_manager import pull
from localm.model_manager.pull import _pull_civitai_file
from localm.model_manager.sources import ResolvedDownload


@pytest.fixture(autouse=True)
def _online(monkeypatch):
    monkeypatch.setenv("LOCALM_NET_MODE", "ask")


@pytest.fixture()
def fake_registry(tmp_path, monkeypatch):
    """Mirrors test_pull_comfy_dest_dir.py's fixture of the same name."""
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
    return store, models_dir


class _FakeStreamResponse:
    def __init__(self, body: bytes, status_code: int = 200, headers=None):
        self.body = body
        self.status_code = status_code
        self.headers = headers or {"content-length": str(len(body))}

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.HTTPError(response=self)

    def iter_content(self, chunk_size):
        for i in range(0, len(self.body), chunk_size):
            yield self.body[i:i + chunk_size]


class _DroppedStream(_FakeStreamResponse):
    """A 200 response whose body stops with a connection error after *keep*
    bytes."""

    def __init__(self, body: bytes, keep: int):
        super().__init__(body)
        self.keep = keep

    def iter_content(self, chunk_size):
        import requests
        yield self.body[:self.keep]
        raise requests.ConnectionError("connection reset by peer")


class _RangeServer:
    """A fake ``pinned_request`` serving *body*: a ``Range: bytes=N-`` GET gets
    the tail from byte N as a 206 (a 416 past the end), any other GET the whole
    body. ``gets`` records every GET's headers."""

    def __init__(self, body: bytes):
        self.body = body
        self.gets: list = []

    def __call__(self, method, url, **kw):
        assert method == "GET", f"unexpected {method} {url}"
        headers = dict(kw.get("headers") or {})
        self.gets.append(headers)
        rng = headers.get("Range")
        if rng is None:
            return _FakeStreamResponse(self.body)
        start = int(rng.removeprefix("bytes=").removesuffix("-"))
        if start >= len(self.body):
            return _FakeStreamResponse(b"", status_code=416)
        return _FakeStreamResponse(self.body[start:], status_code=206)


def _digest(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def _resolved(**over) -> ResolvedDownload:
    body = b"fake-lora-bytes"
    base = dict(
        url="https://civitai.com/api/download/models/135867?fileId=99264",
        filename="add-detail-xl.safetensors",
        source_tag="civitai:135867",
        model_type="lora",
        size_bytes=len(body),
        sha256=hashlib.sha256(body).hexdigest(),
        comfy_subfolder="loras",
        file_id="99264",
    )
    base.update(over)
    return ResolvedDownload(**base)


def _civitai_pull(monkeypatch, resolved: ResolvedDownload, dest_dir: Path, fetch,
                  version_id: str = "135867", **kw):
    """Run _pull_civitai_file for *resolved* into *dest_dir*, with *fetch* as
    the HTTP layer and the redirect resolver passing URLs through."""
    monkeypatch.setattr(
        "localm.model_manager.sources.CivitAISource.resolve_download",
        lambda self, ref, file, **k: resolved)
    monkeypatch.setattr(
        "localm.media.managed_comfy.comfy_models_dest_dir",
        lambda subfolder, cfg=None, plugin=None: dest_dir)
    monkeypatch.setattr(
        "localm.model_manager.pull._ssrf_resolve_final_url", lambda url: url)
    monkeypatch.setattr("localm.netpolicy.pinned_request", fetch)
    return _pull_civitai_file(version_id, None, **kw)


def _interrupt_pull(monkeypatch, resolved: ResolvedDownload, dest_dir: Path,
                    body: bytes, keep: int, version_id: str = "135867") -> Path:
    """Run a pull of *resolved* whose transfer of *body* drops after *keep*
    bytes, and return the partial file it leaves behind."""
    ok = _civitai_pull(monkeypatch, resolved, dest_dir,
                       lambda method, url, **kw: _DroppedStream(body, keep),
                       version_id=version_id)
    part = dest_dir / (resolved.filename + ".part")
    assert ok is False
    assert part.read_bytes() == body[:keep], "the interrupted pull left no partial"
    return part


def _wire_happy_path(monkeypatch, resolved: ResolvedDownload, dest_dir: Path,
                      body: bytes = b"fake-lora-bytes"):
    monkeypatch.setattr(
        "localm.model_manager.sources.CivitAISource.resolve_download",
        lambda self, ref, file, **kw: resolved)
    monkeypatch.setattr(
        "localm.media.managed_comfy.comfy_models_dest_dir",
        lambda subfolder, cfg=None, plugin=None: dest_dir)
    monkeypatch.setattr(
        "localm.model_manager.pull._ssrf_resolve_final_url", lambda url: url)
    monkeypatch.setattr(
        "localm.netpolicy.pinned_request",
        lambda method, url, **kw: _FakeStreamResponse(body))


class TestPullCivitaiFile:
    def test_lands_in_the_resolved_comfy_subfolder_and_registers(
            self, fake_registry, tmp_path, monkeypatch):
        store, _ = fake_registry
        dest_dir = tmp_path / "comfyui-models" / "loras"
        resolved = _resolved()
        _wire_happy_path(monkeypatch, resolved, dest_dir)

        ok = _pull_civitai_file("135867", None)

        assert ok is True
        landed = dest_dir / "add-detail-xl.safetensors"
        assert landed.is_file()
        assert landed.read_bytes() == b"fake-lora-bytes"
        assert sorted(p.name for p in dest_dir.iterdir()) == ["add-detail-xl.safetensors"]
        assert not (fake_registry[1] / "add-detail-xl.safetensors").exists()
        entry = store["add-detail-xl"]
        assert entry["source"] == "civitai:135867"
        assert entry["model_type"] == "lora"
        assert entry["sha256"] == resolved.sha256

    def test_explicit_type_overrides_the_registry_label_not_the_subfolder(
            self, fake_registry, tmp_path, monkeypatch):
        store, _ = fake_registry
        dest_dir = tmp_path / "comfyui-models" / "loras"
        resolved = _resolved(model_type="lora", comfy_subfolder="loras")
        _wire_happy_path(monkeypatch, resolved, dest_dir)

        ok = _pull_civitai_file("135867", None, model_type="unknown")

        assert ok is True
        assert (dest_dir / "add-detail-xl.safetensors").is_file()
        assert store["add-detail-xl"]["model_type"] == "unknown"

    def test_register_false_skips_the_registry(self, fake_registry, tmp_path, monkeypatch):
        store, _ = fake_registry
        dest_dir = tmp_path / "comfyui-models" / "loras"
        _wire_happy_path(monkeypatch, _resolved(), dest_dir)

        ok = _pull_civitai_file("135867", None, register=False)

        assert ok is True
        assert (dest_dir / "add-detail-xl.safetensors").is_file()
        assert store == {}

    def test_already_downloaded_short_circuits_without_a_fetch(
            self, fake_registry, tmp_path, monkeypatch):
        store, _ = fake_registry
        dest_dir = tmp_path / "comfyui-models" / "loras"
        dest_dir.mkdir(parents=True)
        (dest_dir / "add-detail-xl.safetensors").write_bytes(b"fake-lora-bytes")
        resolved = _resolved()
        monkeypatch.setattr(
            "localm.model_manager.sources.CivitAISource.resolve_download",
            lambda self, ref, file, **kw: resolved)
        monkeypatch.setattr(
            "localm.media.managed_comfy.comfy_models_dest_dir",
            lambda subfolder, cfg=None, plugin=None: dest_dir)

        def _forbidden(*a, **kw):
            raise AssertionError("must not fetch when the file already exists")

        monkeypatch.setattr("localm.model_manager.pull._ssrf_resolve_final_url", _forbidden)
        monkeypatch.setattr("localm.netpolicy.pinned_request", _forbidden)

        ok = _pull_civitai_file("135867", None)

        assert ok is True
        assert store["add-detail-xl"]["source"] == "civitai:135867"

    def test_sha256_mismatch_deletes_the_file_and_refuses(
            self, fake_registry, tmp_path, monkeypatch):
        store, _ = fake_registry
        dest_dir = tmp_path / "comfyui-models" / "loras"
        resolved = _resolved(sha256="0" * 64)   # wrong digest for the fake body
        _wire_happy_path(monkeypatch, resolved, dest_dir)

        ok = _pull_civitai_file("135867", None)

        assert ok is False
        assert not (dest_dir / "add-detail-xl.safetensors").exists()
        assert list(dest_dir.iterdir()) == []
        assert store == {}

    def test_no_comfy_folder_configured_refuses_cleanly(
            self, fake_registry, tmp_path, monkeypatch):
        store, _ = fake_registry
        monkeypatch.setattr(
            "localm.model_manager.sources.CivitAISource.resolve_download",
            lambda self, ref, file, **kw: _resolved())
        monkeypatch.setattr(
            "localm.media.managed_comfy.comfy_models_dest_dir",
            lambda subfolder, cfg=None, plugin=None: None)

        ok = _pull_civitai_file("135867", None)

        assert ok is False
        assert store == {}

    def test_resolve_download_error_is_reported_not_raised(
            self, fake_registry, monkeypatch):
        from localm.model_manager.sources import ModelSourceError

        def _refuse(self, ref, file, **kw):
            raise ModelSourceError("excluded")

        monkeypatch.setattr(
            "localm.model_manager.sources.CivitAISource.resolve_download", _refuse)

        assert _pull_civitai_file("135867", None) is False


class TestPullCivitaiFileSSRF:
    """The download redirect must get the SAME per-hop SSRF guard as
    _pull_url, never a trust-the-metadata shortcut."""

    def test_a_refused_redirect_chain_downloads_nothing(
            self, fake_registry, tmp_path, monkeypatch):
        store, _ = fake_registry
        dest_dir = tmp_path / "comfyui-models" / "loras"
        monkeypatch.setattr(
            "localm.model_manager.sources.CivitAISource.resolve_download",
            lambda self, ref, file, **kw: _resolved())
        monkeypatch.setattr(
            "localm.media.managed_comfy.comfy_models_dest_dir",
            lambda subfolder, cfg=None, plugin=None: dest_dir)

        from localm.netpolicy import NetworkPolicyError

        def _refuse(url):
            raise NetworkPolicyError("refused: private address")

        monkeypatch.setattr("localm.model_manager.pull._ssrf_resolve_final_url", _refuse)

        def _forbidden_get(*a, **kw):
            raise AssertionError("must not GET after the redirect resolver refused")

        monkeypatch.setattr("localm.netpolicy.pinned_request", _forbidden_get)

        ok = _pull_civitai_file("135867", None)

        assert ok is False
        assert not dest_dir.exists() or list(dest_dir.iterdir()) == []
        assert store == {}

    def test_real_ssrf_resolver_refuses_a_private_redirect_target(
            self, fake_registry, tmp_path, monkeypatch):
        """A URL that is itself private-IP shaped: even the second,
        immediately-before-connect check_url call alone (the same "revalidate
        right before the GET" pattern _pull_url_locked uses) would catch this
        one, so this proves the OVERALL guarantee - no private address is ever
        connected to - rather than isolating _ssrf_resolve_final_url on its
        own (see the redirect-chain variant below for that)."""
        store, _ = fake_registry
        dest_dir = tmp_path / "comfyui-models" / "loras"
        monkeypatch.setenv("LOCALM_NET_MODE", "allow")   # isolate the IP-class check
        resolved = _resolved(url="http://127.0.0.1:9/whatever")
        monkeypatch.setattr(
            "localm.model_manager.sources.CivitAISource.resolve_download",
            lambda self, ref, file, **kw: resolved)
        monkeypatch.setattr(
            "localm.media.managed_comfy.comfy_models_dest_dir",
            lambda subfolder, cfg=None, plugin=None: dest_dir)

        ok = _pull_civitai_file("135867", None)

        assert ok is False
        assert not dest_dir.exists() or list(dest_dir.iterdir()) == []
        assert store == {}

    def test_real_ssrf_resolver_refuses_a_redirect_TO_a_private_target(
            self, fake_registry, tmp_path, monkeypatch):
        """The representative CivitAI threat model: the STARTING url is the
        legitimate public civitai.com download endpoint -
        exactly what resolve_download() returns for a real pull - and only the
        redirect it answers with points at a private address. This isolates
        _ssrf_resolve_final_url's own per-hop re-validation: the immediately-
        before-connect check_url in _pull_civitai_file never even runs, because
        the resolver itself must refuse before returning."""
        store, _ = fake_registry
        dest_dir = tmp_path / "comfyui-models" / "loras"
        monkeypatch.setenv("LOCALM_NET_MODE", "allow")   # isolate the IP-class check
        resolved = _resolved()   # a real https://civitai.com/... starting URL
        monkeypatch.setattr(
            "localm.model_manager.sources.CivitAISource.resolve_download",
            lambda self, ref, file, **kw: resolved)
        monkeypatch.setattr(
            "localm.media.managed_comfy.comfy_models_dest_dir",
            lambda subfolder, cfg=None, plugin=None: dest_dir)

        def _head_redirects_to_private(method, url, **kw):
            assert method == "HEAD", "only the resolver's own HEAD probe should ever fire here"
            resp = _FakeStreamResponse(b"", status_code=307,
                                       headers={"Location": "http://169.254.169.254/latest/meta-data/"})
            return resp

        monkeypatch.setattr("localm.netpolicy.pinned_request", _head_redirects_to_private)

        ok = _pull_civitai_file("135867", None)

        assert ok is False
        assert not dest_dir.exists() or list(dest_dir.iterdir()) == []
        assert store == {}


_V1 = b"version-111-weights!"
_V2 = b"VERSION-222-WEIGHTS?"


class TestPullCivitaiFileResume:
    """An interrupted pull leaves ``<file>.part`` plus a ``<file>.part.json``
    record of which file it holds, and a later pull appends to that partial
    only when the record matches the file it is pulling."""

    FILE = "char.safetensors"

    def _resolved(self, **over) -> ResolvedDownload:
        base = dict(filename=self.FILE, source_tag="civitai:222", file_id="2",
                    sha256=None, size_bytes=len(_V2))
        base.update(over)
        return _resolved(**base)

    def test_an_interrupted_pull_keeps_the_partial_and_its_record(
            self, fake_registry, tmp_path, monkeypatch):
        dest_dir = tmp_path / "comfyui-models" / "loras"

        _interrupt_pull(monkeypatch, self._resolved(), dest_dir, _V2, 12,
                        version_id="222")

        assert sorted(p.name for p in dest_dir.iterdir()) == [
            self.FILE + ".part", self.FILE + ".part.json"]

    @pytest.mark.parametrize("digest", [None, _digest(_V2)], ids=["no-digest", "digest"])
    def test_the_same_file_resumes_from_its_partial(
            self, digest, fake_registry, tmp_path, monkeypatch):
        store, _ = fake_registry
        dest_dir = tmp_path / "comfyui-models" / "loras"
        resolved = self._resolved(sha256=digest)
        _interrupt_pull(monkeypatch, resolved, dest_dir, _V2, 12, version_id="222")
        server = _RangeServer(_V2)

        ok = _civitai_pull(monkeypatch, resolved, dest_dir, server, version_id="222")

        dest = dest_dir / self.FILE
        assert dest.is_file() and dest.read_bytes() == _V2
        assert sorted(p.name for p in dest_dir.iterdir()) == [self.FILE]
        assert [g.get("Range") for g in server.gets] == ["bytes=12-"]
        assert ok is True
        assert store["char"]["sha256"] == _digest(_V2)

    @pytest.mark.parametrize("first, second, body", [
        pytest.param({"version_id": "111"}, {"version_id": "222"}, _V2,
                     id="other-version"),
        pytest.param({"file_id": "1"}, {"file_id": "2"}, _V2, id="other-file"),
        pytest.param({"sha256": _digest(_V1)}, {"sha256": _digest(_V2)}, _V2,
                     id="other-digest"),
        pytest.param({"size_bytes": len(_V1)}, {"size_bytes": len(_V2) + 1},
                     _V2 + b"+", id="other-size"),
    ])
    def test_a_partial_of_another_file_is_not_resumed(
            self, first, second, body, fake_registry, tmp_path, monkeypatch):
        store, _ = fake_registry
        dest_dir = tmp_path / "comfyui-models" / "loras"

        def _pick(over):
            over = dict(over)
            version = over.pop("version_id", "222")
            return version, self._resolved(source_tag=f"civitai:{version}", **over)

        version, resolved = _pick(first)
        _interrupt_pull(monkeypatch, resolved, dest_dir, _V1, 12, version_id=version)
        version, resolved = _pick(second)
        server = _RangeServer(body)

        exc = None
        try:
            ok = _civitai_pull(monkeypatch, resolved, dest_dir, server,
                               version_id=version)
        except Exception as e:
            ok, exc = None, e

        dest = dest_dir / self.FILE
        assert dest.is_file() and dest.read_bytes() == body, (
            "the new file was not downloaded whole: "
            f"{dest.read_bytes() if dest.is_file() else None!r}")
        assert [g.get("Range") for g in server.gets] == [None]
        assert exc is None, f"the pull raised {exc!r}"
        assert ok is True
        assert store["char"]["sha256"] == _digest(body)

    def test_a_restart_over_another_files_partial_reports_progress_from_zero(
            self, fake_registry, tmp_path, monkeypatch, capsys):
        import json
        dest_dir = tmp_path / "comfyui-models" / "loras"
        _interrupt_pull(monkeypatch, self._resolved(source_tag="civitai:111"),
                        dest_dir, _V1, 12, version_id="111")
        capsys.readouterr()
        monkeypatch.setenv("LOCALM_PROGRESS_JSON", "1")

        ok = _civitai_pull(monkeypatch, self._resolved(), dest_dir, _RangeServer(_V2),
                           version_id="222")

        out = capsys.readouterr().out
        events = [json.loads(line.split(mm.PROGRESS_SENTINEL, 1)[1])
                  for line in out.splitlines() if mm.PROGRESS_SENTINEL in line]
        downloads = [e for e in events if e.get("phase") == "download"]
        assert downloads, f"no download progress was reported: {out!r}"
        assert downloads[0]["downloaded"] == 0, downloads
        assert ok is True

    def test_a_restart_that_cannot_truncate_leaves_no_record_beside_the_old_bytes(
            self, fake_registry, tmp_path, monkeypatch):
        dest_dir = tmp_path / "comfyui-models" / "loras"
        _interrupt_pull(monkeypatch, self._resolved(source_tag="civitai:111"),
                        dest_dir, _V1, 12, version_id="111")
        part = dest_dir / (self.FILE + ".part")
        refused = []

        def _open_refusing_truncate(path, mode="r", *a, **kw):
            if Path(path) == part and "w" in mode:
                refused.append(mode)
                raise PermissionError(13, "file is in use", str(path))
            return open(path, mode, *a, **kw)

        monkeypatch.setattr(pull, "open", _open_refusing_truncate, raising=False)
        ok = _civitai_pull(monkeypatch, self._resolved(), dest_dir, _RangeServer(_V2),
                           version_id="222")
        monkeypatch.delattr(pull, "open")
        assert refused == ["wb"], "the truncate was never refused"
        assert ok is False
        server = _RangeServer(_V2)

        ok = _civitai_pull(monkeypatch, self._resolved(), dest_dir, server,
                           version_id="222")

        dest = dest_dir / self.FILE
        assert dest.is_file() and dest.read_bytes() == _V2
        assert [g.get("Range") for g in server.gets] == [None]
        assert ok is True

    def test_a_record_with_no_partial_beside_it_is_ignored(
            self, fake_registry, tmp_path, monkeypatch):
        dest_dir = tmp_path / "comfyui-models" / "loras"
        resolved = self._resolved()
        _interrupt_pull(monkeypatch, resolved, dest_dir, _V2, 12, version_id="222")
        (dest_dir / (self.FILE + ".part")).unlink()
        server = _RangeServer(_V2)

        ok = _civitai_pull(monkeypatch, resolved, dest_dir, server, version_id="222")

        dest = dest_dir / self.FILE
        assert dest.is_file() and dest.read_bytes() == _V2
        assert sorted(p.name for p in dest_dir.iterdir()) == [self.FILE]
        assert [g.get("Range") for g in server.gets] == [None]
        assert ok is True

    @pytest.mark.parametrize("record", [None, b"{not json", b"[]"],
                             ids=["no-record", "unreadable-record", "not-an-object"])
    def test_a_partial_without_a_usable_record_is_not_resumed(
            self, record, fake_registry, tmp_path, monkeypatch):
        dest_dir = tmp_path / "comfyui-models" / "loras"
        dest_dir.mkdir(parents=True)
        (dest_dir / (self.FILE + ".part")).write_bytes(b"GARBAGE")
        if record is not None:
            (dest_dir / (self.FILE + ".part.json")).write_bytes(record)
        server = _RangeServer(_V2)

        ok = _civitai_pull(monkeypatch, self._resolved(), dest_dir, server,
                           version_id="222")

        dest = dest_dir / self.FILE
        assert dest.is_file() and dest.read_bytes() == _V2
        assert sorted(p.name for p in dest_dir.iterdir()) == [self.FILE]
        assert [g.get("Range") for g in server.gets] == [None]
        assert ok is True

    def test_server_ignoring_range_restarts_clean(self, fake_registry, tmp_path, monkeypatch):
        dest_dir = tmp_path / "comfyui-models" / "loras"
        resolved = _resolved()
        _interrupt_pull(monkeypatch, resolved, dest_dir, b"fake-lora-bytes", 7)
        gets = []

        def _ignores_range(method, url, **kw):
            gets.append(dict(kw.get("headers") or {}))
            return _FakeStreamResponse(b"fake-lora-bytes")

        ok = _civitai_pull(monkeypatch, resolved, dest_dir, _ignores_range)

        dest = dest_dir / "add-detail-xl.safetensors"
        assert dest.read_bytes() == b"fake-lora-bytes"
        assert [g.get("Range") for g in gets] == ["bytes=7-"]
        assert ok is True

    def test_range_not_satisfiable_resets_and_retries(self, fake_registry, tmp_path, monkeypatch):
        dest_dir = tmp_path / "comfyui-models" / "loras"
        resolved = _resolved()
        _interrupt_pull(monkeypatch, resolved, dest_dir, b"fake-lora-bytes", 5)
        calls = []

        def _pinned_req(method, url, **kw):
            calls.append(dict(kw.get("headers") or {}))
            if len(calls) == 1:
                return _FakeStreamResponse(b"", status_code=416)
            return _FakeStreamResponse(b"fake-lora-bytes")

        ok = _civitai_pull(monkeypatch, resolved, dest_dir, _pinned_req)

        dest = dest_dir / "add-detail-xl.safetensors"
        assert dest.read_bytes() == b"fake-lora-bytes"
        assert sorted(p.name for p in dest_dir.iterdir()) == ["add-detail-xl.safetensors"]
        assert [c.get("Range") for c in calls] == ["bytes=5-", None]
        assert ok is True


class TestPullCivitaiFileUnderThePartLock:
    """What a pull decides from its partial and its destination file is read
    after it takes the part lock, and a finished file already in place is never
    replaced by one that failed verification."""

    FILE = "add-detail-xl.safetensors"
    BODY = b"0123456789ABCDEFGHIJ"

    def _resolved(self) -> ResolvedDownload:
        return _resolved(sha256=_digest(self.BODY), size_bytes=len(self.BODY))

    def _before_the_lock(self, monkeypatch, action) -> list:
        """Run *action* immediately before the pull takes the real part lock;
        returns the filenames the hook fired for."""
        real = pull._part_lock
        fired = []

        @contextlib.contextmanager
        def _hooked(filename):
            fired.append(filename)
            action()
            with real(filename):
                yield

        monkeypatch.setattr(pull, "_part_lock", _hooked)
        return fired

    def test_a_file_another_pull_finished_first_is_kept(
            self, fake_registry, tmp_path, monkeypatch):
        store, _ = fake_registry
        dest_dir = tmp_path / "comfyui-models" / "loras"
        resolved = self._resolved()
        part = _interrupt_pull(monkeypatch, resolved, dest_dir, self.BODY, 10)
        dest = dest_dir / self.FILE

        def _other_pull_finishes():
            dest.write_bytes(self.BODY)
            for p in dest_dir.glob(self.FILE + ".part*"):
                p.unlink()

        fired = self._before_the_lock(monkeypatch, _other_pull_finishes)
        server = _RangeServer(self.BODY)

        exc = None
        try:
            ok = _civitai_pull(monkeypatch, resolved, dest_dir, server)
        except Exception as e:
            ok, exc = None, e

        assert fired == [self.FILE], "the other pull was never simulated"
        assert dest.is_file() and dest.read_bytes() == self.BODY, (
            "the finished model file was lost")
        assert not part.exists(), f"a tail-only partial was left: {part.read_bytes()!r}"
        assert exc is None, f"the pull raised {exc!r}"
        assert ok is True
        assert server.gets == [], "a file already in place was downloaded again"
        assert store["add-detail-xl"]["sha256"] == _digest(self.BODY)

    def test_a_partial_that_grew_before_the_lock_resumes_from_its_new_end(
            self, fake_registry, tmp_path, monkeypatch):
        dest_dir = tmp_path / "comfyui-models" / "loras"
        resolved = self._resolved()
        part = _interrupt_pull(monkeypatch, resolved, dest_dir, self.BODY, 10)

        def _other_pull_adds_bytes():
            with open(part, "ab") as f:
                f.write(self.BODY[10:15])

        fired = self._before_the_lock(monkeypatch, _other_pull_adds_bytes)
        server = _RangeServer(self.BODY)

        ok = _civitai_pull(monkeypatch, resolved, dest_dir, server)

        dest = dest_dir / self.FILE
        assert fired == [self.FILE], "the other pull was never simulated"
        assert dest.is_file() and dest.read_bytes() == self.BODY
        assert [g.get("Range") for g in server.gets] == ["bytes=15-"]
        assert ok is True

    def test_a_redownload_that_fails_verification_keeps_the_file_already_there(
            self, fake_registry, tmp_path, monkeypatch):
        dest_dir = tmp_path / "comfyui-models" / "loras"
        dest_dir.mkdir(parents=True)
        dest = dest_dir / self.FILE
        dest.write_bytes(self.BODY)
        server = _RangeServer(b"corrupted-in-transit")

        exc = None
        try:
            ok = _civitai_pull(monkeypatch, self._resolved(), dest_dir, server,
                               redownload=True)
        except Exception as e:
            ok, exc = None, e

        assert dest.is_file() and dest.read_bytes() == self.BODY, (
            "the verified copy already in place was destroyed")
        assert sorted(p.name for p in dest_dir.iterdir()) == [self.FILE]
        assert exc is None, f"the pull raised {exc!r}"
        assert ok is False

    def test_a_redownload_replaces_the_file_already_there(
            self, fake_registry, tmp_path, monkeypatch):
        dest_dir = tmp_path / "comfyui-models" / "loras"
        dest_dir.mkdir(parents=True)
        dest = dest_dir / self.FILE
        dest.write_bytes(b"an-older-copy-of-it!")
        server = _RangeServer(self.BODY)

        exc = None
        try:
            ok = _civitai_pull(monkeypatch, self._resolved(), dest_dir, server,
                               redownload=True)
        except Exception as e:
            ok, exc = None, e

        assert dest.read_bytes() == self.BODY
        assert sorted(p.name for p in dest_dir.iterdir()) == [self.FILE]
        assert exc is None, f"the pull raised {exc!r}"
        assert ok is True


class TestPullModelCivitaiDispatch:
    def test_parses_version_and_file_id(self, monkeypatch):
        monkeypatch.setenv("LOCALM_NET_MODE", "ask")
        calls = []
        monkeypatch.setattr(
            mm, "_pull_civitai_file",
            lambda version_id, name, **kw: calls.append((version_id, kw)) or True)
        assert mm.pull_model("civitai:135867:99264") is True
        assert calls[0][0] == "135867"
        assert calls[0][1]["file_id"] == "99264"

    def test_parses_bare_version_with_no_file_id(self, monkeypatch):
        monkeypatch.setenv("LOCALM_NET_MODE", "ask")
        calls = []
        monkeypatch.setattr(
            mm, "_pull_civitai_file",
            lambda version_id, name, **kw: calls.append((version_id, kw)) or True)
        assert mm.pull_model("civitai:135867") is True
        assert calls[0][0] == "135867"
        assert calls[0][1]["file_id"] is None

    def test_refuses_an_explicit_dest_dir(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LOCALM_NET_MODE", "ask")
        called = []
        monkeypatch.setattr(mm, "_pull_civitai_file", lambda *a, **kw: called.append(1) or True)
        assert mm.pull_model("civitai:135867", dest_dir=tmp_path) is False
        assert called == []

    def test_net_mode_off_refuses_before_dispatch(self, monkeypatch):
        import localm.netpolicy as netpolicy
        monkeypatch.setattr(netpolicy, "network_mode", lambda: "off")
        called = []
        monkeypatch.setattr(mm, "_pull_civitai_file", lambda *a, **kw: called.append(1) or True)
        assert mm.pull_model("civitai:135867") is False
        assert called == []

    def test_mmproj_spec_is_ignored_not_crashed_on(self, monkeypatch):
        monkeypatch.setenv("LOCALM_NET_MODE", "ask")
        monkeypatch.setattr(mm, "_pull_civitai_file", lambda *a, **kw: True)
        assert mm.pull_model("civitai:135867", mmproj_spec="org/repo:mmproj.gguf") is True

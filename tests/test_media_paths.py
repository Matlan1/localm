# SPDX-License-Identifier: AGPL-3.0-or-later
"""Filesystem confinement on the media plugins' caller-supplied paths, and on the
log-export destination.

Three routes reach the filesystem, all of them reachable by a scoped NON-owner
key:

(a) ``input_image`` must not be confined to the whole data dir - which is
    localm's credential store (auth.key, the plaintext owner key; auth.json;
    sessions.json; rag/; coder/; bug-reports/) - because the img2img path
    UPLOADS the file to ComfyUI before any image validation, over an api_url
    sanitize_comfy_url permits to be a LAN or public host on plaintext http.

(b) ``POST /api/{imagine,video,music}/file/{name}/move`` takes ``dest`` verbatim
    into mkdir(parents=True) + shutil.move, gated only on gallery.require_owner -
    which proves ARTIFACT ownership, and passes for ANY caller when the artifact
    has no recorded owner (open mode, legacy or hand-placed files).

(c) ``POST /api/logs/export`` needs require_fs_host, like the /api/fs/dirs picker
    that supplies its ``dest``: a config:write key with fs_access="none" could
    otherwise mkdir + write anywhere, and the exists-or-not 400 is a
    directory-existence oracle for the whole disk.

On the /move tests the dest-not-created assertions are load-bearing twice over:
they are the security property, AND mkdir(parents=True) means a regression would
litter the test tree with directories it was not supposed to make. That is NOT
true of the log-export denials - export_logs 400s on a missing dest before it
mkdirs anything - so those tests point at a dest that EXISTS and assert nothing
was written INTO it, which is the only falsifiable form.

Every path in this file is built from tmp_path, never a real location. Proving
this oracle works means reverting the fix and re-running, and a negative pass
executes the unsafe path for real.
"""

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from localm.media import paths as media_paths
from localm.plugins.gui.web import attach_gui


def _h(key):
    return {"Authorization": f"Bearer {key}"}


def _media_app(tmp_path, monkeypatch, plugin):
    """The gallery-plugin app harness from test_media_gallery_ownership, with the
    data dir pinned inside tmp_path so "outside the data dir" is expressible."""
    home = tmp_path / ".localm"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    import localm.config as _cfg
    monkeypatch.setattr(_cfg, "HOME_DIR", home)
    monkeypatch.setattr(_cfg, "MODELS_DIR", home / "models")
    monkeypatch.setattr(_cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(_cfg, "REGISTRY_FILE", home / "registry.json")
    from localm.plugins.engine import PluginManager
    app = FastAPI()
    PluginManager(app, external_root=tmp_path / "noplugins").install(plugin)

    async def switch_model(name):
        pass

    attach_gui(app, self_url="http://127.0.0.1:9/v1",
               switch_model=switch_model, active_model=lambda: "model-a")
    return app, home


def _key(scope_list, fs_access="none", privileged=False):
    from localm import auth
    return auth.create_key("k", scope_list, fs_access=fs_access,
                           allow_privileged=privileged)["key"]


def _no_upload(monkeypatch):
    """Record any ComfyUI image upload.

    Patched on both generator modules, not on media.comfy_client: they do
    ``from ...comfy_client import _upload_image`` at import time, so the name is
    already bound in their own namespace and patching the source module would
    silently miss.

    Generation is dispatched to a JobManager worker thread (gui/jobs.py
    start_fn) and _upload_image is reached only deep inside it, so an assertion
    on this list right after the HTTP response returns is INERT - it would be
    empty even if confinement had let the file through. A cheap tripwire, not
    the check. ``_job_count`` below is the assertion that can actually fail."""
    calls = []

    def _record(image_path, api_url):
        calls.append(str(image_path))
        raise AssertionError(f"_upload_image was called with {image_path}")

    import localm.image_gen.comfy as _img
    import localm.video_gen.comfy as _vid
    monkeypatch.setattr(_img, "_upload_image", _record)
    monkeypatch.setattr(_vid, "_upload_image", _record)
    return calls


def _dead_backend(monkeypatch):
    """Make the media backends inert for the whole test.

    CONTAINMENT FOR THE NEGATIVE PASS, not for the passing run. Proving this
    oracle works means reverting the fix and re-running, and a negative pass
    executes the UNSAFE path for real. Against unfixed source these input_image
    requests do NOT 400; they return 200 and ``jobs.start_fn`` dispatches a real
    generation on a worker thread, which calls ``_backend.ensure_available`` ->
    ComfyUI probe/launch. The passing run never reaches it. Stubbing
    ensure_available to a hard False means the worker returns immediately on
    either source, so the negative pass cannot dial or spawn anything.

    Payload containment is separate and already handled: every path in this file
    is built from tmp_path, never a real location."""
    from localm.plugins.builtin.image import backend as _ib
    from localm.plugins.builtin.video import backend as _vb
    for mod in (_ib, _vb):
        monkeypatch.setattr(mod, "ensure_available",
                            lambda s, *a, **k: (False, "backend disabled in test"))


def _job_count(app) -> int:
    """How many jobs the GUI job manager holds.

    The falsifiable half of the input_image tests: confinement runs BEFORE
    ``jobs.start_fn``, so a refused request must leave the job list untouched.
    If the confinement regressed, the route would 200 and a job WOULD appear
    here - unlike the _upload_image tripwire, this is observable synchronously."""
    jobs = getattr(app.state, "jobs", None)
    assert jobs is not None, "harness bug: attach_gui did not publish app.state.jobs"
    return len(jobs._jobs)          # gui/jobs.py:214 - the only job registry


# --------------------------------------------------------------------------- #
#  (a) input_image may not name localm's own credential store
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "plugin,route,body_extra",
    [("image", "/api/imagine", {}),
     ("video", "/api/video", {})],
    ids=["image", "video"],
)
@pytest.mark.parametrize("secret_name", ["auth.key", "auth.json", "sessions.json"])
def test_input_image_cannot_name_the_credential_store(
        tmp_path, monkeypatch, plugin, route, body_extra, secret_name):
    app, home = _media_app(tmp_path, monkeypatch, plugin)
    _dead_backend(monkeypatch)
    uploaded = _no_upload(monkeypatch)
    secret = home / secret_name
    secret.write_bytes(b"\x89PNG\r\n\x1a\nowner-key-material")  # readable AND image-shaped
    key = _key([plugin])                       # media scope only, fs_access=none

    with TestClient(app) as c:
        before = _job_count(app)
        r = c.post(route, headers=_h(key),
                   json={"prompt": "x", "input_image": str(secret), **body_extra})
        after = _job_count(app)
    assert r.status_code == 400, r.text
    # Confinement runs BEFORE jobs.start_fn, so a refusal must not have queued
    # generation.
    assert after == before, "a generation job was queued for the credential file"
    assert not uploaded
    assert str(home) not in r.text, "rejection must not disclose the data dir"


@pytest.mark.parametrize(
    "plugin,route",
    [("image", "/api/imagine"), ("video", "/api/video")],
    ids=["image", "video"],
)
def test_input_image_outside_the_data_dir_still_rejected(tmp_path, monkeypatch,
                                                         plugin, route):
    app, _home = _media_app(tmp_path, monkeypatch, plugin)
    _dead_backend(monkeypatch)
    uploaded = _no_upload(monkeypatch)
    outside = tmp_path / "elsewhere" / "secret.png"
    outside.parent.mkdir(parents=True)
    outside.write_bytes(b"\x89PNG\r\n\x1a\nNOTYOURS")
    key = _key([plugin])

    with TestClient(app) as c:
        before = _job_count(app)
        r = c.post(route, headers=_h(key),
                   json={"prompt": "x", "input_image": str(outside)})
        after = _job_count(app)
    assert r.status_code == 400, r.text
    assert after == before, "a generation job was queued for an out-of-root file"
    assert not uploaded


def test_allowed_input_roots_excludes_the_data_dir_root(tmp_path, monkeypatch):
    """The policy itself, independent of any route: the data dir must not be a
    root, and the upload inbox plus all three galleries must be."""
    home = tmp_path / ".localm"
    home.mkdir()
    monkeypatch.setenv("LOCALM_HOME", str(home))
    import localm.config as _cfg
    monkeypatch.setattr(_cfg, "HOME_DIR", home)

    roots = {p.resolve() for p in media_paths.allowed_input_roots()}
    assert home.resolve() not in roots
    for name in ("uploads", "gui_images", "gui_video", "gui_music"):
        assert (home / name).resolve() in roots, name


def test_the_install_directory_is_not_an_allowed_root():
    """The localm install tree must NOT be readable through this route.

    A repo-root allowance guarded on pyproject.toml narrows nothing (pyproject.toml
    is release-include, so an installed copy has one too), and on a `git clone`
    install it admits the whole tree - including the gitignored issues/ and qa/
    directories that hold bug-report screenshots. This test pins its absence; do
    not reintroduce it."""
    install_root = Path(media_paths.__file__).resolve().parents[2]
    assert (install_root / "pyproject.toml").is_file(), "test runs from a checkout"
    roots = {p.resolve() for p in media_paths.allowed_input_roots()}
    assert install_root not in roots
    assert not any(media_paths._under(install_root, r) for r in roots)
    assert not hasattr(media_paths, "source_checkout_root"), \
        "the install-root allowance was reintroduced"


def test_unresolvable_data_dir_fails_closed_with_its_own_reason(tmp_path, monkeypatch):
    """A security step that cannot RUN must not read as a routine policy refusal,
    and must not widen the policy. With every allowed root a subdir OF the data
    dir, losing the data dir means nothing is permitted."""
    monkeypatch.setattr(media_paths, "_resolved_home", lambda: None)
    assert media_paths.allowed_input_roots() == []
    with pytest.raises(Exception) as ei:
        media_paths.confined_input_image(str(tmp_path / "anything.png"))
    # 500 (a fault), NOT the ordinary 400 refusal - the two must stay distinct.
    assert getattr(ei.value, "status_code", None) == 500
    assert "data directory" in str(ei.value.detail)


_UNC_FORMS = [
    "\\\\192.0.2.1\\share\\x.png",     # canonical UNC (RFC5737 documentation address)
    "//192.0.2.1/share/x.png",         # forward-slash UNC
    "\\/192.0.2.1\\share\\x.png",      # mixed separators, form 1
    "/\\192.0.2.1/share/x.png",        # mixed separators, form 2
    "\\\\.\\PhysicalDrive0",           # device namespace
    "\\\\?\\C:\\Windows\\win.ini",     # extended-length namespace
]


@pytest.mark.parametrize("raw", _UNC_FORMS)
def test_unc_input_is_refused_without_ever_touching_the_filesystem(
        tmp_path, monkeypatch, raw):
    """A UNC dest would be refused by the allowlist anyway - but only AFTER
    .resolve() had dialled SMB, which can stall a whole async handler for
    minutes, and against a REACHABLE share Windows also surrenders the host
    net-NTLMv2 credential. So the refusal must happen on the STRING, before any
    syscall.

    All four separator mixes are covered: ntpath parses //h/s, \\/h/s and /\\h/s
    to the same drive, so a check that only looks for a leading \\\\ is bypassed
    by typing the path a different way. No address here is routable."""
    home = tmp_path / ".localm"
    (home / "uploads").mkdir(parents=True)
    monkeypatch.setenv("LOCALM_HOME", str(home))
    import localm.config as _cfg
    monkeypatch.setattr(_cfg, "HOME_DIR", home)

    # Fail the test rather than hang it if a syscall is ever attempted on the
    # attacker-supplied path. Guarding Path.resolve directly (not sleeping or
    # timing) keeps this deterministic and fast on every OS, including Linux
    # where a UNC string is just an odd relative path and would never block.
    real_resolve = Path.resolve

    def _guard(self, *a, **k):
        if media_paths.is_unc_or_device_path(str(self)):
            raise AssertionError(f"resolve() was called on the UNC path {self}")
        return real_resolve(self, *a, **k)
    monkeypatch.setattr(Path, "resolve", _guard)

    with pytest.raises(Exception) as ei:
        media_paths.confined_input_image(raw)
    assert getattr(ei.value, "status_code", None) == 400


@pytest.mark.parametrize("raw", _UNC_FORMS)
def test_is_unc_or_device_path_matches_every_separator_mix(raw):
    assert media_paths.is_unc_or_device_path(raw)


@pytest.mark.parametrize("raw", [
    "", " ", "x.png", "C:/Users/x/pic.png", "/tmp/pic.png", "./rel.png", "\\",
])
def test_is_unc_or_device_path_does_not_over_match(raw):
    """The control for the detector above: it must NOT fire on ordinary local
    paths, or every legitimate input would be refused as a network path."""
    assert not media_paths.is_unc_or_device_path(raw)


def test_a_symlink_out_of_an_allowed_root_is_rejected(tmp_path, monkeypatch):
    """confined_input_image's docstring promises symlinks are resolved first, so
    a link INSIDE an allowed root that targets outside it is still rejected.
    That property rests entirely on the single .resolve() call, and nothing else
    in the function would notice if it stopped resolving links."""
    home = tmp_path / ".localm"
    (home / "uploads").mkdir(parents=True)
    monkeypatch.setenv("LOCALM_HOME", str(home))
    import localm.config as _cfg
    monkeypatch.setattr(_cfg, "HOME_DIR", home)

    secret = home / "auth.key"                     # the credential store itself
    secret.write_bytes(b"\x89PNG\r\n\x1a\nowner-key-material")
    link = home / "uploads" / "innocent.png"       # lexically inside an allowed root
    try:
        link.symlink_to(secret)
    except (OSError, NotImplementedError) as e:
        pytest.skip(f"this OS/account cannot create symlinks: {e}")

    assert link.is_file(), "precondition: the link resolves to a real file"
    with pytest.raises(Exception) as ei:
        media_paths.confined_input_image(str(link))
    assert getattr(ei.value, "status_code", None) == 400


def test_input_image_from_the_gallery_and_uploads_still_accepted(tmp_path, monkeypatch):
    """The legitimate flows must survive the narrowing: an uploaded file, and a
    previously generated image reused as img2img input (the GUI's "use as
    input" button fills the field with a gui_images path)."""
    home = tmp_path / ".localm"
    home.mkdir()
    monkeypatch.setenv("LOCALM_HOME", str(home))
    import localm.config as _cfg
    monkeypatch.setattr(_cfg, "HOME_DIR", home)

    for sub in ("uploads", "gui_images"):
        src = home / sub / "ref.png"
        src.parent.mkdir(parents=True, exist_ok=True)
        src.write_bytes(b"\x89PNG\r\n\x1a\nMINE")
        assert media_paths.confined_input_image(str(src)) == src.resolve()


# --------------------------------------------------------------------------- #
#  (a2) the upload choke point refuses a non-image outright
# --------------------------------------------------------------------------- #

def test_upload_image_refuses_a_non_image_before_transmitting(tmp_path, monkeypatch):
    """The backstop that also covers the CLI and an owner-key caller, for whom
    the path policy above is wider. Nothing may be read into a
    request body or a socket opened for a file that is not an image."""
    import localm.media.comfy_client as cc
    secret = tmp_path / "auth.key"
    secret.write_text("sk-not-an-image")

    def _no_socket(*a, **k):
        raise AssertionError("a request was built for a non-image")
    # _upload_image routes through cc._comfy_urlopen, which builds its own opener
    # and never calls urllib.request.urlopen.
    monkeypatch.setattr(cc, "_comfy_urlopen", _no_socket)
    monkeypatch.setattr(cc.urllib.request, "Request", _no_socket)

    with pytest.raises(ValueError, match="not in a format this upload supports"):
        cc._upload_image(secret, "http://127.0.0.1:8188")

    # The refusal must NOT claim the file is not an image. This allowlist is
    # narrower than image: localm accepts .heic/.heif elsewhere (gui/web.py
    # _SHARE_IMAGE_EXTS, an ordinary iPhone photo), and those land here too.
    with pytest.raises(ValueError) as ei:
        cc._upload_image(secret, "http://127.0.0.1:8188")
    assert "is not an image" not in str(ei.value)


def test_upload_image_transmits_a_real_webp_end_to_end(tmp_path, monkeypatch):
    """Drive _upload_image PAST the gate with a real file, not just the pure
    signature function.

    Narrowing the read window at comfy_client.py (`head = f.read(16)`) to 8 bytes
    would break every real WebP upload - WebP's second signature window is at
    offset 8..12 - while every other test in this file either asserts a REFUSAL
    or calls looks_like_image directly. WebP is the format that needs the wider
    window."""
    import localm.media.comfy_client as cc
    img = tmp_path / "ref.webp"
    img.write_bytes(b"RIFF\x24\x00\x00\x00WEBPVP8 " + b"\x00" * 32)

    sent = {}

    class _FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b'{"name": "uploaded-ref.webp"}'

    def _fake_request(url, data=None, headers=None, method=None):
        sent["url"] = url
        sent["body"] = data
        return "REQ"

    monkeypatch.setattr(cc.urllib.request, "Request", _fake_request)
    monkeypatch.setattr(cc, "_comfy_urlopen",
                        lambda req, timeout=None: _FakeResp())

    assert cc._upload_image(img, "http://127.0.0.1:8188") == "uploaded-ref.webp"
    assert sent["url"].endswith("/upload/image")
    assert b"RIFF" in sent["body"], "the real image bytes were not transmitted"


@pytest.mark.parametrize("head", [
    b"\x89PNG\r\n\x1a\n....",
    b"\xff\xd8\xff\xe0" + b"\x00" * 12,
    b"GIF89a" + b"\x00" * 10,
    b"BM" + b"\x00" * 14,
    b"RIFF\x00\x00\x00\x00WEBP",
    b"II*\x00" + b"\x00" * 12,
    b"MM\x00*" + b"\x00" * 12,
], ids=["png", "jpeg", "gif", "bmp", "webp", "tiff-le", "tiff-be"])
def test_looks_like_image_accepts_every_supported_format(head):
    from localm.media.comfy_client import looks_like_image
    assert looks_like_image(head)


@pytest.mark.parametrize("head", [
    b"", b"sk-live-abcdef", b"{\"key\": \"secret\"}", b"\x89PNG not really",
    b"RIFF\x00\x00\x00\x00WAVE",          # a RIFF container that is NOT WebP
], ids=["empty", "api-key", "json", "truncated-png-sig", "riff-wave"])
def test_looks_like_image_rejects_non_images(head):
    from localm.media.comfy_client import looks_like_image
    assert not looks_like_image(head)


# --------------------------------------------------------------------------- #
#  (b) /move dest may not create directories anywhere on the disk
# --------------------------------------------------------------------------- #

_MOVE_CASES = [
    ("image", "gui_images", "img.png", "/api/imagine/file/{n}/move"),
    ("video", "gui_video", "clip.mp4", "/api/video/file/{n}/move"),
    ("music", "gui_music", "track.flac", "/api/music/file/{n}/move"),
]


@pytest.mark.parametrize("plugin,subdir,fname,route", _MOVE_CASES,
                         ids=[c[0] for c in _MOVE_CASES])
def test_move_dest_outside_the_data_dir_denied_for_a_non_host_key(
        tmp_path, monkeypatch, plugin, subdir, fname, route):
    """An UNOWNED artifact: require_owner passes for any caller then, so it is
    not an authorization gate for the destination."""
    app, home = _media_app(tmp_path, monkeypatch, plugin)
    art = home / subdir / fname
    art.parent.mkdir(parents=True, exist_ok=True)
    art.write_bytes(b"artifact")
    dest = tmp_path / "exfil" / "drop"
    key = _key([plugin])                       # media scope only, fs_access=none

    with TestClient(app) as c:
        r = c.post(route.format(n=fname), headers=_h(key),
                   json={"dest": str(dest)})
    assert r.status_code == 403, r.text
    assert not dest.exists(), "a denied move must not create the destination"
    assert not dest.parent.exists(), "mkdir(parents=True) ran anyway"
    assert art.is_file(), "the artifact must stay put"


@pytest.mark.parametrize("plugin,subdir,fname,route", _MOVE_CASES,
                         ids=[c[0] for c in _MOVE_CASES])
def test_move_inside_the_data_dir_still_allowed_for_a_non_host_key(
        tmp_path, monkeypatch, plugin, subdir, fname, route):
    app, home = _media_app(tmp_path, monkeypatch, plugin)
    art = home / subdir / fname
    art.parent.mkdir(parents=True, exist_ok=True)
    art.write_bytes(b"artifact")
    dest = home / "kept" / plugin
    key = _key([plugin])

    with TestClient(app) as c:
        r = c.post(route.format(n=fname), headers=_h(key),
                   json={"dest": str(dest)})
    assert r.status_code == 200, r.text
    assert (dest / fname).is_file()
    assert not art.exists()


@pytest.mark.parametrize("plugin,subdir,fname,route", _MOVE_CASES,
                         ids=[c[0] for c in _MOVE_CASES])
def test_move_anywhere_still_allowed_for_a_host_fs_key(
        tmp_path, monkeypatch, plugin, subdir, fname, route):
    """"Any folder on this machine" is the documented feature for a principal
    the owner granted host filesystem access - the same dial the /api/fs/dirs
    picker that supplies `dest` already requires. It must NOT be removed."""
    app, home = _media_app(tmp_path, monkeypatch, plugin)
    art = home / subdir / fname
    art.parent.mkdir(parents=True, exist_ok=True)
    art.write_bytes(b"artifact")
    dest = tmp_path / "pictures" / plugin
    key = _key([plugin], fs_access="host")

    with TestClient(app) as c:
        r = c.post(route.format(n=fname), headers=_h(key),
                   json={"dest": str(dest)})
    assert r.status_code == 200, r.text
    assert (dest / fname).is_file()


@pytest.mark.parametrize("plugin,subdir,fname,route", _MOVE_CASES,
                         ids=[c[0] for c in _MOVE_CASES])
def test_move_dest_alias_does_not_clobber_a_different_real_file(
        tmp_path, monkeypatch, plugin, subdir, fname, route):
    """confined_move_dest's own containment check (media.paths._under) has no
    name-preservation walk, unlike pathsafe.confined_under /
    confined_absolute_or_under - so an OS-level short-name alias resolving the
    caller's dest string to a DIFFERENT real directory than it names would defeat
    containment-by-string IF anything downstream trusted the caller's own
    spelling for the write.

    It does not: the eventual write target is built from the ALREADY-.resolve()d
    destination directory (no attacker alias syntax survives that step) joined
    with the SOURCE artifact's own basename - which reached this handler via
    pathsafe.confined_file, never from caller text. The exists-check and the move
    then act on that one concrete object; there is no second, divergent
    resolution for the check to lie about. So an aliased dest can only relocate
    the write to a REAL directory inside the confined data dir - it cannot
    desynchronize the collision check from the write.

    Deterministic simulation: monkeypatch Path.resolve so a typed alias component
    substitutes for the real directory's name, exactly as an actual 8.3 short
    name would."""
    app, home = _media_app(tmp_path, monkeypatch, plugin)
    art = home / subdir / fname
    art.parent.mkdir(parents=True, exist_ok=True)
    art.write_bytes(b"artifact-content")

    real_dir = home / "LongDestinationFolderName"
    real_dir.mkdir(parents=True, exist_ok=True)
    victim = real_dir / fname
    victim.write_bytes(b"VICTIM-DATA-DO-NOT-OVERWRITE")

    alias_name = "LONGDE~1"
    real_resolve = Path.resolve

    def fake_resolve(self, *a, **k):
        parts = list(self.parts)
        if alias_name in parts:
            parts[parts.index(alias_name)] = real_dir.name
            return real_resolve(Path(*parts), *a, **k)
        return real_resolve(self, *a, **k)

    monkeypatch.setattr(Path, "resolve", fake_resolve)
    dest = home / alias_name
    key = _key([plugin])                        # media scope only, fs_access=none
    target_path = route.replace("{n}", fname)

    with TestClient(app) as c:
        r = c.post(target_path, headers=_h(key), json={"dest": str(dest)})
    # The collision is caught (409) rather than silently clobbered, and the
    # victim's content is asserted as well as the status code.
    assert victim.read_bytes() == b"VICTIM-DATA-DO-NOT-OVERWRITE", (
        f"the aliased move clobbered a different real file (status {r.status_code})")
    if r.status_code == 200:
        assert art.exists(), "a 200 that never wrote `victim` must not have moved the source either"


# --------------------------------------------------------------------------- #
#  (c) /api/logs/export needs host filesystem access
# --------------------------------------------------------------------------- #

def test_logs_export_denied_without_fs_host(tmp_path, monkeypatch):
    from localm import scopes as S
    app, home = _media_app(tmp_path, monkeypatch, "image")
    (home / "logs").mkdir(parents=True, exist_ok=True)
    (home / "logs" / "server.log").write_text("secret-ish log content")
    key = _key([S.CONFIG_WRITE], privileged=True)          # fs_access="none"
    # dest must EXIST: export_logs 400s on a missing dest before it ever mkdirs,
    # so only an existing dest exercises the write.
    dest = tmp_path / "logsteal"
    dest.mkdir()

    with TestClient(app) as c:
        r = c.post("/api/logs/export", headers=_h(key), json={"dest": str(dest)})
    assert r.status_code == 403, r.text
    assert list(dest.iterdir()) == [], "denied export still wrote into dest"


def test_logs_export_allowed_with_fs_host(tmp_path, monkeypatch):
    from localm import scopes as S
    app, home = _media_app(tmp_path, monkeypatch, "image")
    (home / "logs").mkdir(parents=True, exist_ok=True)
    (home / "logs" / "server.log").write_text("hello")
    key = _key([S.CONFIG_WRITE], fs_access="host", privileged=True)
    dest = tmp_path / "logs-out"
    dest.mkdir()

    with TestClient(app) as c:
        r = c.post("/api/logs/export", headers=_h(key), json={"dest": str(dest)})
    assert r.status_code == 200, r.text
    assert list(dest.glob("localm-logs-*")), "export produced no folder"

# SPDX-License-Identifier: AGPL-3.0-or-later
"""Filesystem confinement on the media plugins' caller-supplied paths, and on the
log-export destination (CodeQL WS8: alerts 2, 54, 55, 56, 58, 64).

Three routes reached the filesystem outside any meaningful confinement, all of
them reachable by a scoped NON-owner key:

(a) ``input_image`` was confined to the whole data dir - which is localm's
    credential store (auth.key, the plaintext owner key; auth.json;
    sessions.json; rag/; coder/; bug-reports/) - and the img2img path UPLOADS
    the file to ComfyUI before any image validation, over an api_url
    sanitize_comfy_url deliberately permits to be a LAN or public host on
    plaintext http. So the confinement did not remove the arbitrary-file
    read-and-transmit primitive its own docstring named, it retargeted it at
    localm's own secrets.

(b) ``POST /api/{imagine,video,music}/file/{name}/move`` took ``dest`` verbatim
    into mkdir(parents=True) + shutil.move, gated only on gallery.require_owner
    - which proves ARTIFACT ownership, and passes for ANY caller when the
    artifact has no recorded owner (open mode, legacy or hand-placed files).

(c) ``POST /api/logs/export`` had no require_fs_host even though the
    /api/fs/dirs picker that supplies its ``dest`` does, so a config:write key
    with fs_access="none" could mkdir + write anywhere, and the exists-or-not
    400 was a directory-existence oracle for the whole disk.

The dest-not-created assertions are load-bearing twice over: they are the real
security property, AND mkdir(parents=True) means a regression would otherwise
litter the test tree with the directories it was not supposed to make.
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
    """Make any ComfyUI image upload a hard test failure.

    Patched on both generator modules, not on media.comfy_client: they do
    ``from ...comfy_client import _upload_image`` at import time, so the name is
    already bound in their own namespace and patching the source module would
    silently miss."""
    calls = []

    def _boom(image_path, api_url):
        calls.append(str(image_path))
        raise AssertionError(f"_upload_image was called with {image_path}")

    import localm.image_gen.comfy as _img
    import localm.video_gen.comfy as _vid
    monkeypatch.setattr(_img, "_upload_image", _boom)
    monkeypatch.setattr(_vid, "_upload_image", _boom)
    return calls


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
    uploaded = _no_upload(monkeypatch)
    secret = home / secret_name
    secret.write_bytes(b"\x89PNG\r\n\x1a\nowner-key-material")  # readable AND image-shaped
    key = _key([plugin])                       # media scope only, fs_access=none

    with TestClient(app) as c:
        r = c.post(route, headers=_h(key),
                   json={"prompt": "x", "input_image": str(secret), **body_extra})
    assert r.status_code == 400, r.text
    assert not uploaded, "the file was handed to the ComfyUI upload path"
    assert str(home) not in r.text, "rejection must not disclose the data dir"


@pytest.mark.parametrize(
    "plugin,route",
    [("image", "/api/imagine"), ("video", "/api/video")],
    ids=["image", "video"],
)
def test_input_image_outside_the_data_dir_still_rejected(tmp_path, monkeypatch,
                                                         plugin, route):
    app, _home = _media_app(tmp_path, monkeypatch, plugin)
    uploaded = _no_upload(monkeypatch)
    outside = tmp_path / "elsewhere" / "secret.png"
    outside.parent.mkdir(parents=True)
    outside.write_bytes(b"\x89PNG\r\n\x1a\nNOTYOURS")
    key = _key([plugin])

    with TestClient(app) as c:
        r = c.post(route, headers=_h(key),
                   json={"prompt": "x", "input_image": str(outside)})
    assert r.status_code == 400, r.text
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


def test_repo_root_allowance_needs_a_real_checkout_not_just_pyproject():
    """The source-checkout allowance must key on something a release build does
    NOT have. pyproject.toml alone does not qualify: it is release-include (the
    updater's verify_zip requires it), so an installed copy carries one and the
    "source checkout only" guard would never actually narrow. A release zip is
    assembled from git-TRACKED files, so .git is the distinguishing marker."""
    real_root = Path(media_paths.__file__).resolve().parents[2]
    assert (real_root / "pyproject.toml").is_file(), "test runs from a checkout"
    assert media_paths.source_checkout_root() == real_root

    src = Path(media_paths.__file__).read_text(encoding="utf-8")
    assert '(root / ".git").exists()' in src, \
        "pyproject.toml alone would also match an installed copy"


def test_data_dir_is_refused_even_when_it_sits_inside_the_repo_root(tmp_path,
                                                                   monkeypatch):
    """The repo-root allowance can CONTAIN the data dir: with no LOCALM_HOME and
    no localm-home.cfg, localm falls back to <repo>/home. Without an explicit
    re-deny, the source-checkout root would readmit auth.key through the back
    door - defeating the whole narrowing on every from-source install."""
    fake_repo = tmp_path / "checkout"
    home = fake_repo / "home"                 # the fallback layout, verbatim
    (home / "uploads").mkdir(parents=True)
    monkeypatch.setenv("LOCALM_HOME", str(home))
    import localm.config as _cfg
    monkeypatch.setattr(_cfg, "HOME_DIR", home)
    monkeypatch.setattr(media_paths, "source_checkout_root", lambda: fake_repo)

    secret = home / "auth.key"
    secret.write_bytes(b"\x89PNG\r\n\x1a\nowner-key")
    assert fake_repo in {p.resolve() for p in media_paths.allowed_input_roots()}, \
        "precondition: the repo root really is an allowed root here"

    with pytest.raises(Exception) as ei:
        media_paths.confined_input_image(str(secret))
    assert getattr(ei.value, "status_code", None) == 400

    # ...and a legitimate file in an allowed subdir of that same data dir still
    # passes, so the re-deny did not over-reach.
    ok = home / "uploads" / "ref.png"
    ok.write_bytes(b"\x89PNG\r\n\x1a\nMINE")
    assert media_paths.confined_input_image(str(ok)) == ok.resolve()

    # A non-data-dir file elsewhere in the checkout is still allowed (that is
    # what the source-checkout allowance is FOR).
    asset = fake_repo / "examples" / "cat.png"
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b"\x89PNG\r\n\x1a\nASSET")
    assert media_paths.confined_input_image(str(asset)) == asset.resolve()


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
    the path policy above is deliberately wider. Nothing may be read into a
    request body or a socket opened for a file that is not an image."""
    import localm.media.comfy_client as cc
    secret = tmp_path / "auth.key"
    secret.write_text("sk-not-an-image")

    def _no_socket(*a, **k):
        raise AssertionError("a request was built for a non-image")
    monkeypatch.setattr(cc.urllib.request, "urlopen", _no_socket)
    monkeypatch.setattr(cc.urllib.request, "Request", _no_socket)

    with pytest.raises(ValueError, match="not an image"):
        cc._upload_image(secret, "http://127.0.0.1:8188")


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
    """An UNOWNED artifact deliberately: require_owner passes for any caller
    then, which is exactly why it was never an authorization gate for the
    destination."""
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


# --------------------------------------------------------------------------- #
#  (c) /api/logs/export needs host filesystem access
# --------------------------------------------------------------------------- #

def test_logs_export_denied_without_fs_host(tmp_path, monkeypatch):
    from localm import scopes as S
    app, _home = _media_app(tmp_path, monkeypatch, "image")
    key = _key([S.CONFIG_WRITE], privileged=True)          # fs_access="none"
    dest = tmp_path / "logsteal"

    with TestClient(app) as c:
        r = c.post("/api/logs/export", headers=_h(key), json={"dest": str(dest)})
    assert r.status_code == 403, r.text
    assert not dest.exists()


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

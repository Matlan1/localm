# SPDX-License-Identifier: AGPL-3.0-or-later
"""Failure-path coverage for the self-updater that the existing suite did not yet
pin (TEST-2). The self-updater swaps the install tree, so a silent regression on a
recovery path is exactly the costly kind: these tests cover the post-step CRASH
(runner raises, distinct from a non-zero exit), the corrupt-manifest fallback, and
the backend-detection fallback. Everything uses the apply()/download injectables so
no test ever touches the real install or the network."""

import base64
import io
import zipfile

import pytest
from cryptography.hazmat.primitives import serialization as _ser
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from localm import _version, updater
from localm.bugreport import LocalmError

# A throwaway signing key so the apply() crash-path tests pass the signature gate
# and exercise the ROLLBACK logic they are about (not the gate itself, which
# test_updater_signature.py covers).
_PRIV = Ed25519PrivateKey.generate()
_PUB_HEX = _PRIV.public_key().public_bytes(
    encoding=_ser.Encoding.Raw, format=_ser.PublicFormat.Raw).hex()


# --------------------------- read_manifest --------------------------------

def test_read_manifest_corrupt_json_falls_back_to_empty(tmp_path):
    # A corrupt manifest must NOT raise; classify() then relies on auto-detection.
    (tmp_path / "update.json").write_text("{ this is not valid json ", encoding="utf-8")
    assert updater.read_manifest(tmp_path) == {}


def test_read_manifest_absent_is_empty(tmp_path):
    assert updater.read_manifest(tmp_path) == {}


def test_read_manifest_unreadable_dir_is_empty(tmp_path):
    # staged_dir does not exist at all -> still {} (never raises).
    assert updater.read_manifest(tmp_path / "nope") == {}


def test_read_manifest_valid(tmp_path):
    (tmp_path / "update.json").write_text('{"needs": "runtime"}', encoding="utf-8")
    assert updater.read_manifest(tmp_path) == {"needs": "runtime"}


# --------------------------- apply(): crash path --------------------------

def _signed_build(version, deps):
    """A build zip (bytes) + its base64 signature under the test key; returns
    ``(download_opener, signature_b64)`` so apply()'s signature gate accepts it."""
    buf = io.BytesIO()
    pyproject = ('[project]\nname = "localm"\ndependencies = ['
                 + ", ".join(f'"{d}"' for d in deps) + ']\n')
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("VERSION", version)
        z.writestr("pyproject.toml", pyproject)
        z.writestr("localm/__init__.py", f"# {version}")
    data = buf.getvalue()
    sig = base64.b64encode(_PRIV.sign(data)).decode("ascii")

    def dl(url, headers, timeout, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
    return dl, sig


def _pin_sig_env(monkeypatch):
    monkeypatch.setattr(updater, "_UPDATE_PUBKEYS", (_PUB_HEX,))
    monkeypatch.setattr(_version, "read_version", lambda: "0.1.0")


def _fake_install(tmp_path, deps=("click",)):
    inst = tmp_path / "inst"
    (inst / "localm").mkdir(parents=True)
    (inst / "localm" / "__init__.py").write_text("# 0.1.0", encoding="utf-8")
    (inst / "VERSION").write_text("0.1.0", encoding="utf-8")
    (inst / "pyproject.toml").write_text(
        '[project]\nname = "localm"\ndependencies = ['
        + ", ".join(f'"{d}"' for d in deps) + ']\n', encoding="utf-8")
    return inst


def test_apply_rolls_back_when_post_step_raises(tmp_path, monkeypatch):
    """The runner RAISING (not just exiting non-zero) is a distinct branch: it must
    still roll back and surface a 'crashed' LocalmError, never a half-applied tree."""
    monkeypatch.setattr("localm.config.load_config", lambda: {"bugreport_upload_url": "https://w"})
    monkeypatch.setattr("localm.config.home_dir", lambda: tmp_path / "home")
    _pin_sig_env(monkeypatch)
    inst = _fake_install(tmp_path, deps=("click",))

    def crashing_runner(cmd):
        raise RuntimeError("uv exploded")

    op, sig = _signed_build("0.2.0", ["click", "httpx"])   # deps -> post cmd
    with pytest.raises(LocalmError, match="crashed"):
        updater.apply(5, installed=inst, signature=sig,
                      download_opener=op, runner=crashing_runner)
    # Rolled back to the pre-apply state on the crash path too.
    assert (inst / "VERSION").read_text().strip() == "0.1.0"
    assert (inst / "localm" / "__init__.py").read_text() == "# 0.1.0"


def test_apply_crash_then_rollback_failure_demands_manual_recovery(tmp_path, monkeypatch):
    """Worst case: the post step CRASHES and rollback ALSO fails. apply() must say
    manual recovery is needed - never hide a broken install behind a clean message."""
    monkeypatch.setattr("localm.config.load_config", lambda: {"bugreport_upload_url": "https://w"})
    monkeypatch.setattr("localm.config.home_dir", lambda: tmp_path / "home")
    _pin_sig_env(monkeypatch)
    inst = _fake_install(tmp_path, deps=("click",))
    from localm import _apply_update as au
    monkeypatch.setattr(au, "rollback", lambda *a, **k: (_ for _ in ()).throw(OSError("rb boom")))

    def crashing_runner(cmd):
        raise RuntimeError("uv exploded")

    op, sig = _signed_build("0.2.0", ["click", "httpx"])
    with pytest.raises(LocalmError, match="rollback failed"):
        updater.apply(5, installed=inst, signature=sig,
                      download_opener=op, runner=crashing_runner)


# --------------------------- _installed_backend ---------------------------
# _installed_backend() must PRESERVE whatever backend is already installed
# (setup_llama's on-disk marker) rather than re-derive one from the current
# hardware-recommendation policy. See updater.py's own docstring for why this
# is no longer "just call recommended_install_backend()" (that WAS the #833
# fix, and it stopped being correct the moment the recommendation policy
# itself changed for hardware someone was already running on - b8878c2b moved
# NVIDIA-Linux from vulkan to cuda, which would have silently re-provisioned
# an already-working vulkan install onto cuda on the next runtime update).

def test_installed_backend_preserves_what_is_actually_installed(tmp_path, monkeypatch):
    """THE regression test: an install with amd-rocm actually provisioned must
    stay on amd-rocm across an update, even though the CURRENT hardware
    recommendation has since moved to something else entirely. Exercises the
    REAL on-disk marker (setup_llama._record_provisioned_backend /
    _provisioned_backend) rather than a mocked return value, so this proves
    the whole read path, not just the updater's own branching.

    Asserts recommended_install_backend was never even CALLED, from OUTSIDE
    the call (a plain counter, not a raised exception) - _installed_backend()
    wraps both its lookups in a broad except, so an exception raised as the
    mock's side_effect would be silently swallowed and prove nothing (see
    diff-review-discipline item 13)."""
    from localm import hwdetect, setup_llama
    monkeypatch.setattr(setup_llama, "_repo_runtime_lib", lambda: tmp_path)
    setup_llama._record_provisioned_backend(tmp_path, "amd-rocm")

    calls = []
    monkeypatch.setattr(
        hwdetect, "recommended_install_backend",
        lambda *a, **k: calls.append(1) or "cuda-from-a-changed-recommendation")

    assert updater._installed_backend() == "amd-rocm"
    assert calls == [], ("must not re-derive a recommendation when a backend "
                         "is already installed")


def test_installed_backend_falls_back_to_recommendation_when_nothing_provisioned(
        tmp_path, monkeypatch):
    """No marker on disk (a fresh install, or one predating the marker) means
    there is nothing to preserve, so this is a first-time pick - not an
    override - and falls back to the current hardware recommendation."""
    from localm import hwdetect, setup_llama
    monkeypatch.setattr(setup_llama, "_repo_runtime_lib", lambda: tmp_path)   # empty: no marker
    monkeypatch.setattr(hwdetect, "recommended_install_backend",
                        lambda *a, **k: "cuda-from-the-policy")
    assert updater._installed_backend() == "cuda-from-the-policy"


def test_installed_backend_falls_back_to_vulkan_when_everything_fails(monkeypatch):
    """Both the marker read AND the hardware detection raise -> the universal
    default. A detection failure must never break an update."""
    from localm import hwdetect, setup_llama

    def boom_lib_dir():
        raise RuntimeError("runtime lib dir blew up")
    monkeypatch.setattr(setup_llama, "_repo_runtime_lib", boom_lib_dir)

    def boom_recommend(*a, **k):
        raise RuntimeError("detection blew up")
    monkeypatch.setattr(hwdetect, "recommended_install_backend", boom_recommend)

    assert updater._installed_backend() == "vulkan"


# --------------------- download(): real urllib error paths ----------------
# The default opener (opener=None) has its own HTTP/URL error handling; the
# happy-path download test in test_updater.py only exercises an injected opener.

def _config(monkeypatch):
    monkeypatch.setattr("localm.config.load_config",
                        lambda: {"bugreport_upload_url": "https://w", "bugreport_upload_token": "t"})


# download() now streams through a build_opener() (an HTTPS-only redirect handler,
# CHK-UPDATER-INTEGRITY), so these mock OpenerDirector.open (the real transport).
def test_download_http_error_raises_localmerror(tmp_path, monkeypatch):
    import urllib.request
    _config(monkeypatch)

    def boom(self, req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 404, "Not Found", {}, None)
    monkeypatch.setattr(urllib.request.OpenerDirector, "open", boom)
    with pytest.raises(LocalmError, match="download failed"):
        updater.download(7, tmp_path / "b.zip")


def test_download_urlerror_raises_localmerror(tmp_path, monkeypatch):
    import urllib.request
    _config(monkeypatch)

    def boom(self, req, timeout=None):
        raise urllib.error.URLError("connection refused")
    monkeypatch.setattr(urllib.request.OpenerDirector, "open", boom)
    with pytest.raises(LocalmError, match="could not download"):
        updater.download(7, tmp_path / "b.zip")


def test_download_non_2xx_status_raises(tmp_path, monkeypatch):
    import urllib.request
    _config(monkeypatch)

    class _Resp:
        status = 500
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self, n): return b""
    monkeypatch.setattr(urllib.request.OpenerDirector, "open",
                        lambda self, req, timeout=None: _Resp())
    with pytest.raises(LocalmError, match="download failed"):
        updater.download(7, tmp_path / "b.zip")


def test_download_unconfigured_raises(tmp_path, monkeypatch):
    monkeypatch.setattr("localm.config.load_config", lambda: {})
    with pytest.raises(LocalmError, match="not configured"):
        updater.download(7, tmp_path / "b.zip")

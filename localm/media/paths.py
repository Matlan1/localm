# SPDX-License-Identifier: AGPL-3.0-or-later
"""Filesystem policy for the caller-supplied paths the media plugins accept.

Two of them reach the filesystem from the image/music/video routes, and both are
reachable by a NON-privileged, media-scoped key:

* ``input_image`` (img2img / image-to-video) is READ and then UPLOADED to
  ComfyUI. ``sanitize_comfy_url`` permits a LAN or public ``api_url`` over
  plaintext http, so this is a read-AND-TRANSMIT primitive, not a local read.
  ``allowed_input_roots`` confines it to the places a source image legitimately
  comes from, never the whole data dir, which is localm's credential store
  (auth.key, auth.json, sessions.json, rag/, coder/, bug-reports/).

* the ``/move`` ``dest`` comes from the request body and goes into
  ``mkdir(parents=True)`` + ``shutil.move``. ``gallery.require_owner`` gates only
  ARTIFACT ownership (true for the scoped key that generated the file, and for
  any caller when the artifact has no recorded owner), so ``confined_move_dest``
  gates the outside-the-data-dir case on host filesystem access.

Both policies live here, in one place, for every media plugin.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import HTTPException, Request

from localm import pathsafe as _pathsafe

# The gallery subdirectories of the data dir, one per media plugin. Defined here
# and consumed by each plug.py, so the input policy below and the directories the
# plugins write to share one definition.
IMAGE_DIR_NAME = "gui_images"
VIDEO_DIR_NAME = "gui_video"
MUSIC_DIR_NAME = "gui_music"

# <home>/uploads, where POST /api/upload puts files "so models and tools can
# read them". That is precisely this route's intended inbox.
UPLOADS_DIR_NAME = "uploads"


def gallery_dir(dir_name: str) -> Path:
    from localm.config import home_dir
    return home_dir() / dir_name


def _resolved_home() -> Path | None:
    """The resolved data dir, or None when it cannot be resolved."""
    from localm.config import home_dir
    try:
        return home_dir().resolve()
    except OSError as e:
        # Not silently swallowed: without the data dir there is no input root at
        # all, so every input_image is refused until it resolves. That is
        # fail-CLOSED because every allowed root is a subdirectory OF the data
        # dir. A root that is not under home would break that property.
        from localm.debuglog import logger
        logger.warning("cannot resolve the localm data directory (%s); no "
                       "input-image root is available, so every input_image "
                       "will be rejected until it resolves", e)
        return None


def _home_input_roots(home: Path) -> list[Path]:
    """The subdirectories of the data dir an input_image may come from."""
    return [home / name for name in (UPLOADS_DIR_NAME, IMAGE_DIR_NAME,
                                     VIDEO_DIR_NAME, MUSIC_DIR_NAME)]


def _under(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


# check_input_image and confined_move_dest below answer a coarse question: does
# this ALREADY-RESOLVED path fall under one of a caller's own allowed root
# directories, for a READ-and-forward (input_image) or a directory pick (move
# dest), never a specific new or targeted entry. They therefore do not use
# pathsafe.confined_under / confined_absolute_or_under, which confine a
# caller-named STRING to one new or targeted entry. Revisit if either function
# starts gating a delete or an exact-identity write.


# The UNC/device predicate is localm.pathsafe.is_unc_or_device_path, NOT a copy.
# Re-exported here so this module's own callers and its tests read as one policy
# surface.
is_unc_or_device_path = _pathsafe.is_unc_or_device_path


def allowed_input_roots() -> list[Path]:
    """Roots an ``input_image`` may live under.

    NOT the data dir: that holds auth.key (the plaintext owner key) and the rest
    of localm's credential store, and these routes are mounted under the
    non-privileged ``image`` / ``video`` scopes. The allowed set is the upload
    inbox plus the generated-media galleries, which is what the GUI's two real
    flows produce: a file uploaded on the Settings page, and the gallery's "use
    as input" button (it fills the field with a gui_images path).

    The install directory is NOT a root either, source checkout or not. A
    reference image belongs in <home>/uploads.

    The roots are NOT created here - a policy check must not have the side
    effect of making a directory. A root that does not exist simply matches
    nothing.
    """
    home = _resolved_home()
    return list(_home_input_roots(home)) if home is not None else []


class InputImageRefused(ValueError):
    """An ``input_image`` was refused by the policy below.

    A ValueError, not an HTTPException: the policy is NOT HTTP-specific - the
    MCP server reaches the same ComfyUI upload over stdio JSON-RPC.
    ``confined_input_image`` is the thin HTTP wrapper; ``check_input_image`` is
    the policy. Same split ``pathsafe`` draws between ``confined_name`` (raises
    HTTPException) and ``confined_under`` (raises ValueError)."""


class InputImageUnavailable(InputImageRefused):
    """The policy could not be EVALUATED - the data dir would not resolve.

    Distinct from an ordinary refusal, so a FAULT is never reported as a routine
    policy decision. Subclasses InputImageRefused, so a caller that only wants
    "did this pass" still fails closed by catching the base."""


def check_input_image(raw: str) -> Path:
    """Resolve and confine an ``input_image`` to an allowed root, or raise.

    The transport-independent policy. Raises InputImageRefused when the path
    escapes every allowed root or does not point at an existing file, and
    InputImageUnavailable when the data dir cannot be resolved at all. Symlinks
    are resolved first, so a link inside an allowed root that targets outside it
    is still rejected.

    Messages name the allowed LOCATIONS but never an absolute path: the data dir
    contains the OS username, and every caller of this is reachable by a
    principal that is not the owner.

    An unresolvable data dir FAILS CLOSED. Because the allowed roots are exactly
    four subdirectories OF the data dir, "no data dir" really does mean "nothing
    is permitted" - there is no wider root left to fall back to.
    """
    home = _resolved_home()
    if home is None:
        raise InputImageUnavailable(
            "Cannot resolve the localm data directory, so no input image can be "
            "authorised. See the server log for the cause.")
    # BEFORE any syscall on the caller's string: a UNC path would be refused
    # below anyway (it cannot be under a local data dir), but only AFTER
    # .resolve() had already dialled SMB and stalled the event loop for minutes.
    # Skipped only if the data dir is ITSELF a UNC path, where a UNC input can
    # legitimately be under an allowed root.
    if is_unc_or_device_path(raw) and not is_unc_or_device_path(str(home)):
        raise InputImageRefused(
            "Input image must be a local file you uploaded or a generated "
            "image; network (UNC) and device paths are not accepted.")
    try:
        resolved = Path(raw).expanduser().resolve()
    except (OSError, ValueError, RuntimeError):
        raise InputImageRefused("Invalid input image path")
    if not any(_under(resolved, h) for h in _home_input_roots(home)):
        raise InputImageRefused(
            "Input image must be a file you uploaded (the Settings page's "
            "uploads folder) or one of the generated-media galleries. Other "
            "locations, including the localm data directory itself and the "
            "localm install directory, are not readable here.")
    if not resolved.is_file():
        raise InputImageRefused(f"Input image not found: {raw}")
    return resolved


def confined_input_image(raw: str) -> Path:
    """``check_input_image`` for an HTTP route: the same policy, as an
    HTTPException.

    A 500 for InputImageUnavailable and a 400 for everything else, so a fault
    the operator must fix is not returned as a caller mistake."""
    try:
        return check_input_image(raw)
    except InputImageUnavailable as e:
        raise HTTPException(500, str(e))
    except InputImageRefused as e:
        raise HTTPException(400, str(e))


def confined_move_dest(request: Request, raw: str) -> Path:
    """Resolve a ``/move`` destination directory and check the caller may write
    there. Returns the resolved path WITHOUT creating it (the caller does the
    mkdir, once this has passed).

    A principal with host filesystem access - the owner/ADMIN key, open mode, or
    a key the owner explicitly granted ``fs_access=host`` - gets "any folder on
    this machine". The folder picker that supplies ``dest`` in the GUI flow
    (``/api/fs/dirs``) is gated on the same dial.

    Every other principal is confined to the data dir. This SUPPLEMENTS
    ``gallery.require_owner``: that dependency proves ARTIFACT ownership, which
    is a different question from authority over the host filesystem, and it
    passes for ANY caller when the artifact has no recorded owner (open mode,
    legacy or hand-placed files).
    """
    from localm.config import home_dir
    from localm.inference.http_server import effective_fs_access
    # The credential check FIRST, because it costs no syscall, then the
    # pure-string UNC/device check - both before .resolve() touches the caller's
    # string. A non-host caller's UNC dest is refused below regardless (it is not
    # under the data dir), but resolving it first would dial SMB and stall the
    # event loop for minutes before saying so.
    #
    # A host-fs principal is NOT string-checked: moving a generated file to a
    # network share is legitimate for the owner. That path can still block the
    # loop.
    fs_host = effective_fs_access(request) == "host"
    if not fs_host and is_unc_or_device_path(raw):
        raise HTTPException(
            403, "This key cannot move media to a network (UNC) or device "
                 "path; it has no host filesystem access.")
    try:
        resolved = Path(raw).expanduser().resolve()
    except (OSError, ValueError, RuntimeError):
        raise HTTPException(400, "Invalid destination path")
    if fs_host:
        return resolved
    try:
        home = home_dir().resolve()
    except OSError:
        # The boundary itself is unavailable, so the destination cannot be
        # proven inside it. Deny with the REAL reason, not the ordinary "outside
        # the data directory" one.
        #
        # The exception is NOT interpolated: str(OSError) carries .filename, so
        # it would hand the absolute data dir path - and hence the OS username -
        # to the only principal that reaches this branch, a non-owner key that
        # just failed the fs_access check. The cause goes to the server log
        # instead. This branch calls home_dir().resolve() directly rather than
        # _resolved_home(), so it does its OWN logging below.
        from localm.debuglog import logger
        logger.warning("move destination refused: the localm data directory "
                       "could not be resolved", exc_info=True)
        raise HTTPException(
            500, "Cannot resolve the localm data directory, so the destination "
                 "could not be authorised. See the server log for the cause.")
    if not _under(resolved, home):
        raise HTTPException(
            403, "This key cannot move media outside the localm data "
                 "directory (it has no host filesystem access).")
    return resolved

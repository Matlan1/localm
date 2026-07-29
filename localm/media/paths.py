# SPDX-License-Identifier: AGPL-3.0-or-later
"""Filesystem policy for the caller-supplied paths the media plugins accept.

Two of them reach the filesystem from the image/music/video routes, and both
were reachable by a NON-privileged, media-scoped key:

* ``input_image`` (img2img / image-to-video) is READ and then UPLOADED to
  ComfyUI. ``sanitize_comfy_url`` deliberately permits a LAN or public
  ``api_url`` over plaintext http, so this is a read-AND-TRANSMIT primitive, not
  a local read. It used to be confined to the whole data dir - which is
  localm's credential store (auth.key, auth.json, sessions.json, rag/, coder/,
  bug-reports/) - so the confinement did not remove the arbitrary-file primitive
  its own docstring named, it merely retargeted it at localm's own secrets.
  ``allowed_input_roots`` narrows it to the places a source image legitimately
  comes from.

* the ``/move`` ``dest`` was taken verbatim from the request body into
  ``mkdir(parents=True)`` + ``shutil.move``, gated only on ARTIFACT ownership
  (true for the scoped key that generated the file, and for any caller when the
  artifact has no recorded owner). ``confined_move_dest`` gates the
  outside-the-data-dir case on host filesystem access instead.

Both policies live here rather than being copy-pasted per plugin: the
input_image confinement was already duplicated verbatim in image/ and video/,
and the move handler three times over, which is how one reading of one copy can
declare a family of routes safe.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import HTTPException, Request

# The gallery subdirectories of the data dir, one per media plugin. Defined here
# and consumed by each plug.py so the names have a single definition: the input
# policy below has to agree with where the plugins actually write, and two
# independent copies of a name is how that agreement silently breaks.
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
        # Not silently swallowed (AGENTS rule 5): without the data dir there is
        # no input root at all, so every input_image is refused until it
        # resolves. That is genuinely fail-CLOSED now - every allowed root is a
        # subdirectory OF the data dir, so there is no wider root left to fall
        # back to. (It was NOT true while an install-directory root existed
        # alongside: losing home then SKIPPED the data-dir re-deny and widened
        # the policy on the failure path. Do not reintroduce a root that is not
        # under home without revisiting this.)
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


def is_unc_or_device_path(raw: str) -> bool:
    """True for a Windows UNC (``\\\\host\\share``) or device (``\\\\.\\x``,
    ``\\\\?\\x``) path STRING.

    PURE STRING WORK - no filesystem access - so it is safe to run on
    unsanitized caller input BEFORE any syscall, and that ordering is the entire
    point. ``Path.resolve()`` on a UNC path to an unroutable host DIALS SMB and
    blocks for minutes (measured here: a probe of ``\\\\192.0.2.1\\share`` did
    not return inside 120s), and these routes are ``async``, so a single request
    stalls the whole event loop. Against a REACHABLE attacker share, Windows
    also auto-authenticates and surrenders the host net-NTLMv2 credential. The
    refusal is worthless if it happens after the dial.

    All four separator mixes are tested, not just ``\\\\``: ntpath treats them
    alike, so ``//h/s``, ``\\/h/s`` and ``/\\h/s`` all parse to the drive
    ``\\\\h\\s``. Checking only the canonical form is the bug this avoids."""
    s = raw.strip()
    return len(s) >= 2 and s[:2] in ("\\\\", "//", "\\/", "/\\")


def allowed_input_roots() -> list[Path]:
    """Roots an ``input_image`` may live under.

    Deliberately NOT the data dir: that holds auth.key (the plaintext owner key)
    and the rest of localm's credential store, and these routes are mounted
    under the non-privileged ``image`` / ``video`` scopes. The allowed set is the
    upload inbox plus the generated-media galleries, which is what the GUI's two
    real flows produce: a file uploaded on the Settings page, and the gallery's
    "use as input" button (it fills the field with a gui_images path).

    The install directory is NOT a root either, source checkout or not. Earlier
    versions allowed the repo root "for a reference image kept in the checkout",
    guarded on pyproject.toml. That guard never worked (pyproject.toml is
    release-include - release-manifest.toml, the updater's verify_zip requires
    it - so an installed copy has one too), and more importantly the allowance
    is wrong even when it works: README documents `git clone` as the install
    path, so on the PRIMARY topology it admitted the entire tree, including the
    gitignored `issues/` and `qa/` directories that hold bug-report screenshots
    and test-instance data. That is the same read-and-transmit primitive this
    module exists to remove, aimed at a different directory. A reference image
    belongs in <home>/uploads; copying one there is a single command.

    The roots are NOT created here - a policy check must not have the side
    effect of making a directory. A root that does not exist simply matches
    nothing.
    """
    home = _resolved_home()
    return list(_home_input_roots(home)) if home is not None else []


def confined_input_image(raw: str) -> Path:
    """Resolve and confine an ``input_image`` to an allowed root.

    Raises HTTPException(400) when the path escapes every allowed root or does
    not point at an existing file. Symlinks are resolved first, so a link inside
    an allowed root that targets outside it is still rejected.

    The rejection message names the allowed LOCATIONS but never an absolute
    path: the data dir contains the OS username, and this route is reachable by
    a non-owner key.

    An unresolvable data dir FAILS CLOSED with its own distinct error, rather
    than falling through to the ordinary refusal: a fault must not be reported
    as a routine policy decision (AGENTS rule 5). Because the allowed roots are
    now exactly four subdirectories OF the data dir, "no data dir" really does
    mean "nothing is permitted" - there is no wider root left to fall back to.
    """
    home = _resolved_home()
    if home is None:
        raise HTTPException(
            500, "Cannot resolve the localm data directory, so no input image "
                 "can be authorised. See the server log for the cause.")
    # BEFORE any syscall on the caller's string: a UNC path would be refused
    # below anyway (it cannot be under a local data dir), but only AFTER
    # .resolve() had already dialled SMB and stalled the event loop for minutes.
    # Skipped only if the data dir is ITSELF a UNC path, where a UNC input can
    # legitimately be under an allowed root.
    if is_unc_or_device_path(raw) and not is_unc_or_device_path(str(home)):
        raise HTTPException(
            400, "Input image must be a local file you uploaded or a generated "
                 "image; network (UNC) and device paths are not accepted.")
    try:
        resolved = Path(raw).expanduser().resolve()
    except (OSError, ValueError, RuntimeError):
        raise HTTPException(400, "Invalid input image path")
    if not any(_under(resolved, h) for h in _home_input_roots(home)):
        raise HTTPException(
            400, "Input image must be a file you uploaded (the Settings page's "
                 "uploads folder) or one of the generated-media galleries. "
                 "Other locations, including the localm data directory itself "
                 "and the localm install directory, are not readable by this "
                 "route.")
    if not resolved.is_file():
        raise HTTPException(400, f"Input image not found: {raw}")
    return resolved


def confined_move_dest(request: Request, raw: str) -> Path:
    """Resolve a ``/move`` destination directory and check the caller may write
    there. Returns the resolved path WITHOUT creating it (the caller does the
    mkdir, once this has passed).

    A principal with host filesystem access - the owner/ADMIN key, open mode, or
    a key the owner explicitly granted ``fs_access=host`` - keeps "any folder on
    this machine". That is the documented feature, and the folder picker that
    supplies ``dest`` in the GUI flow (``/api/fs/dirs``) is itself gated on the
    same dial, so this route simply stops being the weaker door.

    Every other principal is confined to the data dir. This SUPPLEMENTS
    ``gallery.require_owner`` rather than replacing it: that dependency proves
    ARTIFACT ownership, which is a different question from authority over the
    host filesystem, and it passes for ANY caller when the artifact has no
    recorded owner (open mode, legacy or hand-placed files).
    """
    from localm.config import home_dir
    from localm.inference.http_server import effective_fs_access
    # The credential check FIRST, because it costs no syscall, then the
    # pure-string UNC/device check - both before .resolve() touches the caller's
    # string. A non-host caller's UNC dest is refused below regardless (it is not
    # under the data dir), but resolving it first would dial SMB and stall the
    # event loop for minutes before saying so.
    #
    # A host-fs principal is deliberately NOT string-checked: moving a generated
    # file to a network share is legitimate for the owner, who is not the threat
    # this gate addresses. That path can still block the loop, which is the
    # general "blocking fs work in an async handler" problem being fixed
    # separately for the admin routes; it is noted here rather than half-solved.
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
        # proven inside it. Deny - but with the REAL reason, not the ordinary
        # "outside the data directory" one, which would hide a fault behind a
        # routine-looking refusal (AGENTS rule 5).
        #
        # The exception is NOT interpolated: str(OSError) carries .filename, so
        # it would hand the absolute data dir path - and hence the OS username -
        # to the only principal that reaches this branch, a non-owner key that
        # just failed the fs_access check. The cause goes to the server log
        # instead, which is where the path belongs. (This branch calls
        # home_dir().resolve() directly rather than _resolved_home(), so it does
        # its OWN logging below - _resolved_home's warning is on the input-image
        # path, not this one.)
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

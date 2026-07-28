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
    """The resolved data dir, or None when it cannot be resolved (warned once at
    the call site that needs it)."""
    from localm.config import home_dir
    try:
        return home_dir().resolve()
    except OSError as e:
        # Not silently swallowed (AGENTS rule 5): without the data dir there is
        # no input root at all, so every legitimate input_image starts failing
        # confinement, which reads as a bad path rather than an unresolvable
        # data dir. Fail CLOSED, but say so where it can be found.
        from localm.debuglog import logger
        logger.warning("cannot resolve the localm data directory (%s); no "
                       "input-image root is available, so every input_image "
                       "will be rejected until it resolves", e)
        return None


def _home_input_roots(home: Path) -> list[Path]:
    """The subdirectories of the data dir an input_image may come from."""
    return [home / name for name in (UPLOADS_DIR_NAME, IMAGE_DIR_NAME,
                                     VIDEO_DIR_NAME, MUSIC_DIR_NAME)]


def source_checkout_root() -> Path | None:
    """The repo root when running from a SOURCE CHECKOUT, else None.

    The previous copies of this check keyed on pyproject.toml alone, which does
    not mean what it looks like it means: pyproject.toml is release-include
    (release-manifest.toml - the updater's verify_zip requires it), so an
    INSTALLED copy carries one too and the "source checkout only" allowance
    never actually narrowed. build_release assembles the zip from git-TRACKED
    files, so a release build can never carry a .git; that is the marker that
    distinguishes them. Require both."""
    root = Path(__file__).resolve().parents[2]
    if (root / "pyproject.toml").is_file() and (root / ".git").exists():
        return root
    return None


def _under(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def allowed_input_roots() -> list[Path]:
    """Roots an ``input_image`` may live under.

    Deliberately NOT the data dir: that holds auth.key (the plaintext owner key)
    and the rest of localm's credential store, and these routes are mounted
    under the non-privileged ``image`` / ``video`` scopes. The allowed set is the
    upload inbox plus the generated-media galleries, which is what the GUI's two
    real flows produce: a file uploaded on the Settings page, and the gallery's
    "use as input" button (it fills the field with a gui_images path).

    The roots are NOT created here - a policy check must not have the side
    effect of making a directory. A root that does not exist simply matches
    nothing.
    """
    home = _resolved_home()
    roots: list[Path] = list(_home_input_roots(home)) if home is not None else []
    # When running from a SOURCE CHECKOUT, the repo root is a legitimate place to
    # keep a reference image (e.g. an examples/ asset).
    repo_root = source_checkout_root()
    if repo_root is not None:
        roots.append(repo_root)
    return roots


def confined_input_image(raw: str) -> Path:
    """Resolve and confine an ``input_image`` to an allowed root.

    Raises HTTPException(400) when the path escapes every allowed root or does
    not point at an existing file. Symlinks are resolved first, so a link inside
    an allowed root that targets outside it is still rejected.

    The rejection message names the allowed LOCATIONS but never an absolute
    path: the data dir contains the OS username, and this route is reachable by
    a non-owner key.
    """
    try:
        resolved = Path(raw).expanduser().resolve()
    except (OSError, ValueError, RuntimeError):
        raise HTTPException(400, "Invalid input image path")
    refused = HTTPException(
        400, "Input image must be a file you uploaded (the Settings page's "
             "uploads folder) or one of the generated-media galleries. "
             "Other locations, including the localm data directory itself, "
             "are not readable by this route.")
    if not any(_under(resolved, r) for r in allowed_input_roots()):
        raise refused
    # The source-checkout allowance can CONTAIN the data dir: with no LOCALM_HOME
    # set (and no localm-home.cfg), localm falls back to <repo>/home, so the repo
    # root would re-admit auth.key through the back door - the exact thing the
    # narrowing exists to stop. Re-deny anything under the data dir that is not
    # in one of its four explicitly allowed subdirectories, whatever root let it
    # through. Checked SECOND so the allowed subdirs still pass either way.
    home = _resolved_home()
    if home is not None and _under(resolved, home) \
            and not any(_under(resolved, h) for h in _home_input_roots(home)):
        raise refused
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
    try:
        resolved = Path(raw).expanduser().resolve()
    except (OSError, ValueError, RuntimeError):
        raise HTTPException(400, "Invalid destination path")
    if effective_fs_access(request) == "host":
        return resolved
    try:
        home = home_dir().resolve()
    except OSError as e:
        # The boundary itself is unavailable, so the destination cannot be
        # proven inside it. Deny - but with the REAL reason, not the ordinary
        # "outside the data directory" one, which would hide a fault behind a
        # routine-looking refusal (AGENTS rule 5).
        raise HTTPException(
            500, f"Cannot resolve the localm data directory: {e}")
    if not _under(resolved, home):
        raise HTTPException(
            403, "This key cannot move media outside the localm data "
                 "directory (it has no host filesystem access).")
    return resolved

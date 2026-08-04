# SPDX-License-Identifier: AGPL-3.0-or-later
"""
GUI web layer - API routes and static file serving, attached to the
existing localm FastAPI inference app.

Routes (all under /api, bearer-protected when LOCALM_API_KEY is set):
  GET    /api/models                       registry + active model
  POST   /api/models/load                  switch the active engine

Coder routes (/api/coder/*) live in the builtin "coder" plugin
(localm/plugins/builtin/coder); attach_gui only publishes the shared
services they read via request.app.state.

The static frontend is mounted at / and must be attached AFTER all API
routes so it doesn't shadow them.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from localm.bindhost import is_loopback_host as _is_loopback_host  # noqa: F401  (re-export for back-compat)
from localm.debuglog import logger
from localm.plugins.coder.sessions import SessionManager

STATIC_DIR = Path(__file__).parent / "static"

# SSE keepalive interval - must beat proxy/browser idle timeouts
_KEEPALIVE_S = 15


class _RevalidatingStatic(StaticFiles):
    """Serve the GUI assets with ``Cache-Control: no-cache`` so the browser
    REVALIDATES every load instead of silently serving a stale copy.

    SEAMLESS UPDATES: Starlette's StaticFiles sends an ``ETag`` but no
    ``Cache-Control``, so browsers fall back to HEURISTIC caching and can keep an
    old ``app.js`` / ``sw.js`` / ``style.css`` for a while - the user (or a tester)
    then has to clear the cache by hand to pick up new code. ``no-cache`` does NOT
    disable caching: the browser still caches and revalidates with the ETag, so an
    unchanged file is a cheap ``304`` and a changed one is fetched fresh. It also
    lets a phone's ``serviceWorker.register(...).update()`` actually see a new
    ``sw.js`` (the browser HTTP-caching sw.js is a known update-stickiness cause).
    The server, not the user, is now responsible for delivering current code."""

    async def get_response(self, path, scope):
        resp = await super().get_response(path, scope)
        resp.headers.setdefault("Cache-Control", "no-cache")
        return resp


def _index_html_with_shell_token(token: str) -> str:
    """The SPA shell, optionally seeding the per-process open-mode *token* (the
    shell token) as a JS global so a loopback launch can still perform management
    when no API key is configured. The protected-mode API key is NOT injected
    here - the shell route sets it as an HttpOnly cookie instead, so it never
    reaches page JS / localStorage. An empty *token* injects nothing.

    The token is embedded only in same-origin HTML served to a trusted loopback
    client and is a short-lived per-process secret, not the durable API key."""
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    if not token:
        return html
    # json.dumps escapes quotes/backslashes; also escape "<" so the value can
    # never break out of the <script> element (defence in depth).
    snippet = ("<script>window.__LOCALM_SHELL_TOKEN__="
               + json.dumps(token).replace("<", "\\u003c")
               + ";</script>")
    lower = html.lower()
    i = lower.find("<head>")
    if i != -1:
        cut = i + len("<head>")
        return html[:cut] + snippet + html[cut:]
    return snippet + html


# sw.js's own fetch() handler regex (self.addEventListener("fetch", ...)) for
# API-shaped paths that are NEVER cache-first. Nothing under STATIC_DIR is
# actually named this today (these are server routes, not static files), so
# this exclusion is future-proofing, not dead weight - kept in sync BY HAND
# with the identical pattern inside sw.js itself, since a service worker
# cannot import a shared module.
_SW_UNCACHED_RE = re.compile(r"^(api|v1|plugins|localm-ca\.crt)(/|$)")

# sw.js's own CACHE-constant line, e.g. `const CACHE = "localm-shell-dev";` -
# matches regardless of the literal placeholder text so the source can say
# anything descriptive; the ROUTE substitutes group(1)'s content on every
# request. check_hygiene.py's placeholder-intact check uses the same pattern.
SW_CACHE_LINE_RE = re.compile(r'(const CACHE = ")[^"]+(";)')


def _sw_cacheable_files() -> list[Path]:
    """Every FILE under STATIC_DIR the service worker's fetch handler can cache
    first, sorted for deterministic hashing. Deliberately NOT limited to sw.js's
    own SHELL precache list: the fetch handler runtime-caches ANY same-origin,
    non-API GET into the same versioned cache (see sw.js's fetch listener), so a
    non-SHELL asset (a KaTeX font, /vendor/jsQR.min.js, ...) goes stale exactly
    as hard as a SHELL one and must invalidate the version the same way."""
    out = []
    for p in STATIC_DIR.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(STATIC_DIR).as_posix()
        if rel == "sw.js" or _SW_UNCACHED_RE.match(rel):
            continue
        out.append(p)
    return sorted(out, key=lambda p: p.relative_to(STATIC_DIR).as_posix())


def _compute_sw_cache_value() -> str:
    """A short content digest over every file the service worker can cache-first
    (see ``_sw_cacheable_files``), used as sw.js's served CACHE version.

    Computed FRESH on every request from whatever is currently on disk - never
    hand-typed, never committed to git. This is what makes it immune to the
    incident this replaces: a hand-maintained ``localm-shell-vNN`` line shipped
    stale THREE times (undetected by review) because bumping it depended on a
    human remembering, and even after a hygiene gate was added to catch a
    missed bump, the shared counter line still produced a real merge conflict
    between any two GUI PRs landing close together. A value derived fresh from
    the files being served has nothing checked into git to forget OR to
    conflict over - two branches touching different assets each simply serve a
    correct value; once merged, the tree naturally reflects both changes.

    A file that cannot be read (missing, permission error) feeds a distinct
    ``MISSING:<path>: <error>`` marker into the digest instead of crashing this
    route - a broken asset still forces a cache-bust rather than either a 500
    or, worse, silently serving a digest that looks unchanged. The read failure
    is ALSO logged loudly (AGENTS.md rule 5): a client seeing stale assets
    because of a broken install should be discoverable in the debug log, not
    only inferable from cache behaviour."""
    h = hashlib.sha256()
    for p in _sw_cacheable_files():
        rel = p.relative_to(STATIC_DIR).as_posix()
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        try:
            h.update(p.read_bytes())
        except OSError as e:
            logger.warning(
                "sw.js cache digest: could not read %s (%s) - forcing a "
                "cache-bust rather than silently serving a stale digest",
                rel, e)
            h.update(f"MISSING:{rel}: {e}".encode("utf-8"))
        h.update(b"\0")
    return f"localm-shell-{h.hexdigest()[:16]}"


def _sw_js_response(if_none_match: "str | None" = None) -> Response:
    """sw.js's own bytes with the CACHE constant substituted for a fresh content
    digest (see ``_compute_sw_cache_value``). An unparseable source (the
    placeholder line was edited into some other shape) is a real problem - fail
    LOUD with a clear 500 rather than silently serving the placeholder text as
    a literal cache name that could collide across unrelated deploys.

    Honors a conditional GET (``If-None-Match``) with a real 304 the same way
    ``_RevalidatingStatic`` already does for every other static asset - its own
    docstring's contract is "no-cache does NOT disable caching: the browser
    still caches and revalidates with the ETag, so an unchanged file is a
    cheap 304". Routing this ONE path around that mount must not silently
    drop that contract for it. The ETag is over the FINAL substituted body,
    not just the computed cache value: an edit to sw.js's own logic (with no
    watched asset changing) still changes what gets served and must not
    collide with a stale ETag from before the edit.

    ``read_text`` applies universal-newline translation, so a CRLF-checked-out
    source (this repo's default on Windows) is served with LF line endings -
    harmless (JS does not care) and, since it happens the same way on every
    request, does not affect determinism; noted so a future reader diffing the
    served bytes against the git-tracked file does not mistake the line-ending
    difference for a real bug."""
    text = (STATIC_DIR / "sw.js").read_text(encoding="utf-8")
    value = _compute_sw_cache_value()
    # Count matches BEFORE substituting, not via subn's own return count: subn(...,
    # count=1) reports n=1 whether the pattern matched once or several times (it
    # caps how many it REPLACES, not how many it FOUND), so `if n != 1` can never
    # detect a second match - it would silently substitute the FIRST occurrence
    # (e.g. an illustrative example string added to a future comment) and leave
    # the REAL const CACHE line holding the literal placeholder text, reporting
    # success while quietly defeating the whole mechanism. Exactly one match is
    # required; zero or two-or-more both fail loud.
    n_matches = len(SW_CACHE_LINE_RE.findall(text))
    if n_matches != 1:
        raise HTTPException(
            500, f"sw.js's CACHE constant line matched {n_matches} times "
            '(expected exactly one `const CACHE = "...";`) - cannot safely '
            "substitute the computed cache version. This is a build defect, "
            "not a client error.")
    new_text = SW_CACHE_LINE_RE.sub(rf"\g<1>{value}\g<2>", text, count=1)
    etag = f'"{hashlib.sha256(new_text.encode("utf-8")).hexdigest()[:32]}"'
    headers = {"Cache-Control": "no-cache", "ETag": etag}
    if if_none_match and if_none_match == etag:
        return Response(status_code=304, headers=headers)
    return Response(content=new_text, media_type="text/javascript", headers=headers)


def mint_launch_grant(app, ttl: float = 300.0) -> str:
    """Mint a single-use, short-lived grant that the launcher/CLI puts in the
    browser URL (``/?localm_token=<grant>``) so a just-launched loopback browser
    lands AUTHENTICATED via a real navigation, instead of relying on the implicit
    GET / cookie auto-seed (which a focused-but-not-reloaded tab or a warm service-
    worker cache can skip). Stored in-process on ``app.state.launch_grants`` (dies on
    restart); expired grants are pruned on each mint so the dict cannot grow."""
    import secrets as _secrets
    import time as _time
    grants = getattr(app.state, "launch_grants", None)
    if grants is None:
        grants = {}
        app.state.launch_grants = grants
    now = _time.time()
    for k in [k for k, exp in grants.items() if exp <= now]:
        grants.pop(k, None)
    token = _secrets.token_urlsafe(32)
    grants[token] = now + float(ttl)
    return token


def _consume_launch_grant(app, token: str) -> bool:
    """Redeem a launch grant: SINGLE-USE (popped) and not expired. False for an
    unknown/used/expired token (so a replayed or guessed token simply falls through
    to the normal key gate, never an error that would confirm anything)."""
    import time as _time
    if not token:
        return False
    grants = getattr(app.state, "launch_grants", None) or {}
    exp = grants.pop(token, None)      # single use: remove on redeem
    return exp is not None and _time.time() <= float(exp)


def mint_pull_grant(app, spec: str, ttl: float = 120.0) -> str:
    """Mint a single-use, short-lived grant binding a specific model *spec* to an
    unguessable token (SEC-PULL-CONFIRM). ``localm gui --pull SPEC`` puts this in
    the deep link (``?pull=SPEC&pull_token=...``) so ITS OWN browser tab can
    auto-start the download with zero clicks; a forged ``?pull=`` link (any other
    page, or a hidden iframe on any site while localm runs locally) cannot know
    this secret, so the frontend falls back to an explicit human confirmation
    instead (see init.js) - a download never starts from a URL alone. Stored
    in-process on ``app.state.pull_grants`` (dies on restart); expired grants are
    pruned on each mint so the dict cannot grow."""
    import secrets as _secrets
    import time as _time
    grants = getattr(app.state, "pull_grants", None)
    if grants is None:
        grants = {}
        app.state.pull_grants = grants
    now = _time.time()
    for k in [k for k, (_, exp) in grants.items() if exp <= now]:
        grants.pop(k, None)
    token = _secrets.token_urlsafe(32)
    grants[token] = (spec, now + float(ttl))
    return token


def consume_pull_grant(app, spec: str, token: str) -> bool:
    """Redeem a pull grant: SINGLE-USE, not expired, and bound to the EXACT spec
    it was minted for (so a leaked/observed token cannot be replayed to authorise
    pulling a different model). Only popped on an actual match or once expired -
    a mismatched-spec probe must not burn an otherwise-still-valid grant, or a
    single wrong guess could deny the legitimate redemption that follows it.
    False for an unknown/used/expired/mismatched token."""
    import time as _time
    if not token:
        return False
    grants = getattr(app.state, "pull_grants", None) or {}
    entry = grants.get(token)
    if entry is None:
        return False
    granted_spec, exp = entry
    if _time.time() > float(exp):
        grants.pop(token, None)     # expired: clean up regardless of spec
        return False
    if granted_spec != spec:
        return False                # wrong spec: leave the grant intact
    grants.pop(token, None)          # right spec, still valid: single use
    return True


def _set_session_cookies(response, key: str, *, secure: bool) -> None:
    """Establish the S2 auth cookie on *response* for a loopback owner: mint an
    OPAQUE server-side session for the current owner *key* and set the HttpOnly
    ``localm_session`` cookie to the SESSION ID (never the key, so it never touches
    page JS and rolling the key does not invalidate it). It carries SESSION_MAX_AGE
    so the session PERSISTS across a browser/PWA restart (SEAMLESS). No-op if *key*
    is not a valid key. The CSRF token is DERIVED from the session and fetched by
    the client from GET /api/session, so there is no separate CSRF cookie to set (or
    to fall out of sync with the session, the pre-rework 'missing CSRF token' bug)."""
    from localm import scopes as S, sessions
    from localm.auth import _hash_key, fs_access_for, verify
    from localm.inference.http_server import SESSION_COOKIE, SESSION_MAX_AGE
    held = verify(key)
    if held is None:
        return
    fs = "host" if S.ADMIN in held else fs_access_for(key, "none")
    try:
        sid = sessions.create(scopes=held, key_hash=_hash_key(key), fs_access=fs)
    except Exception as e:
        # The session store could not be written (e.g. a corrupt/unreadable
        # sessions.json). Do NOT 500 the whole GUI shell over a convenience
        # auto-seed: serve the shell WITHOUT a session cookie so the client falls
        # to the key gate (recoverable), and surface the reason. Fail SAFE - no
        # cookie means no access granted, so this never reports a success that did
        # not happen (AGENTS rule 5).
        from localm.debuglog import logger as _dbg
        _dbg.warning("could not establish a browser session (auto-seed): %s; "
                     "serving the shell unauthenticated (the key gate will show)", e)
        return
    response.set_cookie(SESSION_COOKIE, sid, httponly=True, secure=secure,
                        samesite="strict", path="/", max_age=SESSION_MAX_AGE)


class LoadModelRequest(BaseModel):
    model: str


class PullRequest(BaseModel):
    spec: str
    name: str | None = None
    mmproj: str | None = None
    # "copy" | "move" | None (default: register a local path in place, unchanged).
    # Ignored for a HuggingFace/URL spec - only the local-path pull branch uses it.
    store: str | None = None
    # Explicit type hint from the "Find models" search: the result's detected
    # type, or the single Type checkbox the user narrowed to. Bypasses pull-time
    # HF guessing (unreliable for a standalone vae/text-encoder, whose repos
    # carry no type metadata). None lets the pull auto-detect.
    model_type: str | None = None


class PullTokenRedeemRequest(BaseModel):
    spec: str
    token: str


class MediaPreflightRequest(BaseModel):
    """Model-relevant overrides for a pre-generate model-existence check. Only
    the fields that can change WHICH model filename a loader node references -
    prompt text, seed, steps, dimensions, etc. never affect that and are not
    accepted here. Image uses clip_name1/clip_name2/lora_name; music uses
    ckpt_name. ``model_overrides`` is the per-slot node_id/input_name dict from
    the Workflow panel's model dropdowns (see apply_model_overrides()) - shared
    by all three media types, applied first, exactly like the real generate
    call, so a picked-but-not-installed model is caught here too."""
    clip_name1: str | None = None
    clip_name2: str | None = None
    lora_name: str | None = None
    ckpt_name: str | None = None
    model_overrides: dict[str, dict[str, str]] | None = None


class ComfyPullRequest(BaseModel):
    # Only a filename the client saw in a preceding preflight response - NEVER a
    # client-supplied repo/path. The server re-resolves everything else from
    # COMFY_MODEL_SOURCES itself (see model_pull_comfy_source in routes/models.py).
    filename: str


class RemoveModelRequest(BaseModel):
    model: str


class UnloadModelRequest(BaseModel):
    # None (the default, and an empty POST body) unloads every loaded model -
    # unchanged behavior; a name unloads only that one.
    model: str | None = None


class AliasRequest(BaseModel):
    model: str
    alias: str


class RenameModelRequest(BaseModel):
    model: str
    new_name: str


class SetTypeRequest(BaseModel):
    model: str
    model_type: str


class ScanRequest(BaseModel):
    # None (the default, and an empty/absent POST body - the old Scan button
    # sends no body at all) scans the configured comfy_workdir, unchanged. An
    # explicit workdir is a one-off scan of that folder (never written back to
    # config) - gated behind host filesystem access, same trust level as the
    # folder browser, since it lets the caller point the scanner anywhere.
    workdir: str | None = None
    # True previews counts by category and registers nothing.
    dry_run: bool = False


class ShareClearRequest(BaseModel):
    ids: list[str] = []


class LogExportRequest(BaseModel):
    dest: str = ""


# Image types accepted from a phone share-sheet into the chat composer.
_SHARE_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".heic", ".heif"}


def _share_inbox() -> Path:
    """Transient inbox for files shared INTO localm from a phone (PWA share
    target). Lives under the data dir; entries are deleted once the app ingests
    them, so it never accumulates."""
    from localm.config import home_dir
    d = home_dir() / "share_inbox"
    d.mkdir(parents=True, exist_ok=True)
    return d


# No recorded owner ("-") means open mode / untracked - unrestricted, mirroring
# jobs' job_owner_ok "owner=None is unrestricted" semantics.
_SHARE_NO_OWNER = "-"


def _share_entry_name(owner: "str | None", fid: str, filename: str) -> str:
    """Build an inbox filename carrying its creator's principal id, so a later
    request can be checked against job_owner_ok before it is read or cleared."""
    token = owner if owner else _SHARE_NO_OWNER
    return f"{fid}__{token}__{filename}"


def _parse_share_entry(path: Path) -> "tuple[str, str | None, str]":
    """(fid, owner_or_None, filename) from an inbox entry's name.

    maxsplit=2 so a filename that itself contains "__" is not corrupted (fid and
    the owner token are both constructed to never contain "_", so the first two
    separators are unambiguous). Falls back to the pre-ownership two-part format
    (owner=None, i.e. unrestricted) for any entry left over from before this
    field existed, so an old on-disk inbox never breaks the listing."""
    parts = path.name.split("__", 2)
    if len(parts) == 3:
        fid, token, name = parts
        return fid, (None if token == _SHARE_NO_OWNER else token), name
    fid, _, name = path.name.partition("__")
    return fid, None, name


def _multipart_boundary(content_type: str) -> "bytes | None":
    """The boundary token from a multipart/form-data Content-Type, or None."""
    if "multipart/form-data" not in (content_type or "").lower():
        return None
    for part in content_type.split(";"):
        part = part.strip()
        if part.lower().startswith("boundary="):
            b = part[len("boundary="):].strip().strip('"')
            return b.encode("latin-1") if b else None
    return None


def _disp_param(disposition: bytes, key: bytes) -> "bytes | None":
    """Value of a Content-Disposition parameter, e.g. name= or filename=."""
    token = key + b'="'
    i = disposition.find(token)
    if i == -1:
        return None
    i += len(token)
    j = disposition.find(b'"', i)
    return disposition[i:j] if j != -1 else None


def _parse_multipart(body: bytes, boundary: bytes):
    """Minimal multipart/form-data parser - no python-multipart dependency, in
    keeping with localm's self-contained rule (it already hand-builds multipart
    for ComfyUI uploads). Returns (fields: dict[str,str], files: list of
    (filename, content_type, data))."""
    fields: dict = {}
    files: list = []
    for raw in body.split(b"--" + boundary):
        part = raw.strip(b"\r\n")
        if not part or part == b"--":
            continue                       # preamble / closing delimiter
        head, sep, data = part.partition(b"\r\n\r\n")
        if not sep:
            continue
        headers: dict = {}
        for line in head.split(b"\r\n"):
            k, _, v = line.partition(b":")
            headers[k.strip().lower()] = v.strip()
        disposition = headers.get(b"content-disposition", b"")
        name = _disp_param(disposition, b"name")
        filename = _disp_param(disposition, b"filename")
        if filename is not None:
            ctype = headers.get(b"content-type", b"application/octet-stream")
            files.append((filename.decode("utf-8", "replace"),
                          ctype.decode("latin-1"), data))
        elif name is not None:
            fields[name.decode("utf-8", "replace")] = data.decode("utf-8", "replace")
    return fields, files


# R37: a phone (or any browser) uploads files INTO a durable localm folder that
# models and tools can then read - distinct from transient chat attachments and
# the share_inbox. Capped per request because the hand-rolled parser reads the
# whole body into memory; large model weights use the model pull flow, not this.
_MAX_UPLOAD_BYTES = 100 * 1024 * 1024   # 100 MB / request


# Characters never allowed in an uploaded file's basename: the Windows-reserved
# set (a name like "x.txt:stream" would otherwise write to an NTFS alternate data
# stream that the list/delete routes cannot see) plus all control chars (a literal
# NUL would raise ValueError deep in pathlib and surface as a bare 500).
_BAD_NAME_CHARS = set('<>:"|?*\\/') | {chr(c) for c in range(32)}


def _name_is_safe(safe: str) -> bool:
    """True if *safe* (already a basename) is a usable, listable file name.

    A shared security guard, not a local helper. Several call sites depend on it,
    covering both the write paths (/api/upload, /share-target) and the delete
    path (via _confined_upload_path), so widening _BAD_NAME_CHARS widens what all
    of them accept. Grep for the callers before changing it rather than trusting
    this sentence: a list written into a docstring goes stale the moment someone
    adds one.
    """
    return bool(safe) and safe not in (".", "..") and not (set(safe) & _BAD_NAME_CHARS)


def _uploads_dir() -> Path:
    """<home>/uploads/, created on demand."""
    from localm.config import home_dir
    d = home_dir() / "uploads"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _confined_upload_path(name: str) -> Path:
    """Resolve a user-supplied file name to a path INSIDE the uploads dir. Strips
    any directory components (basename only) and rejects anything that would
    resolve outside the uploads dir - so '..', absolute paths, encoded slashes, or
    a name with illegal/control chars cannot traverse out or crash. Raises
    HTTPException(400) on an unsafe name."""
    base = _uploads_dir()
    safe = Path(name or "").name            # basename only, drops any dir parts
    if not _name_is_safe(safe):
        raise HTTPException(400, "Invalid file name.")
    target = (base / safe).resolve()
    if not target.is_relative_to(base.resolve()):
        raise HTTPException(400, "Invalid file name.")
    return target


def _unique_upload_target(base: Path, safe_name: str) -> Path:
    """A non-clobbering path for *safe_name* in *base*: 'note.txt' -> 'note (1).txt'
    if it already exists (mirrors the log-export dedup), so an upload never
    silently overwrites an existing file."""
    target = base / safe_name
    if not target.exists():
        return target
    stem, suffix = Path(safe_name).stem, Path(safe_name).suffix
    n = 1
    while True:
        cand = base / f"{stem} ({n}){suffix}"
        if not cand.exists():
            return cand
        n += 1


def attach_gui(
    app: FastAPI,
    *,
    self_url: str,
    switch_model,
    active_model,
) -> SessionManager:
    """
    Add GUI routes and static serving to *app*.

    Parameters
    ----------
    self_url:
        Base URL of this server's own /v1 API - coder agents talk to the
        model through it (e.g. ``http://127.0.0.1:8642/v1``).
    switch_model:
        ``Callable[[str], Awaitable[None]]`` - swaps the active engine.
    active_model:
        ``Callable[[], str]`` - name of the currently served model.
    """
    manager = SessionManager()

    from .jobs import JobManager
    jobs = JobManager()

    # Shared services that converted plugins (rag/image/music/video) reach via
    # request.app.state: the background-job manager, this server's own /v1 base
    # URL (for self-embedding), and the active-model accessor. The /api/jobs/*
    # SSE endpoints stay in the kernel GUI (routes/jobs.py) so every plugin's
    # jobs stream through the one manager.
    app.state.jobs = jobs
    app.state.self_url = self_url
    app.state.active_model = active_model
    # The builtin "coder" plugin reads these to drive live sessions and
    # per-session model switches; its routes 503 when they're absent
    # (headless / no GUI). The manager is also returned for close_all().
    app.state.switch_model = switch_model
    app.state.coder_sessions = manager
    # One-time launcher -> browser handoff grants (see mint_launch_grant): a small
    # in-process dict of token -> expiry. Created here so both the auto-opened
    # loopback browser and a later remote mint have somewhere to store them.
    if not hasattr(app.state, "launch_grants"):
        app.state.launch_grants = {}

    # Route groups (extracted to localm/plugins/gui/routes/*.py). The active-model
    # accessor, the model-switch callable, and the job manager are the only locals
    # the handlers close over, so they travel to the groups on ctx. Each group's
    # register() appends its routes; registration order does not matter (every
    # path is distinct) as long as the static mount below stays last.
    from types import SimpleNamespace
    ctx = SimpleNamespace(active_model=active_model, switch_model=switch_model,
                          jobs=jobs)
    from .routes import models as _routes_models
    _routes_models.register(app, ctx)
    from .routes import system as _routes_system
    _routes_system.register(app, ctx)
    from .routes import pairing as _routes_pairing
    _routes_pairing.register(app, ctx)
    from .routes import admin as _routes_admin
    _routes_admin.register(app, ctx)
    from .routes import jobs as _routes_jobs
    _routes_jobs.register(app, ctx)
    from .routes import share as _routes_share
    _routes_share.register(app, ctx)
    from .routes import uploads as _routes_uploads
    _routes_uploads.register(app, ctx)
    from .routes import comfy as _routes_comfy
    _routes_comfy.register(app, ctx)

    # R47: "/api/bug-report" POST lives on the core server (http_server.py) so it
    # works in headless `localm serve`; the GUI button targets that one canonical
    # route. Duplicating it here would shadow it (first route wins) and silently drop
    # the user's description + include_log flag.

    # Media gen (/api/imagine*, /api/music*, /api/video*), chat persistence
    # (/api/conversations, /api/memory, /api/prompts), and RAG (/api/rag/*) live in
    # builtin plugins under localm/plugins/builtin/. The chat turn
    # (/v1/chat/completions) stays in the kernel inference server.

    # Mounted last: API routes above take precedence over the SPA files.
    # Pin the MIME types the PWA relies on (some Windows registries map .js to
    # text/plain, and .webmanifest is unknown to mimetypes) so the service
    # worker, app scripts, manifest, and icon are served correctly.
    import mimetypes
    mimetypes.add_type("text/javascript", ".js")
    mimetypes.add_type("application/manifest+json", ".webmanifest")
    mimetypes.add_type("image/svg+xml", ".svg")

    # Serve the SPA shell. On a loopback bind (or for a loopback client on a LAN
    # bind) seed the configured API key into the page so a fresh launch is not
    # locked out when require_auth is on (C1). Registered before the "/" static
    # mount so it wins for the shell document; every other asset hits the mount.
    @app.get("/", include_in_schema=False)
    @app.get("/index.html", include_in_schema=False)
    async def _gui_index(request: Request):
        from localm import auth
        # Only a loopback BIND (the default `localm gui`, reachable solely from
        # this machine) auto-seeds anything. We deliberately do NOT trust
        # request.client.host: behind a same-host reverse proxy it reads as
        # loopback for REMOTE users, which would leak the secret. A non-loopback
        # bind (e.g. -H 0.0.0.0) seeds nothing - the user enters the key in the
        # page, which POSTs it to /api/session to set the session cookie.
        loopback = _is_loopback_host(
            getattr(request.app.state, "bind_host", "127.0.0.1"))
        key = auth.get_api_key() or ""
        # The shell must never be served stale: it carries the per-request shell
        # token and references the current assets, so always revalidate (a new
        # app.js / index.html is then picked up without the user clearing caches).
        headers = {"Cache-Control": "no-cache"}
        # One-time launch handoff: the launcher/CLI opens /?localm_token=<grant>. A
        # valid single-use grant establishes a session and 303-redirects to the clean
        # path (token stripped), so the browser lands authenticated via a REAL
        # navigation a stale tab or warm SW cannot short-circuit. Redeemed on ANY bind:
        # the grant is a 256-bit single-use secret only the launcher knows and only
        # places in the URL it opens on THIS machine, so possessing it IS the
        # authorization (a network client never receives or guesses it) - which is why
        # it works where the keyless loopback-only auto-seed below cannot. A bad/used/
        # expired grant falls through to the normal shell (no error, nothing leaked).
        grant = request.query_params.get("localm_token")
        if grant and key and _consume_launch_grant(request.app, grant):
            from urllib.parse import urlencode
            from fastapi.responses import RedirectResponse
            q = {k: v for k, v in request.query_params.items() if k != "localm_token"}
            clean = request.url.path + (("?" + urlencode(q)) if q else "")
            resp = RedirectResponse(url=clean, status_code=303, headers=headers)
            _set_session_cookies(resp, key, secure=request.url.scheme == "https")
            return resp
        if key and loopback:
            # Protected mode on loopback: establish an HttpOnly session cookie so
            # the key never touches page JS / localStorage. Only MINT a new
            # session when the browser has no valid one, so an ordinary reload does
            # not spawn a session each time - and, crucially, a browser whose
            # session is still valid after an owner-key ROLL stays signed in (the
            # session is decoupled from the key), instead of being bounced to the
            # key gate (the reported bug).
            from localm.inference.http_server import SESSION_COOKIE
            from localm import sessions as _sessions
            existing = (request.cookies.get(SESSION_COOKIE) or "").strip()
            resp = HTMLResponse(_index_html_with_shell_token(""), headers=headers)
            if not (existing and _sessions.lookup(existing) is not None):
                _set_session_cookies(resp, key, secure=request.url.scheme == "https")
            return resp
        if not key and loopback:
            # Open mode on loopback: seed the per-process shell token as a JS
            # global so the loopback SPA can still manage (H5). app.js sends it
            # as a bearer HEADER (the open-mode gate is header-based); it is
            # never persisted.
            token = getattr(request.app.state, "shell_token", "") or ""
            return HTMLResponse(_index_html_with_shell_token(token), headers=headers)
        return HTMLResponse(_index_html_with_shell_token(""), headers=headers)

    @app.get("/sw.js", include_in_schema=False)
    async def _service_worker(request: Request):
        # Registered before the "/" static mount so it wins for this ONE path;
        # every other static asset still goes straight to the mount below.
        # Reads + hashes the whole static tree, so it is offloaded to a worker
        # thread rather than run inline on the event loop (mirrors how other
        # per-request filesystem work in this server is handled).
        from starlette.concurrency import run_in_threadpool
        return await run_in_threadpool(
            _sw_js_response, request.headers.get("if-none-match"))

    app.mount("/", _RevalidatingStatic(directory=str(STATIC_DIR), html=True), name="gui")

    # Single source of truth that the GUI surface is mounted on this app, so the
    # on-demand mount (phase 5 mount_gui_surface) is idempotent whether the GUI
    # was attached at startup (localm gui) or live on a running api instance.
    app.state.gui_mounted = True

    return manager

# SPDX-License-Identifier: AGPL-3.0-or-later
"""The kernel's security middleware: CORS, the origin / open-mode shell-token
gate, the security response headers (an enforcing CSP with a per-request nonce)
and the API-docs disclosure guard.

``create_app()`` adds them in that order. Each ``add_middleware`` wraps what was
added before it, so a request meets the docs guard first and CORS last, and a
refusal from the origin gate still passes out through the security headers.
Every local-trust decision here keys on ``app.state.bind_host`` (what the server
bound to), never ``request.client.host``: behind portmux every peer is
127.0.0.1."""

from __future__ import annotations

import secrets
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import localm.inference.http_server as _hs


def add_cors(app: FastAPI) -> Any:
    """Add CORSMiddleware for the "cors_origins" setting and return the setting,
    which the origin guard reuses so the config is read once per app."""
    # CORS: localhost-only by default. A wildcard here would let ANY website
    # the user visits call this API from browser JS and read the responses
    # (drive-by GPU use, response exfiltration, /v1/models/unload abuse).
    # Override with config "cors_origins": ["https://app.example"] or "*".
    from localm.config import load_config
    cors_cfg = load_config().get("cors_origins")
    cors_kwargs: dict
    if cors_cfg == "*":
        cors_kwargs = {"allow_origins": ["*"]}
    elif isinstance(cors_cfg, list) and cors_cfg:
        cors_kwargs = {"allow_origins": cors_cfg}
    else:
        cors_kwargs = {
            "allow_origin_regex": r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
        }
    app.add_middleware(
        CORSMiddleware,
        allow_methods=["*"],
        allow_headers=["*"],
        **cors_kwargs,
    )
    return cors_cfg


def add_origin_guard(app: FastAPI, cors_cfg: Any) -> None:
    """Add the cross-origin refusal and the open-mode management gate.
    ``_CROSS_ORIGIN_OK`` stays a local of this function: tests read it out of
    ``_origin_guard``'s closure."""
    # CSRF / drive-by guard. The default CORS policy admits ANY localhost:PORT
    # origin, so without this a malicious local web page (a dev server, an npm
    # postinstall server) could drive state-changing endpoints from the user's
    # browser - mint a key, flip require_auth, install a plugin, load/unload the
    # model, browse the filesystem, read files via a plugin route like /api/rag,
    # or drive the coder via /api/coder - even keyless (open-mode scope collapse).
    # So every unsafe-method request must be same-origin (or a configured CORS
    # origin), EXCEPT the OpenAI-compatible inference API, left cross-origin
    # callable for local apps. Allowlist-by-default means a new plugin route is
    # protected the moment it is added. Non-browser clients (CLI / SDK) send no
    # Origin; "cors_origins": "*" opts out entirely.
    _UNSAFE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
    # Every entry below is a full route path, not a directory prefix - matched
    # with str.startswith(), so a prefix entry would silently exempt every
    # FUTURE route added under it too. Exempting a new sibling route must be a
    # deliberate addition here, not an inheritance. Keep in sync with
    # _BESPOKE_GATED_ROUTES. See test_every_kernel_route_is_gated_or_explicitly_allowlisted.
    _CROSS_ORIGIN_OK = (
        "/v1/chat/completions", "/v1/completions", "/v1/embeddings",
        # Surface management (phase 5 on-demand GUI mount) is driven by a local
        # process (the attaching `localm gui`), not the browser shell: no Origin,
        # no shell_token. The route does its OWN strict auth (this instance's
        # attach token, or an owner API key) - that, not the same-origin gate, is
        # the real credential, so it is exempt. A cross-origin page still cannot
        # set Authorization without a secret it cannot read, so no CSRF surface.
        # Also listed in _BESPOKE_GATED_ROUTES - keep both in sync.
        # See test_every_kernel_route_is_gated_or_explicitly_allowlisted.
        "/v1/surfaces/gui",
        # Multi-instance GPU coordination (localm.gpu_registry): a SIBLING localm
        # instance calls this loopback-only, like surface-management above - no
        # Origin, no shell_token (different process). Its own coordination_token
        # (never the API key/shell token) is the real credential, checked in the
        # route, so the same-origin gate is exempt for the same reason.
        # Also listed in _BESPOKE_GATED_ROUTES - keep both in sync.
        # See test_every_kernel_route_is_gated_or_explicitly_allowlisted.
        "/v1/instances/cooperate-unload",
    )
    _cors_allowlist = frozenset(cors_cfg) if isinstance(cors_cfg, list) else frozenset()
    _cors_wildcard = cors_cfg == "*"

    # CWE-200: a short list of UNAUTHENTICATED GETs that disclose host
    # detail and, unlike the /api,/v1 metadata reads below, have NO route-level
    # auth to fall back on. The default CORS policy hands an ACAO to any
    # http(s)://localhost:PORT origin, so without an explicit refusal a drive-by
    # local page could read them cross-origin: /whoami leaks root_dir (an absolute
    # path -> the OS username) on a loopback bind, and /debug/stacks leaks thread
    # stacks. They sit OUTSIDE the /api,/v1 metadata-GET gate, so they are refused
    # here instead - cross-origin, in EVERY mode (they are unauthenticated in
    # protected mode too, so an open-mode-only refusal would miss them).
    _CROSS_ORIGIN_GET_REFUSED = ("/whoami", "/debug/stacks")

    # Same refusal, matched by PREFIX rather than exact path. /api/fs/* is the
    # host filesystem browser: it enumerates the user's disk, which is host
    # detail of exactly the kind above, and it is a GET, so it is exempt from
    # both the CSRF gate (unsafe methods only) and the open-mode shell-token
    # gate. The generic /api,/v1 metadata-GET gate below does cover it, but only
    # inside the `not any_key_configured()` branch - so in PROTECTED mode a
    # cookie-authenticated cross-origin GET would execute. SameSite=strict does
    # not close it either: "site" ignores port, so any other page on a loopback
    # port is same-site, which is the precise actor the comment above at the
    # _cross_origin_refused definition names. Refused in EVERY mode instead.
    _CROSS_ORIGIN_GET_REFUSED_PREFIXES = ("/api/fs/",)
    # Sensitive GETs that sit OUTSIDE the /api,/v1 prefixes the open-mode
    # shell-token gate below keys on, and so are not covered by it.
    # /debug/stacks' own Depends(require_fs_host) is a TAUTOLOGY in keyless mode
    # - effective_fs_access returns "host" for everyone when no key is
    # configured - so without this list it is reachable with no credential at
    # all. Listing it here routes it through the same shell-token + cross-origin
    # check as a management read, which is what makes its fs-host gate mean
    # something. /whoami is NOT here: it is the endpoint the GUI
    # shell calls to discover whether it needs a key at all, so requiring the
    # token to read it would be circular. Its disclosure is handled by the
    # cross-origin refusal above and is a separate, narrower surface.
    # NOTE: enforced only on a LOOPBACK bind - see the comment at token_gated_get
    # below for why answering 403 off loopback would open a new oracle.
    _SHELL_TOKEN_GETS = ("/debug/stacks",)

    def _cross_origin_refused(request) -> bool:
        """True when this request carries an Origin header that is neither
        same-origin nor CORS-allow-listed. Shared by the CSRF check (unsafe
        methods) and the open-mode shell-token gate: the default
        CORS policy lets any http(s)://localhost:PORT / 127.0.0.1:PORT origin
        READ a matching response, so a hostile local page can steal the shell
        token from a plain cross-origin ``GET /`` and replay it - token
        possession alone does not prove the caller IS the loopback GUI shell.
        "cors_origins": "*" opts OUT of this specific check, same
        as it already did for the CSRF check; it does not waive the shell-token
        requirement itself."""
        if _cors_wildcard:
            return False
        origin = request.headers.get("origin")
        if not origin:
            return False
        allowlisted = origin in _cors_allowlist
        host = request.headers.get("host", "")
        same_origin = origin.split("://", 1)[-1] == host
        return not (same_origin or allowlisted)

    @app.middleware("http")
    async def _origin_guard(request, call_next):
        _path = request.url.path
        # Cross-origin refusal (every mode): every state-changing method (CSRF),
        # plus the sensitive GETs in _CROSS_ORIGIN_GET_REFUSED (exact) and
        # _CROSS_ORIGIN_GET_REFUSED_PREFIXES (prefix) - host-detail disclosure,
        # All are subject to the same same-origin / CORS-allowlist
        # check.
        if ((request.method in _UNSAFE_METHODS
             or (request.method == "GET"
                 and (_path in _CROSS_ORIGIN_GET_REFUSED
                      or _path.startswith(_CROSS_ORIGIN_GET_REFUSED_PREFIXES))))
                and not _path.startswith(_CROSS_ORIGIN_OK)):
            if _cross_origin_refused(request):
                return JSONResponse(
                    status_code=403,
                    content={"detail": "Cross-origin request refused "
                             "(only same-origin requests or a configured "
                             "'cors_origins' may use this endpoint)."},
                )
            # Open-mode management gate. With no key configured, management
            # routes still require the per-process shell token (injected into the
            # loopback GUI shell), so a no-Origin local client (curl, a script)
            # can no longer mint a key, flip config, install a plugin, load a
            # model, or browse the filesystem unauthenticated. Protected mode (a
            # key exists) is bearer-auth'd on the route. The token is required even
            # for an allowlisted CORS origin: an Origin header is forgeable, so
            # it is not a management credential - a configured external origin must
            # use an API key for state changes.
        is_unsafe = request.method in _UNSAFE_METHODS
        # A _SHELL_TOKEN_GETS path is token-gated only where it is actually
        # SERVED. /debug/stacks 404s off a loopback bind, and returning 403
        # there instead would tell an unauthenticated NETWORK
        # caller that the endpoint exists - a brand new existence oracle opened
        # in the middle of closing a disclosure, since an unknown path under
        # /debug/ 404s. So off loopback, fall through to the handler's own 404.
        # The handler does the loopback check before computing anything, so
        # nothing is spent serving it.
        token_gated_get = (
            request.method == "GET"
            and request.url.path in _SHELL_TOKEN_GETS
            and _hs._is_loopback_host(
                getattr(request.app.state, "bind_host", "127.0.0.1")))
        is_metadata_get = token_gated_get or (
            request.method == "GET"
            and (request.url.path.startswith("/api/")
                 or request.url.path.startswith("/v1/"))
            and request.url.path != "/api/session"
            and not request.url.path.startswith("/v1/models")
        )
        if (is_unsafe or is_metadata_get) and not request.url.path.startswith(_CROSS_ORIGIN_OK):
            from localm.auth import (any_key_configured, ct_equal,
                                     require_auth_enabled)
            if not any_key_configured() and not require_auth_enabled():
                token = getattr(request.app.state, "shell_token", None)
                # A keyless LOCAL process (`localm status`, the MCP
                # server_activity tool) has no way to obtain shell_token - it is
                # per-process, never persisted, and only ever injected into the
                # browser-served SPA. It DOES already have this instance's own
                # attach token (instances.py's per-instance registry file,
                # 0600/owner-only, read via instances.attach_target/snapshot) -
                # the exact credential /v1/surfaces/gui's mount_gui route
                # already accepts for the same "local process, not a browser"
                # distinction. Accepting it here too turns keyless CLI/MCP
                # activity reads on, without touching what a browser can do:
                # a browser has no filesystem access to the registry file and
                # so can never present this token, unlike shell_token (which
                # DOES reach the browser and needs the cross-origin check
                # below as its own defence).
                #
                # This gate covers every open-mode management route (minting a
                # key, changing config, unloading a model), not just activity -
                # so accepting inst_token here is NOT scoped to /api/activity,
                # it authorizes all of them. That is not an escalation: the
                # PRINCIPAL, not the credential, is what decides this. Anything
                # that can read a 0600 file under the user's own home IS that
                # OS user, and that user can already read the keystore, the
                # config file, and the models directory directly on disk - the
                # token grants a local process nothing it did not already have
                # by other means.
                inst_token = getattr(request.app.state, "instance_token", None)
                presented = _hs._bearer_token(request)
                token_ok = ct_equal(presented, token) or (
                    bool(inst_token) and ct_equal(presented, inst_token))
                # An unsafe-method request already passed the
                # same-origin check above (or is exempt as _CROSS_ORIGIN_OK,
                # which never reaches here); a metadata GET never went through
                # that block at all, so it must pass the identical check here -
                # otherwise a token stolen via CORS (the default policy trusts
                # every localhost:PORT origin to READ a response) is directly
                # replayable cross-origin against every /api/*, /v1/* read.
                # Applies uniformly to both token kinds: a real CLI/MCP client
                # never sends an Origin header at all (that is a browser-only
                # header), so this costs the legitimate case nothing, and it is
                # defence-in-depth against a caller that somehow obtained the
                # (never-served) instance token some other way.
                cross_origin = is_metadata_get and _cross_origin_refused(request)
                if not token_ok or cross_origin:
                    return JSONResponse(
                        status_code=403,
                        content={"detail": "Open-mode management requires the "
                                 "localm GUI shell on this machine, or an API key "
                                 "(run 'localm key generate')."},
                    )
        return await call_next(request)


def add_security_headers(app: FastAPI) -> None:
    """Add nosniff, the CSP with this request's nonce, and cross-origin isolation."""
    # Security response headers. The user-content render path is XSS-safe via
    # DOMPurify; this is the Content-Security-Policy backstop behind it on the
    # GUI shell. nosniff is enforced everywhere (blocks MIME-sniff into
    # executable HTML), and the CSP is ENFORCING, not report-only.
    #
    # script-src carries a PER-REQUEST nonce rather than 'unsafe-inline', so the
    # shell's own inline scripts run and an injected one cannot. Adding
    # 'unsafe-inline' alongside would have no effect: a policy containing a nonce
    # makes browsers IGNORE 'unsafe-inline' entirely.
    #
    # style-src keeps 'unsafe-inline', and the NONCE CANNOT REPLACE IT: CSP3's
    # "is element nonceable" algorithm covers <script>, <style> and <link>
    # ELEMENTS only, while an inline style ATTRIBUTE is reachable only by
    # 'unsafe-inline' or by 'unsafe-hashes' plus a hash per distinct attribute
    # value. index.html relies on such attributes, most of them display:none on
    # elements that must start hidden; under style-src 'self' they stop applying
    # and those elements paint. Unaffected either way: KaTeX styles via CSSOM,
    # which CSP does not govern, as do the app's own el.style.x = y writes. The
    # cost is bounded: DOMPurify passes a model-authored style ATTRIBUTE through,
    # so a reply can restyle its own subtree. That is presentation, not
    # execution, and img-src/connect-src still deny the CSS url() exfiltration
    # path.
    #
    # form-action 'none' because NOTHING in this GUI submits a form - there is
    # not one <form> element in static/, and every mutation goes through fetch().
    # It is not covered by default-src: form-action is a NAVIGATION directive
    # with no fallback, so omitting it allows submission ANYWHERE. DOMPurify's
    # default ALLOWED_TAGS includes <form>, so a model reply rendering
    # <form action="https://elsewhere/" method="post"><input ...> survives
    # sanitisation and its action resolves to that remote origin, with no script
    # involved - neither DOMPurify nor the script-src nonce is in that path.
    # 'none' rather than 'self', since there is no legitimate same-origin
    # submission either, which also closes the same-origin CSRF shape against
    # localm's own /api.
    #
    # NO CDN ORIGIN IS LISTED, AND NOTHING NEEDS ONE. The tts plugin's Kokoro
    # bundle pulls the onnxruntime-web backend with a dynamic import()
    # (ort-wasm-simd-threaded.jsep.mjs), and a dynamic import is a MODULE SCRIPT,
    # so it is governed by script-src rather than connect-src. That runtime is
    # vendored and served from 'self'
    # (localm/plugins/builtin/tts/static/vendor/onnxruntime/, pointed at by the
    # plugin's own wasm_paths default). A TTS load error means the vendored
    # runtime did not resolve; widening the policy hides that fault rather than
    # fixing it.
    #
    # The 'wasm-unsafe-eval' token is REQUIRED and is a SECOND, INDEPENDENT
    # block: allowing an origin only gets the backend DOWNLOADED, while
    # compiling ANY WebAssembly needs its own grant, and onnxruntime-web is
    # WebAssembly on BOTH its wasm and webgpu paths, so without this no backend
    # can start at all. That token is the narrow CSP3 source for exactly this
    # case: it permits WebAssembly compilation only, and does NOT permit dynamic
    # evaluation of JavaScript, so it is strictly tighter than the broader token
    # a browser error text names.
    #
    # `blob:` in script-src is REQUIRED once the page is cross-origin isolated.
    # Isolation gives onnxruntime-web SharedArrayBuffer, so it switches to its
    # THREADED build, which loads its worker as a blob: module, and the load
    # otherwise dies with
    #     no available backend found. ERR: [wasm] TypeError: Failed to fetch
    #     dynamically imported module: blob:http://.../<uuid>
    # `worker-src 'self' blob:` is NOT sufficient for it: the dynamic import of
    # the blob module is governed by script-src. A blob: URL can only be minted
    # by same-origin script that is already executing, so this gives an INJECTED
    # script no new way in - the nonce still gates what may execute at all.
    _CSP_PREFIX = ("default-src 'self'; "
                   "script-src 'self' blob: 'wasm-unsafe-eval' 'nonce-")
    _CSP_SUFFIX = (
        "'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: blob:; "
        "font-src 'self' data:; "
        # blob: here for the same reason it is in img-src/script-src/worker-src:
        # the GUI mints these with its OWN URL.createObjectURL and then reads
        # them back (fetch for "send to chat" / "copy image", and the <video>
        # and <audio> players). A blob: URL is same-origin-scoped and cannot be
        # pointed at a remote origin, so this grants no exfiltration path -
        # connect-src still names every third-party origin explicitly.
        #
        # media-src MUST be spelled out. Without it, media falls back to
        # default-src 'self', which has no blob:, and the failure is SILENT:
        # assigning a blocked src fires an error EVENT on the element rather
        # than throwing, so a try/catch around it cannot see it and the player
        # just sits there dead. That is how this survived unnoticed.
        # huggingface.co / *.hf.co are the MODEL weights (chat models are
        # server-side, but the tts plugin fetches Kokoro's ~86 MB ONNX in the
        # browser and caches it there). The onnxruntime RUNTIME is vendored and
        # same-origin, so no CDN origin belongs here either.
        "connect-src 'self' blob: https://huggingface.co https://*.hf.co; "
        "media-src 'self' blob:; "
        "worker-src 'self' blob:; "
        "frame-src 'self'; "
        "object-src 'none'; "
        "base-uri 'none'; "
        # See the form-action note above: a NAVIGATION directive, no default-src
        # fallback, so leaving it out allowed a sanitiser-surviving model-authored
        # <form> to post off-box. Nothing in the GUI submits a form.
        "form-action 'none'; "
        "frame-ancestors 'none'"
    )

    @app.middleware("http")
    async def _security_headers(request, call_next):
        # The nonce is minted BEFORE call_next, not after, because the shell
        # route has to stamp this exact value onto its inline <script> tags while
        # it builds the body - so the value must already exist when the handler
        # runs. Setting the header afterwards from a value the handler never saw
        # would ship a nonce matching nothing and white-screen the GUI.
        nonce = secrets.token_urlsafe(16)
        request.state.csp_nonce = nonce
        resp = await call_next(request)
        resp.headers.setdefault("X-Content-Type-Options", "nosniff")
        resp.headers.setdefault(
            "Content-Security-Policy", _CSP_PREFIX + nonce + _CSP_SUFFIX)
        # CROSS-ORIGIN ISOLATION, so onnxruntime-web can use more than one
        # thread. Without both of these the document is not isolated,
        # SharedArrayBuffer is unavailable, and onnxruntime falls back to
        # numThreads=1, which makes neural TTS synthesis slower than playback so
        # a long reply stutters.
        #
        # 'credentialless' rather than 'require-corp': require-corp demands a CORP
        # header on EVERY cross-origin subresource, which we do not control for
        # huggingface.co or the onnx CDN. credentialless instead sends those
        # requests WITHOUT credentials, which is both sufficient for isolation and
        # correct here - none of localm's cross-origin fetches are authenticated,
        # they are public model and library downloads.
        resp.headers.setdefault("Cross-Origin-Opener-Policy", "same-origin")
        resp.headers.setdefault("Cross-Origin-Embedder-Policy", "credentialless")
        return resp


def add_docs_loopback_gate(app: FastAPI) -> None:
    """Serve /docs, /redoc and /openapi.json only on a loopback bind."""
    # API-surface disclosure guard. FastAPI's built-in docs (/docs, /redoc,
    # /openapi.json) enumerate every route + schema. Fine on a loopback bind, but
    # on a NETWORK bind it hands an unauthenticated remote a full attack-surface
    # map (every endpoint stays scope-gated, so no access is granted, but it is
    # needless disclosure). Serve docs only on a loopback bind, keyed on the
    # CONFIGURED bind host (never the peer - portmux makes the peer always look
    # loopback). 404 not 403, so they simply do not exist off-loopback. bind_host
    # unset in tests / standalone mount -> default loopback -> docs stay available.
    _DOCS_PATHS = frozenset(
        {"/openapi.json", "/docs", "/redoc", "/docs/oauth2-redirect"})

    @app.middleware("http")
    async def _docs_loopback_only(request, call_next):
        if request.url.path in _DOCS_PATHS:
            host = getattr(request.app.state, "bind_host", "127.0.0.1")
            if not _hs._is_loopback_host(host):
                return JSONResponse(status_code=404, content={"detail": "Not Found"})
        return await call_next(request)

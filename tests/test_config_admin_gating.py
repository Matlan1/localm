# SPDX-License-Identifier: AGPL-3.0-or-later
"""Owner-only gating for the two config keys that decide WHAT THE PROCESS LOADS.

``binary_dir`` names the directory the native llama runtime is loaded from: the
loader prepends it to the OS search path, CDLL()s every ``ggml*`` in it and loads
``llama.dll`` from it, so setting it is arbitrary native code execution in the
server process. ``embedding_model`` names a GGUF this process opens, and
``_sizing`` reads it on the CHAT model load path, so a poisoned value fires on
the owner's own next action.

Both are therefore ``admin_only``, alongside ``rag_index_paths`` /
``net_allow_private`` / ``cors_origins``. ``embedding_model`` has a SECOND writer,
``POST /api/rag/embedding``, mounted under the plugin's ``rag`` scope; ``rag`` is
not in ``scopes.PRIVILEGED_SCOPES`` and ``--scope chat --scope rag`` is the
documented restricted key, so the schema flag alone does not cover it and that
route carries its own gate.

The first two test groups assert MEMBERSHIP in ``admin_only_keys()`` AND drive the
real PATCH route end to end, because set membership alone cannot detect a gate
that stopped reading the set. The ``test_owner_can_still_set_*`` cases are the
other direction: they fail if the gating takes a documented owner capability away.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from localm.inference.http_server import create_app

OWNER_KEY = "owner-admin-key-ws7-abc123"


# --------------------------------------------------------------------------- #
#  (a) the mechanism itself                                                    #
# --------------------------------------------------------------------------- #

def test_load_selecting_keys_are_admin_only():
    """The gate at routes/config.py consults admin_only_keys(); a field that is
    not in that SET is not gated, however owner-ish its help text looks."""
    from localm.settings_schema import admin_only_keys
    keys = admin_only_keys()
    assert "binary_dir" in keys, \
        "binary_dir selects the native library this process loads"
    assert "embedding_model" in keys, \
        "embedding_model selects a file this process opens"


# --------------------------------------------------------------------------- #
#  (b) PATCH /v1/config is refused for a non-owner config:write key            #
# --------------------------------------------------------------------------- #

@pytest.fixture
def app_env(tmp_path, monkeypatch):
    """Protected-mode app: an owner (ADMIN) key via env plus a scoped
    config:read/config:write key in the keystore, under an isolated data dir."""
    import localm.config as cfg
    home = tmp_path / ".localm"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.setattr(cfg, "HOME_DIR", home)
    monkeypatch.setattr(cfg, "MODELS_DIR", home / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", home / "registry.json")
    monkeypatch.setenv("LOCALM_API_KEY", OWNER_KEY)      # owner key -> {ADMIN}
    from localm import auth
    scoped = auth.create_key(
        "dev", ["config:read", "config:write"], allow_privileged=True)["key"]
    app = create_app(None)
    with TestClient(app) as c:
        yield c, scoped, tmp_path


def _owner():
    return {"Authorization": f"Bearer {OWNER_KEY}"}


def _scoped(key):
    return {"Authorization": f"Bearer {key}"}


@pytest.mark.parametrize("key", ["binary_dir", "embedding_model"])
def test_scoped_key_cannot_set_load_selecting_key(app_env, key):
    """403 for the non-owner config:write key AND the stored value is unchanged
    - a refused write that still landed would be the worst of both."""
    from localm.config import load_config
    c, scoped, tmp_path = app_env
    evil = tmp_path / "planted"
    evil.mkdir()
    value = str(evil) if key == "binary_dir" else str(evil / "x.gguf")
    before = load_config().get(key)

    denied = c.patch("/v1/config", headers=_scoped(scoped), json={key: value})
    assert denied.status_code == 403, denied.text
    assert "owner" in denied.text.lower()
    assert load_config().get(key) == before, f"{key} must be UNCHANGED after a 403"

    # The control is not shipped to the scoped key either (no dead GUI control).
    scoped_schema = {f["key"] for f in
                     c.get("/v1/config/schema", headers=_scoped(scoped)).json()["fields"]}
    assert key not in scoped_schema
    # ... and its current value is stripped from the scoped key's GET.
    assert key not in c.get("/v1/config", headers=_scoped(scoped)).json()


@pytest.mark.parametrize("key", ["binary_dir", "embedding_model"])
def test_owner_can_still_set_load_selecting_key(app_env, key):
    """admin_only HIDES a field from a non-owner; it must not remove it. A custom
    llama.cpp build (binary_dir) and a hand-picked GGUF (embedding_model) are
    both supported OWNER setups."""
    from localm.config import load_config
    c, _scoped_key, tmp_path = app_env
    d = tmp_path / "custom-build"
    d.mkdir()
    value = str(d) if key == "binary_dir" else str(d / "mine.gguf")
    ok = c.patch("/v1/config", headers=_owner(), json={key: value})
    assert ok.status_code == 200, ok.text
    assert load_config().get(key) == value
    assert c.get("/v1/config", headers=_owner()).json().get(key) == value


def test_gate_is_specific_not_a_blanket_block(app_env):
    """The same scoped key still writes an ordinary config:write field: this is a
    targeted trust gate, not a general loss of the scope."""
    c, scoped, _ = app_env
    assert c.patch("/v1/config", headers=_scoped(scoped),
                   json={"net_allow": ["x.com"]}).status_code == 200


# --------------------------------------------------------------------------- #
#  (c) POST /api/rag/embedding is refused for a rag-scoped key                 #
# --------------------------------------------------------------------------- #

@pytest.fixture
def rag_app_env(tmp_path, monkeypatch):
    """The rag plugin mounted on a real auth-enforcing app, with a rag-only key.
    This is the documented restricted key shape (docs/cli.md offers
    ``--scope chat --scope rag``)."""
    import localm.config as cfg
    home = tmp_path / ".localm"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.setattr(cfg, "HOME_DIR", home)
    monkeypatch.setattr(cfg, "MODELS_DIR", home / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", home / "registry.json")
    monkeypatch.setenv("LOCALM_API_KEY", OWNER_KEY)
    from localm import auth
    rag_key = auth.create_key("ragbot", ["rag"], allow_privileged=True)["key"]
    app = create_app(None)
    from localm.plugins.engine import PluginManager
    PluginManager(app, external_root=tmp_path / "noplugins").install("rag")
    with TestClient(app) as c:
        yield c, rag_key, tmp_path


def test_rag_scoped_key_cannot_set_embedding_model(rag_app_env):
    """The rag scope reaches this route by mount, but the route writes an
    admin_only key, so it must demand an owner principal itself. Without this the
    plugin route is a back door around the PATCH /v1/config gate."""
    from localm.config import load_config
    c, rag_key, tmp_path = rag_app_env
    before = load_config().get("embedding_model")
    r = c.post("/api/rag/embedding",
               headers={"Authorization": f"Bearer {rag_key}"},
               json={"model": str(tmp_path / "x.gguf")})
    assert r.status_code == 403, r.text
    assert "owner" in r.text.lower()
    assert load_config().get("embedding_model") == before, \
        "embedding_model must be UNCHANGED after a 403"


def test_rag_get_embedding_is_not_a_file_existence_oracle(rag_app_env):
    """GET reports `installed` only for a localm-managed identity (a known key or
    a registered model). For a bare path it must neither stat the file nor echo
    the path back to a non-owner - `embedding_model` is admin_only, so GET
    /v1/config already withholds that same value."""
    from localm.config import update_config
    c, rag_key, tmp_path = rag_app_env
    secret = tmp_path / "owner-only-secret.gguf"
    secret.write_bytes(b"GGUF")
    update_config(lambda cfg: cfg.__setitem__("embedding_model", str(secret)))

    body = c.get("/api/rag/embedding",
                 headers={"Authorization": f"Bearer {rag_key}"}).json()
    assert body["installed"] is None, "a bare path must not yield an existence bit"
    assert body["status"] == "unknown"
    assert str(secret) not in str(body), "the owner's path must not be echoed back"

    # The OWNER still gets the real answer.
    owner_body = c.get("/api/rag/embedding", headers=_owner()).json()
    assert owner_body["installed"] is True
    assert owner_body["model"] == str(secret)

    # A KNOWN key is a localm-managed identity: answered truthfully even for the
    # non-owner, so the picker still works for a scoped client.
    update_config(lambda cfg: cfg.__setitem__("embedding_model", "bge-small-en-v1.5"))
    known = c.get("/api/rag/embedding",
                  headers={"Authorization": f"Bearer {rag_key}"}).json()
    assert known["installed"] is False        # not downloaded in this tmp home
    assert known["status"] == "not_installed"
    assert known["model"] == "bge-small-en-v1.5"


def test_owner_key_can_still_select_the_embedding_model(rag_app_env):
    """The owner is not blocked by the new gate: the route gets past the
    principal check and fails later (or succeeds), never with a 403."""
    c, _rag_key, _tmp = rag_app_env
    r = c.post("/api/rag/embedding", headers=_owner(),
               json={"model": "bge-small-en-v1.5"})
    assert r.status_code != 403, r.text


# --------------------------------------------------------------------------- #
#  (d) resolve_embedding_model_path refuses a non-local spec, no syscall        #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("spec,marker", [
    ("\\\\192.0.2.1\\share\\evil.gguf", "192.0.2.1"),   # UNC
    ("//192.0.2.1/share/evil.gguf", "192.0.2.1"),       # UNC, forward-slash form
    ("\\\\.\\PhysicalDrive0", "PhysicalDrive0"),        # Windows device path
    ("\\\\?\\C:\\evil.gguf", "evil.gguf"),              # extended-length device path
    ("http://evil.example/x.gguf", "evil.example"),     # URL
    ("file:///etc/passwd", "passwd"),
])
def test_resolve_embedding_model_path_rejects_nonlocal_spec(tmp_path, monkeypatch,
                                                            caplog, spec, marker):
    """A single is_file()/stat()/resolve() on a UNC path hands the string to the
    Windows SMB redirector - minutes of stall on an unroutable host, and an
    outbound net-NTLMv2 credential to a reachable one. So the refusal must land
    BEFORE any filesystem call, not after it.

    Asserted by SPYING on the three syscall entry points and checking the spec
    never reached any of them. The spy DELEGATES instead of raising: patching
    Path.stat to raise takes down pytest's own internal Path use."""
    import localm.config as cfg
    from localm.inference import embedder
    home = tmp_path / ".localm"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.setattr(cfg, "HOME_DIR", home)
    monkeypatch.setattr(cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", home / "registry.json")
    cfg.update_config(lambda c: c.__setitem__("embedding_model", spec))

    touched: list[str] = []
    originals = {name: getattr(Path, name) for name in ("is_file", "stat", "resolve")}

    def _spy(name):
        real = originals[name]

        def wrapper(self, *a, **k):
            touched.append(str(self))
            return real(self, *a, **k)
        return wrapper

    for name in originals:
        monkeypatch.setattr(Path, name, _spy(name))
    try:
        with caplog.at_level("WARNING"):
            result = embedder.resolve_embedding_model_path(allow_download=False)
    finally:
        monkeypatch.undo()          # restore before asserting, so a failure prints

    assert result is None
    offenders = [t for t in touched if marker in t]
    assert not offenders, f"the non-local spec reached the filesystem: {offenders}"
    # The reason an ignored setting was refused is logged, not swallowed.
    assert "embedding_model" in caplog.text


def test_resolve_embedding_model_path_still_accepts_a_local_gguf(tmp_path,
                                                                 monkeypatch):
    """The rejection is narrow: an ordinary local path is still resolved, so the
    documented owner behavior (point it at a GGUF anywhere) is intact."""
    import localm.config as cfg
    from localm.inference import embedder
    home = tmp_path / ".localm"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.setattr(cfg, "HOME_DIR", home)
    monkeypatch.setattr(cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", home / "registry.json")
    gguf = tmp_path / "mine.gguf"
    gguf.write_bytes(b"GGUF")
    cfg.update_config(lambda c: c.__setitem__("embedding_model", str(gguf)))
    assert embedder.resolve_embedding_model_path(allow_download=False) == str(gguf)


# --------------------------------------------------------------------------- #
#  the embedding-spec policy that composes pathsafe's shared predicate         #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("spec", [
    "\\\\server\\share\\x.gguf",   # \\server\share\x.gguf
    "//server/share/x.gguf",
    "\\\\.\\PhysicalDrive0",
    "\\\\?\\C:\\x",
    "//./pipe/x",
    "\\/server\\share",            # mixed separators (Windows resolves as UNC)
    "/\\server/share",
    "http://e/x", "https://e/x", "file:///etc/passwd", "smb://h/s",
])
def test_nonlocal_spec_reason_rejects(spec):
    from localm.inference.embedder import _nonlocal_spec_reason
    assert _nonlocal_spec_reason(spec), f"{spec!r} must be rejected"


@pytest.mark.parametrize("spec", [
    "C:\\models\\x.gguf",
    "C://models/x.gguf",      # a drive letter is ONE char, so never a scheme
    "/home/u/x.gguf", "x.gguf", "./rel/x.gguf", "C:x", "~/models/x.gguf",
    "bge-small-en-v1.5", "my-registered-model",
])
def test_nonlocal_spec_reason_allows_ordinary_paths(spec):
    """The over-rejection control. Pointing embedding_model at a GGUF anywhere
    local is DOCUMENTED owner behaviour, so a rule that ate ordinary paths would
    be a worse bug than the one being fixed."""
    from localm.inference.embedder import _nonlocal_spec_reason
    assert _nonlocal_spec_reason(spec) is None, f"{spec!r} must be allowed"


@pytest.mark.parametrize("spec", ["\\/host\\share\\x", "/\\host/share/x"])
def test_mixed_separator_unc_is_rejected(spec):
    """Windows accepts \\ and / interchangeably in a UNC prefix, so these two
    spellings ARE UNC to the OS, and a spec-rejection rule that only matched the
    doubled spellings would be bypassable by writing the path differently.

    The rejection now comes from pathsafe's shared predicate (it originally did
    not - that was a live bypass this lane reported, fixed upstream by a prefix
    table plus an ntpath.splitdrive backstop). The OS-level fact is asserted here
    INDEPENDENTLY of the implementation, so this test cannot drift with the code
    it guards, and it keeps failing loudly if a future pathsafe refactor narrows
    the predicate back to a startswith."""
    import ntpath
    from pathlib import PureWindowsPath
    from localm.inference.embedder import _nonlocal_spec_reason
    # The OS-level fact this rule covers.
    assert PureWindowsPath(spec).drive == "\\\\host\\share"
    assert ntpath.splitdrive(spec)[0] != ""
    assert _nonlocal_spec_reason(spec)


def test_policy_actually_calls_the_shared_pathsafe_predicate(monkeypatch):
    """Guard against the local policy silently drifting into a FORK of the
    shared primitive: it must delegate the plain UNC case to pathsafe rather
    than reimplementing it, so a future fix there reaches this call site."""
    import localm.pathsafe as ps
    from localm.inference import embedder
    calls = []
    real = ps.is_unc_or_device_path
    monkeypatch.setattr(ps, "is_unc_or_device_path",
                        lambda raw: (calls.append(raw), real(raw))[1])
    embedder._nonlocal_spec_reason("\\\\host\\share\\x")
    assert calls == ["\\\\host\\share\\x"], "must delegate to pathsafe, not fork it"


# --------------------------------------------------------------------------- #
#  the MCP stdio tool writes the same key                                      #
# --------------------------------------------------------------------------- #

def test_mcp_setup_embeddings_refuses_a_raw_path(tmp_path, monkeypatch):
    """stdio has no principal to gate on - the MCP client already runs as the
    owner - so the gate there is on the VALUE. The client is normally an LLM
    steerable by injected content, and this writes the admin_only
    `embedding_model` key, so a raw path is refused (loudly, per rule 5) while a
    known key still works."""
    import localm.config as cfg
    home = tmp_path / ".localm"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.setattr(cfg, "HOME_DIR", home)
    monkeypatch.setattr(cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", home / "registry.json")

    from localm.plugins.mcpserver.server import EngineCache, build_tools
    tools = build_tools(EngineCache(default_model=None,
                                    engine_factory=lambda *a, **k: None),
                        enable_images=False, enable_coder=False)
    tool = tools["setup_embeddings"]["handler"]

    before = cfg.load_config().get("embedding_model")
    out = tool({"model": str(tmp_path / "planted.gguf")})
    assert out.get("isError") is True, out
    assert "Refusing" in str(out)
    assert cfg.load_config().get("embedding_model") == before, \
        "a refused MCP call must not have written the key"
    # The restriction is stated in the tool DESCRIPTION too.
    assert "known" in tools["setup_embeddings"]["inputSchema"][
        "properties"]["model"]["description"].lower()


# --------------------------------------------------------------------------- #
#  (e) the owner-driven CLI install path still works end to end                #
# --------------------------------------------------------------------------- #

def test_setup_embeddings_cli_registers_a_known_model(tmp_path, monkeypatch):
    """`localm setup-embeddings` is the owner path and must be untouched: a known
    key downloaded into <home>/models/embeddings is still registered into the
    Model Manager. Also pins the guard alert 127's triage wrongly called a
    sanitizer: the `p.parent == _embeddings_dir()` check is what stops an
    EXTERNAL user-pointed GGUF being auto-registered, and it stays."""
    import localm.config as cfg
    home = tmp_path / ".localm"
    (home / "models" / "embeddings").mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.setattr(cfg, "HOME_DIR", home)
    monkeypatch.setattr(cfg, "MODELS_DIR", home / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", home / "registry.json")

    from localm.inference import embedder
    from localm.cli import maintenance

    landed = home / "models" / "embeddings" / "bge-small-en-v1.5-q8_0.gguf"
    landed.write_bytes(b"GGUF")
    monkeypatch.setattr(embedder, "_embeddings_dir",
                        lambda: home / "models" / "embeddings")
    monkeypatch.setattr(embedder, "resolve_embedding_model_path",
                        lambda **kw: str(landed))

    maintenance.setup_embeddings.callback(model="bge-small-en-v1.5")

    reg = cfg.load_registry()
    assert any(Path(v["path"] if isinstance(v, dict) else v[0]).resolve()
               == landed.resolve() for v in reg.values()), \
        f"the downloaded embedding model should be registered; registry={reg}"


def test_setup_embeddings_cli_does_not_register_an_external_gguf(tmp_path,
                                                                 monkeypatch):
    """The negative control for the guard above: a GGUF OUTSIDE the embeddings dir
    is used but never registered, so a user-pointed external model keeps whatever
    registration it already had."""
    import localm.config as cfg
    home = tmp_path / ".localm"
    (home / "models" / "embeddings").mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.setattr(cfg, "HOME_DIR", home)
    monkeypatch.setattr(cfg, "MODELS_DIR", home / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", home / "registry.json")

    from localm.inference import embedder
    from localm.cli import maintenance

    external = tmp_path / "elsewhere" / "mine.gguf"
    external.parent.mkdir(parents=True)
    external.write_bytes(b"GGUF")
    monkeypatch.setattr(embedder, "_embeddings_dir",
                        lambda: home / "models" / "embeddings")
    monkeypatch.setattr(embedder, "resolve_embedding_model_path",
                        lambda **kw: str(external))

    maintenance.setup_embeddings.callback(model=str(external))
    assert cfg.load_registry() == {}, "an external GGUF must not be auto-registered"


# --------------------------------------------------------------------------- #
#  the loader announces a non-bundled native runtime directory                 #
# --------------------------------------------------------------------------- #

def test_loader_warns_when_the_runtime_dir_is_not_the_bundled_wheel(tmp_path,
                                                                    caplog,
                                                                    monkeypatch):
    """Everything in that directory gets native code execution in this process,
    so an override must be VISIBLE in the log, never silent. It is not an error:
    binary_dir stays a supported owner setting."""
    from localm.inference.backends.llamacpp import _loader
    monkeypatch.setattr(_loader, "_warned_foreign_binary_dir", False)
    monkeypatch.setenv("LLAMA_CPP_LIB", str(tmp_path / "custom" / "llama.dll"))
    with caplog.at_level("WARNING"):
        _loader._warn_if_not_bundled(tmp_path / "custom")
    assert "native llama runtime" in caplog.text
    assert str(tmp_path / "custom") in caplog.text
    assert "LLAMA_CPP_LIB" in caplog.text

    # Warned once, not on every load.
    caplog.clear()
    with caplog.at_level("WARNING"):
        _loader._warn_if_not_bundled(tmp_path / "custom")
    assert caplog.text == ""


def test_loader_is_silent_for_the_bundled_wheel_dir(tmp_path, caplog, monkeypatch):
    """The negative control: the normal self-contained install must NOT warn, or
    the warning is noise everyone learns to ignore."""
    import sys
    import types
    from localm.inference.backends.llamacpp import _loader
    bundled = tmp_path / "wheel-lib"
    bundled.mkdir()
    fake = types.ModuleType("localm_llama_runtime")
    fake.lib_dir = lambda: str(bundled)          # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "localm_llama_runtime", fake)
    monkeypatch.setattr(_loader, "_warned_foreign_binary_dir", False)
    with caplog.at_level("WARNING"):
        _loader._warn_if_not_bundled(bundled)
    assert caplog.text == ""

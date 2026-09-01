# SPDX-License-Identifier: AGPL-3.0-or-later
"""
MCP server over stdio - exposes localm to any MCP client.

Protocol: JSON-RPC 2.0, newline-delimited JSON on stdin/stdout (the MCP
stdio transport - the mirror image of plugins/coder/mcp.py, which is the
client side).

CRITICAL INVARIANT: stdout carries ONLY protocol messages. Everything in
this process that would normally print (model loading banners, VRAM info,
rich progress) must go to stderr - see _redirect_consoles_to_stderr().

Tools exposed (always, unless noted):
    chat             - generate a response with a local model
    list_models      - registered model names with type and size (read-only)
    system_stats     - live CPU/RAM/VRAM/GPU load, for judging model/quant fit (read-only)
    search_models    - search HuggingFace for GGUF repos (read-only)
    list_model_files - a repo's GGUF files with quant/size/VRAM-fit (read-only)
    pull_model       - download + register + (optionally) load a GGUF
    setup_embeddings - install the on-device embedding model
    remove_model     - remove a model, deleting its file if under the models dir (destructive)
    run_doctor       - run localm doctor and return the report (read-only)
    list_plugins     - engine plugins and their active state (read-only)
    install_plugin / enable_plugin / disable_plugin - manage engine plugins
    uninstall_plugin - uninstall a plugin (and its data with delete_data) (destructive)
Conditional:
    embed            - embedding vectors (only when the backend can embed)
    run_coder_task   - delegate a coding task to the coder agent
                       (coder plugin active, unless --no-coder)
    generate_image   - local FLUX via ComfyUI (unless --no-images)
    memory_recall    - read the owner's durable chat memory, read-only
                       (memory plugin active, unless --no-memory)
    memory_append    - offer one fact for that memory (also needs --memory-write)

Both memory tools are refused in privacy mode, and memory_append writes an
UNVERIFIED (source "synth") record: a fact contradicting one the user typed
themselves becomes a pending correction for them to review, never an overwrite.

read-only tools carry readOnlyHint; the two destructive tools carry
destructiveHint, so an MCP client can confirm before a destructive call.
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from localm import pathsafe
from localm.pathsafe import is_unc_or_device_path

PROTOCOL_VERSION = "2025-03-26"
SERVER_NAME = "localm"
SERVER_VERSION = "0.1.5"


def _log(msg: str) -> None:
    """Server-side logging - stderr only, stdout belongs to the protocol."""
    print(f"[localm-mcp] {msg}", file=sys.stderr, flush=True)


def _child_identity_env() -> dict:
    """Env for a ``-m localm`` helper process so it is THE SAME localm as this
    server: same data home, same code.

    Both are otherwise re-resolved from ambient state at every process
    boundary: the data home falls back to a contained default derived from the
    running code's location when nothing is configured, and ``-m`` puts the
    child's cwd first on ``sys.path``, where a ``localm/`` directory in that
    cwd (any other checkout) silently swaps which CODE runs. run_coder_task
    runs its child in the TASK's directory, so without this a server whose own
    home came from ITS location hands the coder chain a different, empty home.

    LOCALM_HOME pins the data home; PYTHONSAFEPATH stops ``-m`` from putting
    the child's cwd on ``sys.path``; the PYTHONPATH entry keeps this server's
    own package importable regardless of cwd (PYTHONSAFEPATH only drops the
    implicit cwd entry, explicit PYTHONPATH entries still apply)."""
    import localm as _pkg
    from localm.config import home_dir
    env = dict(os.environ)
    env["LOCALM_HOME"] = str(home_dir())
    env["PYTHONSAFEPATH"] = "1"
    pkg_root = str(Path(_pkg.__file__).resolve().parent.parent)
    prior = env.get("PYTHONPATH")
    env["PYTHONPATH"] = pkg_root + ((os.pathsep + prior) if prior else "")
    return env


def _redirect_consoles_to_stderr() -> None:
    """
    Re-point every rich Console used during model loading at stderr.
    A single stray print to stdout corrupts the JSON-RPC stream.
    """
    from rich.console import Console
    err = Console(stderr=True)
    import localm.inference.engine as _engine_mod
    import localm.inference.backends.gguf as _gguf_mod
    import localm.model_manager as _mm_mod
    import localm.inference.backends.llamacpp._sizing as _sizing_mod
    _engine_mod.console = err
    _gguf_mod.console = err
    _mm_mod.console = err
    # _sizing's own module-level console prints the "ctx auto" sizing note
    # during GgufBackend's preflight, BEFORE the model process is even spawned -
    # i.e. still in THIS process.
    _sizing_mod.console = err
    try:
        import localm.inference.backends.llamacpp.llama as _llama_mod
        if hasattr(_llama_mod, "console"):
            _llama_mod.console = err
    except Exception:
        pass


# ---------------------------------------------------------------------------
#  Tool implementations
# ---------------------------------------------------------------------------

class EngineCache:
    """
    Lazy, per-model engine cache. Multi-resident, on the shared policy.

    Models stay loaded ALONGSIDE each other whenever free VRAM provably allows
    it. Both servers ask the same module (``inference.residency``) the same two
    questions - may this load with zero eviction, and if not who is the safe
    victim.

    Stacking needs a fresh, measurable reading that clears the requirement plus
    headroom with no split shortfall. On a box that cannot measure VRAM, on an
    inconclusive probe, or for a model whose footprint cannot be read, this
    falls back to single-resident behaviour (evict, wait for the free to land,
    then load). A wrong PERMIT here is a native OOM or a driver hang, not a tidy
    error.
    """

    def __init__(self, default_model: Optional[str] = None,
                 engine_factory: Optional[Callable] = None) -> None:
        self.default_model = default_model
        # Display name -> engine, plus usage order (least-recently-used FIRST,
        # MRU last).
        self._engines: Dict[str, Any] = {}
        self._lru: list = []
        # Injection point for tests - real factory builds a localm Engine
        self._factory = engine_factory or self._build_engine

    # ---- back-compat views over the multi-resident state -------------------
    # _engine and _loaded_name read the most-recently-used resident.

    @property
    def _engine(self):
        return self._engines.get(self._lru[-1]) if self._lru else None

    @property
    def _loaded_name(self) -> Optional[str]:
        return self._lru[-1] if self._lru else None

    @property
    def resident(self) -> list:
        """Resident display names, least-recently-used first."""
        return list(self._lru)

    def _operator_supplied(self, model_name) -> bool:
        """True only for the model the OPERATOR named when starting this server.

        ``default_model`` comes from the ``--model`` flag / the LOCALM_MODEL
        environment variable, so a filesystem path is legitimate there. Every
        OTHER name arrives in a tool call from the MCP client and is treated as
        hostile input: it must be a registered one."""
        return bool(model_name) and model_name == self.default_model

    def _build_engine(self, model_name: str):
        from localm.inference.engine import Engine
        from localm.model_manager import get_model_info, unregistered_model_error
        # Gated here as well as in resolve_model: _build_engine is also
        # reachable directly, so the gate must not depend on having come
        # through resolve_model.
        trusted = self._operator_supplied(model_name)
        if not trusted:
            bad = unregistered_model_error(model_name)
            if bad:
                raise ValueError(bad)
        info = get_model_info(model_name, allow_direct_path=trusted)
        if info is None:
            raise ValueError(f"Model not found: {model_name!r}. "
                             f"Run 'localm list' to see registered models.")
        path, _hint = info
        return Engine(str(path), display_name=model_name)

    def resolve_model(self, requested: Optional[str]) -> str:
        # A client-supplied name must be a registered one; the operator's own
        # --model default is exempt (see _operator_supplied).
        if requested and not self._operator_supplied(requested):
            from localm.model_manager import unregistered_model_error
            bad = unregistered_model_error(requested)
            if bad:
                raise ValueError(bad)
        name = requested or self.default_model
        if name:
            return name
        from localm.config import load_registry
        from localm.model_manager import is_auto_chat_eligible
        reg = load_registry()
        if not reg:
            raise ValueError("No models registered. Run 'localm pull <name>' first.")
        # Auto-pick the first chat-eligible model; a type='unknown' model is never
        # auto-loaded (it stays usable when named explicitly via --model / a request).
        name = next((n for n in sorted(reg) if is_auto_chat_eligible(reg[n])), None)
        if name is None:
            raise ValueError(
                "No chat model registered (all registered models are type 'unknown'). "
                "Name one explicitly, or set a model's type with 'localm set-type'.")
        return name

    def get(self, requested: Optional[str]):
        name = self.resolve_model(requested)
        engine = self._engines.get(name)
        if engine is not None:
            if (getattr(engine, "loaded", True)
                    and getattr(engine, "unloading", False) is not True):
                self._touch(name)      # already resident: never evict to reuse
                return engine
            # Resident but NOT loaded, so it holds no VRAM yet and the
            # free-VRAM probe cannot see it. Run the gate, then hand back the
            # SAME object so the pulled engine is reused rather than silently
            # replaced.
            self._make_room_for(name)
            self._touch(name)
            return engine
        self._make_room_for(name)
        _log(f"loading model {name}")
        engine = self._factory(name)
        self._engines[name] = engine
        self._touch(name)
        return engine

    def _touch(self, name: str) -> None:
        """Mark ``name`` most-recently-used."""
        if name in self._lru:
            self._lru.remove(name)
        self._lru.append(name)

    def _model_required_bytes(self, name: str) -> Optional[int]:
        """
        VRAM ``name`` is expected to occupy once loaded, or None when that
        cannot be determined (unregistered model, unreadable path).

        None is not "zero": it means the fit cannot be PROVEN, which sends the
        caller down the single-resident path. Never let an unknown read as room.
        """
        from localm.inference.residency import (
            model_footprint_bytes, required_vram_bytes)
        try:
            from localm.model_manager import get_model_info
            # Operator-supplied default may be a path; a client name may not,
            # and returns None here.
            info = get_model_info(
                name, allow_direct_path=self._operator_supplied(name))
            if info is None:
                return None
            path, _hint = info
            return required_vram_bytes(model_footprint_bytes(path))
        except Exception as e:
            # Falls back to single-resident; logged so the degradation is
            # traceable rather than invisible.
            from localm.debuglog import logger
            logger.debug("mcp: could not size %s, assuming it needs the card "
                         "to itself: %s", name, e)
            return None

    def _fits_alongside(self, name: str, required: Optional[int]) -> bool:
        """True when ``name`` may load with NO eviction, next to the residents."""
        if required is None:
            return False
        from localm import discover
        from localm.discover import gpu_split_shortfall, vram_capacity
        from localm.inference.residency import (
            DEFAULT_HEADROOM_BYTES, fits_alongside_residents)
        try:
            # The same deadline the HTTP server's gate uses: the probe must be
            # able to wait out a cold ROCm/CUDA init, since a timed-out probe
            # reads as "unmeasurable" and drops to single-resident. No executor
            # hop, unlike http_server: this process is synchronous, so there is
            # no event loop for the probe to stall.
            v_info, probe_status = vram_capacity(
                return_status=True, deadline=discover._GPU_PROBE_CLI_DEADLINE,
                wait_for_inflight=True)
            probe_ok = probe_status == discover.GPU_PROBE_OK
            shortfall = []
            if probe_ok and self._is_gguf(name):
                # Aggregate free can clear the bar while one device of a
                # configured split is short - see gpu_split_shortfall.
                shortfall = gpu_split_shortfall(required + DEFAULT_HEADROOM_BYTES)
            # PROCESS-scoped readings are blind to every OTHER resident model
            # (each lives in its own isolated worker subprocess), so they can
            # only over-report free space and are never trusted for the PERMIT
            # decision.
            return fits_alongside_residents(
                free_vram=v_info.get("free"), vram_required=required,
                probe_ok=probe_ok, shortfall=shortfall,
                is_process_scoped=(
                    v_info.get("free_scope") == discover.FREE_SCOPE_PROCESS))
        except Exception as e:
            # A failed probe must not be read as headroom.
            _log(f"warning: VRAM probe failed ({e}) - loading {name} "
                 f"single-resident")
            return False

    def _is_gguf(self, name: str) -> bool:
        try:
            from localm.inference.engine import _is_gguf
            from localm.model_manager import get_model_info
            # Tells the operator's own --model path from a client-supplied name.
            info = get_model_info(
                name, allow_direct_path=self._operator_supplied(name))
            return bool(info) and _is_gguf(info[0])
        except Exception:
            # Fail CLOSED: this only decides whether to run the per-device split
            # check, a REFUSE-direction guard, so "unknown" must mean RUN it,
            # not skip it. gpu_split_shortfall returns [] when no split
            # resolves.
            return True

    def _make_room_for(self, name: str) -> None:
        """
        Evict resident peers until ``name`` fits, or until none can be freed.

        Returns as soon as the model may load alongside what is already there,
        which on a measurable box with headroom is immediately and with zero
        eviction.
        """
        from localm.config import load_config
        from localm.inference import residency
        cfg = load_config()
        cap = residency.resident_cap(cfg)
        pinned = residency.pinned_model_names(cfg)
        required = self._model_required_bytes(name)
        while self._lru:
            over_cap = residency.exceeds_resident_cap(self._lru, name, cap)
            # Only probe when the cap is satisfied: being over cap already means
            # room is needed regardless of what VRAM says. vram_ok stays None to
            # record that this pass did NOT measure, which the message below
            # relies on so it never reports a shortfall nobody observed.
            vram_ok = None
            if not over_cap:
                vram_ok = self._fits_alongside(name, required)
                if vram_ok:
                    return
            victim = residency.pick_eviction_victim(
                self._lru, self._engines, requested=name, pinned=pinned)
            if victim is None:
                # Nothing evictable (all pinned, or all busy). Load anyway and
                # SAY the policy was missed, rather than pretending it held.
                reasons = []
                if over_cap:
                    reasons.append("the resident cap")
                if vram_ok is False:
                    # _fits_alongside short-circuits to False WITHOUT probing
                    # when the model cannot be sized, so naming the free-VRAM
                    # check there would report a measurement never taken.
                    reasons.append("the free-VRAM check" if required is not None
                                   else "an unsizeable model")
                _log(f"warning: {' and '.join(reasons)} wanted room for {name} "
                     f"but no resident model could be evicted "
                     f"(resident={self._lru}, pinned={sorted(pinned)}) - "
                     f"loading it anyway")
                return
            self._evict(victim, loading=name)

    def _evict(self, victim: str, *, loading: str) -> None:
        """Unload ``victim`` and wait for its VRAM to actually come back."""
        engine = self._engines.pop(victim, None)
        if victim in self._lru:
            self._lru.remove(victim)
        if engine is None:
            return
        _log(f"evicting {victim} to make room for {loading}")
        from localm.vram import _live_free_vram_bytes, _vram_free_reading
        # SEED the wait with the reading even when the probe was not fresh, and
        # poll with the live-only reader, NOT the other way round: for the
        # 'before' SEED, None means "do not wait at all" (wait_for_vram_release
        # short-circuits on before_bytes=None), while for the 'after' POLL None
        # means "cannot verify". Freshness is carried separately, for the
        # REPORT, not the wait. scope is folded into the verdict below, not just
        # the report: a process-scoped reading (blind to the model in its
        # isolated worker) CANNOT observe the free rising after unload.
        before_free, before_fresh, before_scope = _vram_free_reading()
        try:
            engine.unload()
        except Exception as e:
            # Unload is best-effort (we still load the new model), but a
            # cleanup failure must be visible, not silently swallowed.
            _log(f"warning: failed to unload {victim}: {e}")
        # The native unload's VRAM free is asynchronous - loading the next model
        # before it lands can exceed total VRAM and hang the GPU driver.
        # before_free is None only when VRAM is not measurable AT ALL (a
        # CPU-only box), in which case there is nothing to wait for and this is
        # a no-op.
        from localm.discover import FREE_SCOPE_DEVICE
        from localm.vram import wait_for_vram_release
        released, _final = wait_for_vram_release(
            _live_free_vram_bytes, before_bytes=before_free)
        backable = before_fresh and before_scope == FREE_SCOPE_DEVICE
        if released is False and backable:
            # Fresh AND device-global on both ends: "did not rise" is a claim
            # that can be backed. A process-scoped reading cannot see the
            # model's VRAM in its isolated worker and falls to the branch below.
            _log(f"warning: VRAM free did not rise after unloading "
                 f"{victim} within the timeout - loading {loading} anyway")
        elif before_free is not None and (released is None or not backable):
            # Either end came off a timed-out/busy probe, OR the reading is
            # process-scoped (blind to the worker's VRAM), so whether the free
            # landed is unknown. Say that rather than the "did not rise" claim
            # above. The wait still ran; only the verdict is withheld.
            _log(f"warning: could not confirm the VRAM free after unloading "
                 f"{victim} (no live GPU reading) - loading "
                 f"{loading} anyway")

    def unload_all(self) -> None:
        """Free every resident engine (shutdown). N resident means N to free."""
        for name in list(self._lru):
            engine = self._engines.pop(name, None)
            self._lru.remove(name)
            if engine is None:
                continue
            try:
                engine.unload()
            except Exception as e:
                # Process teardown, so nothing downstream can act on this, but a
                # native free that failed leaves VRAM pinned after exit. stderr
                # only; stdout belongs to the protocol.
                _log(f"warning: failed to unload {name} at shutdown: {e}")


def _text_result(text: str, is_error: bool = False) -> dict:
    return {"content": [{"type": "text", "text": text}], "isError": is_error}


@contextlib.contextmanager
def _quiet_stdout():
    """Redirect stdout to stderr for the duration of the block, so a downstream
    call's stray prints never corrupt the JSON-RPC frame stream on stdout."""
    with contextlib.redirect_stdout(sys.stderr):
        yield


def _run_mgr_action(mgr, fn, *, plugin: str):
    """Call fn(mgr) inside the stdout-quieting guard, mapping KeyError to a
    "no such plugin" result and ValueError to its own message. Returns the
    mapped error _text_result, or None on success."""
    with _quiet_stdout():
        try:
            fn(mgr)
        except KeyError:
            return _text_result(f"No such plugin: {plugin}", is_error=True)
        except ValueError as e:
            return _text_result(str(e), is_error=True)
    return None


def _backend_can_embed(engines: "EngineCache") -> bool:
    """True unless the active/default backend explicitly cannot embed.

    Avoids loading the model at startup by checking the registry for GGUF suffix
    if the engine object is not yet instantiated/cached."""
    if getattr(engines, "_factory", None) != getattr(engines, "_build_engine", None):
        try:
            # A custom factory can be a real engine builder, so guard the same
            # as chat()/embed()/pull_model() below.
            with _quiet_stdout():
                backend = getattr(engines.get(None), "_backend", None)
            return getattr(backend, "can_embed", True) is not False
        except Exception as e:
            # Probe failed: assume embeddable rather than hide the embed tool on
            # a transient error, and log so a real capability bug is traceable.
            # The logger writes to the debug file/stderr, never stdout, so the
            # JSON-RPC frame stream stays clean.
            from localm.debuglog import logger
            logger.debug("mcp: embed-capability probe (custom factory) failed, "
                         "assuming embeddable: %s", e)
            return True

    if engines._engine is not None:
        backend = getattr(engines._engine, "_backend", None)
        return getattr(backend, "can_embed", True) is not False

    try:
        name = engines.resolve_model(None)
        from localm.model_manager import get_model_info
        # resolve_model(None) yields the operator's own default, so a path is
        # legitimate here; anything else is already registry-gated upstream.
        info = get_model_info(
            name, allow_direct_path=engines._operator_supplied(name))
        if info is not None:
            path, _hint = info
            if str(path).lower().endswith(".gguf"):
                return False
    except Exception as e:
        # Registry probe failed: assume embeddable rather than hide the tool,
        # and log the cause. The debug logger stays off stdout.
        from localm.debuglog import logger
        logger.debug("mcp: embed-capability probe (registry) failed, assuming "
                     "embeddable: %s", e)
    return True


def _memory_available() -> bool:
    """True when the memory plugin is installed on disk AND enabled, the same
    check the memory CLI and routes sit behind.

    Fails CLOSED (hides both memory tools) so an unreadable plugin config cannot
    expose a personal-data surface by accident; the cause is logged so the
    disappearance is diagnosable rather than silent. The debug logger writes to
    file/stderr, never the JSON-RPC stdout."""
    try:
        from localm.plugins.engine import PluginManager
        return PluginManager(None).is_active("memory")
    except Exception as e:
        from localm.debuglog import logger
        logger.debug("mcp: memory-availability probe failed, hiding the memory "
                     "tools: %s", e)
        return False


def _memory_writes_allowed() -> bool:
    """The privacy gate for chat memory, resolved fresh on every call.

    The surface is "chat", not "server": these tools read and write the SAME
    namespace the chat routes use (agent "chat"), so gating them on a different
    surface would answer to a different mode knob than the store they touch."""
    from localm.memory.gating import writes_allowed
    return writes_allowed("chat")


def _memory_store():
    """The owner's chat-memory store: the same namespace the /api/memory routes
    and `localm memory` open. The four values are load-bearing and deliberately
    not parameterised - a different agent, scope_key or root would read an empty
    store and report that localm has learned nothing about the user."""
    from localm import memory as _mem
    from localm.config import home_dir
    return _mem.open_store(None, "chat", "", root=home_dir() / "memory")


def _memory_embed_fn():
    """The embedding callable for semantic recall, or None - recall then falls
    back to lexical BM25."""
    try:
        from localm.inference.embedder import get_embedder
        emb = get_embedder()
        return emb.embed if emb is not None else None
    except Exception as e:
        from localm.debuglog import logger
        logger.debug("mcp: embedder resolution failed, memory recall stays "
                     "lexical: %s", e)
        return None


# Default facts per memory_recall call; the store caps any request at K_CAP.
_MEMORY_RECALL_DEFAULT_K = 6

_MEMORY_OFF_MSG = (
    "The memory plugin is not active, so localm has no chat memory to read or "
    "write. Install and enable it with: localm plugin install memory")

_MEMORY_PRIVACY_MSG = (
    "Refused: localm is in privacy mode, where chat memory is fully off - no "
    "recall and no writes. This surface has no privacy-mode opt-in, unlike the "
    "in-process chat recall knob. Set mode/chat_mode to 'log' or 'full' to use it.")

_MEMORY_WRITE_OFF_MSG = (
    "Refused: memory_append is not enabled for this MCP server. Writing to your "
    "personal memory from an external client is opt-in and separate from enabling "
    "the memory plugin. Relaunch the server with: localm mcp --memory-write")


def _coder_available() -> bool:
    """True when the coder plugin is installed on disk AND enabled, the same
    check `localm coder` itself does before accepting a task."""
    try:
        from localm.plugins.engine import PluginManager
        return PluginManager(None).is_active("coder")
    except Exception as e:
        # Fails CLOSED (hide the coder tool), so an installed+enabled coder
        # VANISHES from the tool list if this probe raises (e.g. unreadable
        # plugin config). The cause is logged so that is diagnosable, not a
        # silent disappearance. The debug logger writes to file/stderr, never
        # the JSON-RPC stdout.
        from localm.debuglog import logger
        logger.debug("mcp: coder-availability probe failed, hiding coder tool: %s", e)
        return False


def build_tools(engines: EngineCache, enable_images: bool = True,
                 enable_coder: bool = True, enable_memory: bool = True,
                 enable_memory_write: bool = False) -> Dict[str, dict]:
    """Return {tool_name: {schema, handler}} for everything this server offers."""

    def chat(args: dict) -> dict:
        prompt = args.get("prompt", "")
        if not prompt:
            return _text_result("'prompt' is required", is_error=True)
        # engines.get() can trigger a fresh model load, and a GGUF load prints
        # native sizing/context diagnostics (e.g. the "ctx auto" note) straight
        # to stdout - the same stream the JSON-RPC frames travel on.
        with _quiet_stdout():
            engine = engines.get(args.get("model"))
        messages = []
        if args.get("system"):
            messages.append({"role": "system", "content": args["system"]})
        messages.append({"role": "user", "content": prompt})
        gen: dict = {}
        for key in ("max_tokens", "temperature", "seed"):
            if args.get(key) is not None:
                gen[key] = args[key]
        text = "".join(engine.chat_stream(messages, **gen))
        return _text_result(text)

    def memory_recall(args: dict) -> dict:
        """Read the owner's durable chat memory. Never writes."""
        # Re-checked here, not only in build_tools: a stale client tool list, a
        # call by name, or a later refactor that drops the build-time gate must
        # still not reach the store.
        if not _memory_available():
            return _text_result(_MEMORY_OFF_MSG, is_error=True)
        if not _memory_writes_allowed():
            return _text_result(_MEMORY_PRIVACY_MSG, is_error=True)
        query = (args.get("query") or "").strip()
        if not query:
            return _text_result("'query' is required", is_error=True)
        from localm.memory import render_memories
        from localm.memory.store import K_CAP, MAX_TEXT_LEN
        try:
            k = int(args.get("limit") or _MEMORY_RECALL_DEFAULT_K)
        except (TypeError, ValueError):
            k = _MEMORY_RECALL_DEFAULT_K
        k = max(1, min(k, K_CAP))
        store = _memory_store()
        # reinforce=False: a read from an external client must not bump
        # last_used/uses. That is a write, and it would let a client's query
        # pattern reshape which of the owner's facts survive prune().
        with _quiet_stdout():
            records = store.recall(query, k=k, embed_fn=_memory_embed_fn(),
                                   reinforce=False)
        if not records:
            return _text_result(
                f"No remembered facts matched {query!r}. localm has "
                f"{len(store.all())} fact(s) stored.")
        # render_memories neutralises every line and fences the block as
        # data-not-instructions, so an instruction-shaped memory cannot become an
        # instruction in the calling client's prompt. Its default max_chars is the
        # CHAT prompt-injection budget, far below what an explicit request for k
        # facts needs, so the budget is sized to the request here and the block is
        # still bounded (K_CAP * MAX_TEXT_LEN plus fence/bullet overhead).
        block = render_memories(records, max_chars=k * (MAX_TEXT_LEN + 4) + 128)
        rendered = sum(1 for ln in block.splitlines() if ln.startswith("- "))
        if rendered < len(records):
            # Never let the block cap drop facts silently: a short answer would
            # otherwise read as "that is all localm remembers".
            block += (f"\n\n[{len(records) - rendered} more matching fact(s) were "
                      f"not shown: the block size limit was reached.]")
        return _text_result(block)

    def memory_append(args: dict) -> dict:
        """Offer a fact for the owner's durable chat memory.

        Never writes a TRUSTED record and never overwrites one: a fact that
        contradicts an existing user-typed memory becomes a pending correction
        the owner accepts or rejects."""
        if not _memory_available():
            return _text_result(_MEMORY_OFF_MSG, is_error=True)
        # Unreachable while this tool is registered only when the flag is set:
        # the flag comes from argv and cannot change during the process. It
        # guards a later refactor that registers the tool unconditionally and
        # lets the handler decide. It is NOT the live gate - the plugin probe
        # and the privacy check below are, and each re-resolves per call.
        if not enable_memory_write:
            return _text_result(_MEMORY_WRITE_OFF_MSG, is_error=True)
        if not _memory_writes_allowed():
            return _text_result(_MEMORY_PRIVACY_MSG, is_error=True)
        text = (args.get("text") or "").strip()
        if not text:
            return _text_result("'text' is required", is_error=True)

        from localm.memory import N_MAX, MemoryRecord, PendingCorrection
        from localm.memory.consolidate import (MATCH_THRESHOLD, SYNTH_IMP_CAP,
                                               _nearest)
        from localm.memory.store import TRUSTED_SOURCES
        store = _memory_store()
        existing = store.all()

        # A supersession of a TRUSTED (user/import) fact is PROPOSED, never
        # applied: an MCP client is not the user typing, so it may not rewrite
        # what the user typed. Lexical ratio only - no model call, and when the
        # answer is unclear the conservative direction (propose) is already
        # correct because the owner sees it either way.
        idx, ratio = _nearest(text, existing)
        if idx >= 0 and ratio >= MATCH_THRESHOLD and \
                existing[idx].source in TRUSTED_SOURCES:
            target = existing[idx]
            with _quiet_stdout():
                added = store.propose_corrections([PendingCorrection(
                    target_id=target.id, action="update", proposed_text=text,
                    target_text=target.text, confidence=ratio)])
            if not added:
                return _text_result(
                    "Already proposed: this correction is pending review, or was "
                    "previously rejected. Nothing was changed.")
            return _text_result(
                "Queued for review, NOT saved: this contradicts a fact you "
                f"entered yourself ({target.text!r}). localm never overwrites "
                "your own memories from an external client. Accept or reject it "
                "in Settings > Memory, or with `localm memory corrections`.")

        if len(existing) >= N_MAX:
            return _text_result(
                f"Memory is at its {N_MAX}-record cap; delete a fact before "
                "adding another.", is_error=True)
        # source="synth", never "user": a trusted record is exempt from recall
        # decay and from prune() eviction, so a client that could mint one could
        # pin a fact the owner never asserted.
        with _quiet_stdout():
            rec = store.add(
                MemoryRecord(text=text, kind="semantic", source="synth",
                             importance=SYNTH_IMP_CAP, meta={"via": "mcp"}),
                embed_fn=_memory_embed_fn())
        return _text_result(f"Remembered (unverified, id {rec.id}): {rec.text}")

    def server_activity(args: dict) -> dict:
        """What any running localm server on this machine is doing.

        This MCP server is a SEPARATE PROCESS from the HTTP/GUI server and
        shares no memory with it, so it finds the running instances on disk and
        asks each one over HTTP.

        The states are kept apart. "No server is running" is not "nothing is
        running" - there is nothing to ask. "Could not reach it" is not "it is
        idle". Only a server that actually answered can report an empty list,
        and only that case says nothing is running.
        """
        from localm import instances
        from localm.config import home_dir
        from localm.selfclient import read_activity

        # include_token=True: this call ASKS each discovered instance over HTTP
        # (an internal, non-display use), so it needs the attach token a
        # genuinely open (keyless) instance's middleware requires. Never do this
        # for anything a human reads (e.g. `localm ps`, which keeps the
        # default-stripped snapshot()).
        rows = instances.snapshot(home_dir(), include_token=True)
        if not rows:
            return _text_result(
                "No localm server is running on this machine, so there is "
                "nothing to ask. This is not the same as a server reporting "
                "that it is idle.")
        lines = []
        for e in rows:
            from localm.bindhost import self_connect_host, url_host
            _h = url_host(self_connect_host(e.get("host")))
            where = f"{e.get('scheme', 'http')}://{_h}:{e.get('port')}"
            if not e.get("alive"):
                lines.append(f"{where}: registered but not responding; "
                             f"its activity is unknown.")
                continue
            state, payload = read_activity(
                e.get("scheme", "http"), e.get("port"), e.get("token"),
                e.get("host"))
            if state == "unreachable":
                lines.append(f"{where}: could not be reached ({payload}); "
                             f"its activity is unknown.")
            elif state == "unauthorized":
                # Matches the "could not be X" register the other failure
                # branches use, not a "needs a key" requirement statement: this
                # process cannot tell what the server is doing right now.
                lines.append(f"{where}: could not be asked (it requires an "
                             f"API key this process does not have); its "
                             f"activity is unknown.")
            elif state == "unsupported":
                lines.append(f"{where}: does not report activity (older "
                             f"localm); its activity is unknown.")
            elif state != "ok":
                lines.append(f"{where}: could not be read (HTTP {payload}); "
                             f"its activity is unknown.")
                continue
            else:
                ops = (payload or {}).get("operations") or []
                now = (payload or {}).get("now")
                if not ops:
                    lines.append(f"{where}: idle, nothing running.")
                    continue
                lines.append(f"{where}: {len(ops)} operation(s)")
                for op in ops:
                    label = op.get("label") or op.get("kind") or "operation"
                    bits = [op.get("status") or "?"]
                    pct = op.get("pct")
                    # Absent, not zero: an operation that has reported no
                    # progress is at an unknown percentage.
                    if isinstance(pct, (int, float)):
                        bits.append(f"{pct:.0f}%")
                    created = op.get("created_at")
                    # Age against the SERVER's clock; this process may not
                    # share it.
                    if isinstance(now, (int, float)) and isinstance(created, (int, float)):
                        bits.append(f"{int(max(0, now - created))}s elapsed")
                    lines.append(f"  - {label} [{', '.join(bits)}]")
        return _text_result("\n".join(lines))

    def list_models(args: dict) -> dict:
        from localm.config import load_registry
        from localm.model_manager import _entry_path
        reg = load_registry()
        if not reg:
            return _text_result("No models registered.")
        lines = []
        for name, info in sorted(reg.items()):
            epath = _entry_path(info)
            if epath is None:
                # Match the CLI: a single malformed entry is shown corrupt, never
                # allowed to crash / blank the whole listing (removable via the
                # remove_model tool). Guards a hand-edited/half-written registry.
                lines.append(f"{name}  [corrupt]  (malformed registry entry)")
                continue
            # These stats run INLINE: this dispatcher has no event loop to
            # protect, since MCPStdioServer.run_stdio is a synchronous
            # `for line in stdin` loop and handle() is a plain def, so a thread
            # hop would only move the block. What keeps a pathological row (a
            # UNC path that blocks in the SMB redirector) out of this loop is
            # the REGISTRATION gate, not a probe here.
            p = Path(epath)
            if p.is_dir():
                size = "dir (HF format)"
            elif p.is_file():
                b = p.stat().st_size
                size = f"{b/1024**3:.2f} GB" if b >= 1024**3 else f"{b/1024**2:.0f} MB"
            else:
                size = "missing"
            lines.append(f"{name}  [{size}]  {info.get('source', 'local')}")
        return _text_result("\n".join(lines))

    def system_stats(args: dict) -> dict:
        from localm.sysstats import system_stats as _stats
        # A ONE-SHOT call, unlike the GUI's repeating poll: without
        # wait_first_vram the "Live ... VRAM" promise below omits VRAM on a cold
        # first call while the background probe is still running, because there
        # is no later poll here to pick up the landed reading. MCP stdio serves
        # one request at a time with no event loop to stall.
        return _text_result(json.dumps(_stats(wait_first_vram=True)))

    def search_models(args: dict) -> dict:
        from localm.discover import DiscoverError, hf_search
        try:
            results = hf_search(args.get("query", ""), limit=args.get("limit", 20))
        except DiscoverError as e:
            return _text_result(str(e), is_error=True)
        return _text_result(json.dumps(results))

    def list_model_files(args: dict) -> dict:
        repo = args.get("repo", "")
        if not repo:
            return _text_result("'repo' is required (e.g. 'bartowski/Qwen2.5-7B-Instruct-GGUF')",
                                 is_error=True)
        from localm.discover import DiscoverError, fit_label, hf_gguf_files, vram_capacity
        try:
            files = hf_gguf_files(repo)
        except DiscoverError as e:
            return _text_result(str(e), is_error=True)
        total_vram = vram_capacity().get("total")
        for f in files:
            f["fit"] = fit_label(f["size_bytes"], total_vram)
        return _text_result(json.dumps(files))

    def pull_model(args: dict) -> dict:
        repo = args.get("repo", "")
        name = args.get("name", "")
        if not repo:
            return _text_result("'repo' is required", is_error=True)
        if not name:
            return _text_result(
                "'name' is required - pick a short registry name for this model",
                is_error=True)
        # `repo` is an MCP-CLIENT-supplied string, not a path the local user
        # picked, so is_unc_or_device_path's "remote value" contract applies:
        # refuse UNC/device syntax unconditionally, BEFORE the
        # Path(repo).exists() sink below ever runs. On Windows that sink dials
        # SMB and auto-authenticates for a UNC target, and can stall for minutes
        # inline in this handler.
        #
        # NOT gated on os.name, unlike reject_unsafe_path_string's `//`-form
        # check: no real HuggingFace repo id contains a backslash or starts with
        # `//`, on any platform.
        #
        # The message does not echo `repo` back, unlike the local-add message
        # below: this string never reached a safe-to-display check.
        if is_unc_or_device_path(repo):
            return _text_result(
                "'repo' looks like a filesystem path (UNC or device syntax), not "
                "a HuggingFace repo id. pull_model downloads a model by repo id "
                "(e.g. 'owner/name').", is_error=True)
        # A LOCAL PATH IS NOT A PULL. pull_model treats an existing path as a
        # local add, registering an arbitrary directory under a client-chosen
        # name; that name is then a registered model, so it passes the
        # membership check and resolves via the REGISTRY branch of
        # get_model_info rather than the direct-path gate. A refused add still
        # probes the path (config.json read, rglob, sha256). An MCP client pulls
        # from HuggingFace; registering something already on this disk is
        # `localm add`.
        try:
            if Path(repo).expanduser().exists():
                return _text_result(
                    f"{repo!r} is a path on this machine, not a HuggingFace repo. "
                    "pull_model downloads a model by repo id (e.g. "
                    "'owner/name'). To register a model already on this disk, run "
                    "'localm add <path>' on the host.", is_error=True)
        except OSError:
            pass          # unreadable/oversized path: not a local add, fall through
        spec = f"{repo}:{args['file']}" if args.get("file") else repo

        from localm.model_manager.pull import pull_model as _pull

        # pull_model()'s progress bars/messages print via a rich Console (a
        # module-level singleton in model_manager/_shared.py, imported by value
        # into pull.py at load time). redirect_stdout catches it regardless of
        # which Console instance is in play.
        with _quiet_stdout():
            try:
                ok = _pull(spec, name=name)
            except Exception as e:
                return _text_result(f"pull failed: {e}", is_error=True)
        if not ok:
            return _text_result(f"pull failed for {spec!r} - see server stderr for detail",
                                 is_error=True)

        if not args.get("load", True):
            return _text_result(f"pulled and registered as {name!r} (not loaded)")

        try:
            # This load, like chat()/embed()'s, can print native sizing
            # diagnostics straight to stdout.
            #
            # engines.get() only constructs/registers the Engine and runs the
            # VRAM-eviction gate - it does NOT call Engine.load(), so the
            # backend stays unloaded until some later caller (normally
            # chat_stream()'s lazy-load path) touches it. The tool's own
            # description promises "load it - blocks until ready", so pull_model
            # calls .load() itself rather than leaving a resident-but-unloaded
            # engine parked in the cache.
            with _quiet_stdout():
                engine = engines.get(name)
                engine.load()
        except Exception as e:
            return _text_result(
                f"pulled and registered as {name!r}, but loading it failed: {e}",
                is_error=True)
        msg = f"pulled, registered, and loaded {name!r} - ready to use"
        # gpu_placement is None whenever the backend cannot report per-layer
        # placement for this engine - never fabricate a degraded warning without
        # evidence a load actually happened. When it IS known and partial/zero,
        # say so: a model too big to fully fit VRAM still loads, because the
        # backend's own sizing defers to a partial/zero GPU offload rather than
        # refusing.
        placement = getattr(engine, "gpu_placement", None)
        if placement and placement.get("degraded"):
            msg += (f" ({placement['gpu_layers_offloaded']}/"
                    f"{placement['gpu_layers_total']} layers on GPU, "
                    f"the rest on CPU - slower)")
        return _text_result(msg)

    def embed(args: dict) -> dict:
        texts = args.get("texts")
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return _text_result("'texts' is required (string or list)", is_error=True)
        # A fresh embedder load can print to stdout too, like chat() above.
        with _quiet_stdout():
            engine = engines.get(args.get("model"))
        try:
            vecs = engine.embed(texts)
        except NotImplementedError as e:
            return _text_result(str(e), is_error=True)
        return _text_result(json.dumps(vecs))

    def generate_image(args: dict) -> dict:
        prompt = args.get("prompt", "")
        if not prompt:
            return _text_result("'prompt' is required", is_error=True)
        from localm.audit import SessionMode, effective_mode
        from localm.config import home_dir
        from localm.image_gen.comfy import generate_image as gen_img
        from localm.media import paths as _media_paths

        home = home_dir().resolve()

        def _confine(raw: str, label: str):
            """Keep an MCP OUTPUT path inside the localm data dir - this tool is
            driven by an LLM client, so an arbitrary output_path could overwrite
            anything on disk.

            WRITE targets only. ``input_image`` does NOT use this: confining a
            READ to the data dir is far too wide, because the data dir is the
            credential store (auth.key is the plaintext owner key, plus
            auth.json, sessions.json, rag/, coder/).

            Delegates to ``pathsafe.confined_absolute_or_under``, which carries
            the UNC/device guard and additionally closes an NTFS Alternate Data
            Stream / short-name-alias gap. Every rejection reason is folded into
            the SAME message, never echoing the client-supplied string back."""
            expanded = str(Path(raw).expanduser())
            try:
                return pathsafe.confined_absolute_or_under(home, expanded)
            except ValueError:
                raise ValueError(
                    f"{label} must stay within the localm data dir ({home})")

        try:
            out_arg = args.get("output_path")
            out = (_confine(out_arg, "output_path") if out_arg
                   else home / "mcp-images" / f"mcp-{int(time.time())}.png")
            # input_image is a READ that is then UPLOADED to ComfyUI, over an
            # api_url sanitize_comfy_url permits to be a LAN or public host on
            # plaintext http, so it is read-AND-TRANSMIT and the data dir is the
            # wrong boundary for it. Same policy the image/video HTTP routes use
            # (uploads inbox + the generated-media galleries), via the non-HTTP
            # entry point so a refusal becomes an error REPLY here rather than an
            # HTTPException escaping the stdio handler. InputImageRefused is a
            # ValueError, so the existing except catches it.
            input_p = (_media_paths.check_input_image(args["input_image"])
                       if args.get("input_image") else None)
        except ValueError as e:
            return _text_result(str(e), is_error=True)

        is_privacy = effective_mode("mcp") == SessionMode.PRIVACY
        # comfy.generate_image builds its own rich Console / Progress on stdout;
        # the JSON-RPC frame stream lives on stdout too, so route any stray
        # output to stderr or it corrupts the protocol.
        with _quiet_stdout():
            ok, message = gen_img(
                prompt, out,
                guidance=args.get("guidance"),
                negative_prompt=args.get("negative_prompt"),
                seed=args.get("seed"),
                input_image=input_p,
                denoise=args.get("denoise"),
                write_sidecar=not is_privacy,
                delete_outputs=is_privacy,
            )
        return _text_result(message, is_error=not ok)

    def run_coder_task(args: dict) -> dict:
        task = args.get("task", "")
        if not task:
            return _text_result("'task' is required", is_error=True)
        cwd = args.get("cwd", "")
        if not cwd:
            return _text_result("'cwd' is required (the project directory to work in)",
                                 is_error=True)
        # `cwd` is MCP-client-supplied, same as pull_model's `repo` above:
        # refuse UNC/device syntax unconditionally, BEFORE is_dir() below ever
        # runs. is_dir() dials SMB for a UNC target exactly like exists() does.
        if is_unc_or_device_path(cwd):
            return _text_result(
                "'cwd' must be a local directory path, not a UNC or device path.",
                is_error=True)
        cwd_path = Path(cwd).expanduser()
        if not cwd_path.is_dir():
            return _text_result(f"cwd is not a directory: {cwd_path}", is_error=True)

        # Shells out to the `localm coder` single-shot CLI, which reuses the
        # CLI's own project-config resolution and instance attach/spawn logic,
        # and keeps this MCP server's own EngineCache (used by chat/embed) from
        # fighting the coder's separate server process over the same model load.
        cmd = [sys.executable, "-m", "localm", "coder", task,
               "--cwd", str(cwd_path), "--output-format", "json"]
        if args.get("model"):
            # A MODEL NAME FROM A CLIENT IS NOT OPERATOR INPUT, even though it
            # is about to become argv. The coder CLI spawns `localm gui <model>`
            # when no instance is attached for this cwd, and that positional
            # reaches the startup resolver, which opts into allow_direct_path.
            # This string came from an MCP tool call, and the client also
            # chooses `cwd`, so it can select the spawn branch at will.
            from localm.model_manager import unregistered_model_error
            bad = unregistered_model_error(args["model"])
            if bad:
                return _text_result(bad, is_error=True)
            cmd += ["--model", args["model"]]
        if args.get("max_turns") is not None:
            cmd += ["--max-turns", str(args["max_turns"])]
        # Default OFF, matching the CLI's own fail-closed default: without this,
        # file writes still happen but run_shell is denied for lack of a TTY to
        # confirm it.
        if args.get("yes"):
            cmd.append("--yes")
        timeout = args.get("timeout_seconds") or 900

        try:
            # cwd=cwd_path matters beyond the coder's own file/shell tool scope:
            # if no server is already running for this project, the coder CLI
            # auto-spawns one and identifies "this project" by the SPAWNING
            # process's OS working directory, not just the --cwd flag above. Omit
            # this and the auto-spawned server registers under the MCP server's
            # own directory instead, so the coder's own attach-back lookup can
            # never find it (looks like a timeout; it is a project-root mismatch).
            # env=_child_identity_env(): that cwd change must NOT drag the child
            # onto a different data home or different localm code.
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                                  cwd=str(cwd_path), env=_child_identity_env())
        except subprocess.TimeoutExpired:
            return _text_result(f"coder task timed out after {timeout}s", is_error=True)

        # --output-format json pretty-prints with indent=2 (multi-line), and
        # console messages print to stdout BOTH BEFORE it ("attached to running
        # server", the auto-start banner) AND AFTER it (`--mode full`'s "Session
        # transcript saved -> <path>") - so the JSON is neither the whole stdout
        # nor anchored to either end. Find each line that is a lone "{" (the
        # JSON dict is always non-empty, so indent=2 always opens it on its own
        # line), newest first, and raw_decode from there: unlike json.loads,
        # raw_decode stops at the object's closing brace and tolerates whatever
        # trailing console text follows it.
        stdout = proc.stdout.strip()
        payload = None
        if stdout:
            lines = stdout.splitlines()
            decoder = json.JSONDecoder()
            for i in reversed([n for n, ln in enumerate(lines) if ln == "{"]):
                try:
                    payload, _ = decoder.raw_decode("\n".join(lines[i:]))
                    break
                except json.JSONDecodeError:
                    continue

        if payload is None:
            detail = proc.stderr.strip() or stdout or f"exit code {proc.returncode}"
            return _text_result(f"coder task failed to run: {detail}", is_error=True)

        text = payload.get("response", "")
        meta = (f"\n\n[turns={payload.get('turns')} "
                f"tokens={payload.get('total_tokens')} "
                f"success={payload.get('success')}]")
        return _text_result(text + meta, is_error=not payload.get("success", False))

    def setup_embeddings(args: dict) -> dict:
        model = args.get("model")
        from localm.config import load_registry, update_config
        from localm.inference.embedder import (KNOWN_EMBEDDING_MODELS,
                                          resolve_embedding_model_path)
        # `model` is a free-form string chosen by the MCP CLIENT, and this
        # writes the admin_only `embedding_model` key. stdio gives no principal
        # to gate on, so the gate here is on the VALUE: a known key or a
        # registered model name only, never a raw path. Pointing the setting at
        # an arbitrary GGUF stays available to the owner through
        # `localm setup-embeddings` and the GUI. Refuse loudly rather than
        # silently ignoring the argument, so a caller is never told a selection
        # took effect when it did not.
        if model:
            if model not in KNOWN_EMBEDDING_MODELS and model not in load_registry():
                return _text_result(
                    f"Refusing to set the embedding model to {model!r}: over MCP it "
                    f"must be a known key {tuple(KNOWN_EMBEDDING_MODELS)} or an "
                    "already-registered model name, not a filesystem path. Use "
                    "'localm setup-embeddings <path>' or the GUI to point it at a "
                    "GGUF of your own.",
                    is_error=True)
            update_config(lambda c: c.update({"embedding_model": model}))

        with _quiet_stdout():
            try:
                path = resolve_embedding_model_path(allow_download=True)
            except Exception as e:
                return _text_result(f"Failed to setup embeddings: {e}", is_error=True)
        if not path:
            return _text_result(
                "Could not install the embedding model. It must be a known "
                f"key {tuple(KNOWN_EMBEDDING_MODELS)}, a registered model, or a GGUF "
                "path, and network must be enabled (net_mode is not 'off').",
                is_error=True
            )
        return _text_result(f"Embedding model ready: {path}. Memory and RAG will now use semantic search.")

    def _local_hold(model: str, reg: dict):
        """Whether an engine resident in THIS process is holding *model*'s file.

        The MCP server keeps its own residents (chat, embed and the coder tool
        all load through ``engines``), so this process can be the very thing
        holding the file open while it deletes it.
        """
        from localm.model_manager.registry import engine_holding_model_file
        candidates = [
            (name, getattr(engine, "model_path", None))
            for name, engine in list(getattr(engines, "_engines", {}).items())
            if getattr(engine, "loaded", False)
        ]
        return engine_holding_model_file(model, reg, candidates)

    def remove_model(args: dict) -> dict:
        model = args.get("model", "")
        if not model:
            return _text_result("'model' is required", is_error=True)
        from localm.model_manager import remove_model as _rm
        from localm.config import load_registry
        reg = load_registry()
        if model not in reg:
            return _text_result(f"Model not found: {model}", is_error=True)

        # Removing a registered model DELETES ITS FILE when that file lives in
        # the models dir, and nothing downstream of here asks whether anything
        # is still using it: model_manager.remove_model is the same code path
        # `localm rm` runs, with no server and no engine map in front of it.
        # Both holders are checked here - the engines resident in this process,
        # and any running server - and either one refuses.
        hold = _local_hold(model, reg)
        if hold is not None:
            if hold.reason is None:
                why = f"this MCP server still has its file loaded as {hold.key!r}"
            else:
                why = (f"this MCP server has {hold.key!r} loaded and "
                       f"{hold.reason}, so it cannot be ruled out as holding "
                       f"this file")
            return _text_result(
                f"Refusing to remove {model!r}: {why}. Removing it would "
                f"delete the model file while it is in use. Unload it first "
                f"(or restart this MCP server), then try again.",
                is_error=True)
        from localm.selfclient import remote_hold_reason
        remote = remote_hold_reason(model)
        if remote is not None:
            return _text_result(
                f"Refusing to remove {model!r}: {remote}. Removing it could "
                f"delete the model file while it is in use. Unload it there "
                f"(or stop that server), then try again.",
                is_error=True)

        with _quiet_stdout():
            try:
                _rm(model)
            except Exception as e:
                return _text_result(f"Failed to remove model: {e}", is_error=True)
        return _text_result(f"Model '{model}' successfully removed.")

    def run_doctor(args: dict) -> dict:
        cmd = [sys.executable, "-m", "localm", "doctor"]
        try:
            # env=_child_identity_env(): a doctor that re-resolves home/code
            # from ambient state reports on the WRONG install whenever this
            # server's home came from its own location (source-checkout setup).
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60,
                                  env=_child_identity_env())
            output = proc.stdout
            if proc.stderr:
                output += "\n\nStderr:\n" + proc.stderr
            return _text_result(output)
        except subprocess.TimeoutExpired:
            return _text_result("Doctor task timed out after 60s", is_error=True)
        except Exception as e:
            return _text_result(f"Failed to run doctor: {e}", is_error=True)

    def list_plugins(args: dict) -> dict:
        from localm.plugins.engine import PluginManager
        mgr = PluginManager(None)
        state = mgr.api_state()
        plugins = state.get("plugins", [])
        if not plugins:
            return _text_result("No engine plugins discovered.")
        lines = []
        for p in plugins:
            status = "enabled" if p.get("active") else ("disabled" if p.get("installed") else "available")
            desc = f" - {p['description']}" if p.get("description") else ""
            lines.append(f"{p['name']}  [{status}]{desc}")
        return _text_result("\n".join(lines))

    def install_plugin(args: dict) -> dict:
        plugin = args.get("plugin", "")
        if not plugin:
            return _text_result("'plugin' is required", is_error=True)
        from localm.plugins.engine import PluginManager
        mgr = PluginManager(None)
        err = _run_mgr_action(
            mgr, lambda m: m.set_installed_state(plugin, True), plugin=plugin)
        if err is not None:
            return err
        dep_result = None
        with _quiet_stdout():
            with_deps = args.get("with_deps", True)
            if with_deps and mgr.plugin_missing_deps(plugin):
                dep_result = mgr.install_plugin_deps(plugin)
        if dep_result is not None and not dep_result.ok:
            # Left ENABLED rather than rolled back, matching the CLI's own
            # `plugin install` behaviour, which also leaves a plugin installed
            # on a dep failure and points the operator at retrying the extras
            # later. The failure is SURFACED rather than swallowed: folded into
            # the reply below, plus a warning here since this reply is the only
            # place it is seen.
            from localm.debuglog import logger
            logger.warning("install_plugin(%s): pip extras failed to install: %s",
                            plugin, dep_result.error)
            failed = ", ".join(dep_result.failed) or "see error"
            return _text_result(
                f"Plugin '{plugin}' installed and enabled, but its dependencies "
                f"failed to install ({failed}): {dep_result.error}", is_error=True)
        return _text_result(f"Plugin '{plugin}' successfully installed and enabled.")

    def enable_plugin(args: dict) -> dict:
        plugin = args.get("plugin", "")
        if not plugin:
            return _text_result("'plugin' is required", is_error=True)
        from localm.plugins.engine import PluginManager
        mgr = PluginManager(None)
        err = _run_mgr_action(
            mgr, lambda m: m.set_enabled_state(plugin, True), plugin=plugin)
        if err is not None:
            return err
        return _text_result(f"Plugin '{plugin}' successfully enabled.")

    def disable_plugin(args: dict) -> dict:
        plugin = args.get("plugin", "")
        if not plugin:
            return _text_result("'plugin' is required", is_error=True)
        from localm.plugins.engine import PluginManager
        mgr = PluginManager(None)
        err = _run_mgr_action(
            mgr, lambda m: m.set_enabled_state(plugin, False), plugin=plugin)
        if err is not None:
            return err
        return _text_result(f"Plugin '{plugin}' successfully disabled.")

    def uninstall_plugin(args: dict) -> dict:
        plugin = args.get("plugin", "")
        if not plugin:
            return _text_result("'plugin' is required", is_error=True)
        delete_data = args.get("delete_data", False)
        from localm.plugins.engine import PluginManager
        mgr = PluginManager(None)
        # Bypasses _run_mgr_action (unlike install/enable/disable above):
        # uninstall()'s bool is the only signal that the installed directory
        # actually came off disk (a locked file, an AV hold, a permission
        # denial), so it is read here.
        was_installed = mgr.is_installed(plugin)
        with _quiet_stdout():
            try:
                removed = mgr.uninstall(plugin, delete_data=delete_data)
            except KeyError:
                return _text_result(f"No such plugin: {plugin}", is_error=True)
            except ValueError as e:
                return _text_result(str(e), is_error=True)
        if was_installed and not removed:
            return _text_result(
                f"Plugin '{plugin}' was disabled and unloaded, but its installed "
                "files could not be fully removed (a locked file, an AV hold, or "
                "a permission denial); it is not fully uninstalled.", is_error=True)
        return _text_result(f"Plugin '{plugin}' successfully uninstalled.")

    _model_param = {"type": "string",
                    "description": "Registered model name (default: server's configured model)"}

    tools: Dict[str, dict] = {
        "chat": {
            "description": "Generate a response with a local LLM (fully offline).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "prompt":      {"type": "string", "description": "User prompt"},
                    "system":      {"type": "string", "description": "Optional system prompt"},
                    "model":       _model_param,
                    "max_tokens":  {"type": "integer", "description": "Max tokens to generate"},
                    "temperature": {"type": "number", "description": "Sampling temperature"},
                    "seed":        {"type": "integer", "description": "Seed for reproducible output"},
                },
                "required": ["prompt"],
            },
            "handler": chat,
        },
        "server_activity": {
            "description": (
                "What any running localm server on this machine is currently "
                "doing: model downloads, indexing, media generation. Check this "
                "BEFORE starting a long operation - a pull started from the "
                "browser or another client is otherwise invisible here, and "
                "starting a second one wastes bandwidth and disk. Distinguishes "
                "'no server is running' and 'could not reach it' from 'the "
                "server says it is idle'; only the last one means nothing is "
                "happening."
            ),
            "inputSchema": {"type": "object", "properties": {}},
            "annotations": {"readOnlyHint": True, "title": "Server activity"},
            "handler": server_activity,
        },
        "list_models": {
            "description": "List locally registered models with size and source.",
            "inputSchema": {"type": "object", "properties": {}},
            "annotations": {"readOnlyHint": True, "title": "List models"},
            "handler": list_models,
        },
        "system_stats": {
            "description": (
                "Live CPU/RAM/VRAM/GPU load. Use this BEFORE picking a model or "
                "quant for a task: if VRAM is tight, prefer a smaller quant "
                "(Q4/Q6 over Q8) rather than skipping the task or degrading "
                "quality - and prefer evicting the current model over settling "
                "for a worse-fit one when the task genuinely needs it."
            ),
            "inputSchema": {"type": "object", "properties": {}},
            "annotations": {"readOnlyHint": True, "title": "System stats"},
            "handler": system_stats,
        },
        "search_models": {
            "description": "Search HuggingFace for GGUF model repos (empty query = most downloaded).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search text (optional)"},
                    "limit": {"type": "integer", "description": "Max results (default 20, max 50)"},
                },
            },
            "annotations": {"readOnlyHint": True, "title": "Search models"},
            "handler": search_models,
        },
        "list_model_files": {
            "description": (
                "List a HuggingFace repo's GGUF files (quant, size, and a fit "
                "badge - 'fits'/'tight'/'too-big' - against this machine's VRAM) "
                "so you can pick the right quant before pulling."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "repo": {"type": "string",
                             "description": "HuggingFace repo id, e.g. 'bartowski/Qwen2.5-7B-Instruct-GGUF'"},
                },
                "required": ["repo"],
            },
            "annotations": {"readOnlyHint": True, "title": "List repo GGUF files"},
            "handler": list_model_files,
        },
        "pull_model": {
            "description": (
                "Download a GGUF file from HuggingFace, register it under 'name', "
                "and (by default) load it - blocks until ready. Use search_models "
                "+ list_model_files first to pick repo/file/quant."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "repo": {"type": "string", "description": "HuggingFace repo id"},
                    "file": {"type": "string",
                             "description": "Specific GGUF filename from list_model_files (omit for a full snapshot pull)"},
                    "name": {"type": "string", "description": "Registry name to give this model"},
                    "load": {"type": "boolean",
                             "description": "Load it into the engine after pulling (default true)"},
                },
                "required": ["repo", "name"],
            },
            "handler": pull_model,
        },
    }

    # Only advertise embed when the active backend can actually produce vectors.
    # The handler still degrades gracefully if invoked anyway.
    if _backend_can_embed(engines):
        tools["embed"] = {
            "description": "Compute embedding vectors for one or more texts with a local model.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "texts": {"type": "array", "description": "Texts to embed",
                              "items": {"type": "string"}},
                    "model": _model_param,
                },
                "required": ["texts"],
            },
            "handler": embed,
        }

    if enable_coder and _coder_available():
        tools["run_coder_task"] = {
            "description": (
                "Delegate a whole coding task (read/edit files, run shell commands, "
                "git, tests) to localm's own offline agent, running entirely on a "
                "local model. Blocks until the task finishes or times out, then "
                "returns the agent's final result - use this to hand off a "
                "self-contained sub-task instead of doing it turn-by-turn yourself."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "The task to perform"},
                    "cwd":  {"type": "string",
                             "description": "Project directory the agent should work in"},
                    "model": _model_param,
                    "max_turns": {"type": "integer",
                                  "description": "Safety cap on agent iterations (default: 40)"},
                    "yes": {"type": "boolean",
                            "description": ("Auto-approve shell commands too, not just file "
                                             "writes (default false: shell calls are denied "
                                             "without this since there is no TTY to confirm "
                                             "them)")},
                    "timeout_seconds": {"type": "integer",
                                        "description": "Give up after this long (default 900)"},
                },
                "required": ["task", "cwd"],
            },
            "handler": run_coder_task,
        }

    if enable_images:
        tools["generate_image"] = {
            "description": ("Generate an image with the local FLUX model via ComfyUI. "
                            "Returns the saved file path and seed."),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "prompt":          {"type": "string", "description": "Image description"},
                    "output_path":     {"type": "string", "description": "Where to save (default: <data dir>/mcp-images/)"},
                    "negative_prompt": {"type": "string", "description": "Things to avoid"},
                    "seed":            {"type": "integer", "description": "Reproducibility seed"},
                    "guidance":        {"type": "number", "description": "Guidance scale (default 3.5)"},
                    "input_image":     {"type": "string", "description": "Existing image for img2img"},
                    "denoise":         {"type": "number", "description": "img2img change amount 0-1"},
                },
                "required": ["prompt"],
            },
            "handler": generate_image,
        }

    # Chat memory. Hidden unless the memory plugin is active, so a client never
    # sees a personal-data tool on an instance that has no memory to serve.
    # Each handler re-checks this at call time; hiding is UX, the handler check
    # is the gate.
    if enable_memory and _memory_available():
        tools["memory_recall"] = {
            "description": (
                "Search what localm durably remembers about this user (their "
                "stated preferences, projects, tools, recurring goals) and return "
                "the matching facts. Read-only. Use it to carry context between "
                "agents instead of asking the user to repeat themselves."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string",
                              "description": "What to recall facts about"},
                    "limit": {"type": "integer",
                              "description": "Max facts to return "
                                             "(default 6, capped at 32)"},
                },
                "required": ["query"],
            },
            "annotations": {"readOnlyHint": True, "title": "Recall memory"},
            "handler": memory_recall,
        }
        # WRITE is a separate, explicit opt-in: enabling the memory plugin is a
        # decision about localm remembering things, not about a foreign process
        # writing into that memory.
        if enable_memory_write:
            tools["memory_append"] = {
                "description": (
                    "Offer one durable fact about this user for localm's memory. "
                    "Stored as UNVERIFIED (machine-asserted), never as a "
                    "user-typed fact. A fact that contradicts something the user "
                    "entered themselves is queued for their review instead of "
                    "being saved, and never overwrites it."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string",
                                 "description": "One durable fact about the user"},
                    },
                    "required": ["text"],
                },
                "annotations": {"title": "Remember a fact"},
                "handler": memory_append,
            }

    tools["setup_embeddings"] = {
        "description": "Install the on-device embedding model for semantic search (memory + RAG).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "model": {"type": "string", "description": "Optional embedding model to set. Must be a known key (bge-small-en-v1.5, nomic-embed-text-v1.5) or an already-registered model name - a filesystem path is refused here; use 'localm setup-embeddings <path>' or the GUI for that."}
            }
        },
        "handler": setup_embeddings,
    }
    tools["remove_model"] = {
        "description": "Remove a model from the registry (and delete the file if it's in <data dir>/models/).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "model": {"type": "string", "description": "Registered model name to remove"}
            },
            "required": ["model"],
        },
        # Deletes the model file on disk - declare it so an MCP client can prompt
        # for confirmation before calling (confirmation belongs at the client).
        "annotations": {"destructiveHint": True, "title": "Remove model"},
        "handler": remove_model,
    }
    tools["run_doctor"] = {
        "description": "Check system requirements and report any issues (runs localm doctor).",
        "inputSchema": {"type": "object", "properties": {}},
        "annotations": {"readOnlyHint": True, "title": "Run doctor"},
        "handler": run_doctor,
    }
    tools["list_plugins"] = {
        "description": "List engine plugins, their descriptions, and activation status.",
        "inputSchema": {"type": "object", "properties": {}},
        "annotations": {"readOnlyHint": True, "title": "List plugins"},
        "handler": list_plugins,
    }
    tools["install_plugin"] = {
        "description": "Install and enable an engine plugin.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "plugin": {"type": "string", "description": "Plugin name to install"},
                "with_deps": {"type": "boolean", "description": "Also install pip dependencies (default true)"}
            },
            "required": ["plugin"],
        },
        "handler": install_plugin,
    }
    tools["enable_plugin"] = {
        "description": "Enable an installed engine plugin.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "plugin": {"type": "string", "description": "Plugin name to enable"}
            },
            "required": ["plugin"],
        },
        "handler": enable_plugin,
    }
    tools["disable_plugin"] = {
        "description": "Disable an installed engine plugin.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "plugin": {"type": "string", "description": "Plugin name to disable"}
            },
            "required": ["plugin"],
        },
        "handler": disable_plugin,
    }
    tools["uninstall_plugin"] = {
        "description": "Uninstall (deselect) an engine plugin.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "plugin": {"type": "string", "description": "Plugin name to uninstall"},
                "delete_data": {"type": "boolean", "description": "Also delete stored data (default false)"}
            },
            "required": ["plugin"],
        },
        # Removes the plugin (and, with delete_data, its stored data on disk) -
        # declare it so an MCP client can confirm before calling.
        "annotations": {"destructiveHint": True, "title": "Uninstall plugin"},
        "handler": uninstall_plugin,
    }

    return tools


# ---------------------------------------------------------------------------
#  JSON-RPC dispatch
# ---------------------------------------------------------------------------

class MCPStdioServer:
    """Dispatches MCP JSON-RPC messages to tool handlers."""

    def __init__(self, tools: Dict[str, dict]) -> None:
        self.tools = tools

    def handle(self, msg: dict) -> Optional[dict]:
        """Process one message. Returns the response dict, or None for
        notifications (which get no reply)."""
        if not isinstance(msg, dict):
            # A JSON-RPC batch array, a bare scalar, or null all parse fine but
            # are not a request object - reply Invalid Request instead of
            # crashing on msg.get(...).
            return self._error(None, -32600, "Invalid Request: expected a JSON object")
        method = msg.get("method", "")
        mid = msg.get("id")

        if mid is None:
            return None   # notification (e.g. notifications/initialized)

        if method == "initialize":
            return self._result(mid, {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            })

        if method == "ping":
            return self._result(mid, {})

        if method == "tools/list":
            listed = []
            for name, spec in self.tools.items():
                entry = {"name": name,
                         "description": spec["description"],
                         "inputSchema": spec["inputSchema"]}
                # MCP tool annotations (destructiveHint / readOnlyHint / title):
                # emitted only when a tool declares them, so clients can decide
                # when to confirm a destructive call.
                if spec.get("annotations"):
                    entry["annotations"] = spec["annotations"]
                listed.append(entry)
            return self._result(mid, {"tools": listed})

        if method == "tools/call":
            params = msg.get("params", {}) or {}
            name = params.get("name", "")
            spec = self.tools.get(name)
            if spec is None:
                return self._error(mid, -32602, f"Unknown tool: {name}")
            try:
                result = spec["handler"](params.get("arguments", {}) or {})
            except Exception as e:
                _log(f"tool {name} crashed: {e}")
                result = _text_result(f"Tool failed: {e}", is_error=True)
            return self._result(mid, result)

        return self._error(mid, -32601, f"Method not found: {method}")

    @staticmethod
    def _result(mid: Any, result: dict) -> dict:
        return {"jsonrpc": "2.0", "id": mid, "result": result}

    @staticmethod
    def _error(mid: Any, code: int, message: str) -> dict:
        return {"jsonrpc": "2.0", "id": mid,
                "error": {"code": code, "message": message}}

    # ------------------------------------------------------------------ #

    def run_stdio(self, stdin=None, stdout=None) -> None:
        """Blocking loop: read newline-delimited JSON until EOF."""
        stdin = stdin or sys.stdin
        stdout = stdout or sys.stdout
        _log("ready - waiting for MCP client")
        for line in stdin:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                _log("skipping non-JSON input line")
                continue
            # A JSON-RPC payload may be a single request object or a batch
            # array; a bare scalar / null is invalid. handle() replies -32600
            # for any non-dict element rather than crashing the loop.
            if isinstance(msg, list):
                batch = msg or [None]      # empty batch -> one Invalid Request
            else:
                batch = [msg]
            for one in batch:
                response = self.handle(one)
                if response is not None:
                    stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
                    stdout.flush()
        _log("stdin closed - shutting down")


def serve_stdio(model: Optional[str] = None, enable_images: bool = True,
                 enable_coder: bool = True, enable_memory: bool = True,
                 enable_memory_write: bool = False) -> None:
    """Entry point used by the CLI: build everything and block on stdio."""
    _redirect_consoles_to_stderr()
    engines = EngineCache(default_model=model)
    server = MCPStdioServer(build_tools(
        engines, enable_images=enable_images, enable_coder=enable_coder,
        enable_memory=enable_memory,
        enable_memory_write=enable_memory_write))
    try:
        server.run_stdio()
    finally:
        # Every resident engine, not just the most recent one: freeing one of N
        # would leave the rest holding VRAM past exit.
        engines.unload_all()

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

The handlers and their schemas live in the ``tools`` subpackage, one module
per tool family; build_tools() composes the families and applies the gates
above.
"""

from __future__ import annotations

import contextlib
import json
import sys
import threading
import time
from typing import Any, Callable, Dict, Optional

PROTOCOL_VERSION = "2025-03-26"
SERVER_NAME = "localm"
SERVER_VERSION = "0.2.0"

# Wall-clock bound on how long a load waits for a resident that is still
# serving a request to free itself before the load is refused as busy. The
# wait runs on the protocol thread, so the whole server is unresponsive for
# its duration.
BUSY_WAIT_SECONDS = 30.0
BUSY_POLL_SECONDS = 1.0

# Longest run_coder_task budget a client may ask for; the budget covers the
# model load and the agent's construction as well as the run.
MAX_CODER_TIMEOUT_SECONDS = 3600.0


class ModelBusyError(RuntimeError):
    """A load was refused because every evictable resident is still serving a
    request past BUSY_WAIT_SECONDS."""


def _log(msg: str) -> None:
    """Server-side logging - stderr only, stdout belongs to the protocol."""
    print(f"[localm-mcp] {msg}", file=sys.stderr, flush=True)


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
                 engine_factory: Optional[Callable] = None,
                 share_loaded: bool = False) -> None:
        self.default_model = default_model
        # Display name -> engine, plus usage order (least-recently-used FIRST,
        # MRU last).
        self._engines: Dict[str, Any] = {}
        self._lru: list = []
        # Injection point for tests - real factory builds a localm Engine
        self._factory = engine_factory or self._build_engine
        # When True, get_chat() answers with a model another localm instance on
        # this machine already has loaded instead of loading a second copy.
        self.share_loaded = bool(share_loaded)
        # Display name -> a client for another instance's loaded copy.
        self._peers: Dict[str, Any] = {}
        # Display name -> (endpoint, credential) that copy was verified with.
        self._peer_targets: Dict[str, tuple] = {}
        # Held by every method that reads or changes the engines and peers to
        # pick or load one, on the protocol thread and on a coder run's worker
        # thread alike. Never acquired while holding a generation lock
        # (generation_lock), Engine's load lock or the residency pin lock.
        self._lock = threading.RLock()

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

    @staticmethod
    def pin(engine) -> None:
        """Count one in-flight request on *engine*. A pinned engine
        (``active_requests > 0``) is never chosen as an eviction victim."""
        from localm.inference.residency import pin_engine
        pin_engine(engine)

    @staticmethod
    def unpin(engine) -> None:
        """Release one pin taken by ``pin``."""
        from localm.inference.residency import unpin_engine
        unpin_engine(engine)

    def generation_lock(self, engine):
        """The lock that serialises generations on *engine*: one per engine
        object, shared by every request and coder run that drives it."""
        from localm.plugins.coder.backends.shared_engine import engine_lock
        return engine_lock(engine)

    @contextlib.contextmanager
    def serving(self, engine):
        """Hold *engine* for one generation: pinned against eviction, and
        the only generation running on it. Waits for an in-flight generation
        on the same engine to finish first."""
        self.pin(engine)
        try:
            with self.generation_lock(engine):
                yield engine
        finally:
            self.unpin(engine)

    def is_resident(self, name: str, engine) -> bool:
        """True while *engine* is the cache's resident for *name*."""
        return self._engines.get(name) is engine

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
        from localm.model_manager.registry import get_operator_model_info
        # Gated here as well as in resolve_model: _build_engine is also
        # reachable directly, so the gate must not depend on having come
        # through resolve_model.
        if self._operator_supplied(model_name):
            info = get_operator_model_info(self.default_model)
        else:
            bad = unregistered_model_error(model_name)
            if bad:
                raise ValueError(bad)
            info = get_model_info(model_name)
        if info is None:
            raise ValueError(f"Model not found: {model_name!r}. "
                             f"Run 'localm list' to see registered models.")
        path, _hint = info
        from localm.model_manager import get_model_mmproj
        from localm.model_manager.registry import get_operator_model_mmproj
        mmproj = (get_operator_model_mmproj(self.default_model)
                  if self._operator_supplied(model_name)
                  else get_model_mmproj(model_name))
        return Engine(str(path), display_name=model_name, mmproj_path=mmproj)

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

    def route(self, requested: Optional[str], messages: list, *,
              required: tuple = (), pinned: Optional[bool] = None):
        """The capability-routing decision for a request with *messages*,
        without loading anything: which model should answer it.

        A model the client named is pinned unless *pinned* says otherwise;
        the server's default model is not. A request that needs something the
        model it would use lacks (an image, structured tool calls, a longer
        conversation than it was trained for) resolves to an installed model
        that has it. Raises ValueError for a name that is not registered."""
        from localm.inference import capability_routing as cr
        from localm.model_manager import capabilities as caps
        current = self.resolve_model(requested)
        pinned = bool(requested) if pinned is None else bool(pinned)
        needs = cr.request_needs(messages or [], required=required)
        known = {}
        with self._lock:
            eng = self._engines.get(current)
            resident = list(self._lru) + list(self._peers)
        if (eng is not None and getattr(eng, "loaded", False)
                and getattr(eng, "supports_images", False) is True):
            known[caps.VISION] = True
        return cr.plan_route(current, needs, pinned=pinned, resident=resident,
                             current_known=known)

    def get_chat(self, name: str):
        """The engine to answer a chat with model *name*: this server's own
        copy when it has one loaded; otherwise, when ``share_loaded``, another
        localm instance's already-loaded copy if one is available; else this
        server's own (see get)."""
        with self._lock:
            own = self._engines.get(name)
            own_loaded = (own is not None and getattr(own, "loaded", False) is True
                          and getattr(own, "unloading", False) is not True)
            if self.share_loaded and not own_loaded:
                peer = self._peers.get(name) or self._peer_engine(name)
                if peer is not None:
                    return peer
            return self.get(name)

    def is_peer(self, engine) -> bool:
        """True when *engine* is another instance's copy, not one loaded here."""
        return any(e is engine for e in list(self._peers.values()))

    def drop_peer(self, name: str) -> None:
        """Stop using another instance's copy of *name*, e.g. after it stopped
        answering; the next get_chat() looks again or loads it here."""
        with self._lock:
            self._peers.pop(name, None)
            self._peer_targets.pop(name, None)

    def get_loaded_chat(self, name: str):
        """get_chat() with the model ready to answer. Another instance's copy
        found by an earlier call is used only while that instance still
        answers and accepts the credential it was verified with; otherwise it
        is dropped, and another instance's copy is looked for or the model is
        loaded here. This server's own engine is loaded before it is returned;
        one that fails to load is removed from the cache and the load error
        raised."""
        with self._lock:
            if name in self._peers and not self._peer_still_answers(name):
                self.drop_peer(name)
            return self._loaded(name, self.get_chat(name))

    def get_loaded(self, name: str):
        """This server's own engine for *name* (see get), loaded before it is
        returned; one that fails to load is removed from the cache and the
        load error raised."""
        with self._lock:
            return self._loaded(name, self.get(name))

    def load_here_instead_of(self, name: str, peer):
        """This server's own engine for *name*, loaded and pinned (release it
        with unpin()), in place of *peer*, another instance's copy that can no
        longer be used; *peer* is dropped. Raises what building or loading the
        engine raised."""
        with self._lock:
            if self._peers.get(name) is peer:
                self.drop_peer(name)
            engine = self.get_loaded(name)
            self.pin(engine)
            return engine

    def _loaded(self, name: str, engine):
        """*engine*, the cache's engine for *name*, ready to answer: another
        instance's copy as it is, this server's own after ``load()``. An
        engine that fails to load is removed from the cache and the load error
        raised."""
        if self.is_peer(engine) or getattr(engine, "loaded", False) is True:
            return engine
        try:
            engine.load()
        except Exception:
            self._discard(name, engine)
            raise
        return engine

    def _peer_still_answers(self, name: str) -> bool:
        """True while the instance behind the cached copy of *name* answers and
        accepts the credential that copy was verified with. A failed check is
        logged."""
        target, token = self._peer_targets.get(name, (None, None))
        if target is None:
            _log(f"not using the cached copy of {name}: no endpoint is recorded for it")
            return False
        from localm import peer_routing
        try:
            peer_routing.verify_peer_credential(target, token)
        except Exception as e:
            _log(f"not using {name} loaded by the localm instance on port "
                 f"{target.get('port')} any more: {e}")
            return False
        return True

    def _discard(self, name: str, engine) -> None:
        """Remove *engine*, the cache's engine for *name* whose load failed,
        and release whatever the failed load left behind."""
        if self._engines.get(name) is engine:
            self._engines.pop(name, None)
            if name in self._lru:
                self._lru.remove(name)
        try:
            engine.unload()
        except Exception as e:
            _log(f"warning: failed to release {name} after its load failed: {e}")

    def _peer_engine(self, name: str):
        """A client for another localm instance on this machine that has
        *name* loaded (matched by model file), or None.

        An instance of this same install is reached at the address in this
        install's own instance file and authenticated with this install's
        credential; an entry claiming such an instance at any other port is
        not used. Any other instance is used only when it needs no credential
        (open mode). Best-effort: a failed lookup is logged and answers None."""
        try:
            from localm import instances, peer_routing
            from localm.auth import resolve_bearer_token
            from localm.bindhost import self_connect_host
            from localm.config import home_dir, load_registry
            from localm.inference.http_engine import HttpEngine
            reg = load_registry()
            canonical, aliases = peer_routing.registry_name_and_aliases(reg, name)
            peer = peer_routing.find_offer(
                canonical, aliases,
                identity=peer_routing.local_identity(reg, canonical))
            if peer is None:
                return None
            own = {str(e.get("instance_id")): e for e in instances.list_entries(home_dir())}
            same_install = own.get(str(peer.get("instance_id")))
            token = None
            target = peer
            if same_install is not None:
                if str(same_install.get("port")) != str(peer.get("port")):
                    _log(f"not using {name}: the machine-wide registry names this "
                         f"install's instance {peer.get('instance_id')} at port "
                         f"{peer.get('port')}, but it serves port "
                         f"{same_install.get('port')}")
                    return None
                target = {**peer,
                          "host": self_connect_host(same_install.get("host")),
                          "port": same_install.get("port"),
                          "scheme": same_install.get("scheme") or "http"}
                token = resolve_bearer_token(same_install.get("token"))
            try:
                peer_routing.verify_peer_credential(target, token)
            except Exception as e:
                _log(f"not using {name} loaded by the localm instance on port "
                     f"{target.get('port')}: {e}")
                return None
            route = peer_routing.PeerRoute(
                model=name, instance_id=str(target.get("instance_id")),
                host=target.get("host"), port=int(target.get("port")),
                scheme=target.get("scheme") or "http", api_key=token or "")
            base = peer_routing._peer_url(route, "/v1")
            eng = HttpEngine(base, token=token,
                             model=peer.get("matched_model") or name,
                             display_name=name, pin_model=True)
            eng.active_requests = 0
            eng.unloading = False
            self._peers[name] = eng
            self._peer_targets[name] = (target, token)
            _log(f"using {name} already loaded by the localm instance on port "
                 f"{target.get('port')} (no second copy loaded)")
            return eng
        except Exception as e:
            _log(f"warning: looking for another instance with {name} loaded "
                 f"failed: {e}")
            return None

    def get(self, requested: Optional[str]):
        name = self.resolve_model(requested)
        with self._lock:
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
            from localm.model_manager.registry import get_operator_model_info
            # Operator-supplied default may be a path; a client name may not,
            # and returns None here.
            info = (get_operator_model_info(self.default_model)
                    if self._operator_supplied(name) else get_model_info(name))
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
            from localm.model_manager.registry import get_operator_model_info
            # Tells the operator's own --model path from a client-supplied name.
            info = (get_operator_model_info(self.default_model)
                    if self._operator_supplied(name) else get_model_info(name))
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
        eviction. A peer that is still serving a request is waited for, bounded
        by BUSY_WAIT_SECONDS of wall clock (the VRAM probe's own cost counts);
        past the bound the load is refused with ModelBusyError rather than
        stacked on top of the busy peer.
        """
        from localm.config import load_config
        from localm.inference import residency
        cfg = load_config()
        cap = residency.resident_cap(cfg)
        pinned = residency.pinned_model_names(cfg)
        required = self._model_required_bytes(name)
        started = time.monotonic()
        waited = 0.0
        announced = False
        vram_ok = None
        # The resident set only changes through _evict, so the probe is taken
        # once per change, never once per poll of a busy peer.
        probe_pending = True
        while self._lru:
            over_cap = residency.exceeds_resident_cap(self._lru, name, cap)
            # Only probe when the cap is satisfied: being over cap already means
            # room is needed regardless of what VRAM says. vram_ok stays None to
            # record that no pass measured, which the message below relies on
            # so it never reports a shortfall nobody observed.
            if not over_cap and probe_pending:
                vram_ok = self._fits_alongside(name, required)
                probe_pending = False
            if not over_cap and vram_ok:
                return
            victim = residency.pick_eviction_victim(
                self._lru, self._engines, requested=name, pinned=pinned)
            if victim is None:
                busy = [n for n in self._lru
                        if n != name and n not in pinned
                        and residency.is_serving(self._engines.get(n))]
                waited = time.monotonic() - started
                if busy and waited < BUSY_WAIT_SECONDS:
                    if not announced:
                        announced = True
                        _log(f"waiting for {busy} to finish serving before "
                             f"making room for {name}")
                    time.sleep(BUSY_POLL_SECONDS)
                    continue
                if busy:
                    raise ModelBusyError(
                        f"cannot load {name}: {', '.join(busy)} is still serving "
                        f"a request after {waited:.0f}s and no other resident "
                        f"model can be evicted. Retry once it finishes.")
                # Nothing evictable and nothing busy: every remaining peer is
                # pinned by configuration.
                # Load anyway and SAY the policy was missed, rather than
                # pretending it held.
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
            if self._evict(victim, loading=name):
                probe_pending = True

    def _evict(self, victim: str, *, loading: str) -> bool:
        """Unload ``victim`` and wait for its VRAM to actually come back.

        Returns False, with nothing changed, when a request pinned the victim
        between its selection and this call; the residency mark
        (``begin_unload``) and that pin are decided under one lock, so an
        evicted engine can never be claimed for a generation again."""
        from localm.inference.residency import begin_unload
        engine = self._engines.get(victim)
        if engine is None:
            if victim in self._lru:
                self._lru.remove(victim)
            return True
        if not begin_unload(engine):
            return False
        self._engines.pop(victim, None)
        if victim in self._lru:
            self._lru.remove(victim)
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
        return True

    def unload_all(self) -> None:
        """Free every resident engine (shutdown). N resident means N to free."""
        with self._lock:
            for name in list(self._lru):
                engine = self._engines.pop(name, None)
                self._lru.remove(name)
                if engine is None:
                    continue
                try:
                    engine.unload()
                except Exception as e:
                    # Process teardown, so nothing downstream can act on this,
                    # but a native free that failed leaves VRAM pinned after
                    # exit. stderr only; stdout belongs to the protocol.
                    _log(f"warning: failed to unload {name} at shutdown: {e}")


def _text_result(text: str, is_error: bool = False) -> dict:
    return {"content": [{"type": "text", "text": text}], "isError": is_error}


@contextlib.contextmanager
def _quiet_stdout():
    """Redirect stdout to stderr for the duration of the block, so a downstream
    call's stray prints never corrupt the JSON-RPC frame stream on stdout."""
    with contextlib.redirect_stdout(sys.stderr):
        yield


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
        from localm.model_manager.registry import get_operator_model_info
        # resolve_model(None) yields the operator's own default, so a path is
        # legitimate here; anything else is already registry-gated upstream.
        info = (get_operator_model_info(engines.default_model)
                if engines._operator_supplied(name) else get_model_info(name))
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
    """Return {tool_name: {schema, handler}} for everything this server offers.

    Composes the tool families under ``localm.plugins.mcpserver.tools`` and
    then drops the gated tools whose gate is closed: ``embed`` when the backend
    cannot embed, ``run_coder_task`` unless *enable_coder* and the coder plugin
    is active, ``generate_image`` unless *enable_images*, ``memory_recall``
    unless *enable_memory* and the memory plugin is active, and
    ``memory_append`` unless additionally *enable_memory_write*. A tool name
    defined by two families raises ``ToolNameCollision``.
    """
    from .tools import (chat, diagnostics, media_coder, memory,
                        merge_tool_groups, models, plugin_admin)

    tools = merge_tool_groups([
        ("chat", chat.build(engines)),
        ("memory", memory.build(enable_memory_write=enable_memory_write)),
        ("models", models.build(engines)),
        ("media_coder", media_coder.build(engines)),
        ("diagnostics", diagnostics.build()),
        ("plugin_admin", plugin_admin.build()),
    ])

    # Advertisement gates; each probe runs at most once. The memory handlers
    # re-check their plugin and privacy gates on every call; the other gated
    # handlers rely on this list alone.
    can_embed = _backend_can_embed(engines)
    coder_on = enable_coder and _coder_available()
    memory_on = enable_memory and _memory_available()
    gates = {
        "embed": can_embed,
        "run_coder_task": coder_on,
        "generate_image": enable_images,
        "memory_recall": memory_on,
        "memory_append": memory_on and enable_memory_write,
    }
    for name, advertised in gates.items():
        if not advertised:
            del tools[name]
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
                 enable_memory_write: bool = False,
                 share_loaded: bool = False) -> None:
    """Entry point used by the CLI: build everything and block on stdio."""
    _redirect_consoles_to_stderr()
    engines = EngineCache(default_model=model, share_loaded=share_loaded)
    server = MCPStdioServer(build_tools(
        engines, enable_images=enable_images, enable_coder=enable_coder,
        enable_memory=enable_memory,
        enable_memory_write=enable_memory_write))
    # The protocol stream keeps the real stdout; every other print in this
    # process, on any thread, lands on stderr from here on.
    protocol_out = sys.stdout
    sys.stdout = sys.stderr
    try:
        server.run_stdio(stdout=protocol_out)
    finally:
        sys.stdout = protocol_out
        # Every resident engine, not just the most recent one: freeing one of N
        # would leave the rest holding VRAM past exit.
        engines.unload_all()

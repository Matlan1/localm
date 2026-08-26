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
SERVER_VERSION = "0.1.5rc3"


def _log(msg: str) -> None:
    """Server-side logging - stderr only, stdout belongs to the protocol."""
    print(f"[localm-mcp] {msg}", file=sys.stderr, flush=True)


def _child_identity_env() -> dict:
    """Env for a ``-m localm`` helper process so it is THE SAME localm as this
    server: same data home, same code.

    Both are otherwise re-resolved from ambient state at every process
    boundary: the data home falls back to a contained default derived from the
    running code's location when nothing is configured, and ``-m`` puts the
    child's cwd first on ``sys.path``, where a ``localm/`` directory in that cwd
    (any other checkout) silently swaps which CODE runs. run_coder_task runs its
    child in the TASK's directory (see its cwd comment), so without this pin a
    server whose own home came from ITS location would hand the coder chain a
    DIFFERENT, empty home.

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
    it, the same as the HTTP server. Both servers ask the same module
    (``inference.residency``) the same two questions: may this load with zero
    eviction, and if not who is the safe victim.

    Stacking needs a fresh, measurable reading that clears the requirement plus
    headroom with no split shortfall. On a box that cannot measure VRAM, on an
    inconclusive probe, or for a model whose footprint cannot be read, this falls
    back to single-resident behaviour (evict, wait for the free to land, then
    load). A wrong PERMIT here is a native OOM or a driver hang, not a tidy
    error.
    """

    def __init__(self, default_model: Optional[str] = None,
                 engine_factory: Optional[Callable] = None) -> None:
        self.default_model = default_model
        # Display name -> engine, plus usage order (least-recently-used first,
        # most-recently-used last).
        self._engines: Dict[str, Any] = {}
        self._lru: list = []
        # Injection point for tests - real factory builds a localm Engine
        self._factory = engine_factory or self._build_engine

    # ---- back-compat views over the multi-resident state -------------------
    # _engine/_loaded_name read the most-recently-used entry.

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
        environment variable, i.e. from the person who launched the process, so a
        filesystem path is legitimate there and gating it would break
        ``localm mcp --model <path>``. Every OTHER name arrives in a tool call
        from the MCP client, which is normally an LLM that can be steered by
        content it was asked to summarise - so a name from that source is treated
        as hostile input and must be a registered one."""
        return bool(model_name) and model_name == self.default_model

    def _build_engine(self, model_name: str):
        from localm.inference.engine import Engine
        from localm.model_manager import get_model_info, unregistered_model_error
        # The registration gate runs here as well as in resolve_model:
        # _build_engine is also reachable directly.
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
        # A client-supplied name must be registered; the operator's own --model
        # default is exempt.
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
            # Resident but NOT loaded, so it holds no VRAM yet: run the
            # eviction gate, then hand back the SAME object.
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
            # A direct path resolves only for the operator-supplied default;
            # any other name returns None here.
            info = get_model_info(
                name, allow_direct_path=self._operator_supplied(name))
            if info is None:
                return None
            path, _hint = info
            return required_vram_bytes(model_footprint_bytes(path))
        except Exception as e:
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
            # Probes with the CLI deadline and waits out an in-flight probe.
            # Runs inline: this process is synchronous, with no event loop.
            v_info, probe_status = vram_capacity(
                return_status=True, deadline=discover._GPU_PROBE_CLI_DEADLINE,
                wait_for_inflight=True)
            probe_ok = probe_status == discover.GPU_PROBE_OK
            shortfall = []
            if probe_ok and self._is_gguf(name):
                # Per-device check: aggregate free can clear the bar while one
                # device of a configured split is short.
                shortfall = gpu_split_shortfall(required + DEFAULT_HEADROOM_BYTES)
            # PROCESS-scoped readings are blind to the other resident models and
            # never permit a load.
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
            info = get_model_info(
                name, allow_direct_path=self._operator_supplied(name))
            return bool(info) and _is_gguf(info[0])
        except Exception:
            # Fail CLOSED: unknown returns True, so the per-device split check
            # still runs.
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
            # Probe only when the cap is satisfied. vram_ok stays None to record
            # that this pass did not measure, which the message below reads.
            vram_ok = None
            if not over_cap:
                vram_ok = self._fits_alongside(name, required)
                if vram_ok:
                    return
            victim = residency.pick_eviction_victim(
                self._lru, self._engines, requested=name, pinned=pinned)
            if victim is None:
                # Nothing evictable (all pinned, or all busy): load anyway and
                # name the policy that was missed.
                reasons = []
                if over_cap:
                    reasons.append("the resident cap")
                if vram_ok is False:
                    # _fits_alongside returns False WITHOUT probing when the
                    # model cannot be sized, so name the check accordingly.
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
        # SEED the wait from _vram_free_reading(), which also returns a stale
        # reading, and POLL with the live-only reader, never the other way
        # round: before_bytes=None makes wait_for_vram_release skip the wait
        # entirely. Freshness and scope are carried separately and feed the
        # verdict below.
        before_free, before_fresh, before_scope = _vram_free_reading()
        try:
            engine.unload()
        except Exception as e:
            # Unload is best-effort - the new model still loads - but a cleanup
            # failure is reported, not swallowed.
            _log(f"warning: failed to unload {victim}: {e}")
        # The native unload frees VRAM asynchronously; wait for it before the
        # next load. before_free is None only when VRAM is not measurable at
        # all, in which case the wait is a no-op.
        from localm.discover import FREE_SCOPE_DEVICE
        from localm.vram import wait_for_vram_release
        released, _final = wait_for_vram_release(
            _live_free_vram_bytes, before_bytes=before_free)
        backable = before_fresh and before_scope == FREE_SCOPE_DEVICE
        if released is False and backable:
            # Fresh AND device-global on both ends: report that the free did
            # not rise.
            _log(f"warning: VRAM free did not rise after unloading "
                 f"{victim} within the timeout - loading {loading} anyway")
        elif before_free is not None and (released is None or not backable):
            # A timed-out probe, or a process-scoped reading blind to the
            # worker's VRAM, cannot show whether the free landed: report it as
            # unconfirmed. The wait still ran; only the verdict is withheld.
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
                # Log to stderr; stdout carries the JSON-RPC frames.
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
    """Call fn(mgr) inside the stdout-quieting guard, mapping KeyError/ValueError
    the way install/enable/disable/uninstall_plugin all do. Returns the mapped
    error _text_result, or None on success."""
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
            # A custom factory can be a real engine builder, so guard stdout
            # the same as chat()/embed()/pull_model() below.
            with _quiet_stdout():
                backend = getattr(engines.get(None), "_backend", None)
            return getattr(backend, "can_embed", True) is not False
        except Exception as e:
            # Probe failed: assume embeddable and log the cause. The debug
            # logger writes to file/stderr, never stdout.
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
        # resolve_model(None) yields the operator's own default, so a path
        # resolves here; any other name is registry-gated upstream.
        info = get_model_info(
            name, allow_direct_path=engines._operator_supplied(name))
        if info is not None:
            path, _hint = info
            if str(path).lower().endswith(".gguf"):
                return False
    except Exception as e:
        # Registry probe failed: assume embeddable and log the cause. The debug
        # logger stays off stdout.
        from localm.debuglog import logger
        logger.debug("mcp: embed-capability probe (registry) failed, assuming "
                     "embeddable: %s", e)
    return True


def _coder_available() -> bool:
    """True when the coder plugin is installed on disk AND enabled - matches the
    same check `localm coder` itself does before accepting a task (see
    plugins/coder/cli/_main.py), so the tool is only advertised when a call
    would actually work."""
    try:
        from localm.plugins.engine import PluginManager
        return PluginManager(None).is_active("coder")
    except Exception as e:
        # Fails CLOSED, hiding the coder tool, when the probe raises; the cause
        # is logged. The debug logger writes to file/stderr, never stdout.
        from localm.debuglog import logger
        logger.debug("mcp: coder-availability probe failed, hiding coder tool: %s", e)
        return False


def build_tools(engines: EngineCache, enable_images: bool = True,
                 enable_coder: bool = True) -> Dict[str, dict]:
    """Return {tool_name: {schema, handler}} for everything this server offers."""

    def chat(args: dict) -> dict:
        prompt = args.get("prompt", "")
        if not prompt:
            return _text_result("'prompt' is required", is_error=True)
        # engines.get() can trigger a fresh model load, whose native sizing and
        # context diagnostics print straight to stdout - the same stream the
        # JSON-RPC frames travel on.
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

        # include_token=True: this call asks each discovered instance over HTTP
        # and needs the attach token a keyless instance's middleware requires.
        # Display paths use the default token-stripped snapshot().
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
                    # Absent, not zero: an operation reporting no progress is at
                    # an unknown percentage.
                    if isinstance(pct, (int, float)):
                        bits.append(f"{pct:.0f}%")
                    created = op.get("created_at")
                    # Age against the SERVER's clock, which this process may not
                    # share.
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
                # A single malformed entry is shown corrupt rather than
                # crashing or blanking the whole listing.
                lines.append(f"{name}  [corrupt]  (malformed registry entry)")
                continue
            # These per-row filesystem stats run INLINE: run_stdio is a
            # synchronous loop and handle() is a plain def, so there is no event
            # loop to protect.
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
        # A one-shot call: wait_first_vram blocks until the first VRAM reading
        # lands. MCP stdio serves one request at a time and has no event loop.
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
        # `repo` is MCP-client-supplied: refuse UNC or device syntax
        # unconditionally, on every platform, BEFORE the Path(repo).exists()
        # sink below runs. The message does not echo `repo` back.
        if is_unc_or_device_path(repo):
            return _text_result(
                "'repo' looks like a filesystem path (UNC or device syntax), not "
                "a HuggingFace repo id. pull_model downloads a model by repo id "
                "(e.g. 'owner/name').", is_error=True)
        # An existing local path is refused: pull_model otherwise treats it as a
        # local add and registers an arbitrary directory under a client-chosen
        # name.
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

        # pull_model()'s progress bars and messages print via a rich Console
        # singleton; redirect_stdout catches them whichever Console instance is
        # in play.
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
            # This load prints native sizing diagnostics straight to stdout.
            #
            # engines.get() only constructs/registers the Engine and runs the
            # VRAM-eviction gate - it does NOT call Engine.load(), so load it
            # explicitly here.
            with _quiet_stdout():
                engine = engines.get(name)
                engine.load()
        except Exception as e:
            return _text_result(
                f"pulled and registered as {name!r}, but loading it failed: {e}",
                is_error=True)
        msg = f"pulled, registered, and loaded {name!r} - ready to use"
        # gpu_placement is None whenever the backend cannot report per-layer
        # placement for this engine. Only a known partial or zero placement adds
        # the degraded note.
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
        # A fresh embedder load can print to stdout too.
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
            auth.json, sessions.json, rag/, coder/). See below.

            Delegates to ``pathsafe.confined_absolute_or_under``, the same
            primitive coder/tools/base.py's ``_confine`` uses, which carries the
            UNC/device guard and the NTFS Alternate Data Stream /
            short-name-alias guard. Every rejection reason is folded into the
            SAME message, never echoing the client-supplied string back."""
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
            # input_image is READ and then UPLOADED to ComfyUI, so it is checked
            # against the uploads inbox and the generated-media galleries rather
            # than the data dir, through the non-HTTP entry point.
            # InputImageRefused is a ValueError, so the except below catches it.
            input_p = (_media_paths.check_input_image(args["input_image"])
                       if args.get("input_image") else None)
        except ValueError as e:
            return _text_result(str(e), is_error=True)

        is_privacy = effective_mode("mcp") == SessionMode.PRIVACY
        # comfy.generate_image builds its own rich Console / Progress on stdout,
        # where the JSON-RPC frame stream also lives: route stray output to
        # stderr.
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
        # `cwd` is MCP-client-supplied: refuse UNC or device syntax
        # unconditionally, BEFORE the is_dir() call below runs. is_dir() dials
        # SMB for a UNC target exactly like exists() does.
        if is_unc_or_device_path(cwd):
            return _text_result(
                "'cwd' must be a local directory path, not a UNC or device path.",
                is_error=True)
        cwd_path = Path(cwd).expanduser()
        if not cwd_path.is_dir():
            return _text_result(f"cwd is not a directory: {cwd_path}", is_error=True)

        # Shells out to the `localm coder` single-shot CLI, which carries its own
        # project-config resolution and instance attach/spawn logic.
        cmd = [sys.executable, "-m", "localm", "coder", task,
               "--cwd", str(cwd_path), "--output-format", "json"]
        if args.get("model"):
            # A client-supplied model name becomes argv and reaches the startup
            # resolver, which opts into allow_direct_path. Registry-check it
            # here first.
            from localm.model_manager import unregistered_model_error
            bad = unregistered_model_error(args["model"])
            if bad:
                return _text_result(bad, is_error=True)
            cmd += ["--model", args["model"]]
        if args.get("max_turns") is not None:
            cmd += ["--max-turns", str(args["max_turns"])]
        # Defaults OFF, matching the CLI: without it, file writes still happen
        # but run_shell is denied for lack of a TTY to confirm on.
        if args.get("yes"):
            cmd.append("--yes")
        timeout = args.get("timeout_seconds") or 900

        try:
            # cwd=cwd_path: an auto-spawned server identifies the project by the
            # spawning process's OS working directory, not the --cwd flag above,
            # so the coder's attach-back lookup depends on it.
            # env=_child_identity_env() keeps that cwd change from moving the
            # child onto a different data home or different localm code.
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                                  cwd=str(cwd_path), env=_child_identity_env())
        except subprocess.TimeoutExpired:
            return _text_result(f"coder task timed out after {timeout}s", is_error=True)

        # --output-format json pretty-prints with indent=2, and console messages
        # print to stdout both before and after it. Find each line that is a lone
        # opening brace, newest first, and raw_decode from there: raw_decode
        # stops at the closing brace and tolerates trailing console text.
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
        # `model` is a free-form string chosen by the MCP client and this writes
        # the admin_only embedding_model key, so the gate is on the VALUE: a
        # known key or a registered model name only, never a raw path. An
        # unacceptable value is refused, not silently ignored.
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
        holding the file open while it deletes it. That is not a race: it is
        deterministic, and it is the most likely way this tool destroys a
        model, because the agent that just chatted with one is the same agent
        that asks to remove it.
        """
        from localm.model_manager.registry import engine_holding_model_file
        candidates = [
            (name, getattr(engine, "model_path", None))
            for name, engine in list(getattr(engines, "_engines", {}).items())
            if getattr(engine, "loaded", False)
        ]
        return engine_holding_model_file(model, reg, candidates)

    def _remote_hold(model: str):
        """Why a running localm SERVER means this removal must be refused, or
        None when every discovered instance positively ruled itself out.

        This process shares no memory with the HTTP/GUI server, so the only way
        to find out is to ask each running instance over HTTP - the same
        discovery the ``server_activity`` tool uses, for the same reason.

        EVERY OUTCOME THAT IS NOT AN ANSWER IS A REFUSAL, and the message says
        which one it was. "That server reports nothing holds it" and "I could
        not reach that server" are opposite conclusions, and collapsing them
        would delete a live model's file on the strength of never having found
        out. A refused delete costs one command and names the server to go and
        check; a deleted model file is gone.
        """
        from localm import instances
        from localm.bindhost import self_connect_host, url_host
        from localm.config import home_dir
        from localm.selfclient import read_model_file_hold

        # include_token=True: this ASKS each instance over HTTP (an internal,
        # non-display use), so it needs the attach token a genuinely open
        # (keyless) instance's middleware requires. Never for anything a human
        # reads.
        rows = instances.snapshot(home_dir(), include_token=True)
        for e in rows:
            scheme = e.get("scheme", "http")
            where = (scheme + "://"
                     + url_host(self_connect_host(e.get("host")))
                     + ":" + str(e.get("port")))
            if not e.get("alive"):
                # A failed /whoami is NOT proof the process is gone: snapshot()
                # reaps entries whose pid has died before this runs, and
                # instances.py's own comment warns that a transient probe miss
                # must never be read as death. A listed instance that did not
                # answer is therefore a live process of unknown state, and
                # unknown refuses.
                return (f"a localm server at {where} is registered but did not "
                        f"answer an identity check, so whether it has this "
                        f"model loaded could not be established")
            state, payload = read_model_file_hold(
                scheme, e.get("port"), model, e.get("token"), e.get("host"))
            if state == "ok":
                if not payload.get("held"):
                    continue          # this server positively ruled itself out
                key = payload.get("key") or "a loaded model"
                reason = payload.get("reason")
                if reason:
                    return (f"the localm server at {where} has {key!r} loaded "
                            f"and {reason}, so it cannot be ruled out as "
                            f"holding this file")
                return (f"the localm server at {where} still has this model's "
                        f"file loaded as {key!r}")
            if state == "absent":
                continue              # that instance serves a different library
            if state == "unauthorized":
                return (f"the localm server at {where} requires an API key this "
                        f"process does not have, so whether it has this model "
                        f"loaded could not be established")
            if state == "unsupported":
                return (f"the localm server at {where} is an older localm that "
                        f"cannot report which models it holds, so whether it "
                        f"has this one loaded could not be established")
            if state == "unreachable":
                return (f"the localm server at {where} could not be reached "
                        f"({payload}), so whether it has this model loaded "
                        f"could not be established")
            return (f"the localm server at {where} answered HTTP {payload} "
                    f"instead of reporting what it holds, so whether it has "
                    f"this model loaded could not be established")
        return None

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
        # The GUI's remove route guards exactly this before spawning that
        # command; this tool did not. Both holders are checked here - the
        # engines resident in this process, and any running server - and either
        # one refuses.
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
        remote = _remote_hold(model)
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
            # env=_child_identity_env(): pins the child to this server's own
            # home and localm code rather than ambient state.
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
            # The plugin is left installed and enabled rather than rolled back.
            # The dependency failure is folded into the reply below and logged
            # here.
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
        # Bypasses _run_mgr_action to read uninstall()'s bool: it reports
        # whether the installed directory actually came off disk.
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

    # Advertise embed only when the active backend can produce vectors. The
    # handler still degrades gracefully if invoked anyway.
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
        # Deletes the model file on disk; declared so an MCP client can prompt
        # for confirmation before calling.
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
        # Removes the plugin, and with delete_data its stored data on disk;
        # declared so an MCP client can confirm before calling.
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
            # A JSON-RPC batch array, a bare scalar, or null parses fine but is
            # not a request object: reply Invalid Request rather than calling
            # msg.get(...) on it.
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
                # MCP tool annotations (destructiveHint / readOnlyHint / title)
                # are emitted only when a tool declares them.
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
            # array; a bare scalar or null is invalid. handle() replies -32600
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
                 enable_coder: bool = True) -> None:
    """Entry point used by the CLI: build everything and block on stdio."""
    _redirect_consoles_to_stderr()
    engines = EngineCache(default_model=model)
    server = MCPStdioServer(build_tools(engines, enable_images=enable_images,
                                          enable_coder=enable_coder))
    try:
        server.run_stdio()
    finally:
        # Frees every resident engine, not just the most recent one.
        engines.unload_all()

# SPDX-License-Identifier: AGPL-3.0-or-later
"""Find `async def` route handlers that block the event loop.

A FastAPI handler declared `async def` runs ON the event loop. A handler
declared plain `def` is run in a worker thread by FastAPI itself. So a blocking
call is only ever a defect in the first shape, and when it happens EVERY client
of the server stalls, not just the one making the request - every chat stream,
every SSE job feed, every poll.

A grep cannot find these: the blocking call is usually two or three hops down a
helper chain. Coverage cannot either: the line executes, it is simply executing
in the wrong thread.

WHAT COUNTS AS BLOCKING, and why it is not "any lock"
-----------------------------------------------------
Flagging every `with some_lock:` reached from a handler produces roughly 180 of
this repo's 208 async routes, because the auth path alone takes several
short-lived dict guards on every single request. So a lock is only interesting
when its critical section can be held for a long time, and that is computed
rather than declared: pass 1 walks every `with <lock>:` body in the tree and
asks whether it transitively reaches a slow primitive.

    UNBOUNDED   the body spawns a child and waits for it, or loads a native
                library. `localm.inference.embedder._LOCK` is the worked
                example: `get_embedder` holds it across an `IsolatedEmbedder`
                construction, i.e. a process spawn plus a native model load,
                whose own ceiling is 300s. A status probe that takes that lock
                to read one integer therefore blocks the loop for as long as a
                cold load takes.
    BOUNDED     the body runs a short subprocess or writes a file.
                `localm.sessions._LOCK` is the worked example: its writes reach
                `restrict_file_perms`, which shells out to `icacls` on Windows.
                Reported, but it does not fail the gate.

A network call is always UNBOUNDED: the timeouts in this tree are 10-60s, and
`socket.getaddrinfo` takes no timeout at all.

WHAT IS NOT A FINDING
---------------------
Work handed to a worker thread. The analyzer follows the same boundaries the
code does: `run_in_threadpool`, `run_in_threadpool_bounded`, `run_in_executor`,
`anyio.to_thread.run_sync`, a `jobs.start_fn` callback, an explicit
`Thread(target=...)`. A nested `def` that is passed to any of those is not
walked, because its body runs off the loop.

CONFIDENCE
----------
A call of the form `x.method(...)`, where `x` is a local whose type this file
cannot know, is resolved by unique method name across the tree and marked
UNCONFIRMED (`~>` in the printed chain). Dropping the heuristic loses real
findings (`store.record_result`); trusting it invents them (`sidecar.rename`,
which is `pathlib.Path.rename`, resolves to the CLI's `rename` command and
manufactures a network call on three media routes that make none). Read the
chain before acting on an UNCONFIRMED hop.

USAGE
-----
    python scripts/check_event_loop_offload.py            # full report
    python scripts/check_event_loop_offload.py --gate     # exit 1 on a NEW unbounded finding

`tests/test_event_loop_offload.py` pins the gate and proves this file FIRES.
"""

from __future__ import annotations

import argparse
import ast
from collections import defaultdict
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[1]

ROUTE_METHODS = {"get", "post", "put", "patch", "delete", "head", "options"}
ROUTE_RECEIVERS = {"app", "_app", "router", "_router"}

# Calls that hand the callable to a worker thread, so its body is not walked.
OFFLOADERS = {"run_in_threadpool", "run_in_threadpool_bounded", "run_sync",
              "run_in_executor", "start_fn", "submit", "start_thread", "to_thread"}
THREAD_CTORS = {"Thread", "Process"}

# Direct outbound network calls, all treated as unbounded here.
NET_LEAVES = {"safe_fetch_bytes", "safe_fetch_json", "safe_fetch_text",
              "verified_urlopen", "urlopen", "urlretrieve"}
NET_MOD_CALLS = (
    {("requests", c) for c in
     ("get", "post", "put", "delete", "head", "patch", "request")}
    | {("socket", "create_connection"), ("socket", "getaddrinfo"),
       ("httpx", "get"), ("httpx", "post"), ("httpx", "request")}
)

# Primitives that make a lock body slow, split by whether the wait is bounded.
UNBOUNDED_LEAVES = {"spawn_and_load", "load_lib", "CDLL", "WinDLL", "Popen",
                    "getaddrinfo", "create_connection"} | NET_LEAVES
BOUNDED_LEAVES = {"check_output", "check_call", "communicate"}
# `x.join()` / `x.wait()` / `x.poll()` where x names a child, a queue or a
# future. Matched on the receiver's name.
WAIT_CALLS = {"join", "wait", "recv", "poll", "result"}
WAIT_RECEIVER_HINTS = ("proc", "thread", "queue", "conn", "child", "future",
                       "_q", "worker")

# Known exceptions, keyed on (handler qualname, sink function) and never on a
# line number. Every entry carries a reason and a tracking id.
ALLOWED: dict[tuple[str, str], str] = {}
for _handler in (
    "imagine", "imagine_comfy_models", "imagine_comfy_launch",
    "music", "music_comfy_models", "music_comfy_launch",
    "video", "video_comfy_models", "video_comfy_launch",
    "register.media_preflight", "register.model_pull_comfy_source",
):
    ALLOWED[(_handler, "_host_is_link_local")] = (
        "NEW-COMFY-URL-SANITIZE-DNS-ON-THE-LOOP. sanitize_comfy_url resolves the "
        "configured comfy_api_url to refuse a link-local/metadata host, and the "
        "resolution is a blocking getaddrinfo. NOT reachable in the default "
        "config: comfy_api_url defaults to an IP literal, which ipaddress "
        "parses without touching DNS. Only an owner who sets a HOSTNAME pays "
        "it. The DNS lookup itself is load-bearing for the SSRF guard and must "
        "not be removed; the fix is to offload settings()/default_api_url() at "
        "each of these handlers, which is being tracked separately because "
        "those files are under concurrent edit.")


def _dotted(node: ast.AST) -> str:
    parts: list[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    parts.append(cur.id if isinstance(cur, ast.Name) else "<expr>")
    return ".".join(reversed(parts))


class Module:
    def __init__(self, path: Path, root: Path):
        self.path = path
        self.rel = path.relative_to(root).as_posix()
        rel = path.relative_to(root).with_suffix("")
        parts = list(rel.parts)
        if parts and parts[-1] == "__init__":
            parts.pop()
        self.name = ".".join(parts)
        self.tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        self.imports: dict[str, str] = {}
        self.funcs: dict[str, "Func"] = {}
        self.classes: dict[str, ast.ClassDef] = {}
        self.routes: list[tuple[str, str, str]] = []


class Func:
    __slots__ = ("mod", "qual", "node", "is_async", "calls", "locks", "nets",
                 "offloaded", "nested")

    def __init__(self, mod: Module, qual: str, node):
        self.mod = mod
        self.qual = qual
        self.node = node
        self.is_async = isinstance(node, ast.AsyncFunctionDef)
        self.calls: list[tuple[str, int]] = []
        self.locks: list[tuple[str, str, int]] = []
        self.nets: list[tuple[str, int]] = []
        self.offloaded: set[str] = set()
        self.nested: set[str] = set()

    @property
    def key(self) -> str:
        return self.mod.name + "::" + self.qual

    @property
    def where(self) -> str:
        return self.mod.rel + ":" + str(self.node.lineno)


class Finding:
    def __init__(self, route, handler, kind, sink, sink_at, sink_func, chain, confident):
        self.route = route
        self.handler = handler          # Func
        self.kind = kind                # "network" | "unbounded-lock" | "bounded-lock"
        self.sink = sink                # the blocking call, or the lock's identity
        self.sink_at = sink_at
        self.sink_func = sink_func      # qualname of the function CONTAINING the sink
        self.chain = chain
        self.confident = confident

    @property
    def allow_key(self) -> tuple[str, str]:
        # (handler, function that blocks), never a line number.
        return (self.handler.qual, self.sink_func)

    @property
    def allowed(self) -> Optional[str]:
        return ALLOWED.get(self.allow_key)

    @property
    def gating(self) -> bool:
        """Does this fail --gate? Unbounded only, and only when confirmed."""
        return (self.kind in ("network", "unbounded-lock")
                and self.confident and self.allowed is None)


class Analyzer:
    def __init__(self, root: Path, package: str = "localm"):
        self.root = root
        self.mods: dict[str, Module] = {}
        # Files this analyzer could not parse; reported by --gate.
        self.unparseable: list[str] = []
        self.by_leaf: dict[str, list[Func]] = defaultdict(list)
        self.by_class: dict[str, list[Func]] = defaultdict(list)
        self.lock_bodies: dict[str, list[tuple[Func, ast.AST]]] = defaultdict(list)
        self.lock_kind: dict[str, tuple[str, str, str]] = {}
        self._load(root / package)
        self._classify_locks()

    # ---------------------------------------------------------------- parse --
    def _load(self, pkg: Path) -> None:
        for path in sorted(pkg.rglob("*.py")):
            try:
                mod = Module(path, self.root)
            except (SyntaxError, UnicodeDecodeError) as e:
                self.unparseable.append("%s: %s" % (path.relative_to(self.root), e))
                continue
            self._collect_imports(mod)
            self._collect_defs(mod)
            self.mods[mod.name] = mod
        for mod in self.mods.values():
            for func in mod.funcs.values():
                self._scan(func)
        for mod in self.mods.values():
            for qual, func in mod.funcs.items():
                self.by_leaf[qual.rsplit(".", 1)[-1]].append(func)
            for qual in mod.classes:
                init = mod.funcs.get(qual + ".__init__")
                if init is not None:
                    self.by_class[qual.rsplit(".", 1)[-1]].append(init)

    @staticmethod
    def _collect_imports(mod: Module) -> None:
        for node in ast.walk(mod.tree):
            if isinstance(node, ast.ImportFrom):
                if node.level:
                    base = mod.name.split(".")
                    if mod.path.name != "__init__.py":
                        base = base[:-1]
                    if node.level > 1:
                        base = base[: len(base) - (node.level - 1)]
                    target = ".".join(base + ([node.module] if node.module else []))
                else:
                    target = node.module or ""
                for alias in node.names:
                    mod.imports[alias.asname or alias.name] = target + "." + alias.name
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.asname:
                        mod.imports[alias.asname] = alias.name
                    else:
                        head = alias.name.split(".")[0]
                        mod.imports[head] = head

    @staticmethod
    def _collect_defs(mod: Module) -> None:
        def walk(node, prefix: str) -> None:
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    qual = prefix + child.name
                    mod.funcs[qual] = Func(mod, qual, child)
                    for dec in child.decorator_list:
                        if not (isinstance(dec, ast.Call)
                                and isinstance(dec.func, ast.Attribute)
                                and dec.func.attr in ROUTE_METHODS):
                            continue
                        receiver = _dotted(dec.func).rsplit(".", 1)[0]
                        if receiver.split(".")[-1] not in ROUTE_RECEIVERS:
                            continue
                        path = (dec.args[0].value
                                if dec.args and isinstance(dec.args[0], ast.Constant)
                                else "?")
                        mod.routes.append((qual, dec.func.attr.upper(), str(path)))
                    walk(child, qual + ".")
                elif isinstance(child, ast.ClassDef):
                    mod.classes[prefix + child.name] = child
                    walk(child, prefix + child.name + ".")
                else:
                    walk(child, prefix)

        walk(mod.tree, "")

    @staticmethod
    def _own_body(func: Func) -> list[ast.stmt]:
        """func's own statements, with directly-nested def/async def removed.

        A nested def is a separate Func and is usually the thread-side closure,
        so counting its body as part of the parent would report a
        correctly-offloaded handler as blocking."""
        out: list[ast.stmt] = []
        for stmt in func.node.body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func.nested.add(stmt.name)
                continue
            out.append(stmt)
        return out

    def _lock_id(self, mod: Module, name: str) -> str:
        if name.startswith("self.") or name.startswith("cls."):
            return mod.name + "::" + name
        head, _, rest = name.partition(".")
        target = mod.imports.get(head)
        if target:
            return target + ("." + rest if rest else "")
        return mod.name + "." + name

    def _scan(self, func: Func) -> None:
        mod = func.mod
        stack: list[ast.AST] = list(self._own_body(func))
        while stack:
            node = stack.pop()
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    func.nested.add(child.name)
                    continue
                if isinstance(child, ast.Lambda):
                    continue
                stack.append(child)

            if isinstance(node, (ast.With, ast.AsyncWith)):
                for item in node.items:
                    ctx = item.context_expr
                    if not isinstance(ctx, (ast.Name, ast.Attribute)):
                        continue
                    name = _dotted(ctx)
                    if "lock" not in name.lower():
                        continue
                    lock = self._lock_id(mod, name)
                    func.locks.append((lock, name, node.lineno))
                    self.lock_bodies[lock].append((func, node))

            if not isinstance(node, ast.Call):
                continue
            name = _dotted(node.func)
            leaf = name.rsplit(".", 1)[-1]
            head = name.split(".", 1)[0]
            head_mod = mod.imports.get(head, head).split(".")[0]

            if leaf == "acquire" and "lock" in name.lower():
                base = name.rsplit(".", 1)[0]
                func.locks.append((self._lock_id(mod, base), base, node.lineno))
            if (leaf in NET_LEAVES or (head, leaf) in NET_MOD_CALLS
                    or (head_mod, leaf) in NET_MOD_CALLS):
                func.nets.append((name, node.lineno))
            if leaf in OFFLOADERS and node.args:
                first = node.args[0]
                if isinstance(first, ast.Name):
                    func.offloaded.add(first.id)
                elif isinstance(first, ast.Attribute):
                    func.offloaded.add(first.attr)
            if leaf in THREAD_CTORS:
                for kw in node.keywords:
                    if kw.arg == "target" and isinstance(kw.value, (ast.Name, ast.Attribute)):
                        func.offloaded.add(_dotted(kw.value).rsplit(".", 1)[-1])
            func.calls.append((name, node.lineno))

    # -------------------------------------------------------------- resolve --
    def resolve(self, func: Func, name: str) -> list[tuple[Func, bool]]:
        mod = func.mod
        head, _, rest = name.partition(".")
        leaf = name.rsplit(".", 1)[-1]

        if name in func.nested:
            nested = mod.funcs.get(func.qual + "." + name)
            if nested is not None:
                return [(nested, True)]
        if name in mod.funcs:
            return [(mod.funcs[name], True)]

        target = mod.imports.get(head)
        if target:
            tmod, _, tleaf = target.rpartition(".")
            if not rest and tmod in self.mods and tleaf in self.mods[tmod].funcs:
                return [(self.mods[tmod].funcs[tleaf], True)]
            full = target + "." + rest if rest else target
            tmod2, _, tleaf2 = full.rpartition(".")
            if tmod2 in self.mods and tleaf2 in self.mods[tmod2].funcs:
                return [(self.mods[tmod2].funcs[tleaf2], True)]
            if tmod in self.mods and tleaf in self.mods[tmod].classes:
                init = self.mods[tmod].funcs.get(tleaf + ".__init__")
                if init is not None:
                    return [(init, True)]
        if name in mod.classes:
            init = mod.funcs.get(name + ".__init__")
            if init is not None:
                return [(init, True)]

        candidates = self.by_class.get(leaf) or self.by_leaf.get(leaf) or []
        if len(candidates) != 1:
            return []
        opaque = bool(rest) and head not in ("self", "cls") \
            and (mod.imports.get(head) or head) not in self.mods
        return [(candidates[0], not opaque)]

    # ------------------------------------------------------- lock severity --
    def _slow_primitive(self, func: Func, node, depth: int, seen: set) -> Optional[tuple[str, str]]:
        if depth > 3:
            return None
        body = node.body if hasattr(node, "body") else []
        stack: list[ast.AST] = [s for s in body
                                if not isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef))]
        bounded_hit = None
        while stack:
            item = stack.pop()
            for child in ast.iter_child_nodes(item):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                    continue
                stack.append(child)
            if not isinstance(item, ast.Call):
                continue
            name = _dotted(item.func)
            leaf = name.rsplit(".", 1)[-1]
            at = func.mod.rel + ":" + str(item.lineno)
            if leaf in UNBOUNDED_LEAVES:
                return ("unbounded", name + " @" + at)
            if leaf in ("run", "call") and name.split(".", 1)[0] == "subprocess":
                bounded_hit = bounded_hit or ("bounded", name + " @" + at)
            if leaf in BOUNDED_LEAVES:
                bounded_hit = bounded_hit or ("bounded", name + " @" + at)
            if leaf in WAIT_CALLS and isinstance(item.func, ast.Attribute):
                receiver = name.rsplit(".", 1)[0].lower()
                if any(hint in receiver for hint in WAIT_RECEIVER_HINTS):
                    return ("unbounded", name + " @" + at)
            for target, _ok in self.resolve(func, name):
                if target.key in seen:
                    continue
                seen.add(target.key)
                deeper = self._slow_primitive(target, target.node, depth + 1, seen)
                if deeper is None:
                    continue
                kind, why = deeper
                if kind == "unbounded":
                    return ("unbounded", why + "  via " + name)
                bounded_hit = bounded_hit or ("bounded", why + "  via " + name)
        return bounded_hit

    def _classify_locks(self) -> None:
        for lock, entries in self.lock_bodies.items():
            best = None
            for func, node in entries:
                found = self._slow_primitive(func, node, 0, set())
                if found is None:
                    continue
                kind, why = found
                if kind == "unbounded":
                    best = (kind, why, func.mod.rel + ":" + str(node.lineno))
                    break
                if best is None:
                    best = (kind, why, func.mod.rel + ":" + str(node.lineno))
            if best is not None:
                self.lock_kind[lock] = best

    # ----------------------------------------------------------- the sweep --
    def findings(self, max_depth: int = 5) -> list[Finding]:
        out: list[Finding] = []
        for mod in self.mods.values():
            for qual, method, path in mod.routes:
                handler = mod.funcs.get(qual)
                if handler is None or not handler.is_async:
                    continue
                out.extend(self._walk(handler, method + " " + path, max_depth))
        return out

    def _walk(self, handler: Func, route: str, max_depth: int) -> list[Finding]:
        out: list[Finding] = []
        seen: set[str] = set()

        def visit(func: Func, chain: list[str], depth: int, confident: bool) -> None:
            if depth > max_depth or func.key in seen:
                return
            seen.add(func.key)
            for lock, shown, line in func.locks:
                kind = self.lock_kind.get(lock)
                if kind is None:
                    continue
                at = func.mod.rel + ":" + str(line)
                out.append(Finding(
                    route, handler,
                    "unbounded-lock" if kind[0] == "unbounded" else "bounded-lock",
                    lock, at, func.qual,
                    chain + [at + "  " + func.qual + "  (" + shown + ")"],
                    confident))
            for name, line in func.nets:
                at = func.mod.rel + ":" + str(line)
                out.append(Finding(route, handler, "network", name, at, func.qual,
                                   chain + [at + "  " + func.qual + "  (" + name + ")"],
                                   confident))
            for name, line in func.calls:
                leaf = name.rsplit(".", 1)[-1]
                if leaf in OFFLOADERS or leaf in THREAD_CTORS:
                    continue
                if leaf in func.offloaded or name in func.offloaded:
                    continue
                for target, ok in self.resolve(func, name):
                    if target.qual.rsplit(".", 1)[-1] in func.offloaded:
                        continue
                    arrow = "  ->  " if ok else "  ~>  "
                    visit(target,
                          chain + [func.mod.rel + ":" + str(line) + arrow + name],
                          depth + 1, confident and ok)

        visit(handler, [handler.where + "  " + handler.qual], 0, True)
        return out


def _shortest(findings: list[Finding]) -> list[Finding]:
    """One row per (route, kind, sink), keeping the shortest call chain."""
    best: dict[tuple, Finding] = {}
    for f in findings:
        key = (f.route, f.kind, f.sink, f.sink_at, f.sink_func)
        if key not in best or len(f.chain) < len(best[key].chain):
            best[key] = f
    order = {"network": 0, "unbounded-lock": 1, "bounded-lock": 2}
    return sorted(best.values(), key=lambda f: (order[f.kind], f.route, f.sink))


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gate", action="store_true",
                    help="exit 1 when a NEW unbounded finding appears")
    ap.add_argument("--root", default=str(REPO))
    ap.add_argument("--chains", action="store_true", help="print full call chains")
    args = ap.parse_args(argv)

    analyzer = Analyzer(Path(args.root))
    rows = _shortest(analyzer.findings())

    if analyzer.unparseable:
        print("UNPARSEABLE (not swept - these files are NOT covered):")
        for item in analyzer.unparseable:
            print("  " + item)
        print()

    print("locks reached from a route: %d classified slow (%d unbounded)"
          % (len(analyzer.lock_kind),
             sum(1 for k in analyzer.lock_kind.values() if k[0] == "unbounded")))
    for lock, (kind, why, at) in sorted(analyzer.lock_kind.items()):
        print("  %-9s %-58s %s -> %s" % (kind.upper(), lock, at, why))
    print()

    gating = [f for f in rows if f.gating]
    for group, label in (("network", "NETWORK CALL ON THE EVENT LOOP"),
                         ("unbounded-lock", "UNBOUNDED LOCK ON THE EVENT LOOP"),
                         ("bounded-lock", "BOUNDED LOCK ON THE EVENT LOOP (reported, not gated)")):
        chunk = [f for f in rows if f.kind == group]
        if not chunk:
            continue
        print("== %s: %d ==" % (label, len(chunk)))
        for f in chunk:
            flags = []
            if not f.confident:
                flags.append("UNCONFIRMED")
            if f.allowed:
                flags.append("ALLOWED")
            print("  %-46s %-34s %s%s"
                  % (f.route, f.sink, f.sink_at,
                     ("  [" + ",".join(flags) + "]") if flags else ""))
            if args.chains:
                for step in f.chain:
                    print("        " + step)
        print()

    if args.gate:
        if analyzer.unparseable:
            print("FAILED: %d file(s) could not be parsed, so they were not "
                  "swept - an unswept file is not a clean one."
                  % len(analyzer.unparseable))
            return 1
        if gating:
            print("FAILED: %d unoffloaded blocking call(s) on the event loop." % len(gating))
            print("Offload with run_in_threadpool_bounded (see "
                  "localm/inference/_threadpool_timeout.py), or add an ALLOWED "
                  "entry in this file with a reason and a tracking id.")
            return 1
        print("Event-loop offload check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

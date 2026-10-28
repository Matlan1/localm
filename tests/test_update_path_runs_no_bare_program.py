# SPDX-License-Identifier: AGPL-3.0-or-later
"""Nothing the update path starts is looked up on PATH.

Starting from every function in ``localm/updater.py`` and
``localm/_apply_update.py``, the scan follows calls and function references into
other ``localm`` modules, and follows a ``[sys.executable, "-m", "localm", <command>]``
the update path starts into that command's module. It fails on a process start
whose program is a bare name (``"uv"``, ``"powershell"``, ``shutil.which(...)``)
instead of ``sys.executable`` or a full path a resolver returned. A command
counts when it is handed to a process-starting call (directly, or through a
function that passes one of its parameters on as the program), returned as a
list, or bound to a command-like name (``cmd``, ``argv``, ...).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
START_MODULES = ("localm.updater", "localm._apply_update")
# The module behind each ``localm <command>`` the update path starts as a subprocess.
SUBCOMMAND_MODULES = {"setup-llama": "localm.setup_llama.cli"}

_EXEC_CALLS = {"run", "call", "check_call", "check_output", "Popen", "system", "popen",
               "create_subprocess_exec", "create_subprocess_shell", "startfile",
               "execv", "execve", "execvp", "execvpe", "execl", "execle", "execlp", "execlpe",
               "spawnv", "spawnve", "spawnvp", "spawnvpe", "spawnl", "spawnle", "spawnlp",
               "spawnlpe", "posix_spawn", "posix_spawnp"}
_PROGRAM_KEYWORDS = ("args", "program", "executable", "path", "file")
_COMMAND_NAME = re.compile(r"^(cmd|argv|args|command|cmdline)$|_(cmd|argv|command)$")
_FUNCS = (ast.FunctionDef, ast.AsyncFunctionDef)


def _module_file(root: Path, name: str):
    base = root.joinpath(*name.split("."))
    for p in (base.with_suffix(".py"), base / "__init__.py"):
        if p.is_file():
            return p
    return None


def _dotted(node):
    """``a.b.c`` for a Name/Attribute chain, else None."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


class _Module:
    def __init__(self, name: str, path: Path):
        self.name = name
        self.package = name if path.name == "__init__.py" else name.rpartition(".")[0]
        self.tree = ast.parse(path.read_text(encoding="utf-8"))
        self.functions, self.classes = {}, {}
        for node in self.tree.body:
            if isinstance(node, _FUNCS):
                self.functions[node.name] = node
            elif isinstance(node, ast.ClassDef):
                methods = []
                for sub in node.body:
                    if isinstance(sub, _FUNCS):
                        self.functions[f"{node.name}.{sub.name}"] = sub
                        methods.append(f"{node.name}.{sub.name}")
                self.classes[node.name] = methods
        self.aliases = self._imports(self._top_level_nodes())

    def _top_level_nodes(self):
        todo, out = list(self.tree.body), []
        while todo:
            node = todo.pop()
            if isinstance(node, (*_FUNCS, ast.ClassDef)):
                continue
            out.append(node)
            todo.extend(ast.iter_child_nodes(node))
        return out

    def _imports(self, nodes) -> dict:
        aliases = {}
        for node in nodes:
            if isinstance(node, ast.Import):
                for a in node.names:
                    if a.asname:
                        aliases[a.asname] = a.name
                    else:
                        head = a.name.split(".")[0]
                        aliases[head] = head
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    parts = self.package.split(".")
                    parts = parts[:len(parts) - (node.level - 1)]
                    base = ".".join(parts + ([base] if base else []))
                for a in node.names:
                    aliases[a.asname or a.name] = f"{base}.{a.name}"
        return aliases

    def function_aliases(self, fn) -> dict:
        return {**self.aliases, **self._imports(ast.walk(fn))}


class _Graph:
    def __init__(self, root: Path):
        self.root, self.modules = root, {}

    def module(self, name: str):
        if name not in self.modules:
            path = _module_file(self.root, name) if name.split(".")[0] == "localm" else None
            self.modules[name] = _Module(name, path) if path else None
        return self.modules[name]

    def resolve(self, dotted: str, depth: int = 0) -> list:
        """The (module, qualname) functions a fully qualified dotted name denotes."""
        parts = dotted.split(".")
        for cut in range(len(parts), 0, -1):
            mod = self.module(".".join(parts[:cut]))
            if mod is None:
                continue
            rest = ".".join(parts[cut:])
            if not rest:
                return []
            if rest in mod.functions:
                return [(mod.name, rest)]
            if rest in mod.classes:
                return [(mod.name, q) for q in mod.classes[rest]]
            head, _, tail = rest.partition(".")
            if head in mod.aliases and depth < 8:
                return self.resolve(".".join(p for p in (mod.aliases[head], tail) if p), depth + 1)
            return []
        return []

    def resolve_expr(self, mod: _Module, aliases: dict, cls: str, node) -> list:
        """The functions a Name/Attribute expression inside *mod* denotes."""
        dotted = _dotted(node)
        if not dotted:
            return []
        head, _, tail = dotted.partition(".")
        if head == "self" and cls:
            name = tail.partition(".")[0]
            return [(mod.name, f"{cls}.{name}")] if f"{cls}.{name}" in mod.functions else []
        if head in aliases:
            return self.resolve(".".join(p for p in (aliases[head], tail) if p))
        if head in mod.functions:
            return [(mod.name, head)]
        if head in mod.classes:
            return [(mod.name, q) for q in mod.classes[head]]
        return []

    def references(self, mod: _Module, qual: str) -> list:
        fn = mod.functions[qual]
        aliases = mod.function_aliases(fn)
        cls = qual.rpartition(".")[0]
        out = []
        for node in ast.walk(fn):
            if isinstance(node, (ast.Attribute, ast.Name)) and isinstance(node.ctx, ast.Load):
                out += self.resolve_expr(mod, aliases, cls, node)
        return out


def localm_subcommands(fn) -> set:
    """The ``localm <command>`` names *fn* builds as ``[sys.executable, "-m", "localm", <command>, ...]``."""
    out = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.List) and len(node.elts) >= 4 and _dotted(node.elts[0]) == "sys.executable":
            words = node.elts[1:4]
            if all(isinstance(w, ast.Constant) and isinstance(w.value, str) for w in words) \
                    and [w.value for w in words[:2]] == ["-m", "localm"]:
                out.add(words[2].value)
    return out


def _reach(root: Path, subcommand_modules=None):
    """(graph, reached functions, subcommands the update path starts that have no module listed)."""
    subcommand_modules = SUBCOMMAND_MODULES if subcommand_modules is None else subcommand_modules
    graph = _Graph(root)
    todo = [(m, q) for m in START_MODULES for q in graph.module(m).functions]
    seen, started, unmapped = {}, set(), set()
    while todo:
        key = todo.pop()
        if key in seen:
            continue
        mod = graph.module(key[0])
        seen[key] = mod.functions[key[1]]
        todo.extend(k for k in graph.references(mod, key[1]) if k not in seen)
        for sub in localm_subcommands(seen[key]):
            name = subcommand_modules.get(sub)
            if name is None:
                unmapped.add(sub)
            elif name not in started and graph.module(name) is not None:
                started.add(name)
                todo.extend((name, q) for q in graph.module(name).functions)
    return graph, seen, unmapped


def reachable_functions(root: Path = ROOT) -> dict:
    """Every (module, qualname) reached from the update modules, mapped to its AST."""
    return _reach(root)[1]


def unmapped_subcommands(root: Path = ROOT) -> set:
    return _reach(root)[2]


def _call_name(func) -> str:
    if isinstance(func, ast.Attribute):
        return func.attr
    return func.id if isinstance(func, ast.Name) else ""


def _is_path_lookup(node) -> bool:
    return isinstance(node, ast.Call) and _call_name(node.func) == "which"


def _bare_word(text: str) -> bool:
    word = text.strip().split()[0] if text.strip() else ""
    return bool(word) and "/" not in word and "\\" not in word and not re.match(r"^[A-Za-z]:", word)


def _bare_program(node, assigned: dict, depth: int = 0) -> bool:
    """A command's program element that is looked up on PATH: a bare name, a
    ``which(...)`` result, or a name bound to either."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return _bare_word(node.value)
    if isinstance(node, ast.BoolOp):
        return any(_bare_program(v, assigned, depth) for v in node.values)
    if isinstance(node, ast.IfExp):
        return any(_bare_program(v, assigned, depth) for v in (node.body, node.orelse))
    if isinstance(node, ast.Name) and depth < 4:
        return any(_bare_program(v, assigned, depth + 1) for v in assigned.get(node.id, ()))
    return _is_path_lookup(node)


def _bare_command(node, assigned: dict, depth: int = 0) -> bool:
    """Whether *node*, used as a command, starts with a bare program."""
    if isinstance(node, (ast.List, ast.Tuple)):
        return bool(node.elts) and _bare_program(node.elts[0], assigned)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _bare_command(node.left, assigned, depth)
    if isinstance(node, ast.JoinedStr):
        first = node.values[0] if node.values else None
        return isinstance(first, ast.Constant) and _bare_program(first, assigned)
    if isinstance(node, (ast.BoolOp, ast.IfExp)):
        parts = node.values if isinstance(node, ast.BoolOp) else (node.body, node.orelse)
        return any(_bare_command(v, assigned, depth) for v in parts)
    if isinstance(node, ast.Name):
        return depth < 4 and any(_bare_command(v, assigned, depth + 1)
                                 for v in assigned.get(node.id, ()))
    return _bare_program(node, assigned)


def _assignments(fn) -> dict:
    out = {}
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    out.setdefault(t.id, []).append(node.value)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)) and isinstance(node.target, ast.Name) \
                and node.value is not None:
            out.setdefault(node.target.id, []).append(node.value)
    return out


def _params(fn, *, method: bool) -> list:
    names = [a.arg for a in fn.args.posonlyargs + fn.args.args]
    return names[1:] if method and names and names[0] in ("self", "cls") else names


def _exec_programs(call) -> list:
    """The arguments of a process-starting *call* that name its program."""
    return call.args[:1] + [k.value for k in call.keywords if k.arg in _PROGRAM_KEYWORDS]


def _sink_argument(call, sink):
    """The argument of *call* that lands on a sink's (index, name) parameter."""
    index, name = sink
    if index < len(call.args):
        return call.args[index]
    return next((k.value for k in call.keywords if k.arg == name), None)


def _call_sinks(graph, sinks: dict, mod, fn, qual: str) -> dict:
    """Call node -> the sink parameters it reaches, for each call in *fn* that
    resolves to a sink function."""
    aliases, cls = mod.function_aliases(fn), qual.rpartition(".")[0]
    out = {}
    for node in ast.walk(fn):
        if isinstance(node, ast.Call) and _call_name(node.func) not in _EXEC_CALLS:
            found = [sinks[t] for t in graph.resolve_expr(mod, aliases, cls, node.func)
                     if t in sinks]
            if found:
                out[node] = found
    return out


def _sinks(graph, reached: dict) -> dict:
    """(module, qualname) -> (index, name) for each function that starts a process
    with one of its own parameters as the program, directly or through another sink."""
    sinks = {}
    changed = True
    while changed:
        changed = False
        for (mod_name, qual), fn in reached.items():
            if (mod_name, qual) in sinks:
                continue
            params = _params(fn, method="." in qual)
            calls = _call_sinks(graph, sinks, graph.module(mod_name), fn, qual)
            for node in ast.walk(fn):
                if not isinstance(node, ast.Call):
                    continue
                if _call_name(node.func) in _EXEC_CALLS:
                    used = _exec_programs(node)
                else:
                    used = [_sink_argument(node, s) for s in calls.get(node, ())]
                hit = next((u.id for u in used if isinstance(u, ast.Name) and u.id in params), None)
                if hit:
                    sinks[(mod_name, qual)] = (params.index(hit), hit)
                    changed = True
                    break
    return sinks


def bare_program_lines(fn, call_sinks=None) -> list:
    """Line numbers in *fn* where a command starts with a bare program.
    *call_sinks* maps a Call node to the sink parameters it reaches."""
    assigned = _assignments(fn)
    hits = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            if _call_name(node.func) in _EXEC_CALLS:
                cands = _exec_programs(node)
            else:
                cands = [_sink_argument(node, s) for s in (call_sinks or {}).get(node, ())]
            if any(c is not None and _bare_command(c, assigned) for c in cands):
                hits.add(node.lineno)
        elif isinstance(node, ast.Return) and isinstance(node.value, ast.List):
            if _bare_command(node.value, assigned):
                hits.add(node.lineno)
        elif isinstance(node, ast.Assign) and isinstance(node.value, (ast.List, ast.Tuple)):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if any(_COMMAND_NAME.search(n) for n in names) and _bare_command(node.value, assigned):
                hits.add(node.lineno)
    return sorted(hits)


def bare_program_sites(root: Path = ROOT) -> list:
    """``module:qualname:line`` for every bare-program command the update path reaches."""
    graph, reached, _ = _reach(root)
    sinks = _sinks(graph, reached)
    sites = []
    for (mod_name, qual), fn in sorted(reached.items(), key=lambda kv: kv[0]):
        calls = _call_sinks(graph, sinks, graph.module(mod_name), fn, qual)
        sites += [f"{mod_name}:{qual}:{line}" for line in bare_program_lines(fn, calls)]
    return sites


def _lines(source: str) -> list:
    return bare_program_lines(ast.parse(source).body[0])


@pytest.mark.parametrize("source, expected", [
    ('def f():\n    subprocess.run(["uv", "pip", "install"])', [2]),
    ('def f():\n    return ["uv", "pip", "install", "-e", "."]', [2]),
    ('def f():\n    argv = ["powershell", "-File", s]\n    subprocess.run(argv)', [2, 3]),
    ('def f():\n    cmd = ("sh", s)\n    return 0', [2]),
    ('def f():\n    os.execvp("git", ["git", "status"])', [2]),
    ('def f():\n    os.system("uv pip install x")', [2]),
    ('def f():\n    subprocess.run(f"nvidia-smi {q}", shell=True)', [2]),
    ('def f():\n    subprocess.run([shutil.which("uv"), "pip"])', [2]),
    ('def f():\n    exe = shutil.which("nvidia-smi") or "nvidia-smi"\n    subprocess.run([exe, "-q"])', [3]),
    ('def f():\n    subprocess.Popen(args=["localm", "gui"])', [2]),
    ('def f():\n    subprocess.run(["sh"] + rest)', [2]),
    ('def f():\n    subprocess.run([sys.executable, "-m", "localm"])', []),
    ('def f():\n    subprocess.run([uv, "pip", "install"])', []),
    ('def f():\n    exe = find_program("sh")\n    subprocess.run([exe, s])', []),
    ('def f():\n    subprocess.run(["/bin/sh", s])', []),
    ('def f():\n    subprocess.run([r"C:\\\\Windows\\\\System32\\\\x.exe"])', []),
    ('def f():\n    return [p, "pip"]', []),
    ('def f():\n    return "vulkan", False, reason', []),
    ('def f():\n    wanted = ("VERSION", "pyproject.toml")', []),
    ('def f():\n    msg = "run uv pip install -e .[gpu] to add it"', []),
    ('def f():\n    subprocess.run(c, cwd=d)', []),
])
def test_the_scanner_finds_what_it_claims_to(source, expected):
    assert _lines(source) == expected


def _fake_tree(tmp_path, files: dict) -> Path:
    for rel, text in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    return tmp_path


def test_the_graph_follows_calls_and_wrappers_into_other_modules(tmp_path):
    root = _fake_tree(tmp_path, {
        "localm/__init__.py": "",
        "localm/updater.py": ("from localm import helper\n"
                              "def apply():\n    helper.go()\n"),
        "localm/_apply_update.py": ("def post():\n"
                                    "    from localm.helper import Runner\n"
                                    "    return Runner().start\n"),
        "localm/helper.py": ("import subprocess\n"
                             "def go():\n    _inner()\n    _wrap(['lspci'])\n    _wrap2(cmd=['wmic'])\n"
                             "def _inner():\n    subprocess.run(['powershell', '-c', 'x'])\n"
                             "def _wrap(cmd):\n    return subprocess.run(cmd).returncode\n"
                             "def _wrap2(timeout=1, cmd=None):\n    return _wrap(cmd)\n"
                             "class Runner:\n"
                             "    def start(self):\n        return self._argv()\n"
                             "    def _argv(self):\n        return ['git', 'pull']\n"
                             "def unrelated():\n    subprocess.run(['sh'])\n"),
    })
    assert bare_program_sites(root) == ["localm.helper:Runner._argv:16",
                                        "localm.helper:_inner:7",
                                        "localm.helper:go:4",
                                        "localm.helper:go:5"]


def test_the_scan_reaches_the_modules_the_update_path_calls():
    reached = {mod for mod, _ in reachable_functions()}
    assert {"localm.updater", "localm._apply_update", "localm.uv_tool",
            "localm.install_manifest", "localm.hwdetect",
            "localm.setup_llama.cli", "localm.setup_llama.cuda"} <= reached


def test_the_graph_follows_a_localm_subcommand_into_its_module(tmp_path):
    root = _fake_tree(tmp_path, {
        "localm/__init__.py": "",
        "localm/updater.py": "",
        "localm/_apply_update.py": ("import sys\n"
                                    "def post():\n    return [sys.executable, '-m', 'localm', 'fix-it']\n"),
        "localm/fixer.py": "import subprocess\ndef main():\n    subprocess.run(['git', 'pull'])\n",
    })
    assert _reach(root, {})[2] == {"fix-it"}
    graph, reached, unmapped = _reach(root, {"fix-it": "localm.fixer"})
    assert unmapped == set() and ("localm.fixer", "main") in reached


def test_every_localm_subcommand_the_update_path_starts_is_scanned():
    assert unmapped_subcommands() == set()


def test_nothing_the_update_path_starts_is_looked_up_on_path():
    assert bare_program_sites() == []

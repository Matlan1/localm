# SPDX-License-Identifier: AGPL-3.0-or-later
"""Find the directory walks a Python module makes over this checkout.

`tree_walks(source)` returns the (line, kind) of every place *source* lists a
folder of this checkout: `rglob`, `glob`, `iterdir` or `walk` on an anchored
path, `os.walk`, `os.listdir`, `os.scandir`, `glob.glob` or `glob.iglob` of
one, a `git ls-files` call given one, and a call into another module that is
handed an anchored folder (`mod.Analyzer(REPO)`), which is how a test runs a
script's own walk over the real tree.

A path is anchored when it derives from `__file__`, from the `__file__` or
`__path__` of a module of this repository, from an attribute of a module named
like a root (`cp.REPO`, `mod.ROOT`), from `importlib.resources.files` of a
package of this repository, or from a name, parameter, parameter default,
`self` attribute or function return value that is anchored. Derivation follows
`/`, `.parent`, `.parents[n]`, `Path()`, `str()`, `sorted()`, `os.path.join`
and the like, never an arbitrary call, so a value computed from an anchored
path (a subprocess result, a parsed file) is not anchored. Names resolve per
function scope; within one scope the analysis is flow-insensitive, so a name
bound once to an anchored path counts as anchored everywhere in that scope.
Stdlib only.
"""

from __future__ import annotations

import ast
import re

_SOURCE_ROOTS = ("localm", "scripts", "tests")
_WALK_METHODS = {"rglob", "glob", "iterdir", "walk"}
_WALK_FUNCTIONS = {("os", "walk"), ("os", "listdir"), ("os", "scandir"), ("glob", "glob"),
                   ("glob", "iglob")}
_ROOT_NAME = re.compile(r"^_*[A-Z_]*(REPO|ROOT|PKG|PACKAGE)[A-Z_]*$")
_PATH_PARENTS = {"parent", "parents"}
_PATH_WRAPPERS = {"Path", "PurePath", "PosixPath", "WindowsPath", "PurePosixPath", "str", "sorted",
                  "list", "tuple", "next", "iter", "reversed", "set", "frozenset"}
_PATH_METHODS = {"resolve", "absolute", "joinpath", "with_name", "with_suffix", "expanduser"}
_OS_PATH_FUNCTIONS = {"join", "dirname", "abspath", "realpath", "normpath"}
_NOT_A_CALLEE = {"self", "cls", "os", "pathlib", "shutil", "subprocess", "monkeypatch", "ap",
                 "parser"}
_NOT_A_WALK = {"relative_to", "is_relative_to", "setattr", "setenv", "add_argument", "startswith"}
_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
_WALK_TOKEN = re.compile(r"\b(rglob|glob|iglob|iterdir|walk|listdir|scandir)\s*\(|ls-files"
                         r"|(REPO|ROOT|PKG|PACKAGE)\b")


def _bound_names(target: ast.AST):
    if isinstance(target, ast.Name):
        yield target.id
    elif isinstance(target, (ast.Tuple, ast.List)):
        for element in target.elts:
            yield from _bound_names(element)
    elif isinstance(target, ast.Starred):
        yield from _bound_names(target.value)


def _own_package_name(node: ast.AST) -> bool:
    return (isinstance(node, ast.Constant) and isinstance(node.value, str)
            and node.value.split(".", 1)[0] in _SOURCE_ROOTS)


def _a_folder(expr: ast.AST) -> bool:
    """Whether *expr* reads as a folder: a root-like name or attribute, a
    `.parent`, or a `/` join whose last literal part has no file suffix."""
    if isinstance(expr, ast.Name):
        return bool(_ROOT_NAME.match(expr.id))
    if isinstance(expr, ast.Attribute):
        return bool(_ROOT_NAME.match(expr.attr)) or expr.attr in _PATH_PARENTS
    if isinstance(expr, ast.Subscript):
        return _a_folder(expr.value)
    if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Div):
        right = expr.right
        if isinstance(right, ast.Constant) and isinstance(right.value, str):
            return "." not in right.value.rsplit("/", 1)[-1]
        return False
    if isinstance(expr, ast.Call):
        func = expr.func
        if isinstance(func, ast.Name) and func.id in ("Path", "str") and expr.args:
            return _a_folder(expr.args[0])
        if isinstance(func, ast.Attribute) and func.attr == "resolve":
            return _a_folder(func.value)
    return False


class _TreeWalks:
    def __init__(self, tree: ast.Module) -> None:
        self.parent: dict[ast.AST, ast.AST] = {}
        self.assigned: dict[ast.AST, set[str]] = {}
        self.bindings: list[tuple[ast.AST, list[str], ast.AST]] = []
        self.attribute_bindings: list[tuple[ast.AST, str, ast.AST]] = []
        self.calls: list[tuple[ast.AST, ast.Call]] = []
        self.functions: dict[str, list[ast.AST]] = {}
        self.classes: dict[str, list[ast.AST]] = {}
        self.own: set[str] = set()
        self.tainted: set[tuple[ast.AST, str]] = set()
        self.tainted_attributes: set[str] = set()
        self.anchored_functions: set[str] = set()
        self._index(tree, tree)

    def _index(self, node: ast.AST, scope: ast.AST) -> None:
        self.assigned.setdefault(scope, set())
        for child in ast.iter_child_nodes(node):
            inner = scope
            if isinstance(child, ast.ClassDef):
                self.assigned[scope].add(child.name)
                self.classes[child.name] = [
                    f for f in child.body
                    if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef)) and f.name == "__init__"]
            if isinstance(child, _SCOPES):
                inner = child
                self.parent[child] = scope
                args = child.args
                self.assigned[child] = {a.arg for a in args.posonlyargs + args.args + args.kwonlyargs
                                        + [x for x in (args.vararg, args.kwarg) if x]}
                if not isinstance(child, ast.Lambda):
                    self.functions.setdefault(child.name, []).append(child)
                    self.assigned[scope].add(child.name)
                positional = args.posonlyargs + args.args
                for param, default in zip(positional[len(positional) - len(args.defaults):],
                                          args.defaults, strict=True):
                    self.bindings.append((child, [param.arg], default))
                for param, default in zip(args.kwonlyargs, args.kw_defaults, strict=True):
                    if default is not None:
                        self.bindings.append((child, [param.arg], default))
            pairs: list[tuple[ast.AST, ast.AST]] = []
            if isinstance(child, ast.Assign):
                pairs = [(t, child.value) for t in child.targets]
            elif isinstance(child, (ast.AnnAssign, ast.AugAssign)) and child.value is not None:
                pairs = [(child.target, child.value)]
            elif isinstance(child, (ast.For, ast.AsyncFor, ast.comprehension)):
                pairs = [(child.target, child.iter)]
            elif isinstance(child, ast.NamedExpr):
                pairs = [(child.target, child.value)]
            elif isinstance(child, ast.withitem) and child.optional_vars is not None:
                pairs = [(child.optional_vars, child.context_expr)]
            for target, value in pairs:
                if (isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)
                        and target.value.id in ("self", "cls")):
                    self.attribute_bindings.append((scope, target.attr, value))
                    continue
                names = list(_bound_names(target))
                self.assigned[scope].update(names)
                self.bindings.append((scope, names, value))
            if isinstance(child, (ast.Import, ast.ImportFrom)):
                relative = isinstance(child, ast.ImportFrom) and child.level > 0
                for alias in child.names:
                    bound = alias.asname or alias.name.split(".", 1)[0]
                    self.assigned[scope].add(bound)
                    full = (child.module or "") if isinstance(child, ast.ImportFrom) else alias.name
                    if relative or full.split(".", 1)[0] in _SOURCE_ROOTS:
                        self.own.add(bound)
            if isinstance(child, ast.Call):
                self.calls.append((scope, child))
            self._index(child, inner)

    def _owner(self, scope: ast.AST, name: str) -> ast.AST:
        while name not in self.assigned.get(scope, ()) and scope in self.parent:
            scope = self.parent[scope]
        return scope

    def anchored(self, scope: ast.AST, expr: ast.AST) -> bool:
        """Whether *expr*, evaluated in *scope*, is a path inside this checkout."""
        def again(e: ast.AST) -> bool:
            return self.anchored(scope, e)

        if isinstance(expr, ast.Name):
            return expr.id == "__file__" or (self._owner(scope, expr.id), expr.id) in self.tainted
        if isinstance(expr, ast.Attribute):
            base = expr.value
            while isinstance(base, ast.Attribute):
                base = base.value
            if expr.attr in ("__file__", "__path__"):
                return isinstance(base, ast.Name) and base.id in self.own
            if isinstance(expr.value, ast.Name):
                if expr.value.id in ("self", "cls"):
                    return expr.attr in self.tainted_attributes
                if _ROOT_NAME.match(expr.attr):
                    return True
            return expr.attr in _PATH_PARENTS and again(expr.value)
        if isinstance(expr, ast.Subscript):
            return again(expr.value)
        if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Div):
            return again(expr.left)
        if isinstance(expr, (ast.Tuple, ast.List, ast.Set)):
            return any(again(e) for e in expr.elts)
        if isinstance(expr, ast.Starred):
            return again(expr.value)
        if isinstance(expr, ast.IfExp):
            return again(expr.body) or again(expr.orelse)
        if isinstance(expr, ast.BoolOp):
            return any(again(v) for v in expr.values)
        if isinstance(expr, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
            return again(expr.elt)
        if not isinstance(expr, ast.Call):
            return False
        func = expr.func
        args = list(expr.args) + [k.value for k in expr.keywords]
        if isinstance(func, ast.Name):
            if func.id in self.anchored_functions:
                return True
            if func.id in _PATH_WRAPPERS:
                return any(again(x) for x in args)
            return func.id == "files" and bool(expr.args) and _own_package_name(expr.args[0])
        if isinstance(func, ast.Attribute):
            if func.attr == "files" and expr.args and _own_package_name(expr.args[0]):
                return True
            if (func.attr in _PATH_WRAPPERS and isinstance(func.value, ast.Name)
                    and func.value.id in ("pathlib", "builtins")):
                return any(again(x) for x in args)
            if func.attr in _PATH_METHODS or func.attr in _WALK_METHODS:
                return again(func.value)
            if (isinstance(func.value, ast.Attribute) and func.value.attr == "path"
                    and func.attr in _OS_PATH_FUNCTIONS):
                return any(again(x) for x in args)
        return False

    def _propagate(self) -> None:
        changed = True

        def mark(scope: ast.AST, name: str) -> None:
            nonlocal changed
            key = (self._owner(scope, name), name)
            if key not in self.tainted:
                self.tainted.add(key)
                changed = True

        while changed:
            changed = False
            for scope, names, value in self.bindings:
                if self.anchored(scope, value):
                    for name in names:
                        mark(scope, name)
            for scope, attr, value in self.attribute_bindings:
                if attr not in self.tainted_attributes and self.anchored(scope, value):
                    self.tainted_attributes.add(attr)
                    changed = True
            for name, defs in self.functions.items():
                if name in self.anchored_functions:
                    continue
                if any(isinstance(r, ast.Return) and r.value is not None and self.anchored(f, r.value)
                       for f in defs for r in ast.walk(f)):
                    self.anchored_functions.add(name)
                    changed = True
            for scope, call in self.calls:
                func = call.func
                if isinstance(func, ast.Name):
                    targets = [(f, 0) for f in self.functions.get(func.id, ())]
                    targets += [(f, 1) for f in self.classes.get(func.id, ())]
                elif (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name)
                      and func.value.id in ("self", "cls")):
                    targets = [(f, 1) for f in self.functions.get(func.attr, ())]
                else:
                    continue
                for f, skip in targets:
                    params = [a.arg for a in f.args.posonlyargs + f.args.args][skip:]
                    for i, arg in enumerate(call.args):
                        if i < len(params) and self.anchored(scope, arg):
                            mark(f, params[i])
                    for kw in call.keywords:
                        if kw.arg and self.anchored(scope, kw.value):
                            mark(f, kw.arg)
            for defs in self.functions.values():
                for f in defs:
                    for a in f.args.posonlyargs + f.args.args + f.args.kwonlyargs:
                        if a.arg in self.anchored_functions:
                            mark(f, a.arg)

    def walks(self) -> list[tuple[int, str]]:
        self._propagate()
        found: list[tuple[int, str]] = []
        for scope, node in self.calls:
            func = node.func
            if (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name)
                    and (func.value.id, func.attr) in _WALK_FUNCTIONS):
                if node.args and self.anchored(scope, node.args[0]):
                    found.append((node.lineno, f"{func.value.id}.{func.attr}"))
                continue
            if isinstance(func, ast.Attribute) and func.attr in _WALK_METHODS \
                    and self.anchored(scope, func.value):
                found.append((node.lineno, func.attr))
                continue
            arguments = list(node.args) + [k.value for k in node.keywords]
            literals = [c.value for c in ast.walk(node) if isinstance(c, ast.Constant)]
            if "ls-files" in literals and any(self.anchored(scope, a) for a in arguments):
                found.append((node.lineno, "git ls-files"))
                continue
            if (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name)
                    and func.value.id not in _NOT_A_CALLEE
                    and func.attr not in _PATH_METHODS | _WALK_METHODS | _PATH_WRAPPERS | _NOT_A_WALK
                    and not self.anchored(scope, func.value)
                    and any(_a_folder(a) and self.anchored(scope, a) for a in arguments)):
                found.append((node.lineno, f"hands a checkout folder to {func.value.id}.{func.attr}"))
        return sorted(set(found))


def tree_walks(source: str) -> list[tuple[int, str]]:
    """(line, kind) of every walk *source* makes over a folder of this
    checkout, in line order. Empty for source that does not parse."""
    if not _WALK_TOKEN.search(source):
        return []
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return []
    return _TreeWalks(tree).walks()


def script_names(source: str) -> set[str]:
    """The file names ending in `.py` that *source* spells as a whole string
    literal or as the last part of one (`"scripts/x.py"`, `/ "x.py"`)."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return set()
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                and node.value.endswith(".py") and " " not in node.value:
            out.add(node.value.rsplit("/", 1)[-1])
    return out

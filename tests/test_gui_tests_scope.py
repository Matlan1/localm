# SPDX-License-Identifier: AGPL-3.0-or-later
"""The gui-tests job's pull-request scope in .github/workflows/ci.yml covers
every repository file a tests-js test reads.

On an unlabelled pull request the jsdom suite runs only when a changed path
matches the scope step's `grep -qE` pattern. A test under tests-js/ that reads
a file outside that pattern (a built-in plugin's static/ folder, a plugin
manifest, the changelog) goes unrun on the pull request that breaks it. A file
is also covered when another job's scope pattern matches it and that job runs
the test that reads it.

The references are read from the test sources: `join`/`resolve` calls and
`new URL(...)` against the test's own folder or a constant derived from it,
where a non-literal path segment followed by literal ones matches any one
segment, and string literals that spell a tracked path from the repository
root or from tests-js/. A path built from a template string, by string
concatenation, or ending in a non-literal segment is not read.
"""

from __future__ import annotations

import posixpath
import re
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
_CI = REPO_ROOT / ".github" / "workflows" / "ci.yml"
_JS_ROOT = "tests-js"
_SCRIPT_SUFFIXES = (".mjs", ".js", ".cjs")

_STR = r"""(?:"[^"\n]*"|'[^'\n]*')"""
_ARG = rf"(?:{_STR}|[A-Za-z_$][\w$]*)"
_HERE = "__here__"
_HERE_CALL = re.compile(r"dirname\(\s*fileURLToPath\(\s*import\.meta\.url\s*\)\s*\)")
_TO_PATH = re.compile(r"fileURLToPath\(\s*([A-Za-z_$][\w$]*)\s*\)")
_CALL = re.compile(rf"\b(?:path\.)?(?:join|resolve)\(\s*([A-Za-z_$][\w$]*)((?:\s*,\s*{_ARG})+)\s*,?\s*\)")
_URL = re.compile(rf"new URL\(\s*({_ARG})\s*,\s*(import\.meta\.url|[A-Za-z_$][\w$]*)\s*\)")
_ASSIGN = re.compile(r"\b(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*")
_SEGMENT = re.compile(_ARG)
_LITERAL = re.compile(_STR)


def _workflow() -> dict:
    return yaml.safe_load(_CI.read_text(encoding="utf-8"))


def _job_scope(job: dict) -> re.Pattern[str] | None:
    """The one `grep -qE` pattern of *job*'s scope step, or None when it has
    no scope step or the step does not test one such pattern."""
    scope = next((s for s in job.get("steps", []) if s.get("id") == "scope"), None)
    found = re.findall(r"grep -qE '([^']+)'", scope["run"]) if scope else []
    return re.compile(found[0]) if len(found) == 1 else None


def _scope_pattern() -> re.Pattern[str]:
    pattern = _job_scope(_workflow()["jobs"]["gui-tests"])
    assert pattern is not None, "the gui-tests scope step no longer tests one grep -qE pattern"
    return pattern


def _other_scopes() -> list[tuple[re.Pattern[str], str]]:
    """(scope pattern, text of every step's run) for each job but gui-tests
    that has a scope step."""
    out = []
    for name, job in _workflow()["jobs"].items():
        pattern = _job_scope(job)
        if name != "gui-tests" and pattern is not None:
            out.append((pattern, "\n".join(str(s.get("run", "")) for s in job["steps"])))
    return out


def _tracked() -> list[str]:
    out = subprocess.run(["git", "-C", str(REPO_ROOT), "ls-files", "-z"], capture_output=True,
                         check=True).stdout.decode("utf-8")
    return [p for p in out.split("\0") if p]


def _unquote(token: str) -> str | None:
    """The text of a string literal token, or None for an identifier."""
    return token[1:-1] if token[:1] in "\"'" else None


def _resolve(base: str, parts: list[str | None]) -> str | None:
    """*base* joined with *parts*, normalised; a None part matches one
    segment. None when the last part is not a literal."""
    if not parts or parts[-1] is None:
        return None
    path = base
    for part in parts:
        path = posixpath.join(path, "*" if part is None else part)
    return posixpath.normpath(path)


def references(rel: str, text: str) -> list[str]:
    """The repository-relative paths (with `*` for one non-literal segment)
    the tests-js source *rel* reads through `join`, `resolve` or `new URL`,
    resolved against its own folder and the constants it derives from that
    folder, plus its string literals that start with `../` or `localm/`. A
    path outside the repository is left out."""
    here = posixpath.dirname(rel)
    text = _TO_PATH.sub(r"\1", _HERE_CALL.sub(_HERE, text))
    bases: dict[str, str] = {"__dirname": here, _HERE: here}
    for m in _ASSIGN.finditer(text):
        name, rest = m.group(1), text[m.end():m.end() + 300]
        expr = rest.split(";", 1)[0].strip()
        found: str | None = None
        if expr == _HERE:
            found = here
        elif call := _CALL.match(expr):
            if call.group(1) in bases:
                found = _resolve(bases[call.group(1)], [_unquote(t) for t in _SEGMENT.findall(call.group(2))])
        elif url := _URL.match(expr):
            anchor = here if url.group(2) == "import.meta.url" else bases.get(url.group(2))
            if anchor is not None:
                found = _resolve(anchor, [_unquote(url.group(1))])
        if found is not None:
            bases[name] = found
    out: list[str | None] = []
    for call in _CALL.finditer(text):
        if call.group(1) in bases:
            out.append(_resolve(bases[call.group(1)], [_unquote(t) for t in _SEGMENT.findall(call.group(2))]))
    for url in _URL.finditer(text):
        anchor = here if url.group(2) == "import.meta.url" else bases.get(url.group(2))
        if anchor is not None:
            out.append(_resolve(anchor, [_unquote(url.group(1))]))
    for token in _LITERAL.findall(text):
        literal = token[1:-1]
        if literal.startswith("../"):
            out.append(posixpath.normpath(posixpath.join(here, literal)))
        elif literal.startswith("localm/"):
            out.append(posixpath.normpath(literal))
    return [p for p in out if p is not None and p != ".." and not p.startswith("../")]


def _read_files(tracked: list[str]) -> dict[str, set[str]]:
    """{tracked file outside tests-js: tests-js sources that read it}."""
    sources = [p for p in tracked if p.startswith(_JS_ROOT + "/") and p.endswith(_SCRIPT_SUFFIXES)
               and "/node_modules/" not in p]
    files = set(tracked)
    read: dict[str, set[str]] = {}
    for src in sources:
        text = (REPO_ROOT / src).read_text(encoding="utf-8")
        for ref in references(src, text):
            if ref.startswith(_JS_ROOT + "/"):
                continue
            if "*" in ref:
                one = re.compile("[^/]*".join(re.escape(part) for part in ref.split("*")) + r"\Z")
                hits = [f for f in files if one.match(f)]
            else:
                hits = [ref] if ref in files else []
            for f in hits:
                read.setdefault(f, set()).add(src)
    return read


@pytest.fixture(scope="module")
def read_files() -> dict[str, set[str]]:
    return _read_files(_tracked())


def test_the_reference_reader_finds_the_files_the_motivating_tests_read(read_files):
    assert "tests-js/jobs.test.mjs" in read_files["localm/plugins/builtin/jobs/static/jobs.js"]
    assert "tests-js/data-table-one-row-per-record.test.mjs" in \
        read_files["localm/plugins/builtin/jobs/static/jobs.js"]
    assert "tests-js/document-render.test.mjs" in read_files["CHANGELOG.md"]
    assert "tests-js/i18n.test.mjs" in read_files["localm/plugins/builtin/jobs/plugin.toml"]
    assert any(f.startswith("localm/plugins/gui/static/") for f in read_files)


def test_every_file_a_gui_test_reads_is_inside_the_gui_tests_scope(read_files):
    scope = _scope_pattern()
    others = _other_scopes()
    outside = {f: sorted(srcs) for f, srcs in sorted(read_files.items())
               if not scope.match(f)
               and not all(any(p.match(f) and src in run for p, run in others) for src in srcs)}
    assert not outside, (
        "tests-js reads these files, but a pull request changing one does not run the GUI "
        f"tests; widen the gui-tests scope pattern in ci.yml: {outside}")


def test_the_scope_leaves_a_python_module_out_and_takes_a_plugin_static_file_in():
    scope = _scope_pattern()
    assert not scope.match("localm/plugins/builtin/jobs/runner.py")
    assert not scope.match("localm/inference/protocol.py")
    assert scope.match("localm/plugins/builtin/jobs/static/jobs.js")
    assert scope.match("localm/plugins/builtin/jobs/plugin.toml")


def test_references_resolve_constants_urls_literals_and_a_variable_segment():
    text = (
        'const HERE = dirname(fileURLToPath(import.meta.url));\n'
        'const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");\n'
        'const STATIC = join(HERE, "..", "localm", "plugins", "gui", "static");\n'
        'const VENDOR = new URL("../localm/plugins/gui/static/vendor/", import.meta.url);\n'
        'readFileSync(join(STATIC, "app", "main.js"));\n'
        'readFileSync(new URL("marked.min.js", VENDOR));\n'
        'readFileSync(join(ROOT, "localm", "plugins", "builtin", name, "plugin.toml"));\n'
        'readFileSync(join(HERE, "fixtures", "a.html"));\n'
        'readFileSync(join(ROOT, anyFile));\n'
        'const LIST = ["localm/plugins/builtin/jobs/static/jobs.js"];\n'
        'readFileSync(new URL("../../outside.txt", import.meta.url));\n')
    refs = references("tests-js/x.test.mjs", text)
    assert "localm/plugins/gui/static/app/main.js" in refs
    assert "localm/plugins/gui/static/vendor/marked.min.js" in refs
    assert "localm/plugins/builtin/*/plugin.toml" in refs
    assert "localm/plugins/builtin/jobs/static/jobs.js" in refs
    assert "tests-js/fixtures/a.html" in refs
    assert "*" not in refs
    assert not any(r.startswith("..") or "outside" in r for r in refs)

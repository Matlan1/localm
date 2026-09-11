# SPDX-License-Identifier: AGPL-3.0-or-later
"""NEW-DEBUG-LOG-PROMISED-BUT-ABSENT: `_runner.py`, `_hf_runner.py` and
`_embedder_runner.py` all appended "(full trace in the debug log)" to a
native-fault message UNCONDITIONALLY. The trace itself goes to
`logger.error`, but a debug log FILE only exists once `enable_debug()` has
run, which is off by default - so on a default install the message named a
file that was never created, misdirecting every default-mode user chasing
a crash. Verified live: a positive control with `--debug` on DOES produce
that log line; a default-mode run does not.

A second instance of the same defect survived that fix: each of those three
files, plus `mtmd.py`, ALSO raised a second, differently-worded message
("... may be hung (see the debug log)." / "See the debug log for the native
reason.") that hardcoded the same unconditional claim on a different code
path (a load/eval timeout or failure, not the crash-trace path). The
narrower guard below (checking only the literal "(full trace in the debug
log)" string, and only in the three original files) could not see it.
`TestNoHardcodedDebugLogClaimAnywhere` is the general sweep that closes that
gap for any current or future site under `localm/inference/`.
"""

import ast
import re
from pathlib import Path

import localm.debuglog as debuglog


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


class TestNativeFaultHint:
    def test_debug_off_does_not_promise_a_debug_log(self, monkeypatch):
        monkeypatch.delenv("LOCALM_DEBUG", raising=False)
        hint = debuglog.native_fault_hint()
        assert "debug log" not in hint, (
            "must not claim a debug log exists when debug_enabled() is False")
        assert "--debug" in hint, "must say how to actually get one"

    def test_debug_on_names_the_debug_log(self, monkeypatch):
        monkeypatch.setenv("LOCALM_DEBUG", "1")
        assert debuglog.native_fault_hint() == "full trace in the debug log"


class TestAllFourSitesUseTheSharedHint:
    """Source-level guard: the four sites this bug was found in must all
    route through native_fault_hint() rather than re-hardcoding the old
    unconditional text (or a future fifth site being added the old way)."""

    FILES = [
        "localm/inference/_embedder_runner.py",
        "localm/inference/backends/llamacpp/_runner.py",
        "localm/inference/backends/_hf_runner.py",
        "localm/inference/backends/llamacpp/mtmd.py",
    ]

    def test_no_hardcoded_unconditional_debug_log_claim(self):
        for rel in self.FILES:
            body = (_repo_root() / rel).read_text(encoding="utf-8")
            assert "(full trace in the debug log)" not in body, (
                f"{rel} hardcodes the unconditional claim again - use "
                "native_fault_hint() instead")

    def test_every_site_calls_native_fault_hint(self):
        for rel in self.FILES:
            body = (_repo_root() / rel).read_text(encoding="utf-8")
            assert "native_fault_hint()" in body, (
                f"{rel} no longer routes its native-fault message through "
                "native_fault_hint()")


class TestNoHardcodedDebugLogClaimAnywhere:
    """General sweep, not tied to the four FILES above: no raised message
    anywhere under localm/inference/ may claim "(see/See the debug log)"
    unconditionally. Walks the AST rather than the raw text so an ordinary
    explanatory comment or a docstring mentioning the phrase is not a false
    positive - only a string literal that could reach a user (an f-string or
    plain string constant that is not a module/class/function docstring) is
    checked."""

    _PHRASE = re.compile(r"see the debug log", re.IGNORECASE)

    def _docstring_constant_ids(self, tree: ast.AST) -> set:
        ids = set()
        for node in ast.walk(tree):
            if not isinstance(
                node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
            ):
                continue
            body = node.body
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                ids.add(id(body[0].value))
        return ids

    def test_no_non_comment_non_docstring_literal_claims_a_debug_log(self):
        root = _repo_root() / "localm" / "inference"
        hits = []
        for path in sorted(root.rglob("*.py")):
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
            docstring_ids = self._docstring_constant_ids(tree)
            for node in ast.walk(tree):
                if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
                    continue
                if id(node) in docstring_ids:
                    continue
                if self._PHRASE.search(node.value):
                    rel = path.relative_to(_repo_root())
                    hits.append(f"{rel}:{node.lineno}: {node.value!r}")
        assert not hits, (
            "hardcoded, unconditional debug-log claim(s) found outside comments/"
            "docstrings - route through debuglog.native_fault_hint() instead:\n"
            + "\n".join(hits)
        )

#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Select the test files a change affects, so a targeted run is computed rather
than guessed.

A test file is selected when any of these holds:
  - it changed itself;
  - it imports a changed module directly (with --depth N, also a module that
    imports the changed one, up to N hops; when that selection would exceed
    --max-share, the importer that the most test files import directly (a
    hub such as the app factory or the config module) is not followed, then
    the next, until it fits; nothing is left out when the selection exceeds
    --max-share even at depth 0, and a changed module is never left out);
  - it names a route path a changed module registers, or a module that
    imports a changed module directly registers, at every depth (the static
    prefix of the path, so a test that calls the route by URL is found even
    when it never imports the module; a prefix cut by a placeholder after its
    first segment, such as `/api/` of `/api/{media}/workflows`, names no
    route);
  - it names a changed module by its dotted name (a monkeypatch target, an
    import written as a string), a changed script by file name, or a changed
    non-Python file by file name;
  - a change to a module inside a package listed in _REEXPORT_FACADES also
    counts as a change to that package, and a change to a module inside a
    package mapped in _REEXPORT_MODULE_FACADES also counts as a change to the
    module it is mapped from, for both rules above, and that module counts as
    importing every module the package's modules import;
  - it is listed in _TREE_SCANNERS with a pattern a changed path matches (it
    lists a folder of the checkout instead of importing the changed file);
  - it imports a changed dependency, names it in a string literal, imports a
    module that imports it (with --depth N, also that module's importers, up
    to N hops), or sits under the folder of a conftest.py that imports it. A
    changed dependency is a distribution whose requirement in pyproject.toml
    or whose entry in uv.lock differs, and every package in uv.lock that
    depends on one, transitively.

A version bump selects the tests that read the version instead of every test
file: the files that spell the project version (VERSION, localm/__init__.py
when only its __version__ string differs, the [project] version of
pyproject.toml, the version of the project's own editable entry in uv.lock)
select the test files that reference __version__, name the VERSION, pyproject.toml,
uv.lock or CHANGELOG.md files, or spell the old or the new version.

A change to tests/conftest.py affects every test file. So does a change to
pyproject.toml or uv.lock outside the [project] requirement lists, the
[tool.ruff], [tool.basedpyright], [tool.mutmut] and [tool.uv] tables (a
change to [tool.mutmut] selects the tests listed in _TOOL_TABLE_READERS), the lock's
resolution-markers and supported-markers and the locked packages, either file
missing or unparsable on either side, a listed
pyproject.toml or uv.lock that does not differ from the base (a committed
change named with --files), and a changed dependency the project declares
outside the dev extra that the rules above reach no test file from. A changed
dependency declared only in the dev extra selects only the test files those
rules reach, possibly none.

Changes are read from git: the diff from the merge base with --base (default
origin/master) to HEAD, plus staged, unstaged and untracked work. --files takes
an explicit list instead.

    python scripts/affected_tests.py                   list the selection
    python scripts/affected_tests.py --why             list with the reasons
    python scripts/affected_tests.py --files a.py b.py
    pytest $(python scripts/affected_tests.py) -m "not integration"

The selection is printed one path per line so it can be substituted into a
pytest command line; the summary goes to stderr. Stdout never leaves pytest
with an empty or a whole-suite argument list: when nothing is affected the one
line printed is tests/NO_TEST_FILE_IS_AFFECTED; when the selection is wider
than --max-share of all test files (the change touches a module most tests
import, so a run that wide is not a targeted run) the one line printed is
tests/SELECTION_TOO_WIDE_FOR_A_TARGETED_RUN_SEE_STDERR and the exit status is
3, with --list-wide printing the selection instead; when the script itself
fails the one line printed is tests/AFFECTED_TESTS_FAILED_SEE_STDERR and the
exit status is 1. None of those paths exists, so pytest refuses each one.

Stdlib only.
"""

from __future__ import annotations

import argparse
import ast
import fnmatch
import functools
import json
import os
import re
import subprocess
import sys
import tomllib
import traceback
from importlib import metadata
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
_ROUTE_METHODS = {"get", "post", "put", "delete", "patch", "websocket", "api_route"}
_EVERYTHING = {"tests/conftest.py"}
# Packages whose callers import and patch the package itself, which re-exports
# its submodules' names. Each counts as importing every module its own modules
# import.
_REEXPORT_FACADES = ("localm.setup_llama", "localm.bugreport")
_DEPENDENCY_FILES = ("pyproject.toml", "uv.lock")
_UNREAD_TOOL_TABLES = ("ruff", "basedpyright")
_RESOLVER_TOOL_TABLES = ("uv",)
# Tool tables of pyproject.toml that only the listed test files read. A change
# confined to one selects those tests instead of every test file.
_TOOL_TABLE_READERS = {
    "mutmut": ("tests/test_check_mutation_floors.py", "tests/test_mutation_scope.py"),
}
_LOCK_FORK_KEYS = ("resolution-markers", "supported-markers")
_VERSION_FILE = "VERSION"
_PACKAGE_INIT = "localm/__init__.py"
_PROJECT_LOCK_SOURCE = {"editable": "."}
_VERSION_FILE_NAMES = ("pyproject.toml", "uv.lock", "CHANGELOG.md")
_VERSION_FILE_WORD = re.compile(r"(?<![A-Za-z0-9_])VERSION(?![A-Za-z0-9_])")
_DEV_EXTRA = "dev"
_REQUIREMENT_KEYS = ("dependencies", "optional-dependencies")
# Modules whose callers import and patch the module itself, which re-exports the
# names of the modules in the package mapped to it.
_REEXPORT_MODULE_FACADES = {"localm.rag.store": "localm.rag._store"}
# Test files that list a folder of the checkout instead of importing the module
# that changed, so neither the import graph nor a file-name match reaches a file
# added there. Each maps to the fnmatch patterns of the repository-relative
# paths it reads (`*` also matches `/`) and what it does with them. A file
# added, changed or deleted at a matching path selects the test.
# tests/test_tree_scanners_registered.py fails when a test file walks a folder
# of the checkout and is in neither this table nor _NOT_TREE_SCANNERS.
_PY = ("localm/*.py", "tests/*.py", "scripts/*.py")
_LOCALM_PY = ("localm/*.py",)
_TESTS_PY = ("tests/*.py",)
_GUI_STATIC = "localm/plugins/gui/static/"
_GUI_JS = (_GUI_STATIC + "*.js",)
_BUILTIN_PLUGINS = ("localm/plugins/builtin/*",)
_WORKFLOWS = (".github/workflows/*",)
_RERANK_CORPUS = ("tests/fixtures/rag_rerank_eval/*",)
_TREE_SCANNERS: dict[str, tuple[tuple[str, ...], str]] = {
    "tests/test_check_pins.py": (_LOCALM_PY, "greps every tracked localm/*.py for pin-shaped constants"),
    "tests/test_streaming_response_closes_body.py": (_LOCALM_PY, "reads every localm/*.py for StreamingResponse uses"),
    "tests/test_update_steps_run_in_the_new_version.py": (
        _LOCALM_PY, "scans every localm module for updater.apply calls"),
    "tests/test_amd_windows_torch_pin_one_source.py": (
        ("localm/*", "docs/*", "scripts/*", "installer/*", "docker/*", "README.md", "pyproject.toml",
         "uv.lock", "setup.bat", "setup.sh", "setup-gui.bat", "setup-gui.sh"),
        "greps the checkout for AMD ROCm torch builds"),
    "tests/test_no_gitignored_path_leak.py": (_PY, "scans the string literals of every tracked .py"),
    "tests/test_no_stray_encoding_cookie.py": (_PY, "reads the first lines of every .py"),
    "tests/test_check_hygiene.py": (("*",), "runs the hygiene checks over every tracked file"),
    "tests/test_check_hygiene_layering.py": (_LOCALM_PY, "checks import direction over every localm module"),
    "tests/test_check_hygiene_response_keys.py": (_LOCALM_PY, "checks the JSON keys of every tracked route"),
    "tests/test_check_hygiene_import_baseline.py": (_LOCALM_PY, "counts the deferred imports of every localm module"),
    "tests/test_check_hygiene_import_cycles.py": (_LOCALM_PY, "checks every localm module for import cycles"),
    "tests/test_check_hygiene_sw_cache.py": ((_GUI_STATIC + "*",), "checks the service worker against the static tree"),
    "tests/test_bump_uv_pin.py": (("*",), "lists every tracked file outside tests/ and scripts/ that pins uv"),
    "tests/test_bump_koboldcpp_pin.py": (("*",), "greps every tracked file for the koboldcpp pin"),
    "tests/test_bump_sdcpp_pin.py": (("*",), "lists every tracked file for the sd.cpp pin"),
    "tests/test_bump_tts_bundle.py": (("localm/plugins/builtin/tts/static/vendor/*",),
                                      "lists the vendored TTS runtime"),
    "tests/test_bump_vendored_js.py": ((_GUI_STATIC + "vendor/*",), "copies the vendored GUI libraries"),
    "tests/test_merge_policy.py": (("*",), "checks every merge-policy pattern against the tracked files"),
    "tests/test_native_stderr_redirect_scope.py": (_LOCALM_PY, "walks the AST of every localm .py for stderr redirects"),
    "tests/test_terminal_control_safety.py": (_LOCALM_PY, "walks the AST of every localm .py for rich Console imports"),
    "tests/test_model_name_gate.py": (_LOCALM_PY, "reads every localm .py for trust_remote_code=True"),
    "tests/test_no_unguarded_json_parse.py": (_LOCALM_PY, "walks the AST of every localm .py for unguarded JSON parses"),
    "tests/test_owner_predicate_single_owner.py": (_LOCALM_PY, "walks the AST of every localm .py for inline owner checks"),
    "tests/test_no_bare_secret_chmod.py": (_LOCALM_PY, "walks the AST of every localm .py for chmod 0o600 calls"),
    "tests/test_private_write_survives_a_reader.py": (
        _LOCALM_PY, "walks the AST of every localm .py for owner-only store writes"),
    "tests/test_no_untracked_authorization_header_writes.py": (
        _LOCALM_PY, "walks the AST of every localm .py for Authorization header writes"),
    "tests/test_hwdetect_recommended_is_legacy.py": (_LOCALM_PY, "walks the AST of every localm .py for .recommended reads"),
    "tests/test_install_manifest.py": (_LOCALM_PY, "walks the AST of every localm .py for data-dir children"),
    "tests/test_gui_js_route_contract.py": (_LOCALM_PY + _GUI_JS,
                                            "reads every localm .py for declared routes and every GUI script"),
    "tests/test_ctrlc_clean_stop.py": (_LOCALM_PY, "reads every localm .py for parent-death watchdog installs"),
    "tests/test_residency.py": (_LOCALM_PY, "greps every localm .py for active_requests mutations"),
    "tests/test_discover.py": (_LOCALM_PY, "walks the AST of every localm .py for calls to the discovery helpers"),
    "tests/test_outbound_call_sites_contract.py": (_LOCALM_PY, "classifies every outbound call site under localm/"),
    "tests/test_one_server_switch_path.py": (_LOCALM_PY, "checks every localm module for a second server switch path"),
    "tests/test_session_mode_sentinel.py": (_LOCALM_PY, "scans every localm .py for session-mode writes"),
    "tests/test_settings_place_names.py": (_LOCALM_PY + ("docs/*", "README.md"),
                                           "checks the Settings places every localm .py and docs page names"),
    "tests/test_fault_trace_threads.py": (_LOCALM_PY, "walks the AST of every localm .py for faulthandler calls"),
    "tests/test_media_backend_capabilities.py": (_LOCALM_PY, "checks which localm modules reach the media generators"),
    "tests/test_no_bare_uv_command.py": (_LOCALM_PY, "scans every localm .py for a bare uv command"),
    "tests/test_media_delete_outputs_setting.py": (_LOCALM_PY,
                                                   "checks every media call site under localm/ for delete_outputs"),
    "tests/test_sysprog.py": (_LOCALM_PY, "walks the AST of every localm .py for system program calls"),
    "tests/test_speech_engine.py": (_LOCALM_PY, "walks the AST of every localm .py for speech engine calls"),
    "tests/test_remote_backend_gate.py": (_LOCALM_PY, "scans every localm .py for remote backend calls"),
    "tests/test_event_loop_offload.py": (_LOCALM_PY, "runs the event-loop offload sweep over localm/"),
    "tests/test_config_reads_off_the_loop.py": (_LOCALM_PY,
                                                "walks the AST of every localm .py for store writes in async functions"),
    "tests/test_mtp_arch_allowlist.py": (_LOCALM_PY, "runs the MTP architecture gate over localm/"),
    "tests/test_config_write.py": (("localm/plugins/*.py",), "scans every plugin module for direct config writes"),
    "tests/test_rag_store_live_lookups.py": (("localm/rag/_store/*.py",), "reads every module of the RAG store"),
    "tests/test_setup_llama_live_lookups.py": (("localm/setup_llama/*.py",), "reads every module of setup_llama"),
    "tests/test_docs_reaudit_enumerations.py": (
        ("docs/*", "localm/inference/routes/*.py", "localm/inference/backends/llamacpp/*.py"),
        "compares the documented route and backend modules with the folders"),
    "tests/test_pretokenizer_guard.py": (("localm/inference/routes/*.py",), "reads every inference route module"),
    "tests/test_models_page.py": (_BUILTIN_PLUGINS + ("localm/media/*", "localm/inference/backends/*.py"),
                                  "lists every plugin, media runtime and backend"),
    "tests/test_builtin_manifests.py": (_BUILTIN_PLUGINS, "reads every built-in plugin manifest"),
    "tests/test_builtin_plugins_contract.py": (_BUILTIN_PLUGINS, "reads every built-in plugin manifest"),
    "tests/test_scopes.py": (_BUILTIN_PLUGINS, "reads every built-in plugin manifest"),
    "tests/test_gui.py": (_GUI_JS, "serves one discovered GUI script"),
    "tests/test_gui_esm_integrity.py": (_GUI_JS, "parses every GUI app and page module"),
    "tests/test_gui_no_import_reassignment.py": (_GUI_JS, "parses every GUI app and page module"),
    "tests/test_gui_render_loop.py": (_GUI_JS, "reads every GUI script"),
    "tests/test_gui_slash_menu.py": (_GUI_JS, "reads every GUI script"),
    "tests/test_gui_no_native_dialogs.py": ((_GUI_STATIC + "*", "localm/plugins/builtin/*/static/*"),
                                            "reads every GUI script and page for native browser dialogs"),
    "tests/test_static_cache.py": (_GUI_JS, "checks the cache headers of every GUI script"),
    "tests/test_i18n_language_setting.py": ((_GUI_STATIC + "i18n/*",), "lists the GUI translation catalogs"),
    "tests/test_protocol.py": ((_GUI_STATIC + "i18n/*",), "reads every GUI translation catalog"),
    "tests/test_third_party_notices.py": (
        (_GUI_STATIC + "vendor/*", "localm/plugins/builtin/tts/static/vendor/*"),
        "lists the licence files of the vendored libraries"),
    "tests/test_bugreport.py": (("docs/*",), "reads every docs page"),
    "tests/test_docs_cli_examples_use_real_flags.py": (("docs/*", "README.md"),
                                                       "checks every command line in the docs"),
    "tests/test_export_openapi.py": (("docs/*",), "lists every docs page against the site navigation"),
    "tests/test_ci_schedule_split.py": (_WORKFLOWS, "reads every workflow"),
    "tests/test_supply_chain_workflows.py": (_WORKFLOWS, "checks the action pins of every workflow"),
    "tests/test_workflow_pip_installs_are_hash_pinned.py": (
        _WORKFLOWS + (".github/requirements/*",), "checks every workflow and requirement set"),
    "tests/test_installer_batch.py": (("*.bat",), "checks every batch file at the repository root"),
    "tests/test_rag_store_contract.py": (("tests/fixtures/rag_store_v1/*",), "lists the stored collections"),
    "tests/test_routing_reading_pinned.py": (("tests/*.py",),
                                            "greps every test file for a direct routing reading override"),
    "tests/test_rag_rerank_gate_real.py": (_RERANK_CORPUS, "ingests the evaluation corpus"),
    "tests/test_rag_rerank_replay.py": (_RERANK_CORPUS, "ingests the evaluation corpus"),
    "tests/test_rag_rerank_stage.py": (_RERANK_CORPUS, "ingests the evaluation corpus"),
    "tests/test_affected_tests.py": (_PY, "selects over the real import graph of the checkout"),
    "tests/test_affected_tests_indirect_reach.py": (_PY, "selects over the real import graph of the checkout"),
    "tests/test_gui_tests_scope.py": (("tests-js/*", "localm/plugins/builtin/*"),
                                      "resolves the files every tests-js source reads"),
    "tests/test_tree_scanners_registered.py": (_PY, "finds the folder walks of every test and script"),
    "tests/test_native_fault_debug_log_hint.py": (("localm/inference/*.py",), "scans every localm/inference .py"),
    "tests/test_debug_scrub.py": (_TESTS_PY, "walks the AST of every test file for delenv of LOCALM_DEBUG"),
    "tests/test_real_gguf_gate.py": (_TESTS_PY, "scans every test file for broad skips around model loads"),
    "tests/test_tk_root.py": (_TESTS_PY, "scans every test file that builds Tk windows"),
    "tests/test_bugreport_live_lookups.py": (_TESTS_PY + ("localm/bugreport/*.py",),
                                             "scans every test file that mentions bugreport"),
    "tests/test_no_console_window.py": (_TESTS_PY, "scans every test file for window-opening calls"),
    "tests/test_ci_runner_files.py": (("scripts/*.py",), "reads every scripts/*.py for runner-file variable reads"),
    "tests/test_bugreport_network_off.py": (_LOCALM_PY,
                                            "scans the string literals of every localm module for a "
                                            "net-off refusal not taken from config"),
    "tests/test_pin_weekly.py": (("scripts/*.py", "tests/*.py"),
                                 "reads the pin, bump and confirm scripts for gitignored paths"),
    "tests/test_store_lock_create_refused.py": (_LOCALM_PY,
                                                "walks the AST of every localm .py for O_EXCL lock-file creates"),
}
# Test files that walk a folder of the checkout, or load a module that does,
# while their result does not depend on what that folder holds, with how.
_NOT_TREE_SCANNERS: dict[str, str] = {
    "tests/test_check_hygiene_non_ascii_paths.py": "runs the hygiene file list on a repository in tmp_path",
    "tests/test_check_hygiene_no_child_console_print.py": "reads the fixed module list check_hygiene names",
    "tests/test_conftest_isolation_guards.py": "runs one named test file",
    "tests/test_confirm_sdcpp_runtime.py": "reads the sd.cpp pin file only",
    "tests/test_confirm_comfyui_runtime.py": "writes a source record into a ComfyUI tree in tmp_path",
    "tests/test_bump_gguf_node_pin.py": "runs the pin scan on a copy in tmp_path",
    "tests/test_dependabot_groups.py": "uses the selector's requirement parsing only",
    "tests/test_run_affected_tests.py": "runs the selector with a stand-in or for one named file",
    "tests/test_rollback_standalone.py": "lists the update backup folder of an install in tmp_path",
    "tests/test_update_watchdog.py": "lists the update backup folder of an install in tmp_path",
    "tests/test_update_all_or_nothing.py": "lists the update backup folder of an install in tmp_path",
    "tests/test_bugreport_email_redaction.py": "the report script lists home/logs, which is not tracked",
    "tests/test_bugreport_host_redaction.py": "the report script lists home/logs, which is not tracked",
    "tests/test_bugreport_scrub_parity.py": "the report script lists home/logs, which is not tracked",
    "tests/test_report_issue_net_mode.py": "lists the bug-reports folder of a clone in tmp_path",
    "tests/test_bugreport_stdin_interactive.py": "the report script lists home/logs, which is not tracked",
    "tests/test_report_issue_standalone.py": "the report script lists home/logs, which is not tracked",
    "tests/test_report_issue_url.py": "the report script lists home/logs, which is not tracked",
    "tests/test_installer_gui_journal.py": "reads the install journal of a root in tmp_path",
    "tests/test_docker_image_files.py": "puts the checkout path into an environment mapping",
    "tests/test_update_components.py": "reads named pin and archive files",
    "tests/test_updater.py": "asks whether the checkout is a site-packages install",
    "tests/test_uv_tool.py": "reads the uv pin file",
    "tests/test_shell_entrypoints_executable.py": "reads the index mode of a fixed list of files",
}
_ROOT_DECLARATIONS = ("dependencies", "optional-dependencies", "dev-dependencies")
_ROOT_METADATA_DECLARATIONS = ("requires-dist", "requires-dev", "provides-extras")
_SOURCE_ROOTS = ("localm", "scripts", "tests")
_TESTS_ROOT = "tests"
_TEST_FILE = re.compile(r"(^|/)test_[^/]*\.py$")
_WIDE_EXIT = 3
_NOTHING_AFFECTED = "tests/NO_TEST_FILE_IS_AFFECTED"
_TOO_WIDE = "tests/SELECTION_TOO_WIDE_FOR_A_TARGETED_RUN_SEE_STDERR"
_FAILED = "tests/AFFECTED_TESTS_FAILED_SEE_STDERR"


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True,
                          text=True, encoding="utf-8", errors="replace", check=True).stdout


def _tracked(prefix: str) -> list[str]:
    """Tracked plus untracked, not ignored, files under *prefix*."""
    files: list[str] = []
    for extra in ((), ("--others", "--exclude-standard")):
        out = subprocess.run(["git", "-C", str(REPO), "ls-files", "-z", *extra, "--", prefix],
                             capture_output=True, check=True).stdout
        files.extend(p for p in out.decode("utf-8").split("\0") if p)
    return files


def module_name(rel: str) -> str:
    """The dotted module name of a repository-relative .py path."""
    parts = list(Path(rel).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def imported_names(source: str, own: str, is_package: bool) -> set[str]:
    """Every dotted name *source* imports: plain and from-imports (each
    imported attribute also as `module.attr`, so a submodule import resolves),
    relative imports made absolute against *own*, and a literal
    `import_module("x")` target."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return set()
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = own.split(".") if is_package else own.split(".")[:-1]
                base = base[:len(base) - node.level + 1]
                mod = ".".join(base + ([node.module] if node.module else []))
            else:
                mod = node.module or ""
            out.add(mod)
            out.update(f"{mod}.{a.name}" for a in node.names)
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
              and node.func.attr == "import_module" and node.args
              and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)):
            out.add(node.args[0].value)
    return out


def _top_levels(names: set[str]) -> set[str]:
    """The first component of each dotted name in *names*."""
    return {n.split(".", 1)[0] for n in names if n}


def route_prefixes(source: str) -> set[str]:
    """The static prefix of every literal or f-string route path a decorated
    handler in *source* registers. A prefix cut short by a placeholder that
    keeps fewer than two path segments (`/api/` of `/api/{media}/workflows`)
    is left out."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return set()
    out: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for dec in node.decorator_list:
            if not (isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute)
                    and dec.func.attr in _ROUTE_METHODS and dec.args):
                continue
            arg = dec.args[0]
            text = None
            cut = False
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                text = arg.value
            elif isinstance(arg, ast.JoinedStr) and arg.values \
                    and isinstance(arg.values[0], ast.Constant):
                text = str(arg.values[0].value)
                cut = len(arg.values) > 1
            if text and text.startswith("/"):
                prefix = text.split("{", 1)[0]
                cut = cut or prefix != text
                if cut and len([seg for seg in prefix.split("/") if seg]) < 2:
                    continue
                if len(prefix) > 1:
                    out.add(prefix)
    return out


class Graph:
    """Import edges over the tracked source files (localm/, scripts/, and the
    helper modules under tests/) and the test files (tests/**/test_*.py)."""

    def __init__(self) -> None:
        self.sources: dict[str, str] = {}          # module name -> relative path
        self.aliases: dict[str, str] = {}          # importable spelling -> module name
        self.reverse: dict[str, set[str]] = {}     # module -> modules importing it
        self.test_files: list[str] = []
        self.test_imports: dict[str, set[str]] = {}
        self.source_top: dict[str, set[str]] = {}  # module name -> top-level names it imports
        self.test_top: dict[str, set[str]] = {}    # test file -> top-level names it imports
        for root in _SOURCE_ROOTS:
            for rel in _tracked(root):
                if not rel.endswith(".py"):
                    continue
                if _TEST_FILE.search(rel):
                    self.test_files.append(rel)
                    continue
                mod = module_name(rel)
                self.sources[mod] = rel
                self.aliases[mod] = mod
                if rel.startswith(_TESTS_ROOT + "/") and "." in mod:
                    self.aliases[mod.split(".", 1)[1]] = mod
        self.test_files.sort()
        for mod, rel in self.sources.items():
            names = imported_names(_read(rel), mod, rel.endswith("__init__.py"))
            self.source_top[mod] = _top_levels(names)
            for dep in self.resolve(names):
                if dep != mod:
                    self.reverse.setdefault(dep, set()).add(mod)
        for facade, pkg in _REEXPORT_MODULE_FACADES.items():
            if facade not in self.sources:
                continue
            inside = {m for m in self.sources if m == pkg or m.startswith(pkg + ".")}
            for dep, users in self.reverse.items():
                if dep != facade and dep not in inside and users & inside:
                    users.add(facade)
        for rel in self.test_files:
            names = imported_names(_read(rel), module_name(rel), False)
            self.test_top[rel] = _top_levels(names)
            self.test_imports[rel] = self.resolve(names)
        for facade in _REEXPORT_FACADES:
            if facade not in self.sources:
                continue
            for dep, users in self.reverse.items():
                inside = dep == facade or dep.startswith(facade + ".")
                if not inside and any(u.startswith(facade + ".") for u in users):
                    users.add(facade)

    def resolve(self, names: set[str]) -> set[str]:
        """The known modules the dotted *names* refer to, by longest prefix."""
        found: set[str] = set()
        for name in names:
            parts = name.split(".")
            while parts:
                candidate = self.aliases.get(".".join(parts))
                if candidate:
                    found.add(candidate)
                    break
                parts.pop()
        return found

    def tests_importing(self, modules: set[str] | dict[str, int]) -> set[str]:
        """The test files that import any of *modules* directly."""
        wanted = set(modules)
        return {t for t in self.test_files if self.test_imports[t] & wanted}

    def importers(self, modules: set[str], depth: int) -> dict[str, int]:
        """{module: hops} for *modules* (0 hops) and everything that imports
        them within *depth* hops."""
        hops = {m: 0 for m in modules}
        frontier = set(modules)
        for hop in range(1, depth + 1):
            nxt: set[str] = set()
            for m in frontier:
                for importer in self.reverse.get(m, ()):
                    if importer not in hops:
                        hops[importer] = hop
                        nxt.add(importer)
            frontier = nxt
        return hops


def _read(rel: str) -> str:
    """The text of the repository file *rel*; empty for a path outside the
    repository, an unreadable file, or non-UTF-8 content."""
    root = os.path.normpath(str(REPO))
    full = os.path.normpath(os.path.join(root, rel))
    if not full.startswith(root + os.sep):
        return ""
    try:
        with open(full, encoding="utf-8") as fh:
            return fh.read()
    except (OSError, UnicodeDecodeError):
        return ""


def _read_at(ref: str, rel: str) -> str:
    try:
        return _git("show", f"{ref}:{rel}")
    except subprocess.CalledProcessError:
        return ""


def dist_key(name: str) -> str:
    """The normalized (PEP 503) form of a distribution name."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _requirements(pyproject: dict) -> dict[str, list[tuple[str, str]]] | None:
    """{distribution: sorted (extra, requirement) pairs} over [project]
    dependencies (extra "") and optional-dependencies; None when a
    requirement does not start with a distribution name."""
    project = pyproject.get("project", {})
    sections = [("", project.get("dependencies", []))]
    sections += sorted(project.get("optional-dependencies", {}).items())
    out: dict[str, list[tuple[str, str]]] = {}
    for extra, reqs in sections:
        for req in reqs:
            m = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", req) if isinstance(req, str) else None
            if not m:
                return None
            out.setdefault(dist_key(m.group(1)), []).append((extra, req))
    return {name: sorted(pairs) for name, pairs in out.items()}


def _pyproject_rest(pyproject: dict) -> dict:
    """*pyproject* without the [project] requirement lists, the tool tables no
    test reads, the tool tables in _TOOL_TABLE_READERS, which
    changed_tool_table_readers() handles, the [project] version, which
    project_versions() handles, and the [tool.uv] table, whose effect on the locked packages
    dependency_change() reads from the lock instead."""
    project = {k: v for k, v in pyproject.get("project", {}).items()
               if k not in _REQUIREMENT_KEYS and k != "version"}
    skipped = _UNREAD_TOOL_TABLES + _RESOLVER_TOOL_TABLES + tuple(_TOOL_TABLE_READERS)
    tool = {k: v for k, v in pyproject.get("tool", {}).items() if k not in skipped}
    return {**pyproject, "project": project, "tool": tool}


def changed_tool_table_readers(old_pyproject: str, new_pyproject: str) -> dict[str, list[str]]:
    """{test file: reasons} for the test files that read a [tool.*] table of
    _TOOL_TABLE_READERS that differs between the two pyproject.toml texts.
    Empty when either side is empty or does not parse."""
    try:
        old, new = tomllib.loads(old_pyproject), tomllib.loads(new_pyproject)
    except tomllib.TOMLDecodeError:
        return {}
    out: dict[str, list[str]] = {}
    for table, readers in _TOOL_TABLE_READERS.items():
        if old.get("tool", {}).get(table) != new.get("tool", {}).get(table):
            for t in readers:
                out.setdefault(t, []).append(f"reads pyproject.toml [tool.{table}]: it changed")
    return out


def _lock_packages(lock: dict, project: str) -> dict[str, list[str]]:
    """{distribution: the canonical JSON of each of its uv.lock entries}, with
    the dependency declarations left out of *project*'s own entry, and the
    version too when that entry is the project's editable one."""
    out: dict[str, list[str]] = {}
    for entry in lock.get("package", []):
        name = dist_key(str(entry.get("name", "")))
        if name == project:
            meta = {k: v for k, v in entry.get("metadata", {}).items()
                    if k not in _ROOT_METADATA_DECLARATIONS}
            skipped = _ROOT_DECLARATIONS + (("version",) if entry.get("source") == _PROJECT_LOCK_SOURCE
                                            else ())
            entry = {**{k: v for k, v in entry.items() if k not in skipped},
                     "metadata": meta}
        out.setdefault(name, []).append(json.dumps(entry, sort_keys=True))
    return {name: sorted(entries) for name, entries in out.items()}


def _lock_dependents(lock: dict) -> dict[str, set[str]]:
    """{distribution: the uv.lock packages that depend on it}."""
    out: dict[str, set[str]] = {}
    for entry in lock.get("package", []):
        groups = [entry.get("dependencies", [])]
        for key in ("optional-dependencies", "dev-dependencies"):
            groups.extend(entry.get(key, {}).values())
        for group in groups:
            for dep in group:
                out.setdefault(dist_key(str(dep.get("name", ""))), set()).add(
                    dist_key(str(entry.get("name", ""))))
    return out


def _project_version_pair(pyproject: dict, lock: dict, project: str) -> tuple[object, object, bool]:
    """(the [project] version, the version of the project's editable entry in
    *lock*, whether such an entry exists)."""
    for entry in lock.get("package", []):
        if dist_key(str(entry.get("name", ""))) == project and entry.get("source") == _PROJECT_LOCK_SOURCE:
            return pyproject.get("project", {}).get("version"), entry.get("version"), True
    return pyproject.get("project", {}).get("version"), None, False


def _differing_project_versions(py_old: dict, py_new: dict, lock_old: dict, lock_new: dict,
                                project: str) -> set[str] | None:
    """The project version strings, old and new, of each of pyproject.toml and
    the lock's editable entry whose version differs between the two sides; None
    when a version is not a string on a side or the entry exists on one side
    only."""
    pv_old, lv_old, has_old = _project_version_pair(py_old, lock_old, project)
    pv_new, lv_new, has_new = _project_version_pair(py_new, lock_new, project)
    pairs = [(pv_old, pv_new)] + ([(lv_old, lv_new)] if has_old and has_new else [])
    if has_old != has_new or not all(isinstance(v, str) for pair in pairs for v in pair):
        return None
    return {v for old, new in pairs if old != new for v in (old, new)}


def project_versions(old_pyproject: str, new_pyproject: str, old_lock: str,
                     new_lock: str) -> set[str]:
    """The project version strings, old and new, that pyproject.toml and the
    lock's editable entry for the project change. Empty when neither changes
    its version, a side is empty or does not parse, or a version is not a
    string."""
    if not (old_pyproject and new_pyproject and old_lock and new_lock):
        return set()
    try:
        py_old, py_new = tomllib.loads(old_pyproject), tomllib.loads(new_pyproject)
        lock_old, lock_new = tomllib.loads(old_lock), tomllib.loads(new_lock)
    except tomllib.TOMLDecodeError:
        return set()
    project = dist_key(str(py_new.get("project", {}).get("name", "")))
    return _differing_project_versions(py_old, py_new, lock_old, lock_new, project) or set()


def init_version_change(old_source: str, new_source: str) -> tuple[str, str] | None:
    """(old, new) version when the two texts of localm/__init__.py differ only
    in the string assigned to the module-level `__version__`; None otherwise."""
    versions: list[str] = []
    dumps: list[str] = []
    for source in (old_source, new_source):
        try:
            tree = ast.parse(source)
        except (SyntaxError, ValueError):
            return None
        found = [node for node in tree.body
                 if isinstance(node, ast.Assign) and len(node.targets) == 1
                 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "__version__"
                 and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)]
        if len(found) != 1:
            return None
        versions.append(found[0].value.value)
        found[0].value = ast.Constant("")
        dumps.append(ast.dump(tree))
    if versions[0] == versions[1] or dumps[0] != dumps[1]:
        return None
    return versions[0], versions[1]


def version_bump(changed: list[str], base_ref: str) -> tuple[set[str], set[str]]:
    """(the changed files that are only a version bump, the old and new project
    versions): VERSION when its text differs from *base_ref*'s, and
    localm/__init__.py when only its __version__ string differs."""
    files: set[str] = set()
    versions: set[str] = set()
    if _VERSION_FILE in changed:
        old, new = _read_at(base_ref, _VERSION_FILE).strip(), _read(_VERSION_FILE).strip()
        if old and new and old != new:
            files.add(_VERSION_FILE)
            versions.update((old, new))
    if _PACKAGE_INIT in changed:
        pair = init_version_change(_read_at(base_ref, _PACKAGE_INIT), _read(_PACKAGE_INIT))
        if pair:
            files.add(_PACKAGE_INIT)
            versions.update(pair)
    return files, versions


def _references_version(source: str) -> bool:
    """Whether *source* reads or names `__version__`."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return False
    return any((isinstance(n, ast.Name) and n.id == "__version__")
               or (isinstance(n, ast.Attribute) and n.attr == "__version__")
               or (isinstance(n, ast.ImportFrom) and any(a.name == "__version__" for a in n.names))
               or (isinstance(n, ast.Constant) and n.value == "__version__")
               for n in ast.walk(tree))


def version_readers(graph: Graph, versions: set[str], test_text) -> dict[str, list[str]]:
    """{test file: reasons} for the test files that read the project version: one
    that references `__version__`, names the VERSION, pyproject.toml, uv.lock or
    CHANGELOG.md file, or spells one of *versions*. *test_text* reads a test
    file."""
    spelled = [(v, re.compile(r"(?<![\w.])v?" + re.escape(v) + r"(?!\w|\.\d)")) for v in sorted(versions)]
    out: dict[str, list[str]] = {}
    for t in graph.test_files:
        text = test_text(t)
        why: list[str] = []
        if _references_version(text):
            why.append("reads __version__: the project version changed")
        if _VERSION_FILE_WORD.search(text):
            why.append("names VERSION: the project version changed")
        why += [f"names {name}: the project version changed"
                for name in _VERSION_FILE_NAMES if name in text]
        why += [f"spells {v}: the project version changed" for v, rx in spelled if rx.search(text)]
        if why:
            out[t] = why
    return out


def dependency_change(old_pyproject: str, new_pyproject: str, old_lock: str,
                      new_lock: str) -> tuple[set[str], set[str], set[str]] | None:
    """(changed, affected, runtime) for a change to pyproject.toml and uv.lock,
    as normalized distribution names: the distributions whose requirement or
    locked entry differs; those plus every locked package that depends on one
    of them, transitively; and the distributions the project requires outside
    the dev extra, on either side. None when either file differs in anything
    besides the [project] requirement lists and the locked packages (the
    project's own locked entry counts only for its dependency declarations),
    or when a side is empty or does not parse."""
    if not (old_pyproject and new_pyproject and old_lock and new_lock):
        return None
    try:
        py_old, py_new = tomllib.loads(old_pyproject), tomllib.loads(new_pyproject)
        lock_old, lock_new = tomllib.loads(old_lock), tomllib.loads(new_lock)
    except tomllib.TOMLDecodeError:
        return None
    req_old, req_new = _requirements(py_old), _requirements(py_new)
    if req_old is None or req_new is None or _pyproject_rest(py_old) != _pyproject_rest(py_new):
        return None
    project = dist_key(str(py_new.get("project", {}).get("name", "")))
    if _differing_project_versions(py_old, py_new, lock_old, lock_new, project) is None:
        return None
    if ({k: v for k, v in lock_old.items() if k != "package" and k not in _LOCK_FORK_KEYS}
            != {k: v for k, v in lock_new.items() if k != "package" and k not in _LOCK_FORK_KEYS}):
        return None
    pkg_old, pkg_new = _lock_packages(lock_old, project), _lock_packages(lock_new, project)
    if pkg_old.get(project) != pkg_new.get(project):
        return None
    changed = {n for n in req_old.keys() | req_new.keys() if req_old.get(n) != req_new.get(n)}
    changed |= {n for n in pkg_old.keys() | pkg_new.keys() if pkg_old.get(n) != pkg_new.get(n)}
    dependents = _lock_dependents(lock_old)
    for name, parents in _lock_dependents(lock_new).items():
        dependents.setdefault(name, set()).update(parents)
    affected, frontier = set(changed), list(changed)
    while frontier:
        for parent in dependents.get(frontier.pop(), ()):
            if parent != project and parent not in affected:
                affected.add(parent)
                frontier.append(parent)
    runtime = {n for reqs in (req_old, req_new) for n, pairs in reqs.items()
               if any(extra != _DEV_EXTRA for extra, _ in pairs)}
    return changed, affected, runtime


@functools.cache
def _installed() -> dict[str, list[str]]:
    """{top-level import name: the installed distributions providing it}."""
    return metadata.packages_distributions()


def import_names(dist: str) -> set[str]:
    """The top-level names the distribution *dist* can be imported under: its
    name with separators as underscores, and the names its installed files
    provide."""
    names = {dist.replace("-", "_")}
    names.update(top for top, dists in _installed().items()
                 if any(d and dist_key(d) == dist for d in dists))
    return names


def _names_in_a_string(names: set[str]) -> re.Pattern[str]:
    """Matches any of *names* opening a string literal, as a whole name."""
    alternatives = "|".join(sorted(map(re.escape, names), key=len, reverse=True))
    return re.compile(r"""["'](?:""" + alternatives + r""")(?=["'.\[<>=!~;\s])""", re.IGNORECASE)


def _dependency_reasons(graph: Graph, base_ref: str, listed: list[str], depth: int,
                        test_text) -> dict[str, list[str]] | None:
    """{test file: reasons} for the test files a change to the *listed*
    dependency files reaches: a test that imports an affected distribution,
    names it in a string literal, or imports (within *depth* hops) a module
    that imports it, and every test file under the folder of a conftest.py
    that imports it. None when every test file is affected:
    dependency_change() returned None, a listed file does not differ from
    *base_ref*, or an affected distribution the project requires outside the
    dev extra reaches no test file. *test_text* reads a test file."""
    texts = {rel: (_read_at(base_ref, rel), _read(rel)) for rel in _DEPENDENCY_FILES}
    if any(texts[rel][0] == texts[rel][1] for rel in listed):
        return None
    change = dependency_change(*texts["pyproject.toml"], *texts["uv.lock"])
    if change is None:
        return None
    changed, affected, runtime = change
    reasons: dict[str, list[str]] = {}
    for dist in sorted(affected):
        names = import_names(dist)
        users = {m for m, top in graph.source_top.items() if top & names}
        named = _names_in_a_string(names | {dist})
        hops = graph.importers(users, depth)
        folders = sorted(graph.sources[m].rsplit("/", 1)[0] + "/" for m in users
                         if graph.sources[m].endswith("/conftest.py"))
        label = f"changed dependency {dist}" if dist in changed else \
            f"dependency {dist} (it depends on a changed package)"
        hits: dict[str, list[str]] = {}
        for t in graph.test_files:
            why = [f"every test file under {folder}: its conftest.py imports {dist}: {label}"
                   for folder in folders if t.startswith(folder)]
            why += [f"imports {name}: {label}" for name in sorted(graph.test_top[t] & names)]
            for mod in sorted(graph.test_imports[t] & set(hops), key=lambda m: (hops[m], m)):
                why.append(f"imports {mod}, which imports {dist}: {label}" if hops[mod] == 0
                           else f"imports {mod} ({hops[mod]} hop(s) from a module importing "
                                f"{dist}): {label}")
            if named.search(test_text(t)):
                why.append(f"names {dist}: {label}")
            if why:
                hits[t] = why
        if not hits and dist in runtime:
            return None
        for t, why in hits.items():
            reasons.setdefault(t, []).extend(why)
    return reasons


def changed_files(base: str) -> tuple[list[str], str]:
    """Repository-relative paths changed since the merge base with *base*,
    plus staged, unstaged and untracked files; and the merge-base ref the
    committed half was diffed against (HEAD when *base* cannot be resolved)."""
    try:
        merge_base = _git("merge-base", base, "HEAD").strip()
    except subprocess.CalledProcessError:
        merge_base = "HEAD"
    files: set[str] = set()
    files.update(_git("diff", "--name-only", merge_base, "HEAD").splitlines())
    files.update(_git("diff", "--name-only", "HEAD").splitlines())
    files.update(_git("ls-files", "--others", "--exclude-standard").splitlines())
    return sorted(f.replace("\\", "/") for f in files if f), merge_base


def follow(graph: Graph, modules: set[str], depth: int, max_share: float | None = None,
           others: set[str] | frozenset[str] = frozenset()) -> tuple[dict[str, int], dict[str, float]]:
    """{module: hops} for *modules* and their importers within *depth* hops,
    and {module: share} for the importers left out.

    With *max_share*, while the test files importing a followed module plus
    *others* (the test files other rules select) exceed that share of all test
    files, the followed importer that the most test files import directly (a
    hub, ties broken by name) is left out with every module reached only
    through it, until the selection fits or only *modules* remain. *modules*
    themselves are never left out, and nothing is left out when the selection
    exceeds the share even at depth 0. The share recorded for a left-out
    importer is the share of the test files importing it directly."""
    held: dict[str, float] = {}
    total = len(graph.test_files)

    def reach() -> dict[str, int]:
        hops = {m: 0 for m in modules}
        frontier = set(modules)
        for hop in range(1, depth + 1):
            nxt: set[str] = set()
            for m in frontier:
                for importer in graph.reverse.get(m, ()):
                    if importer not in hops and importer not in held:
                        hops[importer] = hop
                        nxt.add(importer)
            frontier = nxt
        return hops

    hops = reach()
    if not (depth and max_share is not None and total):
        return hops, held
    if len(set(others) | graph.tests_importing(modules)) > max_share * total:
        return hops, held
    direct: dict[str, int] = {}
    while len(set(others) | graph.tests_importing(hops)) > max_share * total:
        free = [m for m, hop in hops.items() if hop]
        if not free:
            break
        for m in free:
            if m not in direct:
                direct[m] = len(graph.tests_importing({m}))
        hub = min(free, key=lambda m: (-direct[m], m))
        held[hub] = direct[hub] / total
        hops = reach()
    return hops, held


def select(changed: list[str], graph: Graph, depth: int = 0, base_ref: str = "HEAD",
           max_share: float | None = None,
           held: dict[str, float] | None = None) -> dict[str, list[str]]:
    """{test file: reasons} for the test files *changed* affects. With
    *max_share*, the importers of the changed modules that the most test files
    import are left out until the selection fits that share of the test
    files, as `follow` describes; the left-out importers and their shares go
    into *held* when a dict is passed."""
    reasons: dict[str, list[str]] = {}

    def add(test: str, why: str) -> None:
        reasons.setdefault(test, []).append(why)

    test_text = functools.cache(_read)
    whole = [c for c in changed if c in _EVERYTHING]
    listed = [c for c in changed if c in _DEPENDENCY_FILES]
    dependency_reasons: dict[str, list[str]] = {}
    if listed:
        found = _dependency_reasons(graph, base_ref, listed, depth, test_text)
        if found is None:
            whole += listed
        else:
            dependency_reasons = found
    if whole:
        which = ", ".join(sorted(whole))
        for t in graph.test_files:
            add(t, f"every test file: {which} changed")
        return reasons

    bump_files, versions = version_bump(changed, base_ref)
    if listed:
        versions |= project_versions(_read_at(base_ref, "pyproject.toml"), _read("pyproject.toml"),
                                     _read_at(base_ref, "uv.lock"), _read("uv.lock"))
    if versions:
        for t, whys in version_readers(graph, versions, test_text).items():
            for why in whys:
                add(t, why)

    if "pyproject.toml" in listed:
        readers = changed_tool_table_readers(_read_at(base_ref, "pyproject.toml"),
                                             _read("pyproject.toml"))
        for t, whys in readers.items():
            if t in graph.test_imports:
                for why in whys:
                    add(t, why)

    for rel in changed:
        for t, (patterns, why) in _TREE_SCANNERS.items():
            under = [p for p in patterns if fnmatch.fnmatchcase(rel, p)]
            if under and t in graph.test_imports:
                add(t, f"{why} under {under[0]}: {rel} changed")

    changed_modules: set[str] = set()
    facades: set[str] = set()
    needles: list[tuple[str, re.Pattern[str]]] = []
    routes: set[str] = set()
    for rel in changed:
        if rel in graph.test_imports:
            add(rel, "changed")
            continue
        if rel in bump_files:
            continue
        if rel.endswith("/conftest.py"):
            folder = rel.rsplit("/", 1)[0] + "/"
            for t in graph.test_files:
                if t.startswith(folder):
                    add(t, f"every test file under {folder}: its conftest.py changed")
            continue
        under_source = rel.split("/", 1)[0] in _SOURCE_ROOTS
        if rel.endswith(".py") and under_source:
            mod = module_name(rel)
            changed_modules.add(mod)
            facades.update(f for f in _REEXPORT_FACADES
                           if mod.startswith(f + ".") and f in graph.sources)
            facades.update(f for f, pkg in _REEXPORT_MODULE_FACADES.items()
                           if (mod == pkg or mod.startswith(pkg + "."))
                           and f in graph.sources)
            text = _read(rel) + "\n" + _read_at(base_ref, rel)
            for prefix in sorted(route_prefixes(text) - routes):
                routes.add(prefix)
                needles.append((f"names route {prefix}", re.compile(re.escape(prefix))))
            if rel.startswith("localm/"):
                needles.append((f"names {mod}", re.compile(re.escape(mod) + r"\b")))
            else:
                needles.append((f"names {Path(rel).name}", re.compile(re.escape(Path(rel).name))))
        else:
            name = Path(rel).name
            if name:
                needles.append((f"names {name}", re.compile(re.escape(name))))
    for facade in sorted(facades - changed_modules):
        changed_modules.add(facade)
        needles.append((f"names {facade}", re.compile(re.escape(facade) + r"\b")))
    for mod in sorted(changed_modules):
        for user in sorted(graph.reverse.get(mod, set()) - changed_modules):
            for prefix in sorted(route_prefixes(_read(graph.sources[user])) - routes):
                routes.add(prefix)
                needles.append((f"names route {prefix} of {user}, which imports {mod}",
                                re.compile(re.escape(prefix))))

    named = {t: [label for label, rx in needles if rx.search(test_text(t))]
             for t in graph.test_files} if needles else {}
    others = set(reasons) | set(dependency_reasons) | {t for t, labels in named.items() if labels}
    hops, held_here = follow(graph, changed_modules, depth, max_share, others)
    if held is not None:
        held.update(held_here)
    for t in graph.test_files:
        hit = graph.test_imports[t] & set(hops)
        for mod in sorted(hit, key=lambda m: (hops[m], m)):
            add(t, f"imports {mod}" if hops[mod] == 0 else f"imports {mod} ({hops[mod]} hop(s) from a change)")
        for label in named.get(t, ()):
            add(t, label)
        for why in dependency_reasons.get(t, []):
            add(t, why)
    return reasons


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--base", default="origin/master",
                    help="ref the committed changes are diffed from (via merge-base)")
    ap.add_argument("--files", nargs="*", help="use these changed paths instead of git")
    ap.add_argument("--depth", type=int, default=0,
                    help="follow importers of a changed module this many hops (default 0)")
    ap.add_argument("--why", action="store_true", help="print the reason next to each file")
    ap.add_argument("--max-share", type=float, default=0.25,
                    help="exit 3 when the selection exceeds this share of all test files")
    ap.add_argument("--list-wide", action="store_true",
                    help="print a selection wider than --max-share instead of the sentinel")
    args = ap.parse_args(argv)

    # Every line is LF-terminated on every platform.
    sys.stdout.reconfigure(newline="\n")
    try:
        return _run(args)
    except Exception:
        print(_FAILED)
        traceback.print_exc()
        return 1


def _run(args) -> int:
    graph = Graph()
    if args.files is not None:
        changed, base_ref = sorted(f.replace("\\", "/") for f in args.files), "HEAD"
    else:
        changed, base_ref = changed_files(args.base)
    held: dict[str, float] = {}
    reasons = select(changed, graph, depth=args.depth, base_ref=base_ref,
                     max_share=args.max_share, held=held)
    selected = sorted(reasons)
    total = len(graph.test_files)
    share = len(selected) / total if total else 0.0

    wide = share > args.max_share
    if wide and not args.list_wide:
        print(_TOO_WIDE)
    elif selected:
        for t in selected:
            print(f"{t}  # {'; '.join(reasons[t])}" if args.why else t)
    else:
        print(_NOTHING_AFFECTED)
    print(f"{len(selected)} of {total} test files affected by {len(changed)} changed file(s)",
          file=sys.stderr)
    for mod, mod_share in sorted(held.items()):
        print(f"not followed: {mod}, which {mod_share:.0%} of the test files import, so "
              f"following it would exceed --max-share {args.max_share:.0%}", file=sys.stderr)
    if wide:
        print(f"WIDE: {share:.0%} of the suite exceeds --max-share {args.max_share:.0%}; "
              "the change touches a module most tests import, so a targeted run "
              "cannot stand in for the suite" + ("" if args.list_wide else
                                                  "; --list-wide prints the selection"),
              file=sys.stderr)
        return _WIDE_EXIT
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

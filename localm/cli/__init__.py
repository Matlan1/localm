# SPDX-License-Identifier: AGPL-3.0-or-later
"""localm CLI package.

This package replaces the former single-file ``localm/cli.py``; it is split by
command area (chat, serve, models, media, rag, keys, doctor, maintenance,
plugins, completion) around a shared ``_core`` module that owns the root
``main`` group and the cross-cutting helpers.

``localm.cli`` keeps the same public surface the rest of the app and the test
suite rely on:
  * ``localm.cli:main`` (the entry point / ``localm/__main__.py``),
  * the helpers ``gui/cli.py`` imports (``_exposed_bind_warning``,
    ``_setup_tls_or_exit``),
  * the names tests import directly (``run``, ``serve``, ``doctor``, ``add``,
    ``plugin_setup``, ``_handle_command``, ``_ThinkPrinter``, ...), and
  * the names tests monkeypatch on this module (``console``, ``HOME_DIR``,
    ``find_binary_dir``, ``add_local``, ``load_config``, ``sys``, ``click``,
    ...). The few call sites that consume a monkeypatched name resolve it from
    this package at call time (see chat._handle_command, doctor.doctor,
    models.add) so a patch on ``localm.cli.<name>`` reaches the call site.
"""
# Expose sys / click on the package so tests can monkeypatch localm.cli.sys
# and localm.cli.click, matching the old single-module surface. (F401: kept on
# the package surface on purpose, not for use here.)
import sys  # noqa: F401
import click  # noqa: F401

# Config + model-manager names live on the package so tests can monkeypatch
# localm.cli.<name> and so the call sites that resolve them via this package
# (chat._handle_command, doctor.doctor, models.add) see the patched value.
from ..config import (  # noqa: F401
    HOME_DIR, find_binary_dir, load_config, save_config,
)
from ..model_manager import (  # noqa: F401
    add_local, get_model_info, list_models, pull_model,
    remove_model, show_shortcuts, sync_models_dir,
)

# Shared core: the root group + the cross-cutting helpers. ``main`` and
# ``console`` are used below; the others are re-exported (gui/cli.py imports
# _exposed_bind_warning / _setup_tls_or_exit from here).
from ._core import (  # noqa: F401
    main, console_main, console, _GracefulGroup, _read_version_for_cli,
    _exposed_bind_warning, _resolve_tls, _setup_tls_or_exit,
    _complete_model_name,
)

# Import the command submodules for their import-time side effect of
# registering commands on ``main``. Aliased to avoid clashing with the names we
# re-export below; the side-effect-only ones are flagged unused by ruff.
from . import (  # noqa: F401
    chat as _chat,
    serve as _serve,
    models as _models,
    media as _media,
    comfy as _comfy,
    rag as _rag,
    completion as _completion,
    keys as _keys,
    doctor as _doctor,
    maintenance as _maint,
    plugins as _plugins,
)

# Re-export the names tests (and other callers) import directly from localm.cli.
run = _chat.run
_attach_fallback_note = _chat._attach_fallback_note
_ThinkPrinter = _chat._ThinkPrinter
_handle_command = _chat._handle_command
serve = _serve.serve
doctor = _doctor.doctor
add = _models.add
plugin_setup = _plugins.plugin_setup
_parse_plugin_selection = _plugins._parse_plugin_selection
# Resolved from the package by the plugin commands (so tests that monkeypatch
# localm.cli._engine_manager reach the call sites in plugins.py).
_engine_manager = _plugins._engine_manager


# Register external plugin commands at import time so they show in --help.
# A broken plugin must never take down the CLI - warnings only. This MUST run
# last, after every built-in command submodule above has registered on ``main``.
try:
    from ..plugins.loader import register_external_plugins as _register_ext

    for _warning in _register_ext(main):
        console.print(f"[yellow]plugin warning:[/yellow] {_warning}")
except Exception as _e:  # pragma: no cover - absolute last resort
    console.print(f"[yellow]plugin discovery failed:[/yellow] {_e}")

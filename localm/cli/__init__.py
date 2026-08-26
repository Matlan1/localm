# SPDX-License-Identifier: AGPL-3.0-or-later
"""localm CLI package.

Split by command area (chat, serve, models, media, rag, keys, doctor,
maintenance, plugins, completion) around a shared ``_core`` module that owns
the root ``main`` group and the cross-cutting helpers.

The public surface:
  * ``localm.cli:main`` (the entry point / ``localm/__main__.py``),
  * the helpers ``gui/cli.py`` imports (``_exposed_bind_warning``,
    ``_setup_tls_or_exit``),
  * the names tests import directly (``run``, ``serve``, ``doctor``, ``add``,
    ``plugin_setup``, ``_handle_command``, ``_ThinkPrinter``, ...), and
  * the names tests monkeypatch on this module (``console``, ``HOME_DIR``,
    ``find_binary_dir``, ``add_local``, ``load_config``, ``sys``, ``click``,
    ...). The call sites that consume a monkeypatched name resolve it from
    this package at call time, so a patch on ``localm.cli.<name>`` reaches the
    call site.
"""
# sys/click on the package surface for test monkeypatch.
import sys  # noqa: F401
import click  # noqa: F401

# Config + model-manager names on the package for test monkeypatch.
from ..config import (  # noqa: F401
    HOME_DIR, find_binary_dir, load_config, save_config,
)
from ..model_manager import (  # noqa: F401
    add_local, get_model_info, list_models, pull_model,
    remove_model, show_shortcuts, sync_models_dir,
)

# Shared core: root group + cross-cutting helpers.
from ._core import (  # noqa: F401
    main, console_main, console, _GracefulGroup, _read_version_for_cli,
    _bind_preflight_error, _exposed_bind_warning, _resolve_bind_host,
    _resolve_tls, _setup_tls_or_exit, _complete_model_name,
)

# Command submodules imported for their import-time side effect: registering
# commands on ``main``. Aliased so they do not clash with the re-exports below.
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
# Resolved from the package by plugins.py so a monkeypatch on
# localm.cli._engine_manager reaches the call site.
_engine_manager = _plugins._engine_manager

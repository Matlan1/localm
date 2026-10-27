# SPDX-License-Identifier: AGPL-3.0-or-later
"""What every setup variant asks and installs.

setup.bat, setup.sh, setup-gui.bat, setup-gui.sh and installer/gui.py offer the
same install. This module holds the parts they share as data: the questions with
their choices and defaults, the extras installed, the llama.cpp backend menu and
the plugin list with its preselected set.

Standard library only: setup.bat and setup.sh call it as
``python -m localm.setup_spec ...`` on the bare interpreter of a fresh ``.venv``
before localm is installed into it, and installer/gui.py imports it on uv's
managed Python.

Every prompt in a variant carries a ``SETUP-Q: <id>`` marker naming its
question here; tests/test_setup_variants_parity.py checks each variant against
QUESTIONS.

CLI (one answer per line on stdout, exit 0; exit 2 on a usage error):

    python -m localm.setup_spec backend-menu <recommended>
        "<number>\\t<backend>\\t<label>" per menu entry, in menu order
    python -m localm.setup_spec backend-hint
        the offered backends as "a|b|c", or "backend" when there are none
    python -m localm.setup_spec extras [app-window]
        the pip extras to install, comma-separated
    python -m localm.setup_spec plugins
        the plugins a user can choose, space-separated
    python -m localm.setup_spec plugin-defaults
        the preselected plugins, comma-separated
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

ALL_PLATFORMS = ("win32", "linux", "darwin")
POSIX = ("linux", "darwin")

CONSOLE = "console"
GUI = "gui"

RECOMMENDED_PLUGINS = ("coder", "rag", "web", "tts")

# The graphical installer runs `localm setup-llama` with ASK_ENV=stdio; each
# question it asks is one ASK_PREFIX + JSON line on stdout, answered by one line
# on stdin.
ASK_ENV = "LOCALM_SETUP_ASK"
ASK_PREFIX = "@@localm-ask "


@dataclass(frozen=True)
class Question:
    """One question setup asks.

    ``default`` is the answer Enter (console) or an untouched control (window)
    gives: a menu number for a numbered menu, ``y`` or ``n`` for a yes/no
    question, ``""`` for a blank-means-skip text answer. ``choices`` are the
    ``(answer, meaning)`` pairs of a numbered menu. ``variants`` are the kinds of
    setup that ask it: ``console`` (setup.bat, setup.sh) and ``gui``
    (setup-gui.bat or setup-gui.sh plus installer/gui.py)."""
    id: str
    default: str
    choices: tuple = ()
    platforms: tuple = ALL_PLATFORMS
    variants: tuple = (CONSOLE, GUI)


QUESTIONS: tuple = (
    Question("existing-install", "1",
             (("1", "repair"), ("2", "uninstall"), ("3", "cancel"))),
    Question("uninstall-data", "n"),
    Question("uninstall-confirm", "n"),
    Question("uninstall-no-python", "n", variants=(CONSOLE,)),
    Question("tooling-store", "1", (("1", "portable"), ("2", "shared"))),
    Question("uv-install", "y"),
    Question("replace-venv", "n"),
    Question("venv-lockers", "y", platforms=("win32",)),
    Question("venv-retry", "y"),
    Question("keep-data", "y"),
    Question("data-offline", "1", (("1", "stop"), ("2", "choose"))),
    Question("data-folder", "1", (("1", "portable"), ("2", "custom"))),
    Question("custom-path-confirm", "y", variants=(CONSOLE,)),
    Question("window-mode", "1", (("1", "browser"), ("2", "app-window"))),
    Question("gpu-use", "y", variants=(CONSOLE,)),
    Question("backend", "1"),
    Question("own-build-dir", ""),
    Question("runtime-failed", "y"),
    Question("shortcut", "1", (("1", "launcher"), ("2", "gui"), ("3", "none")),
             platforms=("win32", "linux")),
    Question("global-command", "n"),
    Question("plugins", ",".join(RECOMMENDED_PLUGINS)),
    Question("plugin-deps", "y"),
    Question("report-problem", "y", variants=(CONSOLE,)),
)

_BY_ID = {q.id: q for q in QUESTIONS}


def question(qid: str) -> Question:
    """The question named *qid*. Raises KeyError for an unknown id."""
    return _BY_ID[qid]


def default_choice(qid: str) -> str:
    """The meaning of *qid*'s default menu answer (``"portable"`` for
    tooling-store). Raises KeyError for an unknown id or a question that is not
    a numbered menu."""
    q = _BY_ID[qid]
    return dict(q.choices)[q.default]


def default_yes(qid: str) -> bool:
    """Whether *qid*'s default answer is yes. Raises KeyError for an unknown id,
    ValueError for a question that is not yes/no."""
    d = _BY_ID[qid].default
    if d not in ("y", "n"):
        raise ValueError(f"{qid} is not a yes/no question")
    return d == "y"


BASE_EXTRAS = "coder,voice,monitor"
DESKTOP_EXTRA = "desktop"


def extras(app_window: bool = False) -> str:
    """The pip extras setup installs into the venv; ``desktop`` only with the app
    window."""
    return BASE_EXTRAS + ("," + DESKTOP_EXTRA if app_window else "")


# Backend menu order, with each label; a dict maps a platform to its own label.
BACKENDS: tuple = (
    ("vulkan", "any GPU (AMD/NVIDIA/Intel), no vendor toolkit"),
    ("cuda", "NVIDIA, peak performance (fetches the CUDA runtime for you)"),
    ("hip", "AMD ROCm/HIP, peak performance (needs the ROCm runtime)"),
    ("amd-rocm", "AMD RX 6000 (gfx103X), self-contained"),
    ("sycl", {"win32": "Intel GPU (incl. integrated), often faster than Vulkan, self-contained",
              "": "Intel GPU (incl. integrated), often faster than Vulkan "
                  "(needs a system oneAPI install)"}),
    ("metal", "Apple Silicon, native GPU acceleration"),
    ("cpu", "no GPU"),
)
OWN = "own"
OWN_LABEL = "I will build / provide my own (skip the download)"
RECOMMENDED_SUFFIX = "(recommended for your hardware)"
SAME_SUFFIX = "(same as [1])"


def _hw():
    from localm import hwdetect
    return hwdetect


def backend_label(key: str, plat: str | None = None) -> str:
    """The menu label of backend *key* on *plat* (default: this machine).
    Raises KeyError for an unknown key."""
    if key == OWN:
        return OWN_LABEL
    label = dict(BACKENDS)[key]
    if isinstance(label, dict):
        plat = plat if plat is not None else _hw().platform_key()
        return label.get(plat, label[""])
    return label


def offered_backends(plat: str | None = None, arch: str | None = None) -> list:
    """The BACKENDS keys setup-llama has a prebuilt build of for *plat* and
    *arch* (default: this machine), in menu order."""
    hw = _hw()
    return [k for k, _ in BACKENDS if hw.has_prebuilt(k, plat, arch)]


def backend_menu(recommended: str, plat: str | None = None,
                 arch: str | None = None) -> list:
    """The backend menu as ``(number, key, label)`` rows: 1 is *recommended*,
    then every offered backend in BACKENDS order (the twin of 1 marked), then
    ``own``. Numbers are consecutive strings starting at ``"1"``."""
    rec_label = (backend_label(recommended, plat)
                 if recommended in dict(BACKENDS) else recommended)
    rows = [("1", recommended, f"{recommended} - {rec_label}   {RECOMMENDED_SUFFIX}")]
    for key in offered_backends(plat, arch):
        same = f"   {SAME_SUFFIX}" if key == recommended else ""
        rows.append((str(len(rows) + 1), key, f"{key} - {backend_label(key, plat)}{same}"))
    rows.append((str(len(rows) + 1), OWN, OWN_LABEL))
    return rows


def backend_hint(plat: str | None = None, arch: str | None = None) -> str:
    """The offered backends joined with ``|`` for a "provision later" hint, or
    ``backend`` when none is offered."""
    return "|".join(offered_backends(plat, arch)) or "backend"


def setup_plugins() -> list:
    """``(name, description)`` for every plugin setup offers: the catalog's
    entries that are offered and not preinstalled, in catalog order."""
    from localm.plugins import catalog
    return [(e.name, e.description) for e in catalog.CATALOG
            if e.offered and not e.preinstalled]


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    cmd = args[0] if args else ""
    if cmd == "backend-menu" and len(args) == 2:
        for row in backend_menu(args[1]):
            print("\t".join(row))
        return 0
    if cmd == "backend-hint" and len(args) == 1:
        print(backend_hint())
        return 0
    if cmd == "extras" and len(args) in (1, 2):
        if len(args) == 2 and args[1] != "app-window":
            print(f"unknown option: {args[1]}", file=sys.stderr)
            return 2
        print(extras(len(args) == 2))
        return 0
    if cmd == "plugins" and len(args) == 1:
        print(" ".join(n for n, _ in setup_plugins()))
        return 0
    if cmd == "plugin-defaults" and len(args) == 1:
        print(",".join(RECOMMENDED_PLUGINS))
        return 0
    print("usage: python -m localm.setup_spec backend-menu <recommended> | "
          "backend-hint | extras [app-window] | plugins | plugin-defaults",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

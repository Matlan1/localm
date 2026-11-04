# SPDX-License-Identifier: AGPL-3.0-or-later
"""Python packages a Hugging Face model folder's own code imports.

A model whose ``config.json`` or ``tokenizer_config.json`` carries an ``auto_map``
runs Python bundled in its folder (only when ``hf_trust_remote_code`` is on).
transformers checks that code's imports before running it and, when a package is
absent, fails with a message that tells the reader to run pip. This module reads
the same imports ahead of time, so a model whose code needs a package this
install lacks is refused at add time and before a load starts, with a sentence a
localm user can act on, and rewrites transformers' own message when it still
reaches us.

The scan mirrors transformers' ``check_imports``: every ``import`` and ``from``
statement of the entry modules named by ``auto_map`` and of the sibling modules
they import relatively, except those inside a ``try`` or behind an availability
check (see :func:`imports_of`).
"""

from __future__ import annotations

import ast
import functools
import importlib.util
import json
import platform
import re
import sys
import warnings
from pathlib import Path
from typing import Callable, Collection, Iterable, Optional

CONFIG_FILES = ("config.json", "tokenizer_config.json")

# Top-level module names that localm's [hf] and [gpu] extras install, directly or
# through a dependency. A module listed here that is missing from an install means
# the install is incomplete, not that localm cannot run the model.
SHIPPED_MODULES = frozenset({
    "torch", "torchvision", "transformers", "tokenizers", "accelerate",
    "psutil", "num2words", "compressed_tensors", "kernels", "numpy", "packaging",
    "safetensors", "huggingface_hub", "regex", "requests", "yaml", "tqdm", "filelock",
    "PIL", "pydantic", "fsspec", "jinja2", "sympy", "networkx", "typing_extensions",
    "tiktoken", "sentencepiece", "google", "einops", "peft", "librosa", "addict",
    "easydict", "matplotlib", "scipy", "decord", "datasets", "backoff", "cv2",
    "lmdb", "pretty_midi", "mir_eval", "sklearn", "monai", "onnxruntime", "x_transformers",
    "nltk", "g2p_en", "lightning", "ftfy", "icecream", "av", "soundfile",
})

# Modules of SHIPPED_MODULES that the [hf] and [gpu] extras leave out on some
# platforms, mapped to whether a ``(sys.platform, platform.machine())`` pair gets
# them. Must agree with the extras' markers: see
# test_platform_gaps_match_the_extras_markers.
_WINDOWS_X64_OR_LINUX_X64 = frozenset({("linux", "x86_64"), ("win32", "AMD64")})
PLATFORM_GAPS: dict[str, Callable[[str, str], bool]] = {
    "decord": lambda plat, machine: (plat, machine) in _WINDOWS_X64_OR_LINUX_X64,
    "tiktoken": lambda plat, machine: (plat, machine) != ("win32", "ARM64"),
    "cv2": lambda plat, machine: (plat, machine) != ("win32", "ARM64"),
    "librosa": lambda plat, machine: (plat, machine) != ("win32", "ARM64"),
    "datasets": lambda plat, machine: (plat, machine) != ("win32", "ARM64"),
    "monai": lambda plat, machine: (plat, machine) != ("darwin", "x86_64"),
    "x_transformers": lambda plat, machine: (plat, machine) != ("darwin", "x86_64"),
}

_OS_NAMES = {"linux": "Linux", "win32": "Windows", "darwin": "macOS"}
_ARCH_NAMES = {"x86_64": "x64", "amd64": "x64", "aarch64": "ARM64", "arm64": "ARM64"}


def shipped_here(module: str, plat: Optional[str] = None,
                 machine: Optional[str] = None) -> bool:
    """True when the [hf] and [gpu] extras install *module* on the platform
    *plat* / *machine* (``sys.platform`` and ``platform.machine()`` when omitted).
    False for a module outside SHIPPED_MODULES."""
    if module not in SHIPPED_MODULES:
        return False
    gate = PLATFORM_GAPS.get(module)
    if gate is None:
        return True
    return gate(sys.platform if plat is None else plat,
                platform.machine() if machine is None else machine)


def platform_label(plat: Optional[str] = None, machine: Optional[str] = None) -> str:
    """This platform as a user names it, such as ``Linux ARM64`` or ``Windows x64``."""
    plat = sys.platform if plat is None else plat
    machine = platform.machine() if machine is None else machine
    os_name = _OS_NAMES.get(plat, plat)
    arch = _ARCH_NAMES.get(machine.lower(), machine)
    return f"{os_name} {arch}".strip()

_PIP_SENTENCE = re.compile(r"\s*(?:Run|Please run|Try running)?\s*`?pip install[^`.\n]*`?\.?",
                           re.IGNORECASE)
_REQUIRES_PACKAGES = re.compile(
    r"requires the following packages that were not found in your environment:\s*"
    r"([A-Za-z0-9_.,\s-]+?)\s*(?:\.|Run|$)")
_REQUIRES_LIBRARY = re.compile(r"requires the ([^\n`.:]{2,40}?) library(?! version)", re.IGNORECASE)
_NO_MODULE = re.compile(r"No module named ['\"]([A-Za-z0-9_.-]+)['\"]")


@functools.lru_cache(maxsize=1)
def _availability_names() -> frozenset:
    """Names defined in transformers' ``utils/import_utils.py`` (its ``is_*_available``
    helpers and the like), read from the file without importing transformers; empty
    when transformers is not installed."""
    try:
        spec = importlib.util.find_spec("transformers")
        origin = spec.origin if spec else None
        if not origin:
            return frozenset()
        tree = ast.parse(
            (Path(origin).parent / "utils" / "import_utils.py").read_text(encoding="utf-8"))
    except (OSError, ValueError, SyntaxError, ImportError):
        return frozenset()
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return frozenset(names)


def _is_availability_guard(node: ast.If, known: frozenset) -> bool:
    for call in ast.walk(node.test):
        if isinstance(call, ast.Call):
            name = getattr(call.func, "id", "")
            if (name.startswith("is_flash_attn") and name.endswith("available")) or name in known:
                return True
    return False


def imports_of(source: str, known: Optional[frozenset] = None) -> tuple[set[str], set[str]]:
    """``(absolute, relative)`` imports of the Python *source*: the top-level names
    of absolute imports and the module names of same-folder relative imports.
    Mirrors transformers' import check: nothing inside a ``try`` is read, nor inside
    an ``if`` whose test calls an ``is_flash_attn*_available`` helper or a name
    defined in transformers' ``import_utils`` (*known*, read from the installed
    transformers when omitted). A source that does not parse yields two empty sets."""
    if known is None:
        known = _availability_names()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError):
        return set(), set()
    absolute: set[str] = set()
    relative: set[str] = set()

    def visit(node: ast.AST) -> None:
        if isinstance(node, ast.Try):
            return
        if isinstance(node, ast.If) and _is_availability_guard(node, known):
            return
        if isinstance(node, ast.Import):
            for alias in node.names:
                absolute.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                absolute.add(node.module.split(".")[0])
            elif node.level == 1 and node.module:
                relative.add(node.module.split(".")[0])
        for child in ast.iter_child_nodes(node):
            visit(child)

    try:
        visit(tree)
    except RecursionError:
        return set(), set()
    return absolute, relative


def entry_modules(configs: Iterable[dict]) -> list[str]:
    """The same-folder modules the ``auto_map`` entries of *configs* name, in order,
    without duplicates. *configs* are the parsed ``config.json`` and
    ``tokenizer_config.json``: a processor's code is left out because a model whose
    processor cannot be loaded still loads with its plain tokenizer. An entry that
    points at another repository (``owner/repo--module.Class``) names no file of
    this folder and is skipped."""
    found: list[str] = []
    for config in configs:
        auto_map = config.get("auto_map") if isinstance(config, dict) else None
        if not isinstance(auto_map, dict):
            continue
        for value in auto_map.values():
            for ref in (value if isinstance(value, (list, tuple)) else [value]):
                if not isinstance(ref, str) or "--" in ref or "." not in ref:
                    continue
                module = ref.rsplit(".", 1)[0]
                if module not in found:
                    found.append(module)
    return found


def third_party_imports(read: Callable[[str], Optional[str]],
                        entries: Iterable[str]) -> set[str]:
    """Top-level non-standard-library module names the code of *entries* imports,
    following relative imports. *read* returns the source text of a module name
    (``"modeling_x"``) or None when the folder has no such file."""
    seen: set[str] = set()
    pending = list(entries)
    needed: set[str] = set()
    while pending:
        module = pending.pop()
        if module in seen:
            continue
        seen.add(module)
        source = read(module)
        if source is None:
            continue
        absolute, relative = imports_of(source)
        needed.update(absolute)
        pending.extend(relative)
    return {m for m in needed if m not in sys.stdlib_module_names and m not in seen
            and read(m) is None}


def is_importable(module: str) -> bool:
    """True when *module* can be located in this interpreter."""
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _read_json(path: Path) -> Optional[dict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError):
        return None
    return data if isinstance(data, dict) else None


def _folder_reader(folder: Path) -> Callable[[str], Optional[str]]:
    def read(module: str) -> Optional[str]:
        if not module.isidentifier():
            return None
        try:
            return (folder / f"{module}.py").read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None
    return read


def folder_third_party_imports(folder: Path) -> set[str]:
    """Third-party modules the bundled code of the model *folder* imports; empty
    for a folder with no ``auto_map``."""
    configs = [c for c in (_read_json(folder / f) for f in CONFIG_FILES) if c is not None]
    entries = entry_modules(configs)
    if not entries:
        return set()
    return third_party_imports(_folder_reader(folder), entries)


def missing_modules(folder: Path) -> list[str]:
    """Sorted top-level modules the bundled code of *folder* imports that this
    interpreter cannot import."""
    try:
        needed = folder_third_party_imports(folder)
    except (OSError, ValueError, RecursionError):
        return []
    return sorted(m for m in needed if not is_importable(m))


_NAMES = {"PIL": "pillow", "yaml": "pyyaml", "sklearn": "scikit-learn", "cv2": "opencv",
          "fla": "flash-linear-attention", "google": "protobuf"}


def package_label(module: str) -> str:
    """The name a user knows the package by for the importable *module*."""
    return _NAMES.get(module, module.replace("_", "-"))


def refusal_for(model_name: str, modules: list[str],
                addable: Optional[Collection[str]] = None) -> str:
    """The sentence that says model *model_name* needs the packages behind
    *modules*, which this install lacks, and what the user can do about it:
    repair the install (a package localm ships), add them with
    ``localm install-packages`` or the Models page (every other package localm
    can add here), or nothing (a package localm cannot add, with the reason, or
    one it ships on other platforms but not on this one).

    *addable* is the set of modules ``model_packages.folder_state`` offers to add
    for this model. When given, the install is offered only if every module localm
    could otherwise add is in it; a module outside it (needed by the model's
    built-in support, or imported behind a guard) gets a sentence saying localm
    cannot add it for this model, because the install action would find nothing
    to do. None means the caller already derived *modules* from ``folder_state``."""
    from localm.model_manager import model_packages
    labels = ", ".join(package_label(m) for m in modules)
    noun = "package" if len(modules) == 1 else "packages"
    elsewhere = [m for m in modules if m in SHIPPED_MODULES and not shipped_here(m)]
    if elsewhere:
        names = ", ".join(package_label(m) for m in elsewhere)
        verb = "is" if len(elsewhere) == 1 else "are"
        return (f"{model_name} cannot be loaded: its code needs the Python "
                f"{'package' if len(elsewhere) == 1 else 'packages'} {names}, which "
                f"{verb} not available for this platform ({platform_label()}). "
                "localm cannot run this model here.")
    shipped = [m for m in modules if m in SHIPPED_MODULES]
    if len(shipped) == len(modules):
        return (f"{model_name} cannot be loaded: its code needs the Python {noun} "
                f"{labels}, which localm normally includes but this install is "
                "missing. Update localm, or run its setup again, to restore it.")
    others = [m for m in modules if m not in SHIPPED_MODULES]
    blocked = [(m, model_packages.unavailable_reason(m)) for m in others]
    blocked = [(m, why) for m, why in blocked if why is not None]
    if blocked:
        reasons = "; ".join(f"{package_label(m)}: {why}" for m, why in blocked)
        return (f"{model_name} cannot be loaded: its own code needs the Python {noun} "
                f"{labels}, and localm cannot add "
                f"{'it' if len(blocked) == 1 else 'them'} to its environment ({reasons}). "
                "localm cannot run this model.")
    noun = "package" if len(others) == 1 else "packages"
    labels = ", ".join(package_label(m) for m in others)
    unreachable = [m for m in others if addable is not None and m not in addable]
    if unreachable:
        return (f"{model_name} cannot be loaded: its own code needs the Python {noun} {labels}, "
                f"which this install lacks, and localm cannot add "
                f"{'it' if len(others) == 1 else 'them'} for this model: localm adds packages only "
                "for modules it finds imported by the model's bundled code, and it finds no such "
                f"import for {', '.join(package_label(m) for m in unreachable)}. "
                "localm cannot run this model.")
    them = "it" if len(others) == 1 else "them"
    repair = (" It also needs " + ", ".join(package_label(m) for m in shipped)
              + ", which localm normally includes: update localm or run its setup again first."
              if shipped else "")
    return (f"{model_name} cannot be loaded yet: its own code needs the Python {noun} "
            f"{labels}, which localm can add to its own environment. Add {them} with "
            f"`localm install-packages {model_name}`, or with Install packages on the "
            f"Models page.{repair}")


def folder_refusal(folder: Path) -> Optional[str]:
    """The reason the bundled code of *folder* cannot run here (it imports packages
    this install lacks), or None. Looks only when the folder declares bundled code
    and ``hf_trust_remote_code`` is on: with it off the folder is refused for
    needing custom code before its packages matter."""
    if not folder.is_dir():
        return None
    from localm.inference.backends.hf import _declares_custom_code, _trust_remote_code_enabled
    if not _declares_custom_code(str(folder)) or not _trust_remote_code_enabled():
        return None
    modules = missing_modules(folder)
    return refusal_for(folder.name, modules) if modules else None


_LIBRARY_MODULES = {"protobuf": "google", "pytorch": "torch", "pyav": "av", "scikit-learn": "sklearn",
                    "sentencepiece": "sentencepiece", "beautiful soup": "bs4", "beautifulsoup4": "bs4",
                    "pillow": "PIL", "pyyaml": "yaml", "opencv-python": "cv2",
                    "opencv-python-headless": "cv2"}


def _module_for_library(display: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_ -]", "", display).strip().lower()
    return _LIBRARY_MODULES.get(name, name.replace("-", "_").replace(" ", ""))


def rewrite_import_error(model_name: str, text: str, folder: Optional[str] = None) -> Optional[str]:
    """The localm sentence for a load failure whose *text* is a missing-package
    ImportError (transformers' remote-code check or a bare ``No module named``),
    or None when *text* is neither. Never contains pip instructions. The install
    is offered only when the model *folder* is given and
    ``model_packages.folder_state`` lists the missing package as one it can add."""
    match = _REQUIRES_PACKAGES.search(text)
    if match:
        modules = [m for m in re.split(r"[,\s]+", match.group(1)) if m]
    else:
        match = _NO_MODULE.search(text)
        modules = [match.group(1).split(".")[0]] if match else []
        modules = [m for m in modules if m != "transformers_modules"]
        library = _REQUIRES_LIBRARY.search(text)
        if not modules and library:
            modules = [_module_for_library(library.group(1))]
    if not modules:
        return None
    return refusal_for(model_name, modules, _addable_for(folder))


def _addable_for(folder: Optional[str]) -> set[str]:
    """The modules ``folder_state`` offers to add for the model *folder*; empty
    without a folder or when the folder cannot be examined."""
    if not folder:
        return set()
    from localm.model_manager import model_packages
    try:
        return set(model_packages.folder_state(Path(folder)).addable)
    except (OSError, ValueError):
        return set()


def strip_pip_instructions(text: str) -> str:
    """*text* with any "pip install ..." instruction removed; a text that still
    mentions pip afterwards is replaced by a sentence that names no tool."""
    stripped = " ".join(_PIP_SENTENCE.sub("", text).split())
    if re.search(r"\bpip\b", stripped, re.IGNORECASE):
        return "The model needs a Python package this install does not have."
    return stripped

#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Census of the Python packages Hugging Face transformers-format models import.

For the most downloaded and the trending transformers-format repositories of
several tasks, plus the repositories localm's shortcuts offer, list the packages
their bundled modeling and tokenizer code imports (the ``auto_map`` modules, as
transformers' own import check reads them) and the packages their tokenizer
files need, and check each against what localm's extras install (resolved from
``uv.lock``). A package that is neither installed by the extras, added on
request (``ON_DEMAND``) nor listed in ``ACCEPTED_REFUSALS`` is a gap: exit status 1.

    python scripts/hf_remote_code_census.py --limit 150
    python scripts/hf_remote_code_census.py --repo moonshotai/Kimi-Linear-48B-A3B-Instruct

Repository files are read as data over HTTPS and are never executed.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import tomllib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from localm.model_manager.missing_packages import (  # noqa: E402
    CONFIG_FILES,
    entry_modules,
    package_label,
    third_party_imports,
)
from localm.model_manager.model_packages import ON_DEMAND, REFUSED  # noqa: E402

TASKS = ("text-generation", "image-text-to-text", "feature-extraction", "sentence-similarity",
         "automatic-speech-recognition", "text-to-speech", "text-classification",
         "fill-mask", "text2text-generation", "any-to-any", "audio-text-to-text",
         "token-classification", "zero-shot-image-classification", "image-classification")

EXTRAS = ("hf", "audio")

# Modules the extras do not install: ON_DEMAND are added on the user's request, the
# rest are refused early with the reason REFUSED gives. Reviewed at every release: a
# new REFUSED entry is a decision to leave those models unsupported.
ACCEPTED_REFUSALS = REFUSED

# Importable module name -> distribution name where they differ.
DIST_FOR_MODULE = {"PIL": "pillow", "yaml": "pyyaml", "sklearn": "scikit-learn",
                   "cv2": "opencv-python-headless", "open_clip": "open-clip-torch", "google": "protobuf", "fla": "flash-linear-attention",
                   "Cryptodome": "pycryptodomex", "bs4": "beautifulsoup4", "skimage": "scikit-image",
                   "attr": "attrs", "dateutil": "python-dateutil", "jax": "jax",
                   "flash_attn": "flash-attn", "mamba_ssm": "mamba-ssm",
                   "causal_conv1d": "causal-conv1d"}

# Modules every transformers install provides through the packages below, which the
# lock may not list under the module's own name.
PROVIDED = {"transformers", "tokenizers", "torch", "torchvision", "huggingface_hub"}


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def shipped_distributions(lock: Path) -> set[str]:
    """Distributions uv.lock resolves from localm's base dependencies and the
    ``EXTRAS``, on any platform."""
    data = tomllib.loads(lock.read_text(encoding="utf-8"))
    graph: dict[str, set[str]] = {}
    roots: set[str] = set()
    for pkg in data.get("package", []):
        name = _norm(pkg["name"])
        deps = {_norm(d["name"]) for d in pkg.get("dependencies", [])}
        for group in (pkg.get("optional-dependencies") or {}).values():
            deps.update(_norm(d["name"]) for d in group)
        graph[name] = deps
        if name == "localm":
            roots.update(_norm(d["name"]) for d in pkg.get("dependencies", []))
            for extra in EXTRAS:
                roots.update(_norm(d["name"])
                             for d in (pkg.get("optional-dependencies") or {}).get(extra, []))
    seen: set[str] = set()
    stack = list(roots)
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        seen.add(name)
        stack.extend(graph.get(name, ()))
    return seen


def module_is_shipped(module: str, shipped: set[str]) -> bool:
    if module in PROVIDED:
        return True
    dist = _norm(DIST_FOR_MODULE.get(module, module))
    return dist in shipped


class Fetcher:
    def __init__(self, cache: Path | None = None) -> None:
        import requests
        self._session = requests.Session()
        self._cache = cache

    def text(self, repo: str, path: str) -> str | None:
        if self._cache is None:
            return self._get(repo, path)
        entry = self._cache / repo.replace("/", "__") / path
        absent = entry.with_name(entry.name + ".absent")
        if entry.is_file():
            return entry.read_text(encoding="utf-8")
        if absent.is_file():
            return None
        body = self._get(repo, path)
        entry.parent.mkdir(parents=True, exist_ok=True)
        if body is None:
            absent.write_text("", encoding="utf-8")
        else:
            entry.write_text(body, encoding="utf-8")
        return body

    def _get(self, repo: str, path: str) -> str | None:
        url = f"https://huggingface.co/{repo}/resolve/main/{path}"
        for _ in range(6):
            r = self._session.get(url, timeout=30)
            if r.status_code == 200:
                return r.text
            if r.status_code == 404:
                return None
            if r.status_code != 429:
                raise RuntimeError(f"HTTP {r.status_code} for {repo}/{path}")
            time.sleep(min(float(r.headers.get("Retry-After") or 20), 60) + 1)
        raise RuntimeError(f"rate limited for {repo}/{path}")


def retrying(call, attempts: int = 8):
    """Run *call*, waiting out the Hub's 429 rate limit between attempts."""
    from huggingface_hub.errors import HfHubHTTPError
    for attempt in range(attempts):
        try:
            return call()
        except HfHubHTTPError as e:
            response = getattr(e, "response", None)
            if response is None or response.status_code != 429 or attempt == attempts - 1:
                raise
            time.sleep(min(float(response.headers.get("Retry-After") or 60), 300) + 2)


def census_repo(api, fetch: Fetcher, repo: str, files: set[str] | None,
                downloads: int | None) -> dict:
    try:
        return _census_repo(api, fetch, repo, files, downloads)
    except Exception as e:
        return {"repo": repo, "error": f"{type(e).__name__}: {str(e)[:120]}"}


def _census_repo(api, fetch: Fetcher, repo: str, files: set[str] | None,
                 downloads: int | None) -> dict:
    if files is None:
        try:
            info = retrying(lambda: api.model_info(repo))
        except Exception as e:
            return {"repo": repo, "error": f"{type(e).__name__}: {str(e)[:120]}"}
        files = {s.rfilename for s in info.siblings or []}
        downloads = info.downloads
    configs = []
    for name in CONFIG_FILES:
        if name in files:
            body = fetch.text(repo, name)
            try:
                value = json.loads(body) if body else None
            except ValueError:
                value = None
            if isinstance(value, dict):
                configs.append(value)
    entries = entry_modules(configs)
    cache: dict[str, str | None] = {}

    def read(module: str) -> str | None:
        if module not in cache:
            cache[module] = (fetch.text(repo, f"{module}.py")
                             if f"{module}.py" in files else None)
        return cache[module]

    code = sorted(third_party_imports(read, entries)) if entries else []
    tokenizer = []
    if "tokenizer.json" not in files and files & {"tokenizer.model", "spiece.model",
                                                  "sentencepiece.bpe.model"}:
        tokenizer.append("sentencepiece")
    if any(f.endswith(".tiktoken") for f in files) or "tiktoken.model" in files:
        tokenizer.append("tiktoken")
    return {"repo": repo, "remote_code": bool(entries), "code": code, "tokenizer": tokenizer,
            "downloads": downloads}


def collect_repos(api, limit: int) -> dict[str, tuple[set[str] | None, int | None]]:
    """Repository id -> (file names, downloads); file names are None for a
    repository whose listing did not carry them."""
    repos: dict[str, tuple[set[str] | None, int | None]] = {}

    def add(models) -> None:
        for m in models:
            if m.id not in repos:
                repos[m.id] = ({s.rfilename for s in m.siblings or []}, m.downloads)

    for task in TASKS:
        for sort in ("downloads", "trending_score"):
            try:
                add(retrying(lambda task=task, sort=sort: list(api.list_models(
                    filter="transformers", pipeline_tag=task, sort=sort, limit=limit,
                    expand=["downloads", "siblings"]))))
            except Exception as e:
                print(f"listing {task}/{sort} failed: {type(e).__name__}: {e}", file=sys.stderr)
    for spec in shortcut_specs():
        repo = spec.split(":")[0]
        if "gguf" not in repo.lower() and repo not in repos:
            repos[repo] = (None, None)
    return repos


def cached_repos(api, limit: int, cache: Path | None) -> dict:
    listing = cache / "listing.json" if cache else None
    if listing is not None and listing.is_file():
        raw = json.loads(listing.read_text(encoding="utf-8"))
        return {r: (None if f is None else set(f), d) for r, (f, d) in raw.items()}
    repos = collect_repos(api, limit)
    if listing is not None:
        listing.parent.mkdir(parents=True, exist_ok=True)
        listing.write_text(json.dumps(
            {r: (None if f is None else sorted(f), d) for r, (f, d) in repos.items()}),
            encoding="utf-8")
    return repos


def shortcut_specs() -> list[str]:
    """The repository specs of ``MODEL_SHORTCUTS``, read from the source without
    importing localm."""
    import ast
    tree = ast.parse((ROOT / "localm" / "model_manager" / "registry.py").read_text(encoding="utf-8"))
    for node in tree.body:
        target = node.target if isinstance(node, ast.AnnAssign) else None
        if target is not None and getattr(target, "id", "") == "MODEL_SHORTCUTS":
            return [v.value for v in node.value.values if isinstance(v, ast.Constant)]
    return []


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--limit", type=int, default=100, help="repos per task and ordering")
    ap.add_argument("--repo", action="append", default=[], help="census this repo only (repeatable)")
    ap.add_argument("--lock", type=Path, default=ROOT / "uv.lock")
    ap.add_argument("--json", type=Path, help="write the raw per-repo results here")
    ap.add_argument("--cache", type=Path, help="folder that keeps the files and listings read, "
                    "so a re-run classifies again without fetching")
    ap.add_argument("--from-json", type=Path,
                    help="classify the results of an earlier run instead of fetching")
    args = ap.parse_args()

    if args.from_json:
        results = json.loads(args.from_json.read_text(encoding="utf-8"))
    else:
        from huggingface_hub import HfApi
        api = HfApi()
        repos = {r: (None, None) for r in args.repo} or cached_repos(api, args.limit, args.cache)
        fetch = Fetcher(args.cache)
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda kv: census_repo(api, fetch, kv[0], *kv[1]),
                                    repos.items()))
        if args.json:
            args.json.write_text(json.dumps(results, indent=1), encoding="utf-8")

    shipped = shipped_distributions(args.lock)
    need: dict[str, list[str]] = defaultdict(list)
    errors = [r for r in results if "error" in r]
    for r in results:
        if "error" in r:
            continue
        for module in set(r["code"]) | set(r["tokenizer"]):
            need[module].append(r["repo"])
    print(f"{len(results) - len(errors)} repos read, {len(errors)} unreadable, "
          f"{sum(1 for r in results if r.get('remote_code'))} with bundled code")
    gaps = 0
    print(f"{'module':24} {'repos':>5}  status")
    for module, users in sorted(need.items(), key=lambda kv: -len(kv[1])):
        if module_is_shipped(module, shipped):
            status = "shipped"
        elif module in ON_DEMAND:
            status = f"added on request ({ON_DEMAND[module].project})"
        elif module in ACCEPTED_REFUSALS:
            status = f"refused early: {ACCEPTED_REFUSALS[module]}"
        else:
            status = "GAP"
            gaps += 1
        print(f"{package_label(module):24} {len(users):>5}  {status}   e.g. {users[0]}")
    if errors:
        print(f"\n{len(errors)} repo(s) could not be read, e.g. {errors[0]['repo']}: "
              f"{errors[0]['error']}: FAIL")
        return 1
    if gaps:
        print(f"\n{gaps} package(s) neither shipped, added on request nor refused: FAIL")
        return 1
    print("\nEvery imported package is shipped, added on request or refused early: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

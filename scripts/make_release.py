#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Cut a SIGNED localm release build (NEW-RELEASE-FILEMANIFEST + CHK-UPDATER-INTEGRITY).

One command for the release signer: assemble the build.zip from release-manifest.toml,
sign it with the offline Ed25519 private key, and (optionally) publish both to a GitHub
Release. A self-updating client downloads the build via the proxy, reads the signature
from the proxy's /update JSON, and verifies it against the PUBLIC key pinned in
localm/updater.py before applying.

  export LOCALM_SIGNING_KEY=/path/to/update_signing_key.pem   # kept OUT of the repo
  python scripts/make_release.py                              # build + sign -> dist/
  python scripts/make_release.py --publish                    # + gh release create

The signing-key PATH comes from --key or $LOCALM_SIGNING_KEY - deliberately NOT baked
into this tracked file (which must stay machine-path-free). Before publishing, this
SELF-CHECKS that the signature verifies against the key pinned in
updater._UPDATE_PUBKEYS, so a build the shipped clients would reject is never released.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # sibling scripts
import build_release  # noqa: E402
import sign_release    # noqa: E402

REPO = Path(__file__).resolve().parent.parent

# Critical non-.py files a working release must carry (a spot-check on top of the
# import smoke below - these are the "omitted and unnoticed until later" class).
_MUST_SHIP = (
    "VERSION", "pyproject.toml", "localm/__init__.py", "localm/__main__.py",
    "localm/plugins/gui/static/index.html", "assets/localm.svg",
    "scripts/report_issue.py",
)


def smoke_test(zip_path: Path) -> None:
    """Prove the built release IMPORTS AND RUNS on its own, so a runtime-needed file
    that was omitted (mis-classified as dev-only, or gitignored) is caught HERE at
    build time, not by a user later.

    Extracts the build.zip to a throwaway dir and, importing ONLY from that tree
    (cwd + PYTHONPATH = the extracted release, so it shadows any dev/editable install):
    runs ``python -m localm --help`` (imports the whole CLI command tree) and imports
    the heavy runtime modules (server app, plugin loader, updater, setup). Also spot-
    checks a few critical assets. Raises SystemExit on any failure - the manifest gate
    proves every file is CLASSIFIED; this proves the included set is actually COMPLETE.

    Uses the current interpreter (sys.executable), which must be able to import localm's
    dependencies - i.e. run make_release from the project venv."""
    tmp = Path(tempfile.mkdtemp(prefix="localm-relcheck-"))
    try:
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(tmp)
        missing = [m for m in _MUST_SHIP if not (tmp / m).is_file()]
        if missing:
            raise SystemExit(f"release smoke: missing critical file(s) from the build: {missing}")
        if not any((tmp / "localm/plugins/gui/static").rglob("*.js")):
            raise SystemExit("release smoke: no GUI JavaScript shipped")
        if not any((tmp / "docs").glob("*.md")):
            raise SystemExit("release smoke: no docs shipped")

        env = {**os.environ, "PYTHONPATH": str(tmp), "LOCALM_HOME": str(tmp / "_home")}
        checks = (
            (["-m", "localm", "--help"], "localm --help (CLI command tree)"),
            (["-c", "from localm.inference.http_server import create_app; "
                    "from localm.plugins.loader import discover_plugins; "
                    "import localm.setup_llama, localm.updater, localm._apply_update, localm.bugreport"],
             "runtime modules (server + loader + updater + setup)"),
        )
        for args, what in checks:
            r = subprocess.run([sys.executable, *args], cwd=str(tmp), env=env,
                               capture_output=True, text=True, timeout=180)
            if r.returncode != 0:
                raise SystemExit(
                    f"release smoke FAILED - the extracted release does not run [{what}].\n"
                    "A runtime-needed file may be omitted (mis-classified dev-only, or "
                    f"gitignored). Details:\n{(r.stderr or r.stdout)[-1600:]}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _pinned_pubkeys() -> tuple:
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from localm import updater
    return tuple(updater._UPDATE_PUBKEYS)


def _verify_against_pinned(zip_path: Path, sig_path: Path) -> None:
    """Confirm the freshly-made signature verifies against a key pinned in the shipped
    updater. This is the gate that stops us publishing a build clients cannot verify -
    e.g. signed with a key whose public half was never pinned, or a stale pin."""
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    keys = _pinned_pubkeys()
    if not keys:
        raise SystemExit(
            "updater._UPDATE_PUBKEYS is EMPTY: pin the release public key before cutting a "
            "signed release, or a keyed client cannot verify this build.")
    data = zip_path.read_bytes()
    sig = base64.b64decode(sig_path.read_text(encoding="utf-8").strip(), validate=True)
    for hexkey in keys:
        try:
            Ed25519PublicKey.from_public_bytes(bytes.fromhex(str(hexkey).strip())).verify(sig, data)
            return
        except (InvalidSignature, ValueError):
            continue
    raise SystemExit(
        "the signature does NOT verify against any key pinned in updater._UPDATE_PUBKEYS - "
        "the signing key and the pinned public key disagree. Refusing to publish a build the "
        "shipped clients would reject.")


def _gh(args):
    """Run a gh subcommand; return the CompletedProcess (never raises)."""
    return subprocess.run(["gh", *args], capture_output=True, text=True)


def _run_ids(runner, workflow):
    """The set of existing run databaseIds for *workflow* (empty on any error)."""
    r = runner(["run", "list", "--workflow", workflow, "--limit", "15", "--json", "databaseId"])
    if getattr(r, "returncode", 1) != 0:
        return set()
    try:
        return {int(x["databaseId"]) for x in json.loads(r.stdout or "[]")}
    except Exception:
        return set()


def require_ci_green(ref: str = "master", *, workflow: str = "ci.yml", runner=_gh,
                     sleeper=time.sleep, poll_s: int = 15, appear_timeout_s: int = 180) -> None:
    """RULE: before a release is PUBLISHED, run ONE full CI pass over the whole repo and
    require it green. Enables the workflow first if a maintainer disabled it (once). A
    release is never published over red or un-run CI.

    Raises SystemExit if gh is unavailable, CI cannot be started, the run never appears,
    or it does not conclude success. *runner*/*sleeper* are injectable so the flow is
    unit-tested without any live GitHub call."""
    if shutil.which("gh") is None:
        raise SystemExit("release CI gate: the gh CLI (authenticated) is required to run CI before publishing.")
    # 1. enable the CI workflow if it was disabled
    q = runner(["api", "repos/{owner}/{repo}/actions/workflows"])
    if getattr(q, "returncode", 1) != 0:
        raise SystemExit(f"release CI gate: could not query workflows: {getattr(q, 'stderr', '').strip()}")
    try:
        wfs = json.loads(q.stdout).get("workflows", [])
    except Exception as e:
        raise SystemExit(f"release CI gate: could not parse the workflows list: {e}")
    ci = next((w for w in wfs if str(w.get("path", "")).endswith("/" + workflow)), None)
    if ci is None:
        raise SystemExit(f"release CI gate: no {workflow} workflow found in the repo.")
    if ci.get("state") != "active":
        en = runner(["workflow", "enable", workflow])
        if getattr(en, "returncode", 1) != 0:
            raise SystemExit(f"release CI gate: could not enable CI (was {ci.get('state')}): "
                             f"{getattr(en, 'stderr', '').strip()}")
        print(f"release CI gate: enabled the CI workflow (was {ci.get('state')}).")
    # 2. snapshot existing runs, then trigger a run on *ref*
    before = _run_ids(runner, workflow)
    tr = runner(["workflow", "run", workflow, "--ref", ref])
    if getattr(tr, "returncode", 1) != 0:
        raise SystemExit(f"release CI gate: could not start CI on '{ref}': {getattr(tr, 'stderr', '').strip()}")
    print(f"release CI gate: started a full CI run on '{ref}'; waiting for it to finish ...")
    # 3. wait for the newly-triggered run to register
    run_id, waited = None, 0
    while waited < appear_timeout_s:
        sleeper(poll_s)
        waited += poll_s
        new = _run_ids(runner, workflow) - before
        if new:
            run_id = max(new)
            break
    if run_id is None:
        raise SystemExit("release CI gate: the CI run did not appear in time; check GitHub Actions.")
    # 4. block until it completes; require success
    w = runner(["run", "watch", str(run_id), "--exit-status"])
    if getattr(w, "returncode", 1) != 0:
        raise SystemExit(f"release CI gate: CI did NOT pass (run {run_id}). Refusing to publish - "
                         "fix CI, then re-run make_release --publish.")
    print(f"release CI gate: full CI passed (run {run_id}).")


def _git(args, runner=None):
    """Run a git subcommand in the repo; returns the CompletedProcess. *runner* is
    injectable (tests pass a fake) so the release gates need no live git."""
    run = runner or (lambda a: subprocess.run(["git", *a], cwd=str(REPO),
                                              capture_output=True, text=True))
    return run(args)


def _require_clean_tree() -> None:
    """A release must be cut from a clean tree so CI tests the SAME code that ships.
    Refuse --publish when there are uncommitted TRACKED changes."""
    r = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                       cwd=str(REPO), capture_output=True, text=True)
    if r.returncode == 0 and r.stdout.strip():
        raise SystemExit("release: the working tree has uncommitted tracked changes. Cut a release "
                         "from a clean, pushed tree so CI validates the same code. Commit or stash first.")


def _require_head_matches_origin(ref: str = "master", *, runner=None) -> None:
    """The release build is assembled from the LOCAL working tree, but the CI gate runs
    on origin/<ref> and the verification record + tag key on local HEAD. If local HEAD
    is not origin/<ref>, CI could pass on code that is NOT what ships. Enforce that they
    are the SAME commit, so CI validated exactly the built artifact. *runner* injectable."""
    _git(["fetch", "origin", ref], runner)   # refresh; a fetch failure surfaces as a mismatch below
    head = (getattr(_git(["rev-parse", "HEAD"], runner), "stdout", "") or "").strip()
    remote = (getattr(_git(["rev-parse", f"origin/{ref}"], runner), "stdout", "") or "").strip()
    if not head or not remote:
        raise SystemExit(f"release: could not resolve HEAD or origin/{ref} to confirm they match; "
                         f"cut the release from a checkout of the pushed {ref} branch.")
    if head != remote:
        raise SystemExit(
            f"release: HEAD ({head[:12]}) is not origin/{ref} ({remote[:12]}). CI runs on "
            f"origin/{ref} while the build is your LOCAL tree, so they must be the same commit. "
            f"Check out and pull origin/{ref} (or push HEAD to {ref}) before publishing.")


def _tag_commit(tag: str, *, runner=None) -> str:
    """The COMMIT a tag points at (peeled through an annotated tag), preferring origin
    and falling back to a local tag; "" when the tag does not exist. Peeling matters so
    an annotated tag compares equal to the HEAD commit, not to its tag object."""
    ls = _git(["ls-remote", "--tags", "origin", tag], runner)
    remote = ""
    for line in (getattr(ls, "stdout", "") or "").splitlines():
        sha, _, name = line.partition("\t")
        # The peeled "refs/tags/<tag>^{}" line (annotated tags) comes second and wins.
        if name.strip() in (f"refs/tags/{tag}", f"refs/tags/{tag}^{{}}"):
            remote = sha.strip()
    if remote:
        return remote
    loc = _git(["rev-parse", "-q", "--verify", f"refs/tags/{tag}^{{commit}}"], runner)
    return (getattr(loc, "stdout", "") or "").strip() if getattr(loc, "returncode", 1) == 0 else ""


def _require_tag_available(tag: str, *, runner=None) -> None:
    """A fresh release must not reuse an existing tag that points at a DIFFERENT commit
    (e.g. the never-distributed v0.1.1 micro-tag): publishing would attach the signed
    build to a stale commit. Refuse when *tag* already exists at a commit other than
    HEAD; a tag already AT HEAD is fine (idempotent re-publish)."""
    existing = _tag_commit(tag, runner=runner)
    if not existing:
        return
    head = (getattr(_git(["rev-parse", "HEAD"], runner), "stdout", "") or "").strip()
    if head and existing != head:
        raise SystemExit(
            f"release: tag {tag} already exists at {existing[:12]}, not the release commit "
            f"{head[:12]}. localm cuts {tag} fresh at the release commit, so delete the stale "
            f"tag first:\n  git push origin --delete {tag}\n  git tag -d {tag}\n"
            "then re-run make_release --publish.")


def _require_verification_record() -> None:
    """RULE: a release must not publish until the build has been COLD-INSTALLED and every
    changelog feature exercised for REAL (a model loads AND things actually work front to
    back, not just import). CI cannot cover that; a human/agent does it per
    RELEASE.md and records the verdict. Publishing refuses unless
    a PASSING record exists for the EXACT release commit (a stale record for older code is
    keyed to a different sha and does not count)."""
    import release_verify as rv   # sibling; already on sys.path (build_release import above)
    sha = rv.current_sha(REPO)
    if not sha:
        raise SystemExit("release verify: cannot determine HEAD sha (not a git checkout).")
    if not rv.has_passing_record(sha, REPO):
        raise SystemExit(
            f"release verify: no PASSING functional-verification record for {sha[:12]} at "
            f"{rv.record_path(sha, REPO)}.\nCold-install the build and exercise every changelog "
            "item per RELEASE.md (real use, not just 'it loads'), record "
            "the verdict, then re-run make_release --publish.")
    print(f"release verify: functional-verification record found for {sha[:12]} (PASS).")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Assemble + sign a localm release build.")
    p.add_argument("--key", type=Path, default=None,
                   help="Ed25519 private key PEM (else $LOCALM_SIGNING_KEY)")
    p.add_argument("--out", type=Path, default=None,
                   help="build.zip path (default: dist/localm-<version>.zip)")
    p.add_argument("--publish", action="store_true",
                   help="gh release create vX.Y.Z with the zip + .sig (runs the CI gate first)")
    p.add_argument("--ci-ref", default="master",
                   help="git ref the pre-publish CI run targets (default: master)")
    args = p.parse_args(argv)

    keypath = args.key
    if keypath is None and os.environ.get("LOCALM_SIGNING_KEY"):
        keypath = Path(os.environ["LOCALM_SIGNING_KEY"])
    if keypath is None:
        raise SystemExit("no signing key: pass --key or set LOCALM_SIGNING_KEY (kept OUT of the repo).")
    if not keypath.is_file():
        raise SystemExit(f"signing key not found: {keypath}")

    version = (REPO / "VERSION").read_text(encoding="utf-8").strip()
    out = args.out or (REPO / "dist" / f"localm-{version}.zip")
    tag = f"v{version}"

    # Pre-publish gates, cheapest-first, fail fast before the heavy build/sign/CI:
    #   1. clean tree (no uncommitted tracked changes),
    #   2. HEAD == origin/<ci-ref> (so the CI run below validates the EXACT built commit,
    #      not a diverged local tree),
    #   3. the tag is free at the release commit (no collision with an old/reused tag),
    #   4. a live functional-verification record for THIS commit (cold-install + exercise
    #      every changelog item by hand - the gate CI cannot cover),
    #   5. the FIRST heavy gate: ONE full CI pass over the whole repo (enabling the
    #      runners if a maintainer disabled them).
    if args.publish:
        _require_clean_tree()
        _require_head_matches_origin(args.ci_ref)
        _require_tag_available(tag)
        _require_verification_record()
        require_ci_green(args.ci_ref)

    # 1. assemble from the manifest (refuses a dirty manifest; self-verifies verify_zip).
    #    CHANGELOG.md ships VERBATIM as a manifest release-include: this tooling never
    #    generates, rewrites, or truncates it. The changelog is APPEND-ONLY and hand-
    #    maintained (AGENTS.md) - a new version's section is added ABOVE the prior ones
    #    by hand, and check_hygiene.py fails the build if any shipped entry is removed.
    members = build_release.build(out, force=True)
    # 2. sign it (writes <out>.sig)
    sig_path = Path(str(out) + ".sig")
    if sign_release._sign(out, keypath, sig_path) != 0:
        raise SystemExit("signing failed")
    # 3. self-check: the signed build must verify against the SHIPPED pinned key
    _verify_against_pinned(out, sig_path)
    # 4. smoke: the release must IMPORT AND RUN on its own (catches an omitted runtime
    #    file before it reaches a user). Gates publish - refuses a build that will not run.
    smoke_test(out)
    print(f"built + signed {out} ({len(members)} files) and {sig_path.name}")
    print("signature verifies against the pinned key; release imports + runs (smoke OK)")

    if args.publish:
        # CI already passed (the FIRST gate above) and the artifact is built + signed +
        # smoke-verified; publish it.
        cmd = ["gh", "release", "create", tag, str(out), str(sig_path),
               "--title", version, "--notes", f"localm {version}"]
        print("publishing:", " ".join(cmd))
        if subprocess.run(cmd, cwd=str(REPO)).returncode != 0:
            raise SystemExit("gh release create failed")
        print(f"published {tag}")
    else:
        print(f"\nnext: gh release create {tag} {out} {sig_path} --title {version} --notes ...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

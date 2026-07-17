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

With --publish, the build is assembled from the exact commit pinned by the HEAD==
origin gate (via build_release.build(..., commit=...) / git archive), not from the
live working tree - so nothing edited on disk during the multi-minute pre-publish CI
wait can end up in the signed artifact. A plain (non-publish) build still reads the
working tree, for quick local iteration.
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
    # Recovery tooling must ship, or a broken update cannot be rolled back.
    "scripts/rollback_update.py", "rollback.bat", "rollback.sh",
    # The post-restart health watchdog must ship, or a broken auto-update is
    # never detected/rolled back (LM-DA-011).
    "scripts/update_watchdog.py",
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


def _run_ids(runner, workflow, *, required=False):
    """The set of existing run databaseIds for *workflow*.

    A FAILED query (non-zero gh exit, or unparseable output) is NOT the same as a
    successful query that returned zero runs. For the BEFORE-snapshot that
    require_ci_green() diffs against (*required* set), a failed query raises SystemExit
    instead of masquerading as an empty set: with ``before == set()`` the first poll
    treats every PRE-EXISTING run as 'new', so ``max(new)`` can select a stale already-
    green run and ``gh run watch`` returns 0 immediately, printing 'CI passed' without
    ever validating the run this gate just triggered (AGENTS.md rule 5: a gate that
    passes without actually checking is the exact harm). A non-required (polling) call
    keeps returning an empty set on a transient error - that just means 'no new run seen
    yet', and the appear-timeout below still fires if it never recovers, so it fails loud
    too, just later."""
    r = runner(["run", "list", "--workflow", workflow, "--limit", "15", "--json", "databaseId"])
    if getattr(r, "returncode", 1) != 0:
        if required:
            raise SystemExit(
                "release CI gate: could not snapshot existing CI runs before triggering a "
                f"new one: {getattr(r, 'stderr', '').strip()}. Refusing to publish - a failed "
                "snapshot would make a pre-existing green run look like the one just triggered.")
        return set()
    try:
        return {int(x["databaseId"]) for x in json.loads(r.stdout or "[]")}
    except Exception as e:
        if required:
            raise SystemExit(
                "release CI gate: could not parse the CI run list for the before-snapshot "
                f"({e}). Refusing to publish - see above.")
        return set()


def _actions_enabled(runner):
    """The repo-level GitHub Actions switch (repos/{owner}/{repo}/actions/permissions
    -> {"enabled": bool}), which is INDEPENDENT of a workflow's own enabled/disabled
    `state` queried in require_ci_green() below. Returns True/False, or None if the
    query itself failed or was unparseable - the caller treats None as 'could not
    tell', distinct from either extreme."""
    r = runner(["api", "repos/{owner}/{repo}/actions/permissions"])
    if getattr(r, "returncode", 1) != 0:
        return None
    try:
        return bool(json.loads(r.stdout).get("enabled"))
    except Exception:
        return None


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
    # 0. GitHub has TWO independent switches gating whether Actions can run: this
    #    repo-level permission and the per-workflow `state` checked in step 1 below. A
    #    workflow can report state == "active" while Actions is disabled for the WHOLE
    #    repo, in which case `gh workflow run` exits 0, no run is ever created, and step
    #    3's appear-timeout below fires ~3 minutes later blaming "the run did not
    #    appear" - true, but not the real cause (verified live 2026-07-17 during the
    #    0.1.2 release: the repo switch was {"enabled": false} while `gh workflow list`
    #    reported ci.yml as "active" throughout). Check the repo-level switch FIRST so a
    #    misconfigured repo fails immediately with the actual cause and the exact fix,
    #    not a misdiagnosed timeout mid-publish. Unlike the workflow-level switch below,
    #    this one is NOT auto-enabled here: it can gate ALL Actions repo-wide for a
    #    maintainer's deliberate reason (cost, org policy), so flipping it is the
    #    maintainer's call, not this script's - re-enabling one specific workflow is a
    #    much smaller hammer than re-enabling Actions for the entire repo.
    enabled = _actions_enabled(runner)
    if enabled is False:
        raise SystemExit(
            "release CI gate: GitHub Actions is disabled for this repo at the repo level "
            "(repos/{owner}/{repo}/actions/permissions -> enabled=false), independent of "
            "the workflow's own enabled/disabled state. A triggered run would never "
            "start. Re-enable it, then retry:\n"
            "  gh api -X PUT repos/{owner}/{repo}/actions/permissions -f enabled=true")
    if enabled is None:
        print("release CI gate: could not read the repo-level Actions permission "
              "(continuing - a genuinely disabled repo is still caught by the "
              "appear-timeout below, just without this earlier diagnosis).")
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
    # 2. snapshot existing runs, then trigger a run on *ref*. The snapshot is
    #    load-bearing (the poll below diffs against it), so a FAILED snapshot query
    #    must abort here, not silently become an empty set (see _run_ids).
    before = _run_ids(runner, workflow, required=True)
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
    # Refresh origin/<ref> from the remote FIRST. A fetch failure must abort: the old
    # "a fetch failure surfaces as a mismatch below" reasoning is FALSE when the stale
    # local origin/<ref> still equals HEAD (e.g. after an agent squash-merged via `gh`,
    # which never updates local remote-tracking refs) - the compare below would then
    # pass while GitHub's real <ref> has moved on, which is exactly the TOCTOU this gate
    # exists to close. So we cannot trust origin/<ref> unless the fetch actually ran.
    fetch = _git(["fetch", "origin", ref], runner)
    if getattr(fetch, "returncode", 1) != 0:
        raise SystemExit(
            f"release: could not refresh origin/{ref} (git fetch failed: "
            f"{getattr(fetch, 'stderr', '').strip()}). Refusing to publish against a "
            f"possibly-stale ref - a stale local origin/{ref} that still equals HEAD would "
            "otherwise let this gate pass while the real remote has moved on.")
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
    #      not a diverged local tree) - and PIN that commit's sha right here, so the
    #      build below reads from the git object database at this exact commit, not
    #      from the working tree as it happens to look after the CI wait below (TOCTOU:
    #      CI takes minutes; a tracked-file edit landing on disk during that wait must
    #      never silently ship in the signed artifact),
    #   3. the tag is free at the release commit (no collision with an old/reused tag),
    #   4. a live functional-verification record for THIS commit (cold-install + exercise
    #      every changelog item by hand - the gate CI cannot cover),
    #   5. the FIRST heavy gate: ONE full CI pass over the whole repo (enabling the
    #      runners if a maintainer disabled them).
    release_sha = None
    if args.publish:
        _require_clean_tree()
        _require_head_matches_origin(args.ci_ref)
        head = _git(["rev-parse", "HEAD"])
        release_sha = (getattr(head, "stdout", "") or "").strip()
        if not release_sha:
            raise SystemExit("release: could not resolve HEAD to pin the build commit.")
        _require_tag_available(tag)
        _require_verification_record()
        require_ci_green(args.ci_ref)

    # 1. assemble from the manifest (refuses a dirty manifest; self-verifies verify_zip).
    #    CHANGELOG.md ships VERBATIM as a manifest release-include: this tooling never
    #    generates, rewrites, or truncates it. The changelog is APPEND-ONLY and hand-
    #    maintained (AGENTS.md) - a new version's section is added ABOVE the prior ones
    #    by hand, and check_hygiene.py fails the build if any shipped entry is removed.
    #    --publish: build.zip is assembled from release_sha (the commit pinned above,
    #    the SAME one CI just validated), via git archive, not from live disk - so the
    #    signed artifact provably matches the verified commit regardless of what has
    #    since changed on disk. A plain (non-publish) build still reads the working
    #    tree, for quick local iteration.
    members = build_release.build(out, force=True, commit=release_sha)
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

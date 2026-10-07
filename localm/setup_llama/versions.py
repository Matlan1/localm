# SPDX-License-Identifier: AGPL-3.0-or-later
"""Which llama.cpp release is installed: the stored pin, tracking upstream's
newest release, the rollback history, the update check, and the --tag and
--rollback requests.
"""

from __future__ import annotations

import json
import re
import time
import urllib.request
from typing import Optional

import click

from localm import config
from localm.debuglog import logger
from localm.setup_llama._common import console
from localm.setup_llama.pins import (_PINNED_TAG, _ROCM_BUILD, _ROCM_TAG, _TRACK_DEFAULT,
                                     _TRACK_LATEST, _UPSTREAM_REPO)
import localm.setup_llama as _sl

# How many past provisions to remember. Rollback only ever needs the previous
# DISTINCT tag, but keeping a short run of them means a user who rolled back and
# then re-pinned can still see where they have been, and it bounds a config key
# that would otherwise grow without limit on a box that re-provisions often.
_RUNTIME_HISTORY_MAX = 20


# A release tag is interpolated straight into a GitHub API path and a download
# URL, so it is validated as a PATH SEGMENT, not merely as "looks like a tag": a
# value carrying '/', '..', '?' or '#' would silently retarget the request at a
# different endpoint. Broader than upstream's own bNNNNN shape: the check is
# about what is safe in a URL, which is the part that must never be relaxed.
_TAG_SAFE_RE = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")


# What a usable tag looks like, in one sentence, for whoever has to REFUSE one.
# Shared by the CLI (which raises a ClickException) and the GUI route (which
# raises a 400), so both state the same rule.
TAG_HELP = ("Use a tag as upstream publishes it, for example 'b10355' (letters, "
            "digits, dot, dash and underscore only), or "
            f"{_TRACK_DEFAULT!r} for the build localm ships and confirmed, or "
            f"{_TRACK_LATEST!r} for upstream's newest.")


def is_safe_tag(tag: "Optional[str]") -> bool:
    """Whether *tag* is safe to interpolate into a release URL path segment.

    PUBLIC: the GUI's runtime route accepts a caller-supplied tag and must
    refuse a bad one with a 400 BEFORE dispatching a job, and it has to refuse
    EXACTLY what the CLI refuses. One predicate, one answer, whichever surface
    asks; _validated_tag delegates here rather than carrying its own regex."""
    tag = (tag or "").strip()
    return bool(_TAG_SAFE_RE.match(tag)) and ".." not in tag


def tracks_latest() -> bool:
    """Whether this install has opted IN to upstream's newest release rather than
    the confirmed build localm ships (``setup-llama --tag latest``).

    A SEPARATE accessor from pinned_tag() rather than a third possible return
    value from it: every caller of pinned_tag() interpolates what it gets into a
    release URL path segment, so the sentinel must never leak through there.

    Never raises, same contract as pinned_tag(): an unreadable config degrades to
    "not tracking", which is the conservative answer - it means the shipped,
    confirmed pin."""
    try:
        raw = config.load_config().get("llama_runtime_pin") or ""
    except Exception:
        return False
    return str(raw).strip().lower() == _TRACK_LATEST


def pinned_tag() -> "Optional[str]":
    """The exact llama.cpp release tag the user has pinned, or None when they
    have not pinned one (the default, which installs _PINNED_TAG, and the
    ``--tag latest`` tracking mode, which is tracks_latest()'s business).

    VALIDATED ON READ, not only where --tag writes it. The CLI flag is not the
    only way this value can arrive: the key is HIDDEN with no coercion branch, so
    PATCH /v1/config stores whatever it is handed (owner-gated, but stored
    verbatim - see test_config_plugin_state_gate.py), and config.json is a plain
    file a user can edit by hand, which no route can police. Checking here covers
    every entry point at the one place the value is actually used, so a tag that
    would escape its URL path segment can never reach _release_assets. An unsafe
    stored value is treated as NO PIN and said out loud rather than silently
    obeyed or silently dropped.

    Never raises: a pin is read on the provisioning path, and an unreadable
    config must degrade to "no pin" rather than break setup entirely."""
    try:
        raw = config.load_config().get("llama_runtime_pin") or ""
    except Exception:
        return None
    raw = str(raw).strip()
    if not raw or raw.lower() == _TRACK_LATEST:
        return None
    if not is_safe_tag(raw):
        console.print(f"[yellow]Warning:[/yellow] ignoring the stored llama.cpp "
                      f"pin {raw!r} - it is not a usable release tag. Set one "
                      "with [bold]localm setup-llama --tag <tag>[/bold].")
        logger.warning("ignoring an unsafe llama_runtime_pin from config: %r", raw)
        return None
    return raw


def set_pinned_tag(tag: "Optional[str]") -> None:
    """Store the user's build choice: an exact tag, the _TRACK_LATEST sentinel,
    or falsy to clear it back to the shipped _PINNED_TAG. Raises on a config
    write failure: unlike recording history, a choice the user explicitly asked
    for must never silently fail to stick (a silently-unpinned install is exactly
    the surprise this whole area exists to remove)."""
    value = (tag or "").strip()
    config.update_config(lambda cfg: cfg.__setitem__("llama_runtime_pin", value))


def _record_runtime_history(backend: str, tag: "Optional[str]") -> None:
    """Append a successful provision to the rollback history. Best-effort by
    design - the provision itself already succeeded, so failing to journal it
    must not turn a working install into an error - but a failure is LOGGED
    rather than swallowed, because the visible symptom otherwise is a --rollback
    that cannot find a build the user knows they had.

    A repeat of the newest entry is collapsed rather than appended, so
    re-running setup-llama on the same build does not push the previous distinct
    tag out of a bounded list and quietly destroy the rollback target."""
    if not tag:
        # Nothing to roll back TO. A tagless provision (--from, --url, an
        # unrecorded backend) is a real event, but it cannot name a build, and
        # journalling it as an entry with no tag would let --rollback offer a
        # target it cannot install.
        return
    entry = {"backend": backend, "tag": tag, "at": int(time.time())}

    def _mutate(cfg: dict) -> None:
        hist = cfg.get("llama_runtime_history")
        hist = list(hist) if isinstance(hist, list) else []
        if hist and isinstance(hist[-1], dict) and \
                hist[-1].get("backend") == backend and hist[-1].get("tag") == tag:
            hist[-1] = entry
        else:
            hist.append(entry)
        cfg["llama_runtime_history"] = hist[-_RUNTIME_HISTORY_MAX:]

    try:
        config.update_config(_mutate)
    except Exception as e:
        logger.debug("could not record the runtime history entry %r: %s", entry, e)
        # Visible, not only in the debug log: without this the failure shows up
        # much later as "no earlier build is recorded ... nothing to roll back
        # to", which collapses two different situations into one message - "you
        # have only ever had this build" and "we could not write down that you
        # had another one". Said here, at the moment it happens, the user can
        # act on it; said later by --rollback, it is indistinguishable from
        # normal. Still not fatal: the install itself succeeded.
        console.print(f"[yellow]Warning:[/yellow] installed {backend} {tag}, but "
                      f"could not record it for rollback ({e}). "
                      "[bold]localm setup-llama --rollback[/bold] will not offer "
                      "this build later.")


def runtime_history() -> list:
    """The recorded provisions, oldest first. Filtered to well-formed entries
    whose tag is SAFE, so neither a hand-edited config nor a verbatim PATCH can
    make --rollback offer a nonsense - or hostile - target.

    The safety filter matters here and not only in pinned_tag(): --rollback takes
    a tag from this list and pins it, so an unchecked entry would become a pin by
    a route that never passed through _validated_tag."""
    try:
        raw = config.load_config().get("llama_runtime_history")
    except Exception:
        return []
    if not isinstance(raw, list):
        return []
    return [e for e in raw
            if isinstance(e, dict) and is_safe_tag(str(e.get("tag") or ""))]


def previous_tag(backend: str) -> "Optional[str]":
    """The most recent recorded tag for *backend* that is NOT the one currently
    installed - i.e. what --rollback goes back to. None when there is no such
    build to return to.

    Compared against the MARKER (what is actually on disk), not against the
    newest history entry, so a rollback still works after a history write failed
    or after the runtime dir was re-provisioned by something that did not
    journal. The marker is the ground truth for "what is installed"; history is
    only the list of candidates."""
    current = _sl.installed_build()
    for entry in reversed(runtime_history()):
        if entry.get("backend") != backend:
            continue
        tag = str(entry.get("tag")).strip()
        if tag and tag != current:
            return tag
    return None


def check_runtime_update() -> dict:
    """Compare the installed llama.cpp runtime against what ``setup-llama``
    would install right now, without provisioning anything: the read-only
    counterpart to a real re-provision, for a "check for updates" surface (the
    GUI's runtime-update card; see localm/plugins/gui/routes/runtime.py).

    The comparison target is whatever ``setup-llama`` would install right now,
    which is the point - the two must never disagree, or the GUI offers an
    update to a build the command would not install. So: an exact PIN if one is
    set (an install that pinned away from a broken release must not be told a
    newer build is "available" - that newer build is exactly what it pinned away
    from), else upstream's newest when the user opted into tracking, else the
    shipped ``_PINNED_TAG``. ``amd-rocm`` compares against its fixed
    ``_ROCM_BUILD`` (the lemonade-sdk build plus its SIMD CPU backend), since
    that build is never resolved from an upstream tag at all.

    ONLY THE TRACKING CASE MAKES A NETWORK CALL. The default path answers from a
    constant, so the GUI's runtime-update card does not reach GitHub on every
    check. The offload in localm/plugins/gui/routes/runtime.py still applies: a
    tracking install needs it, and a route cannot know which kind of install it
    is serving.

    This target is a CANDIDATE, not a proof that it loads on THIS machine: that
    is only established by attempting the provision, which is what
    ``_provision_with_fallback`` does on every install path. This function only
    says whether the installed build differs from the candidate, and never
    re-provisions anything.

    Returns ``{installed, backend, current, target, newer, pinned, previous}``.
    ``installed`` is False when nothing has been provisioned yet (there is no
    "update" for a runtime that was never set up - that is initial setup, a
    different action). ``previous`` is ``--rollback``'s own target
    (``previous_tag(backend)``), included here so the same read-only check
    that powers the "update available" card can also decide whether a
    rollback affordance has anything to offer, without a second round trip.
    Never raises: an unreadable pin/marker degrades to the safe "nothing to
    report" shape rather than breaking the check (mirrors ``pinned_tag()``'s
    own never-raises contract)."""
    backend = _sl.installed_backend()
    if not backend:
        return {"installed": False, "backend": None, "current": None,
                "target": None, "newer": False, "pinned": None, "previous": None}
    current = _sl.installed_build()
    pin = _sl.pinned_tag()
    if backend == "amd-rocm":
        target = _ROCM_BUILD
    elif pin:
        target = pin
    elif tracks_latest():
        target = _sl._latest_tag()
    else:
        target = _PINNED_TAG
    newer = bool(target) and target != current
    return {"installed": True, "backend": backend, "current": current,
            "target": target, "newer": newer, "pinned": pin,
            "previous": previous_tag(backend)}


def _tag_for(backend: str) -> str:
    """The upstream llama.cpp release tag to provision for *backend*: the user's
    exact pin when one is set, else upstream's newest if they opted into tracking
    it, else the confirmed build localm ships (_PINNED_TAG).

    THE ONLY PLACE THAT DECIDES A TAG for the upstream-resolved backends, so a
    choice cannot be honoured on one code path and ignored on another. Note it is
    NOT consulted for amd-rocm, whose build comes from lemonade-sdk's own
    release numbering (_ROCM_TAG) - a different tag space entirely, in which an
    upstream bNNNNN means nothing. _pin_note_for_backend says so out loud rather
    than letting the pin look applied.

    THE DEFAULT BRANCH MAKES NO NETWORK CALL, and that is the substance of the
    change rather than a side benefit: while this function ended in a live
    release-listing lookup, what an install got was decided by whoever had
    published most recently, on every machine, for every released localm - so a
    build nobody here had ever run could arrive without a single localm change.
    A constant cannot do that."""
    pin = _sl.pinned_tag()
    if pin:
        return pin
    if tracks_latest():
        return _sl._latest_tag()
    return _PINNED_TAG


def _pin_note_for_backend(backend: str) -> None:
    """Say plainly when a pin the user set does not apply to the backend being
    provisioned, instead of dropping it silently. Only amd-rocm is in that
    position today: its tag is lemonade-sdk's, not upstream's."""
    if backend != "amd-rocm":
        return
    pin = _sl.pinned_tag()
    if pin:
        console.print(
            f"[yellow]Note:[/yellow] the pinned llama.cpp build {pin} does not "
            f"apply to the amd-rocm backend - it ships from lemonade-sdk's own "
            f"release numbering ({_ROCM_TAG}), a different tag series. The pin "
            "stays set and applies to every other backend.")
    elif tracks_latest():
        # Same reason, other choice: --tag latest is equally inapplicable here,
        # and saying nothing would let a user believe this install is tracking
        # upstream when this backend cannot.
        console.print(
            "[yellow]Note:[/yellow] '--tag latest' does not apply to the "
            f"amd-rocm backend - it ships from lemonade-sdk's own release "
            f"numbering ({_ROCM_TAG}), a different tag series, fixed by the "
            "localm release you are running. The setting stays and applies to "
            "every other backend.")


def _latest_tag() -> str:
    """The newest ggml-org/llama.cpp release tag that actually has its build
    assets uploaded, or _PINNED_TAG if no such release can be found (offline,
    rate-limited, etc.).

    ONLY REACHED WHEN THE USER OPTED IN with ``--tag latest``; it is not the
    default (see _tag_for). What it returns is by construction a build nobody
    here has run.

    The unavailable case falls back to _PINNED_TAG and not to some older release,
    so a failed lookup yields the confirmed pin rather than a build that is BOTH
    unconfirmed and not what was asked for; the message below says which they
    got.

    Upstream publishes a release (tag + notes) as soon as it is cut, then its CI
    matrix uploads the ~25 platform archives afterwards - which can take a while.
    Right after publish, ``/releases/latest`` can point at a tag whose ``assets``
    array is still genuinely empty even though the release body already lists
    the (soon-to-exist) download URLs, and resolving to that tag produces a
    confident-looking match that 404s. So this scans recent releases
    newest-first and uses the first one that already has assets, skipping any
    still-uploading release."""
    tags = _sl._recent_tags()
    if tags:
        return tags[0]
    # Surface it: the user asked to track upstream and is not getting upstream's
    # newest, which they would otherwise discover much later as "localm installed
    # an old build". Name what they got.
    console.print(f"[yellow]No ggml-org/llama.cpp release with an uploaded build "
                  f"was found among the most recent releases. Installing localm's "
                  f"confirmed build {_PINNED_TAG} instead - rerun later for "
                  "upstream's newest.[/yellow]")
    return _PINNED_TAG


# Upstream build tags are "b" plus a monotonically increasing build number.
# Matches scripts/check_llama_pin.py's _TAG_RE.
_RELEASE_TAG_RE = re.compile(r"^b(\d+)$")


def _recent_tags(limit: int = 10) -> list:
    """Upstream release tags that already have their build assets uploaded,
    NEWEST FIRST. Empty when the lookup is unavailable.

    Split out of _latest_tag rather than added beside it: that function ALREADY
    fetched ten releases and returned only the first, discarding the rest. The
    tag walk-back needs those discarded entries, so exposing them costs ZERO
    extra requests - the same shape as the tag that was already resolved and
    thrown away in the record/pin unit. One list, one call, one skip rule, so
    "which releases are candidates" cannot be answered two different ways."""
    api = f"https://api.github.com/repos/{_UPSTREAM_REPO}/releases?per_page={int(limit)}"
    out: list = []
    try:
        req = urllib.request.Request(api, headers={"Accept": "application/vnd.github+json",
                                                   "User-Agent": "localm-setup-llama"})
        with _sl.verified_urlopen(req, timeout=10) as r:
            releases = json.loads(r.read().decode("utf-8"))
        for rel in releases:
            if not isinstance(rel, dict) or rel.get("draft"):
                continue
            tag = rel.get("tag_name")
            # Not excluded on prerelease: upstream flags every real build
            # release prerelease=true. See
            # test_recent_tags_includes_a_prerelease_flagged_release.
            #
            # The asset check is the whole point of scanning rather than taking
            # /releases/latest: a release is published before its CI uploads the
            # ~25 archives, so a tag with an empty assets array 404s on download.
            if isinstance(tag, str) and _RELEASE_TAG_RE.match(tag) and rel.get("assets"):
                out.append(tag)
    except Exception as e:
        # Best-effort like its two siblings (assets._release_assets,
        # cuda._pypi_wheel_url_and_sha), and logged like them: every caller has a pinned fallback, so
        # this must not raise, but "the lookup was unavailable" must stay
        # discoverable - a refused downgrade redirect here would otherwise leave
        # no trace at all.
        logger.debug("release tag listing failed for %s (%s)", api, e)
        return []
    return out


def _validated_tag(raw: str) -> str:
    """*raw* as a usable release tag, or a ClickException naming the problem.

    The CLI-facing half of is_safe_tag: same predicate, so the flag and the
    stored-value check can never disagree about what a usable tag is."""
    tag = (raw or "").strip()
    if not is_safe_tag(tag):
        raise click.ClickException(
            f"{raw!r} is not a usable release tag. {TAG_HELP}")
    return tag


def _apply_version_request(tag: Optional[str], rollback: bool, backend: str,
                           from_dir: Optional[str], url: Optional[str]) -> None:
    """Act on --tag / --rollback BEFORE any provisioning: validate them, resolve
    what --rollback means, and move the pin.

    The pin is written FIRST, so the rest of main() provisions through the
    normal _tag_for() path with no special-casing - one code path decides a tag
    whether it came from a flag or from a pin set weeks ago. That is also what
    makes the pin apply to `localm update`, which re-invokes this command with
    nothing but --backend (see _apply_update.post_swap_command).

    Anything this cannot honour is REFUSED with a reason rather than ignored: a
    silently-dropped --tag would leave the user believing a build is pinned when
    it is not, which is worse than the drift the pin exists to stop."""
    # `tag is None` (the flag was not passed) is distinguished from
    # `tag == ""` (it was passed empty, e.g. a shell variable that expanded to
    # nothing). Treating the empty string as "no request" would DROP a request
    # the user made, which is the exact failure this function exists to prevent;
    # it falls through to _validated_tag and is refused with a reason.
    if tag is not None and rollback:
        raise click.ClickException(
            "--tag and --rollback both choose a build; pass only one. "
            "--rollback goes to the previous recorded build, --tag names one.")
    if tag is None and not rollback:
        return
    if from_dir or url:
        # --from/--url install an artifact this command did not resolve from a
        # release, so there is no tag to record or pin. Refusing beats accepting
        # a flag that could not take effect.
        which = "--from" if from_dir else "--url"
        raise click.ClickException(
            f"{'--tag' if tag is not None else '--rollback'} selects an upstream "
            f"llama.cpp release, so it cannot be combined with {which}, which "
            "installs a build you supply. Run them separately.")

    if tag is not None:
        # TWO WORDS, because they name two different destinations. Before the pin
        # existed there was only one: clearing the pin meant tracking upstream's
        # newest, so "latest" could mean both "unpin" and "track upstream" at
        # once. Now the unpinned default is the build localm confirmed, so a user
        # who wants upstream's newest is asking for something the default is NOT,
        # and reusing one word for both would silently give one of them the other.
        #
        # Both are spelled as words rather than an empty --tag so the intent is
        # visible in shell history and in a script, and so a shell variable that
        # expanded to nothing cannot silently change what an install tracks.
        if tag.strip().lower() == _TRACK_DEFAULT:
            set_pinned_tag(None)
            console.print(f"[green]Back to localm's confirmed build[/green] "
                          f"({_PINNED_TAG}) - the one this release was tested "
                          "with. Re-run setup-llama --force to install it now.")
            return
        if tag.strip().lower() == _TRACK_LATEST:
            set_pinned_tag(_TRACK_LATEST)
            console.print("[yellow]Now tracking upstream's newest llama.cpp "
                          "release.[/yellow] That build is whatever ggml-org "
                          "published most recently and localm has NOT tested it; "
                          "upstream has shipped releases this code cannot load. "
                          f"Go back with: [bold]localm setup-llama --tag "
                          f"{_TRACK_DEFAULT}[/bold] ({_PINNED_TAG}).")
            return
        wanted = _validated_tag(tag)
        set_pinned_tag(wanted)
        console.print(f"[green]Pinned[/green] llama.cpp {wanted} - setup-llama "
                      "and localm update will keep this build until you run "
                      f"[bold]localm setup-llama --tag {_TRACK_DEFAULT}[/bold] "
                      f"(back to localm's confirmed {_PINNED_TAG}).")
        return

    # --rollback. The backend is the one the user named, else whatever is
    # installed: history is per-backend, because a cuda tag and a vulkan tag are
    # different builds even when the tag string matches.
    which = backend.lower() if backend and backend.lower() != "auto" else _sl.installed_backend()
    if not which:
        raise click.ClickException(
            "--rollback needs to know which backend to roll back, and nothing is "
            "recorded as installed on this machine. Name it explicitly, for "
            "example: localm setup-llama --rollback --backend vulkan")
    if which == "amd-rocm":
        # amd-rocm's build is fixed by the _ROCM_TAG CONSTANT in pins.py, not
        # resolved from a release listing, so there is exactly one amd-rocm build
        # per localm release and a pin cannot move it. Without this refusal the
        # command printed "Rolling back the amd-rocm runtime to llama.cpp b1288"
        # and then, moments later, the pin note saying that build does not apply
        # to amd-rocm - two contradictory sentences for one action that changed
        # nothing. Refusing with the real reason beats promising and retracting.
        raise click.ClickException(
            f"the amd-rocm backend cannot be rolled back: its build is fixed by "
            f"the localm release you are running ({_ROCM_TAG}, from lemonade-sdk), "
            "not chosen from upstream llama.cpp releases. To try a different "
            "llama.cpp build on this machine, switch backend, for example: "
            "localm setup-llama --backend vulkan --tag <tag>")
    prev = previous_tag(which)
    if not prev:
        current = _sl.installed_build()
        have = f" The build installed now is {current}." if current else ""
        raise click.ClickException(
            f"no earlier llama.cpp build is recorded for the {which} backend, so "
            f"there is nothing to roll back to.{have} Install a specific build "
            "instead, for example: localm setup-llama --tag b10355")
    set_pinned_tag(prev)
    console.print(f"[green]Rolling back[/green] the {which} runtime to llama.cpp "
                  f"{prev}, and pinning it.")

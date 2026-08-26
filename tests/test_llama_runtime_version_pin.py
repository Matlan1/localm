# SPDX-License-Identifier: AGPL-3.0-or-later
"""Recording, pinning and rolling back the provisioned llama.cpp build.

localm resolved the runtime version dynamically at every install and recorded
nothing, so an upstream build that broke a machine arrived on the next install
with no way to name it, pin away from it, or go back. Three separate breakages
in one week came through that mechanism (the llama_context_params ABI shift, the
Windows AMD asset rename at b10356, and a suspected Vulkan fault on b10361).

The tests here are grouped by the property they defend:

  * the tag is RECORDED, for every backend, at no extra network cost;
  * a PIN is honoured by later installs, including the updater's re-provision;
  * ROLLBACK returns to a build that was actually installed before;
  * a request that CANNOT be honoured is refused with a reason, never dropped.
"""

from pathlib import Path

import click
import pytest

from localm import config as cfg
from localm import setup_llama as sl


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A throwaway LOCALM_HOME with config.py's frozen module paths redirected.

    config.py binds CONFIG_FILE at import, so setting the env var alone leaves
    load_config()/update_config() writing the real user's file - the same reason
    conftest's cli_runner fixture patches both.
    """
    h = tmp_path / ".localm"
    h.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LOCALM_HOME", str(h))
    monkeypatch.setattr(cfg, "HOME_DIR", h)
    monkeypatch.setattr(cfg, "CONFIG_FILE", h / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", h / "registry.json")
    return h


def _release_listing(tag, name="llama-x-bin-win-vulkan-x64.zip"):
    """One release asset shaped like the GitHub API's, for a given tag."""
    return [{"name": name,
             "browser_download_url": f"https://example/{tag}/{name}",
             "digest": "sha256:" + "a" * 64}]


# --------------------------------------------------------------------------- #
#  RECORD - the tag reaches the marker, for every backend                      #
# --------------------------------------------------------------------------- #

def test_provision_returns_the_tag_it_installed(monkeypatch, tmp_path, home):
    """_provision_backend reports WHICH build it fetched, so the caller can
    record it. Before this it returned None and the marker could only ever carry
    a version for amd-rocm."""
    monkeypatch.setattr(sl, "_platform_key", lambda: "win32")
    monkeypatch.setattr(sl, "_release_assets",
                        lambda tag, repo=sl._UPSTREAM_REPO: _release_listing(tag))
    monkeypatch.setattr(sl, "_fetch_verified", lambda url, target, sha, what: None)

    assert sl._provision_backend("vulkan", tmp_path, None, False) == sl._PINNED_TAG


def test_recording_the_tag_costs_no_release_lookup_at_all(monkeypatch, tmp_path, home):
    """Recording the tag must add NO network call at all: the pin is a constant,
    so the release-lookup count must be ZERO, not one - "one" would still pass if
    the pin were re-resolved.

    Counted from OUTSIDE, not asserted by raising inside the call: an exception
    raised in a stub is an input to the code under test, and anything catching
    broadly between here and pytest would swallow it.
    """
    calls = []
    listings = []
    monkeypatch.setattr(sl, "_platform_key", lambda: "win32")
    monkeypatch.setattr(sl, "_latest_tag",
                        lambda: (calls.append("latest"), "b10361")[1])
    monkeypatch.setattr(sl, "_release_assets",
                        lambda tag, repo=sl._UPSTREAM_REPO:
                            (listings.append(tag), _release_listing(tag))[1])
    monkeypatch.setattr(sl, "_fetch_verified", lambda url, target, sha, what: None)

    tag = sl._provision_backend("vulkan", tmp_path, None, False)

    assert tag == sl._PINNED_TAG
    assert calls == [], "a default provision must not resolve upstream's newest"
    assert listings == [sl._PINNED_TAG], (
        f"exactly one asset listing, for the pin; got {listings}")


def test_amd_rocm_provision_makes_no_upstream_lookup_at_all(monkeypatch, tmp_path, home):
    """amd-rocm resolves against lemonade-sdk's own numbering, so it must not
    reach for an upstream tag - a call made purely to decorate a marker is still
    a call."""
    calls = []
    monkeypatch.setattr(sl.sys, "platform", "win32")
    monkeypatch.setattr(sl, "_latest_tag",
                        lambda: (calls.append("latest"), "b10361")[1])
    monkeypatch.setattr(sl, "_release_assets", lambda tag, repo=None: [])
    monkeypatch.setattr(sl, "_fetch_verified", lambda url, target, sha, what: None)

    tag = sl._provision_backend("amd-rocm", tmp_path, None, False)

    assert calls == [], "amd-rocm must not resolve an upstream release tag"
    assert tag is None, "amd-rocm has no upstream tag to report"


def test_marker_round_trips_backend_and_build(tmp_path):
    """The recorded pair reads back through the public accessors' own reader."""
    sl._record_provisioned_backend(tmp_path, "vulkan", build="b10361")
    assert sl._provisioned_backend(tmp_path) == "vulkan"
    assert sl._provisioned_build(tmp_path) == "b10361"


def test_installed_build_reads_the_real_runtime_dir(monkeypatch, tmp_path):
    """installed_build() is the public lookup doctor and the bug reporter use.
    It must resolve the target dir the same way provisioning does, or it reads a
    marker nothing wrote."""
    monkeypatch.setattr(sl, "_repo_runtime_lib", lambda: tmp_path)
    sl._record_provisioned_backend(tmp_path, "cuda", build="b10355")
    assert sl.installed_build() == "b10355"
    assert sl.installed_backend() == "cuda"


def test_installed_build_is_none_when_the_marker_predates_tags(monkeypatch, tmp_path):
    """A one-token marker is NORMAL, not corruption. None means 'not recorded'
    and every reader must be able to say so instead of guessing a version."""
    monkeypatch.setattr(sl, "_repo_runtime_lib", lambda: tmp_path)
    sl._record_provisioned_backend(tmp_path, "vulkan")
    assert sl.installed_backend() == "vulkan"
    assert sl.installed_build() is None


def test_fallback_records_the_fallback_tag_not_the_chosen_one(monkeypatch, tmp_path, home):
    """A cuda pick that falls back to vulkan has VULKAN on disk. Recording the
    tag the chosen backend would have used would name a build that was never
    installed - the marker has to describe the disk."""
    seen = []

    def fake_provision(backend, target, sha256, with_cudart, cuda_line=sl._CUDA_LINE, tag=None):
        seen.append(backend)
        (target / sl._lib_name()).write_text("x")
        return {"cuda": "b10361", "vulkan": "b10300"}[backend]

    monkeypatch.setattr(sl, "_provision_backend", fake_provision)
    monkeypatch.setattr(sl, "_clear_target_or_refuse", lambda t: None)
    monkeypatch.setattr(sl, "_install_runtime_wheel", lambda d: True)
    loads = iter([(False, "cuda did not load"), (True, "")])
    monkeypatch.setattr(sl, "_native_loads_ok", lambda: next(loads))

    backend, tag = sl._provision_with_fallback("cuda", tmp_path, None, True,
                                               assume_yes=True)

    assert seen == ["cuda", "vulkan"]
    assert backend == "vulkan"
    assert tag == "b10300", "the recorded build must be the one that loaded"


# --------------------------------------------------------------------------- #
#  PIN - a chosen build sticks, including through the updater's re-provision   #
# --------------------------------------------------------------------------- #

def test_tag_for_prefers_the_pin_over_a_release_lookup(monkeypatch, home):
    """A pin must not merely win the comparison - it must make the lookup
    unnecessary, so a pinned install still works when the release API does not."""
    monkeypatch.setattr(sl, "_latest_tag",
                        lambda: pytest.fail("a pinned install must not query releases"))
    sl.set_pinned_tag("b10355")
    assert sl._tag_for("vulkan") == "b10355"


def test_tag_for_installs_the_confirmed_pin_with_no_user_choice(monkeypatch, home):
    """THE DEFAULT, and it makes NO NETWORK CALL. The _latest_tag stand-in FAILS
    the test if it is reached: 'the default resolves to the pin' and 'the default
    asks nobody' are two different properties, and only the second one stops an
    untested upstream build arriving the moment a third party publishes it."""
    monkeypatch.setattr(
        sl, "_latest_tag",
        lambda: pytest.fail("the default path must not query upstream at all"))
    assert sl.pinned_tag() is None
    assert sl.tracks_latest() is False
    assert sl._tag_for("vulkan") == sl._PINNED_TAG


def test_tag_for_tracks_latest_only_when_the_user_opted_in(monkeypatch, home):
    """Bleeding edge stays available - it just has to be asked for."""
    monkeypatch.setattr(sl, "_latest_tag", lambda: "b10361")
    sl.set_pinned_tag(sl._TRACK_LATEST)
    assert sl.tracks_latest() is True
    assert sl.pinned_tag() is None, "the sentinel must never read as an exact tag"
    assert sl._tag_for("vulkan") == "b10361"


def test_a_pin_survives_a_bare_reprovision(monkeypatch, tmp_path, home):
    """The updater re-invokes `setup-llama --backend <installed>` with no tag
    (see _apply_update.post_swap_command), so the pin is only real if a bare
    provision reads it. This drives the real resolver, not _tag_for directly."""
    monkeypatch.setattr(sl, "_platform_key", lambda: "win32")
    monkeypatch.setattr(sl, "_latest_tag",
                        lambda: pytest.fail("the pin must short-circuit the lookup"))
    seen = {}

    def fake_assets(tag, repo=sl._UPSTREAM_REPO):
        seen["tag"] = tag
        return _release_listing(tag)

    monkeypatch.setattr(sl, "_release_assets", fake_assets)
    monkeypatch.setattr(sl, "_fetch_verified", lambda url, target, sha, what: None)
    sl.set_pinned_tag("b10355")

    assert sl._provision_backend("vulkan", tmp_path, None, False) == "b10355"
    assert seen["tag"] == "b10355", "the pinned tag must be the one resolved against"


def test_tag_flag_sets_the_pin(home):
    sl._apply_version_request("b10355", False, "vulkan", None, None)
    assert sl.pinned_tag() == "b10355"


def test_tag_latest_clears_the_pin(home):
    sl.set_pinned_tag("b10355")
    sl._apply_version_request("latest", False, "vulkan", None, None)
    assert sl.pinned_tag() is None


def test_pin_that_cannot_apply_is_reported_not_dropped(monkeypatch, capsys, home):
    """amd-rocm ships from lemonade-sdk's tag series, so an upstream pin cannot
    apply to it, and that must be reported: saying nothing leaves the user
    believing a build is pinned when it is not."""
    sl.set_pinned_tag("b10355")
    sl._pin_note_for_backend("amd-rocm")
    out = capsys.readouterr().out
    assert "b10355" in out and "amd-rocm" in out
    assert sl.pinned_tag() == "b10355", "the pin stays set for other backends"


def test_no_pin_note_when_the_pin_applies(monkeypatch, capsys, home):
    sl.set_pinned_tag("b10355")
    sl._pin_note_for_backend("vulkan")
    assert capsys.readouterr().out.strip() == ""


# --------------------------------------------------------------------------- #
#  ROLLBACK - back to a build that was really installed                        #
# --------------------------------------------------------------------------- #

def test_history_records_each_provision(home):
    sl._record_runtime_history("vulkan", "b10300")
    sl._record_runtime_history("vulkan", "b10361")
    assert [(e["backend"], e["tag"]) for e in sl.runtime_history()] == [
        ("vulkan", "b10300"), ("vulkan", "b10361")]


def test_history_collapses_a_repeat_instead_of_appending(home):
    """Re-running setup-llama on the same build must not push the previous
    DISTINCT tag out of a bounded list - that would silently destroy the only
    thing --rollback can return to."""
    sl._record_runtime_history("vulkan", "b10300")
    for _ in range(sl._RUNTIME_HISTORY_MAX + 5):
        sl._record_runtime_history("vulkan", "b10361")
    tags = [e["tag"] for e in sl.runtime_history()]
    assert tags == ["b10300", "b10361"]


def test_a_failed_history_write_is_reported_not_only_logged(monkeypatch, capsys, home):
    """A silent journal failure resurfaces much later as --rollback's "nothing to
    roll back to", which reads identically to "you have only ever had this
    build". Two different situations, one message - so say it when it happens.
    Still not fatal: the install itself succeeded."""
    def boom(mutator):
        raise OSError("config is read-only")

    monkeypatch.setattr(cfg, "update_config", boom)

    sl._record_runtime_history("vulkan", "b10361")   # must not raise

    out = capsys.readouterr().out
    assert "could not record it for rollback" in out
    assert "b10361" in out and "read-only" in out, "name the build and the cause"


def test_history_ignores_a_tagless_provision(home):
    """A --from/--url install has no tag, so journalling it would let --rollback
    offer a target it cannot install."""
    sl._record_runtime_history("custom", None)
    assert sl.runtime_history() == []


def test_previous_tag_skips_the_installed_build(monkeypatch, tmp_path, home):
    monkeypatch.setattr(sl, "_repo_runtime_lib", lambda: tmp_path)
    sl._record_runtime_history("vulkan", "b10300")
    sl._record_runtime_history("vulkan", "b10361")
    sl._record_provisioned_backend(tmp_path, "vulkan", build="b10361")
    assert sl.previous_tag("vulkan") == "b10300"


def test_previous_tag_is_per_backend(monkeypatch, tmp_path, home):
    """A cuda b10361 and a vulkan b10361 are different builds; rolling a vulkan
    install back onto a cuda entry would install the wrong thing."""
    monkeypatch.setattr(sl, "_repo_runtime_lib", lambda: tmp_path)
    sl._record_runtime_history("cuda", "b10200")
    sl._record_runtime_history("vulkan", "b10361")
    sl._record_provisioned_backend(tmp_path, "vulkan", build="b10361")
    assert sl.previous_tag("vulkan") is None
    assert sl.previous_tag("cuda") == "b10200"


def test_previous_tag_compares_against_the_marker_not_the_newest_entry(
        monkeypatch, tmp_path, home):
    """The marker is ground truth for what is installed; history is only the
    list of candidates. If a history write failed, the newest entry is NOT what
    is on disk, and rolling back relative to it would go to the wrong build."""
    monkeypatch.setattr(sl, "_repo_runtime_lib", lambda: tmp_path)
    sl._record_runtime_history("vulkan", "b10300")
    sl._record_runtime_history("vulkan", "b10361")
    # Disk says b10300 - the b10361 provision was journalled but never landed.
    sl._record_provisioned_backend(tmp_path, "vulkan", build="b10300")
    assert sl.previous_tag("vulkan") == "b10361"


def test_rollback_pins_the_previous_build(monkeypatch, tmp_path, home):
    monkeypatch.setattr(sl, "_repo_runtime_lib", lambda: tmp_path)
    sl._record_runtime_history("vulkan", "b10300")
    sl._record_runtime_history("vulkan", "b10361")
    sl._record_provisioned_backend(tmp_path, "vulkan", build="b10361")

    sl._apply_version_request(None, True, "auto", None, None)

    assert sl.pinned_tag() == "b10300"


def test_rollback_with_nothing_to_go_back_to_is_refused(monkeypatch, tmp_path, home):
    monkeypatch.setattr(sl, "_repo_runtime_lib", lambda: tmp_path)
    sl._record_provisioned_backend(tmp_path, "vulkan", build="b10361")
    with pytest.raises(click.ClickException) as e:
        sl._apply_version_request(None, True, "auto", None, None)
    assert "nothing to roll back to" in str(e.value)
    assert "b10361" in str(e.value), "say which build is installed now"
    assert sl.pinned_tag() is None, "a refused rollback must not move the pin"


def test_rollback_is_refused_for_amd_rocm(monkeypatch, tmp_path, home):
    """amd-rocm's build is fixed by a constant in the shipped code, so a pin
    cannot move it and the command must refuse rather than announce a rollback
    it cannot perform. The history entries here use LEMONADE's numbering, which
    is not an upstream tag series at all."""
    monkeypatch.setattr(sl, "_repo_runtime_lib", lambda: tmp_path)
    sl._record_runtime_history("amd-rocm", "b1288")
    sl._record_runtime_history("amd-rocm", "b1307")
    sl._record_provisioned_backend(tmp_path, "amd-rocm", build="b1307")

    with pytest.raises(click.ClickException) as e:
        sl._apply_version_request(None, True, "auto", None, None)

    assert "cannot be rolled back" in str(e.value)
    assert sl._ROCM_TAG in str(e.value), "name the build it is fixed at"
    assert "--backend vulkan" in str(e.value), "offer the route that does work"
    assert sl.pinned_tag() is None, "a refused rollback must not move the pin"


def test_rollback_without_a_known_backend_is_refused(monkeypatch, tmp_path, home):
    monkeypatch.setattr(sl, "_repo_runtime_lib", lambda: tmp_path)
    with pytest.raises(click.ClickException) as e:
        sl._apply_version_request(None, True, "auto", None, None)
    assert "--backend" in str(e.value), "tell the user how to name it"


# --------------------------------------------------------------------------- #
#  REFUSALS - a request that cannot be honoured never passes silently          #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("bad", [
    "../../etc/passwd",          # escapes the release path segment
    "b10355/../../other",
    "b10355?x=1",                # a query would retarget the API request
    "b10355#frag",
    "has space",
    "",
])
def test_a_tag_that_is_unsafe_in_a_url_is_refused(bad, home):
    """The tag is interpolated into a GitHub API path and a download URL, so it
    is validated as a path SEGMENT, not merely as 'looks like a tag'."""
    with pytest.raises(click.ClickException):
        sl._apply_version_request(bad, False, "vulkan", None, None)
    assert sl.pinned_tag() is None, "a refused tag must not be pinned"


@pytest.mark.parametrize("good", ["b10355", "b9870", "v1.2.3", "master-1a2b3c"])
def test_a_plausible_tag_shape_is_accepted(good, home):
    """Broader than upstream's own bNNNNN: the check is about what is safe in a
    URL, so a future tag scheme is not refused cosmetically."""
    assert sl._validated_tag(good) == good


def test_tag_and_rollback_together_are_refused(home):
    with pytest.raises(click.ClickException) as e:
        sl._apply_version_request("b10355", True, "vulkan", None, None)
    assert "only one" in str(e.value)
    assert sl.pinned_tag() is None


@pytest.mark.parametrize("from_dir,url", [("/some/dir", None),
                                          (None, "https://example/x.zip")])
def test_tag_with_a_supplied_build_is_refused(from_dir, url, home):
    """--from/--url install an artifact this command did not resolve from a
    release, so there is no tag to pin. Refusing beats accepting a flag that
    could not take effect."""
    with pytest.raises(click.ClickException) as e:
        sl._apply_version_request("b10355", False, "vulkan", from_dir, url)
    assert ("--from" in str(e.value)) if from_dir else ("--url" in str(e.value))
    assert sl.pinned_tag() is None


@pytest.mark.parametrize("planted", [
    "../../../../evil", "b10355/../../other", "b10355?x=1", "b10355#f", "a b",
])
def test_an_unsafe_pin_STORED_IN_CONFIG_never_reaches_a_url(planted, monkeypatch,
                                                            capsys, tmp_path, home):
    """--tag is NOT the only way a pin arrives, so validating there is not enough.

    `llama_runtime_pin` is HIDDEN with no coercion branch, so PATCH /v1/config
    stores whatever it is handed (owner-gated but verbatim - see
    test_config_plugin_state_gate.py), and config.json is a plain file a user can
    edit by hand, which no route can police. The value is interpolated into a
    GitHub API path and a download URL, so a tag carrying '/', '..', '?' or '#'
    would retarget the request. Checked on READ, at the one place it is used.

    Asserts on WHERE THE REQUEST WENT, not merely on the return value: the
    property is "this string never reached a URL", and a resolver call is the
    thing that would carry it there."""
    cfg.update_config(lambda c: c.__setitem__("llama_runtime_pin", planted))
    resolved = []
    monkeypatch.setattr(sl, "_platform_key", lambda: "win32")
    monkeypatch.setattr(sl, "_release_assets",
                        lambda tag, repo=sl._UPSTREAM_REPO:
                            (resolved.append(tag), _release_listing(tag))[1])
    monkeypatch.setattr(sl, "_fetch_verified", lambda url, target, sha, what: None)

    tag = sl._provision_backend("vulkan", tmp_path, None, False)

    assert resolved == [sl._PINNED_TAG], (
        f"the planted value must never be resolved against; got {resolved}")
    assert tag == sl._PINNED_TAG
    assert planted not in "".join(resolved)
    assert "ignoring the stored llama.cpp pin" in capsys.readouterr().out, (
        "an ignored pin must be said out loud, not silently dropped")


def test_an_unsafe_history_entry_cannot_become_a_pin(monkeypatch, tmp_path, home):
    """--rollback takes a tag from history and PINS it, so an unchecked entry
    would become a pin by a route that never passed through --tag's validator."""
    monkeypatch.setattr(sl, "_repo_runtime_lib", lambda: tmp_path)
    cfg.update_config(lambda c: c.__setitem__("llama_runtime_history", [
        {"backend": "vulkan", "tag": "../../evil"},
        {"backend": "vulkan", "tag": "b10361"}]))
    sl._record_provisioned_backend(tmp_path, "vulkan", build="b10361")

    assert [e["tag"] for e in sl.runtime_history()] == ["b10361"]
    assert sl.previous_tag("vulkan") is None, (
        "the only other entry is unsafe, so there is nothing to roll back to")


def test_no_version_request_leaves_the_pin_alone(home):
    sl.set_pinned_tag("b10355")
    sl._apply_version_request(None, False, "vulkan", None, None)
    assert sl.pinned_tag() == "b10355"


def test_a_hand_edited_history_cannot_produce_a_nonsense_target(monkeypatch, tmp_path, home):
    """runtime_history filters malformed entries, so a config someone edited by
    hand cannot make --rollback offer an empty or non-dict target."""
    monkeypatch.setattr(sl, "_repo_runtime_lib", lambda: tmp_path)
    cfg.update_config(lambda c: c.__setitem__("llama_runtime_history", [
        "not-a-dict", {"backend": "vulkan"}, {"backend": "vulkan", "tag": "  "},
        {"backend": "vulkan", "tag": "b10300"}]))
    assert [e["tag"] for e in sl.runtime_history()] == ["b10300"]
    assert sl.previous_tag("vulkan") == "b10300"


# --------------------------------------------------------------------------- #
#  SURFACES - doctor and the bug report can name the build                     #
# --------------------------------------------------------------------------- #

def test_bug_report_carries_the_backend_and_build(monkeypatch, tmp_path, home):
    """The bug report names the backend and the build directly, so a field report
    does not have to infer it from versioned library filenames."""
    from localm import bugreport
    monkeypatch.setattr(sl, "_repo_runtime_lib", lambda: tmp_path)
    sl._record_provisioned_backend(tmp_path, "vulkan", build="b10361")
    sl.set_pinned_tag("b10361")

    diag = bugreport.collect_diagnostics({})

    assert diag["native_runtime_backend"] == "vulkan"
    assert diag["native_runtime_build"] == "b10361"
    assert diag["native_runtime_pin"] == "b10361"


def test_bug_report_says_not_recorded_rather_than_omitting(monkeypatch, tmp_path, home):
    """An install predating build recording is a real, distinguishable state; a
    missing field reads as a collector that failed."""
    from localm import bugreport
    monkeypatch.setattr(sl, "_repo_runtime_lib", lambda: tmp_path)
    sl._record_provisioned_backend(tmp_path, "vulkan")

    diag = bugreport.collect_diagnostics({})

    assert diag["native_runtime_build"] == "not recorded"
    assert "native_runtime_pin" not in diag


def test_rendered_report_shows_the_build(monkeypatch, tmp_path, home):
    """The label must be wired into the rendered body, not merely collected -
    a diagnostics key with no label entry never reaches the report."""
    from localm import bugreport
    monkeypatch.setattr(sl, "_repo_runtime_lib", lambda: tmp_path)
    sl._record_provisioned_backend(tmp_path, "vulkan", build="b10361")

    body = bugreport.build_report("test", context={})

    assert "Native runtime build: b10361" in body


def test_doctor_prints_the_build_and_the_pin(monkeypatch, tmp_path, capsys, home):
    # localm.cli exposes a click Command named `doctor`, which shadows the
    # submodule of the same name - import the function directly.
    from localm.cli.doctor import _check_runtime_build
    monkeypatch.setattr(sl, "_repo_runtime_lib", lambda: tmp_path)
    sl._record_provisioned_backend(tmp_path, "vulkan", build="b10361")
    sl.set_pinned_tag("b10361")

    _check_runtime_build(lambda: tmp_path)

    out = capsys.readouterr().out
    assert "llama.cpp build: b10361" in out
    assert "pinned to b10361" in out


def test_doctor_flags_a_pin_the_disk_disagrees_with(monkeypatch, tmp_path, capsys, home):
    """A pin set but never provisioned through is a real state. Reporting only
    'pinned to X' would assert something the filesystem contradicts."""
    # localm.cli exposes a click Command named `doctor`, which shadows the
    # submodule of the same name - import the function directly.
    from localm.cli.doctor import _check_runtime_build
    monkeypatch.setattr(sl, "_repo_runtime_lib", lambda: tmp_path)
    sl._record_provisioned_backend(tmp_path, "vulkan", build="b10300")
    sl.set_pinned_tag("b10361")

    _check_runtime_build(lambda: tmp_path)

    out = capsys.readouterr().out
    assert "pinned to b10361 but b10300 is installed" in out


def test_doctor_says_unrecorded_rather_than_guessing(monkeypatch, tmp_path, capsys, home):
    # localm.cli exposes a click Command named `doctor`, which shadows the
    # submodule of the same name - import the function directly.
    from localm.cli.doctor import _check_runtime_build
    monkeypatch.setattr(sl, "_repo_runtime_lib", lambda: tmp_path)
    sl._record_provisioned_backend(tmp_path, "vulkan")

    _check_runtime_build(lambda: tmp_path)

    assert "build not recorded" in capsys.readouterr().out


def test_doctor_stays_quiet_when_nothing_is_provisioned(capsys, home):
    """_check_llama_lib has already said the dir is missing; repeating it adds
    noise to a report people read under stress."""
    # localm.cli exposes a click Command named `doctor`, which shadows the
    # submodule of the same name - import the function directly.
    from localm.cli.doctor import _check_runtime_build
    _check_runtime_build(lambda: None)
    assert capsys.readouterr().out == ""


# --------------------------------------------------------------------------- #
#  END TO END through the CLI                                                  #
# --------------------------------------------------------------------------- #

def _wire_cli(monkeypatch, target, tags_seen, latest=None):
    """Stub provisioning down to the release resolution so a CLI run exercises
    main()'s real flag handling, guard and recording without any network."""
    monkeypatch.setattr(sl, "_repo_runtime_lib", lambda: target)
    monkeypatch.setattr(sl, "_platform_key", lambda: "win32")
    monkeypatch.setattr(sl, "_lib_name", lambda: "llama.dll")
    monkeypatch.setattr(sl, "_auto_backend", lambda: "vulkan")
    # Fails rather than returning a tag unless a caller explicitly passes
    # `latest=`; only the --tag latest test opts into tracking upstream.
    monkeypatch.setattr(
        sl, "_latest_tag",
        (lambda: latest) if latest else
        (lambda: pytest.fail("this CLI path must not resolve upstream's newest")))
    monkeypatch.setattr(sl, "_clear_target_or_refuse", lambda t: None)
    monkeypatch.setattr(sl, "_install_runtime_wheel", lambda d: True)
    monkeypatch.setattr(sl, "_native_loads_ok", lambda: (True, ""))
    monkeypatch.setattr(sl, "_verify", lambda: None)
    monkeypatch.setattr(sl, "_warn_off_profile", lambda c: None)

    def fake_assets(tag, repo=sl._UPSTREAM_REPO):
        tags_seen.append(tag)
        return _release_listing(tag)

    monkeypatch.setattr(sl, "_release_assets", fake_assets)

    def fake_fetch(url, target_dir, sha, what):
        Path(target_dir).mkdir(parents=True, exist_ok=True)
        (Path(target_dir) / "llama.dll").write_text("x")

    monkeypatch.setattr(sl, "_fetch_verified", fake_fetch)


def test_cli_tag_installs_and_records_that_build(monkeypatch, tmp_path, cli_runner):
    target = tmp_path / "rt"
    target.mkdir()
    tags = []
    _wire_cli(monkeypatch, target, tags)

    res = cli_runner.invoke(sl.main, ["--backend", "vulkan", "--tag", "b10355", "-y"])

    assert res.exit_code == 0, res.output
    assert tags == ["b10355"], "the requested tag is what gets resolved"
    assert sl._provisioned_build(target) == "b10355"
    assert sl.pinned_tag() == "b10355"


def test_cli_tag_reprovisions_an_already_provisioned_box(monkeypatch, tmp_path, cli_runner):
    """The already-provisioned guard compares BACKENDS. An explicit --tag changes
    the BUILD with the backend unchanged, so without forcing, setup would print
    'Already provisioned', change nothing, and leave the pin disagreeing with the
    disk - the config and the filesystem out of step with no sign of it."""
    target = tmp_path / "rt"
    target.mkdir()
    (target / "llama.dll").write_text("old")
    sl._record_provisioned_backend(target, "vulkan", build="b10361")
    tags = []
    _wire_cli(monkeypatch, target, tags)

    res = cli_runner.invoke(sl.main, ["--backend", "vulkan", "--tag", "b10355", "-y"])

    assert res.exit_code == 0, res.output
    assert "Already provisioned" not in res.output
    assert sl._provisioned_build(target) == "b10355"


def test_cli_rollback_returns_to_the_previous_build(monkeypatch, tmp_path, cli_runner):
    target = tmp_path / "rt"
    target.mkdir()
    (target / "llama.dll").write_text("old")
    tags = []
    _wire_cli(monkeypatch, target, tags)
    # An install history: b10300, then b10361.
    sl._record_runtime_history("vulkan", "b10300")
    sl._record_runtime_history("vulkan", "b10361")
    sl._record_provisioned_backend(target, "vulkan", build="b10361")

    res = cli_runner.invoke(sl.main, ["--rollback", "-y"])

    assert res.exit_code == 0, res.output
    assert tags == ["b10300"]
    assert sl._provisioned_build(target) == "b10300"
    assert sl.pinned_tag() == "b10300"


def test_cli_bare_run_records_the_resolved_tag(monkeypatch, tmp_path, cli_runner):
    """The default path - no flags - must record too, or nothing is recorded on
    the installs that matter most."""
    target = tmp_path / "rt"
    target.mkdir()
    tags = []
    _wire_cli(monkeypatch, target, tags)

    res = cli_runner.invoke(sl.main, ["-y"])

    assert res.exit_code == 0, res.output
    assert sl._provisioned_build(target) == sl._PINNED_TAG
    assert [(e["backend"], e["tag"]) for e in sl.runtime_history()] == [
        ("vulkan", sl._PINNED_TAG)]


def _updater_argv():
    """The exact flags `localm update` re-provisions with, read from the updater
    itself: a retyped copy would keep passing if the real command changed."""
    from localm._apply_update import post_swap_command
    argv = post_swap_command("runtime", backend="vulkan")
    return argv[argv.index("setup-llama") + 1:]


def test_update_reprovision_honours_the_pin(monkeypatch, tmp_path, cli_runner):
    """The updater re-invokes setup-llama with --backend --force --yes (see
    _apply_update.post_swap_command), so the pin is only real if that
    invocation reads it. Driven with the updater's OWN argv."""
    target = tmp_path / "rt"
    target.mkdir()
    tags = []
    _wire_cli(monkeypatch, target, tags)
    # A cuda install being re-provisioned onto vulkan: the backends differ, so
    # this re-fetches even without --force. The pin is the property under test.
    (target / "llama.dll").write_text("old")
    sl._record_provisioned_backend(target, "cuda", build="b10200")
    sl.set_pinned_tag("b10355")

    res = cli_runner.invoke(sl.main, _updater_argv())

    assert res.exit_code == 0, res.output
    assert tags == ["b10355"], "the pinned build, not upstream's newest"
    assert sl._provisioned_build(target) == "b10355"


def test_update_reprovision_actually_reprovisions_when_backend_matches(
        monkeypatch, tmp_path, cli_runner):
    """`updater.classify()` only escalates to the "runtime" class when the release
    manifest DECLARES the native binaries need re-provisioning, so once the
    updater's re-invocation runs at all it must actually replace the build - not
    keep whatever is already on disk because the backend string happens to match.
    That requires --force on _apply_update.post_swap_command, or setup-llama's own
    "already provisioned, same backend" guard short-circuits the invocation and
    the update reports success having changed nothing. Driven with the updater's
    OWN argv, like its pin-honouring sibling above."""
    target = tmp_path / "rt"
    target.mkdir()
    (target / "llama.dll").write_text("old")
    sl._record_provisioned_backend(target, "vulkan", build="b10300")
    tags = []
    _wire_cli(monkeypatch, target, tags)

    res = cli_runner.invoke(sl.main, _updater_argv())

    assert res.exit_code == 0, res.output
    assert tags == [sl._PINNED_TAG], "the updater's re-invocation must resolve a release"
    assert sl._provisioned_build(target) == sl._PINNED_TAG, (
        "a 'runtime'-class update must actually re-provision, not keep the old build")


def test_cli_tag_latest_opts_in_to_upstreams_newest(monkeypatch, tmp_path, cli_runner):
    """--tag latest is an opt-IN to bleeding edge, not a way to clear a pin, so it
    must both STICK as a stored choice and actually resolve upstream. The stored
    value is checked through tracks_latest() and NOT through pinned_tag(), which
    must stay None: the sentinel reaching pinned_tag() would be interpolated into
    a release URL by every caller."""
    target = tmp_path / "rt"
    target.mkdir()
    (target / "llama.dll").write_text("old")
    tags = []
    _wire_cli(monkeypatch, target, tags, latest="b10361")
    sl.set_pinned_tag("b10300")

    res = cli_runner.invoke(sl.main, ["--backend", "vulkan", "--tag", "latest", "-y"])

    assert res.exit_code == 0, res.output
    assert sl.tracks_latest() is True, "the choice must stick, not just apply once"
    assert sl.pinned_tag() is None, "the sentinel must never read as an exact tag"
    assert tags == ["b10361"], "tracking, it resolves upstream's newest"
    assert "NOT tested" in res.output or "has NOT tested" in res.output, (
        "opting out of the confirmed build must say what that costs")


def test_cli_tag_default_returns_to_the_confirmed_pin(monkeypatch, tmp_path, cli_runner):
    """The way back from --tag latest: clearing a pin returns to the confirmed
    build, so a user who opted into upstream once does not have to name the tested
    tag by hand."""
    target = tmp_path / "rt"
    target.mkdir()
    (target / "llama.dll").write_text("old")
    tags = []
    _wire_cli(monkeypatch, target, tags)
    sl.set_pinned_tag(sl._TRACK_LATEST)

    res = cli_runner.invoke(sl.main, ["--backend", "vulkan", "--tag", "default", "-y"])

    assert res.exit_code == 0, res.output
    assert sl.tracks_latest() is False and sl.pinned_tag() is None
    assert tags == [sl._PINNED_TAG], "and it installs the confirmed build"


# --------------------------------------------------------------------------- #
#  CHECK - the read-only "is a different build available" surface             #
# --------------------------------------------------------------------------- #

def test_check_runtime_update_reports_not_installed_when_nothing_provisioned(
        monkeypatch, tmp_path):
    """No marker at all means no runtime has ever been provisioned - that is
    initial setup's job, not an update, and must not be reported as one."""
    monkeypatch.setattr(sl, "_repo_runtime_lib", lambda: tmp_path)
    assert sl.check_runtime_update() == {
        "installed": False, "backend": None, "current": None,
        "target": None, "newer": False, "pinned": None, "previous": None}


def test_check_runtime_update_compares_against_the_confirmed_pin_by_default(
        monkeypatch, tmp_path, home):
    """The card must offer what setup-llama would actually install. If this
    compared against upstream's newest while the command installs the pin, the
    GUI would advertise an update that re-provisioning does not deliver - and
    would keep advertising it forever."""
    monkeypatch.setattr(sl, "_repo_runtime_lib", lambda: tmp_path)
    monkeypatch.setattr(
        sl, "_latest_tag",
        lambda: pytest.fail("the default check must not query releases"))
    sl._record_provisioned_backend(tmp_path, "vulkan", build="b10300")

    result = sl.check_runtime_update()

    assert result == {"installed": True, "backend": "vulkan", "current": "b10300",
                      "target": sl._PINNED_TAG, "newer": True, "pinned": None,
                      "previous": None}


def test_check_runtime_update_compares_against_latest_when_tracking(
        monkeypatch, tmp_path, home):
    monkeypatch.setattr(sl, "_repo_runtime_lib", lambda: tmp_path)
    monkeypatch.setattr(sl, "_latest_tag", lambda: "b10361")
    sl.set_pinned_tag(sl._TRACK_LATEST)
    sl._record_provisioned_backend(tmp_path, "vulkan", build="b10300")

    result = sl.check_runtime_update()

    assert result == {"installed": True, "backend": "vulkan", "current": "b10300",
                      "target": "b10361", "newer": True, "pinned": None,
                      "previous": None}


def test_check_runtime_update_reports_up_to_date_when_matching(monkeypatch, tmp_path, home):
    monkeypatch.setattr(sl, "_repo_runtime_lib", lambda: tmp_path)
    sl._record_provisioned_backend(tmp_path, "vulkan", build=sl._PINNED_TAG)

    result = sl.check_runtime_update()

    assert result["newer"] is False
    assert result["current"] == result["target"] == sl._PINNED_TAG


def test_check_runtime_update_prefers_the_pin_over_a_release_lookup(
        monkeypatch, tmp_path, home):
    """A build pinned away from a broken release must never be told THAT
    release is "available" again - the pin, not upstream's newest, is the
    correct comparison target. No release lookup should even happen: an
    install that pinned specifically because the network/API was unreliable
    must still get an honest check."""
    monkeypatch.setattr(sl, "_repo_runtime_lib", lambda: tmp_path)
    monkeypatch.setattr(sl, "_latest_tag",
                        lambda: pytest.fail("a pinned install must not query releases"))
    sl._record_provisioned_backend(tmp_path, "vulkan", build="b10300")
    sl.set_pinned_tag("b10355")

    result = sl.check_runtime_update()

    assert result == {"installed": True, "backend": "vulkan", "current": "b10300",
                      "target": "b10355", "newer": True, "pinned": "b10355",
                      "previous": None}


def test_check_runtime_update_amd_rocm_compares_against_the_fixed_tag(monkeypatch, tmp_path):
    """amd-rocm's build is fixed by the localm release (_ROCM_TAG), never
    resolved from an upstream release listing - the check must not query one
    for this backend either."""
    monkeypatch.setattr(sl, "_repo_runtime_lib", lambda: tmp_path)
    monkeypatch.setattr(sl, "_latest_tag",
                        lambda: pytest.fail("amd-rocm must not query releases"))
    sl._record_provisioned_backend(tmp_path, "amd-rocm", build="b1288")

    result = sl.check_runtime_update()

    assert result["target"] == sl._ROCM_TAG
    assert result["newer"] is True


def test_check_runtime_update_never_raises_on_an_unreadable_config(monkeypatch, tmp_path):
    """pinned_tag() degrades an unreadable config to 'no pin' instead of raising,
    and this check must inherit that.

    tracks_latest() must degrade the SAME way and in the SAME direction, asserted
    here as well as in its own test: an unreadable config falling through to
    "tracking" would answer a broken read by reaching for upstream's untested
    newest."""
    monkeypatch.setattr(sl, "_repo_runtime_lib", lambda: tmp_path)
    monkeypatch.setattr(
        sl, "_latest_tag",
        lambda: pytest.fail("an unreadable config must not become 'track latest'"))
    monkeypatch.setattr(sl.config, "load_config", lambda: (_ for _ in ()).throw(OSError("nope")))
    sl._record_provisioned_backend(tmp_path, "vulkan", build="b10300")

    result = sl.check_runtime_update()

    assert result["pinned"] is None
    assert result["target"] == sl._PINNED_TAG

# SPDX-License-Identifier: AGPL-3.0-or-later
"""The ComfyUI readiness cache must not report a DEAD ComfyUI as running for the
rest of the process lifetime.

The cache exists so ensure_comfy() stops pinging /system_stats on every call
(each media submission pings twice back-to-back - once at the route-handler layer
via ensure_available, once at the top of the generator). Its premise, stated in
comfy_client's module docstring, is that "ComfyUI does not appear or disappear on
its own between requests". On this stack it does: an OOM on a large render, a
ZLUDA/ROCm torch crash, the user closing its window, or VRAM eviction all kill it
mid-session.

Once ``mark_comfy_alive`` has run (opening the Image page warms it via
GET /v1/comfy/status), an unbounded confirmation makes ``ensure_comfy`` return
"ComfyUI is running." WITHOUT pinging. Every later generation then submits a
prompt to a dead server and fails deep in the submit path with an opaque
ConnectionError, losing both recovery routes: the auto-launch from comfy_workdir,
and the clean actionable "not reachable ... point localm at your install"
message. Nothing invalidates the entry on a generation failure
(``mark_comfy_dead`` has only two call sites: stop_comfy, and the non-cached
branch that is unreachable once confirmed), and the CLI and the coder's
generate_image path have no /v1/comfy/status trigger at all.

The confirmation is therefore time-bounded. That keeps the win the cache was
added for - the back-to-back double-ping inside ONE submission is milliseconds
apart, far inside the window - while bounding staleness to seconds, so a
mid-session death is detected on the next generation and auto-relaunch works.
is_comfy_confirmed has exactly one consumer (ensure_comfy); /v1/comfy/status
always does a real ping and merely primes the cache.
"""

from __future__ import annotations

import time

import pytest

from localm.media import comfy_client

URL = "http://127.0.0.1:8188"


@pytest.fixture
def no_launch(monkeypatch, tmp_path):
    """No launch command configured, so ensure_comfy's "cannot launch" branch
    returns the clean actionable message instead of trying to start anything."""
    monkeypatch.setattr("localm.config.load_config", lambda: {})
    monkeypatch.setattr(comfy_client, "discover_launch_cmd", lambda p: None)
    return tmp_path


def _pings(monkeypatch, alive: bool):
    """Patch the reachability probe and count how often it is actually called."""
    calls: list = []

    def _probe(api_url, timeout=3.0):
        calls.append(api_url)
        return alive

    monkeypatch.setattr(comfy_client, "_comfy_alive", _probe)
    return calls


# --------------------------------------------------------------------------- #
#  A stale confirmation is re-probed                                           #
# --------------------------------------------------------------------------- #

def test_ensure_comfy_rechecks_after_a_confirmed_comfy_dies(monkeypatch, no_launch):
    """Image page opened (cache warmed) -> ComfyUI dies -> next generation must
    NOT be told "ComfyUI is running.". It must re-probe and report the truth."""
    comfy_client.mark_comfy_alive(URL)
    assert comfy_client.is_comfy_confirmed(URL) is True

    # ComfyUI dies. Age the confirmation past its freshness window.
    _age_confirmation(URL)
    calls = _pings(monkeypatch, alive=False)

    ok, msg = comfy_client.ensure_comfy(api_url=URL)

    assert calls, "ensure_comfy trusted a stale confirmation and never re-probed"
    assert ok is False, "a DEAD ComfyUI was reported as running"
    assert "not reachable" in msg, msg


def test_a_dead_comfy_that_comes_back_is_detected(monkeypatch, no_launch):
    """The cache must not latch dead either: once it is reachable again the next
    call sees it."""
    comfy_client.mark_comfy_dead(URL)
    _pings(monkeypatch, alive=True)
    ok, msg = comfy_client.ensure_comfy(api_url=URL)
    assert ok is True and "running" in msg
    assert comfy_client.is_comfy_confirmed(URL) is True


def test_stale_confirmation_does_not_block_auto_relaunch(monkeypatch, tmp_path):
    """The other capability the stale cache silently disabled: with a launch
    command configured, a dead-but-confirmed ComfyUI must be RELAUNCHED rather
    than reported running."""
    comfy_client.mark_comfy_alive(URL)
    _age_confirmation(URL)

    monkeypatch.setattr("localm.config.load_config",
                        lambda: {"comfy_launch_cmd": "launch-comfy",
                                 "comfy_workdir": str(tmp_path)})
    launched: list = []

    class _Proc:
        returncode = 0

        def poll(self):
            return None

    def _popen(argv, **kw):
        launched.append(argv)
        return _Proc()

    monkeypatch.setattr("subprocess.Popen", _popen)
    monkeypatch.setattr(comfy_client, "_comfy_alive",
                        lambda api_url, timeout=3.0: bool(launched))
    monkeypatch.setattr(time, "sleep", lambda s: None)

    ok, _msg = comfy_client.ensure_comfy(api_url=URL, wait_seconds=5)
    assert launched, "a stale confirmation suppressed the auto-launch"
    assert ok is True


# --------------------------------------------------------------------------- #
#  Back-to-back calls still share one probe                                    #
# --------------------------------------------------------------------------- #

def test_back_to_back_calls_still_share_one_probe(monkeypatch, no_launch):
    """The reason the cache exists: one submission calls ensure_comfy twice
    (route handler, then the generator), milliseconds apart. That must still cost
    exactly ONE probe - the fix bounds staleness, it does not remove the cache."""
    calls = _pings(monkeypatch, alive=True)

    assert comfy_client.ensure_comfy(api_url=URL)[0] is True
    assert comfy_client.ensure_comfy(api_url=URL)[0] is True
    assert comfy_client.ensure_comfy(api_url=URL)[0] is True

    assert len(calls) == 1, (
        "the back-to-back dedupe the readiness cache was added for is gone")


def test_confirmation_is_fresh_immediately_after_marking():
    comfy_client.mark_comfy_alive(URL)
    assert comfy_client.is_comfy_confirmed(URL) is True


def test_confirmation_expires_once_stale():
    comfy_client.mark_comfy_alive(URL)
    _age_confirmation(URL)
    assert comfy_client.is_comfy_confirmed(URL) is False, (
        "a confirmation must not be trusted indefinitely")


def test_mark_dead_clears_the_confirmation():
    comfy_client.mark_comfy_alive(URL)
    comfy_client.mark_comfy_dead(URL)
    assert comfy_client.is_comfy_confirmed(URL) is False


def test_confirmation_is_per_url():
    other = "http://127.0.0.1:9999"
    comfy_client.mark_comfy_alive(URL)
    assert comfy_client.is_comfy_confirmed(other) is False


def test_trailing_slash_is_normalised():
    comfy_client.mark_comfy_alive(URL + "/")
    assert comfy_client.is_comfy_confirmed(URL) is True
    comfy_client.mark_comfy_dead(URL)
    assert comfy_client.is_comfy_confirmed(URL + "/") is False


def _age_confirmation(url: str) -> None:
    """Push a confirmation's timestamp far enough into the past that it is stale,
    standing in for real time passing between two generations."""
    key = url.rstrip("/")
    assert key in comfy_client._confirmed_alive
    comfy_client._confirmed_alive[key] = (
        time.monotonic() - comfy_client._CONFIRM_TTL_SECONDS - 60)

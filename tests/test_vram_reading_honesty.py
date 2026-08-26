# SPDX-License-Identifier: AGPL-3.0-or-later
"""A VRAM number we could not actually measure must never be reported as fact.

These cover the 'AFTER' end of the before/after pair.

discover.list_gpus() is deadline-bounded, and on an overrun serves its
last-known-good reading with status GPU_PROBE_TIMEOUT/BUSY. When the before-read
ANSWERS and only the after-polls lose the driver, after == before and no rise is
observed; a status-blind poll reader would report "vram_freed": false with
before_fresh=True and therefore NO uncertainty flag.

The before-probe is itself what warms the driver, and the native unload between
the two reads is the kind of work that wedges it. Once ANY probe overruns,
discover._gpu_probe_inflight stays True until the abandoned thread returns, so
every later poll is served the frozen value the before-read just cached.

These assert the OBSERVABLE - the response body the endpoint returns, and the
value the reader hands back - never merely that a log line appeared.
"""

import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from localm.discover import GPU_PROBE_BUSY, GPU_PROBE_OK, GPU_PROBE_TIMEOUT
from localm.inference.http_server import create_app
from localm.vram import _live_free_vram_bytes, wait_for_vram_release

GB = 1024 ** 3

# A frozen reading: an idle RX 6900 XT (15.85 GB free of 15.98 GB), served by a
# timed-out probe long after a model had in fact loaded on top of it.
_IDLE = {"index": 0, "name": "AMD Radeon RX 6900 XT",
         "total": 17179869184, "free": 17015111680}


def _list_gpus_double(gpus, status):
    """A faithful double of list_gpus()'s real two-shape contract: a bare list by
    default, ``(gpus, status)`` when return_status=True. A plain return_value
    cannot express the status these tests turn on."""
    def _inner(*, deadline=None, return_status=False):
        served = [dict(g) for g in gpus]
        return (served, status) if return_status else served
    return _inner


def _impatient_wait(read_free, **kw):
    """The REAL wait_for_vram_release with its clock compressed - same logic, same
    branches, without spending the endpoint's full 5s poll budget per test. The
    outcome under test (released=None) is produced BY that function, so it is
    wrapped rather than stubbed."""
    kw["timeout_s"] = 0.05
    kw["sleep"] = lambda _s: None
    return wait_for_vram_release(read_free, **kw)


def _make_mock_engine(loaded=True):
    engine = MagicMock()
    state = {"loaded": loaded}
    engine.unload.side_effect = lambda: state.update(loaded=False)
    engine.load.side_effect = lambda: state.update(loaded=True)
    engine.display_name = "test-model"
    type(engine).loaded = property(lambda self: state["loaded"])
    return engine


class TestLiveFreeReader(unittest.TestCase):
    """The reader wait_for_vram_release() polls must refuse to pass a served-stale
    number off as a current one."""

    def test_returns_none_when_the_probe_timed_out(self):
        with patch("localm.discover.list_gpus",
                   side_effect=_list_gpus_double([_IDLE], GPU_PROBE_TIMEOUT)):
            self.assertIsNone(_live_free_vram_bytes())

    def test_returns_none_when_the_probe_was_busy(self):
        """BUSY is the same class of non-reading as TIMEOUT: no fresh probe was
        taken, so the served value is last-known-good, i.e. possibly frozen. It is
        also the state a wedged box SETTLES into - _gpu_probe_inflight stays True
        until the abandoned thread returns, so every later poll reports BUSY."""
        with patch("localm.discover.list_gpus",
                   side_effect=_list_gpus_double([_IDLE], GPU_PROBE_BUSY)):
            self.assertIsNone(_live_free_vram_bytes())

    def test_returns_the_number_when_the_probe_is_live(self):
        with patch("localm.discover.list_gpus",
                   side_effect=_list_gpus_double([_IDLE], GPU_PROBE_OK)):
            self.assertEqual(_live_free_vram_bytes(), 17015111680)


class TestWaitForVramReleaseCannotClaimNotReleased(unittest.TestCase):
    """'released=False' is a positive claim that VRAM did NOT drop. A reading that
    never refreshed cannot support it; None ('cannot verify') is the honest answer
    and is already this function's documented meaning for an unmeasurable read."""

    def test_unmeasurable_reading_is_unverifiable_not_false(self):
        released, final = wait_for_vram_release(
            lambda: None, before_bytes=10 * GB,
            timeout_s=0.0, sleep=lambda _s: None)
        self.assertIsNone(released,
                          "an unreadable free-VRAM poll must report 'cannot "
                          "verify', never a confident 'VRAM was not freed'")
        self.assertIsNone(final)

    def test_live_reading_with_no_rise_still_reports_false(self):
        """The honest negative must survive: a real, fresh reading that genuinely
        did not rise is still a truthful vram_freed=false. If this regresses to
        None the fix has thrown away a real signal instead of qualifying it."""
        released, final = wait_for_vram_release(
            lambda: 10 * GB, before_bytes=10 * GB,
            timeout_s=0.0, sleep=lambda _s: None)
        self.assertIs(released, False)
        self.assertEqual(final, 10 * GB)

    def test_rise_is_still_detected(self):
        released, final = wait_for_vram_release(
            lambda: 20 * GB, before_bytes=10 * GB,
            timeout_s=0.0, sleep=lambda _s: None)
        self.assertIs(released, True)
        self.assertEqual(final, 20 * GB)


class _UnloadCase(unittest.TestCase):
    def setUp(self):
        self.engine = _make_mock_engine(loaded=True)
        app = create_app(self.engine)
        self.client = TestClient(
            app, headers={"Authorization": f"Bearer {app.state.shell_token}"})

    def assertNoUnqualifiedNotFreed(self, body):
        """The contract for an unverifiable outcome: never a bare "VRAM was not
        freed", and never an unflagged number."""
        self.assertIsNot(body.get("vram_freed"), False,
                         f"claimed VRAM was not freed from a reading that never "
                         f"refreshed: {body}")
        self.assertTrue(body.get("vram_reading_uncertain"),
                        f"an unverifiable reading must be flagged: {body}")
        self.assertTrue(body.get("vram_note"), "the flag needs its explanation")


def _probe_dies_after_first_read():
    """list_gpus() answers the endpoint's BEFORE read, then loses the driver for
    every AFTER poll - serving the frozen value the before-read just cached."""
    calls = {"n": 0}

    def _inner(*, deadline=None, return_status=False):
        calls["n"] += 1
        served = [dict(_IDLE, free=10 * GB)]
        status = GPU_PROBE_OK if calls["n"] == 1 else GPU_PROBE_TIMEOUT
        return (served, status) if return_status else served
    return _inner, calls


class TestUnloadAllEndpointHonesty(_UnloadCase):
    def test_reading_lost_after_a_live_before_is_not_reported_as_not_freed(self):
        """Before IS fresh, so a check that flags only `not before_fresh`
        emits vram_freed: false with no flag at all."""
        double, calls = _probe_dies_after_first_read()
        with patch("localm.discover.list_gpus", side_effect=double), \
             patch("localm.vram.wait_for_vram_release", _impatient_wait):
            r = self.client.post("/v1/models/unload")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["status"], "unloaded")   # the unload itself worked
        self.assertGreater(calls["n"], 1, "the after-poll never ran - the "
                                          "scenario under test did not happen")
        self.assertNoUnqualifiedNotFreed(body)

    def test_fully_stale_probe_is_flagged(self):
        with patch("localm.discover.list_gpus",
                   side_effect=_list_gpus_double([_IDLE], GPU_PROBE_TIMEOUT)), \
             patch("localm.vram.wait_for_vram_release", _impatient_wait):
            r = self.client.post("/v1/models/unload")
        self.assertNoUnqualifiedNotFreed(r.json())

    def test_live_probe_still_reports_the_real_numbers(self):
        """The fix must not blind the working path: a live probe showing a real
        rise must still report vram_freed=true with both byte counts, unflagged."""
        reads = iter([[dict(_IDLE, free=10 * GB)]])
        after = [dict(_IDLE, free=20 * GB)]

        def _live(*, deadline=None, return_status=False):
            served = next(reads, after)
            return (served, GPU_PROBE_OK) if return_status else served

        with patch("localm.discover.list_gpus", side_effect=_live):
            r = self.client.post("/v1/models/unload")
        body = r.json()
        self.assertTrue(body["vram_freed"])
        self.assertEqual(body["vram_before_bytes"], 10 * GB)
        self.assertEqual(body["vram_after_bytes"], 20 * GB)
        self.assertNotIn("vram_reading_uncertain", body)

    def test_live_probe_with_no_rise_still_reports_the_honest_false(self):
        """A genuinely-not-freed result must survive the fix intact, unflagged -
        but ONLY on a DEVICE-GLOBAL reading, which is now asserted, not assumed.

        The honest-negative is only honest where the driver's free-VRAM figure is
        device-global (Linux, or NVIDIA, or an AMD/Windows reading corrected by the
        device-global source). There, a fresh reading that does not rise genuinely
        means nothing was freed, and reporting vram_freed=false without an
        uncertainty flag is correct. `free_scope="device"` on the reading below is
        what makes that true rather than merely fresh.

        The blind case - a process-scoped reading, where a fresh no-rise proves
        nothing because the model's VRAM is in another process - is its own test
        (test_process_scoped_no_rise_is_flagged_not_an_honest_false), and there
        the SAME no-rise MUST be flagged uncertain."""
        with patch("localm.discover.list_gpus",
                   side_effect=_list_gpus_double(
                       [dict(_IDLE, free=10 * GB, free_scope="device")],
                       GPU_PROBE_OK)), \
             patch("localm.vram.wait_for_vram_release", _impatient_wait):
            r = self.client.post("/v1/models/unload")
        body = r.json()
        self.assertIs(body["vram_freed"], False)
        self.assertEqual(body["vram_before_bytes"], 10 * GB)
        self.assertNotIn("vram_reading_uncertain", body)

    def test_process_scoped_no_rise_is_flagged_not_an_honest_false(self):
        """On a PROCESS-SCOPED reading a no-rise is NOT proof that nothing was
        freed.

        torch.cuda.mem_get_info reports `total - THIS process's own allocations`
        and is blind to every other process. Every GGUF model loads in an
        isolated worker subprocess, so the server's own reading cannot see the
        model's VRAM and a real unload produces this identical no-rise. The SAME
        reading and the SAME no-rise as the test above, but tagged
        process-scoped, MUST carry the uncertainty flag."""
        with patch("localm.discover.list_gpus",
                   side_effect=_list_gpus_double(
                       [dict(_IDLE, free=10 * GB, free_scope="process")],
                       GPU_PROBE_OK)):
            r = self.client.post("/v1/models/unload")
        body = r.json()
        # The figure is still reported (it is the best we have), but flagged, never
        # presented as fact.
        self.assertEqual(body["vram_before_bytes"], 10 * GB)
        self.assertIs(body["vram_reading_uncertain"], True)
        self.assertIn("worker process", body["vram_note"])

    def test_box_with_no_vram_telemetry_says_nothing_at_all(self):
        """A CPU-only box (or the Windows registry tier, which reports total but
        never free) has a probe that COMPLETES and has no free reading to give.
        The answer there is silence, not an uncertainty flag."""
        with patch("localm.discover.list_gpus",
                   side_effect=_list_gpus_double([], GPU_PROBE_OK)), \
             patch("localm.discover.vram_info", return_value={}):
            r = self.client.post("/v1/models/unload")
        body = r.json()
        self.assertEqual(body["status"], "unloaded")
        for k in ("vram_freed", "vram_before_bytes", "vram_after_bytes",
                  "vram_reading_uncertain", "vram_note"):
            self.assertNotIn(k, body, f"{k} must not appear on a box with no "
                                      f"VRAM telemetry: {body}")


class TestUnloadOneEndpointHonesty(_UnloadCase):
    """unload_one_model() carries its own reader + reporter - the per-model
    Unload button.

    `model` is a QUERY param on POST /v1/models/unload (a bare `str | None` with
    no Body()), so it MUST go through params=; a json body is silently ignored
    and the route falls through to unload_all_models(), which flags on its own
    reader and would make these pass WITHOUT touching unload_one_model at all."""

    def test_reading_lost_after_a_live_before_is_not_reported_as_not_freed(self):
        from localm.inference import http_server as hs
        double, calls = _probe_dies_after_first_read()
        with patch.dict(hs._engines, {"test-model": self.engine}, clear=False), \
             patch("localm.discover.list_gpus", side_effect=double), \
             patch("localm.vram.wait_for_vram_release", _impatient_wait):
            r = self.client.post("/v1/models/unload", params={"model": "test-model"})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["status"], "unloaded")
        self.assertGreater(calls["n"], 1, "the after-poll never ran")
        self.assertNoUnqualifiedNotFreed(body)

    def test_process_scoped_no_rise_is_flagged_here_too(self):
        """The per-model button must not lie where "unload all" is honest. All
        three unload paths thread before_scope through the same
        _add_vram_fields; this pins the per-model one."""
        from localm.inference import http_server as hs
        with patch.dict(hs._engines, {"test-model": self.engine}, clear=False), \
             patch("localm.discover.list_gpus",
                   side_effect=_list_gpus_double(
                       [dict(_IDLE, free=10 * GB, free_scope="process")],
                       GPU_PROBE_OK)), \
             patch("localm.vram.wait_for_vram_release", _impatient_wait):
            r = self.client.post("/v1/models/unload", params={"model": "test-model"})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["vram_before_bytes"], 10 * GB)
        self.assertIs(body["vram_reading_uncertain"], True)
        self.assertIn("worker process", body["vram_note"])


class TestUnloadEmbedderEndpointHonesty(_UnloadCase):
    """_unload_embedder_if_matches() is the THIRD unload path, and the one the two
    classes above do not exercise: the per-row Unload button for a resident
    EMBEDDING model, which lives outside _engines on its own lifecycle. It carries
    its own copy of the reader + _add_vram_fields call, so it can lie while the
    other two are honest. unload_one_model reaches it only when _engines has no
    live engine for the name, so these install NO engine and stand the embedder
    up instead (contrast TestUnloadOneEndpointHonesty, which puts an engine in
    _engines and therefore never gets here)."""

    _EMB = "/models/embedding-model.gguf"

    def _drive(self, list_gpus_side_effect):
        from localm.inference import http_server as hs
        # No engine for this name -> unload_one_model falls through to the embedder
        # branch. Stand up a resident embedder whose resolved path matches the name.
        with patch.dict(hs._engines, {}, clear=False), \
             patch("localm.inference.embedder.loaded_path", lambda: self._EMB), \
             patch("localm.inference.embedder.active_requests", lambda: 0), \
             patch("localm.inference.embedder.reset_embedder", lambda force=True: True), \
             patch("localm.config.load_registry",
                   return_value={"emb-model": {"path": self._EMB}}), \
             patch("localm.model_manager._entry_path", lambda entry: self._EMB), \
             patch("localm.discover.list_gpus", side_effect=list_gpus_side_effect), \
             patch("localm.vram.wait_for_vram_release", _impatient_wait):
            # `model` is a QUERY param on this route (a bare `str | None` with no
            # Body()), so it MUST be passed as params: a json body is silently
            # ignored and the route falls through to unload_all_models(), whose own
            # reader would set the flag without ever reaching the embedder path.
            r = self.client.post("/v1/models/unload", params={"model": "emb-model"})
        return r

    def test_reading_lost_after_a_live_before_is_not_reported_as_not_freed(self):
        """Fresh device-scoped BEFORE, driver lost for the AFTER polls, on the
        embedder path. The poll reader must return None so vram_freed is
        null-and-flagged, never a bare false."""
        calls = {"n": 0}

        def _double(*, deadline=None, return_status=False):
            calls["n"] += 1
            # The BEFORE read is fresh AND device-scoped, so the only thing that
            # can raise the flag is the lost after-poll (released is None), which
            # isolates the staleness axis from the scope axis.
            served = [dict(_IDLE, free=10 * GB, free_scope="device")]
            status = GPU_PROBE_OK if calls["n"] == 1 else GPU_PROBE_TIMEOUT
            return (served, status) if return_status else served

        r = self._drive(_double)
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["status"], "unloaded")   # the unload itself worked
        self.assertGreater(calls["n"], 1, "the after-poll never ran - the scenario "
                                          "under test did not happen")
        self.assertNoUnqualifiedNotFreed(body)

    def test_process_scoped_no_rise_is_flagged_here_too(self):
        """The embedder button must not lie where the other two unload paths are
        honest. All three thread before_scope through the same _add_vram_fields;
        this pins the third."""
        r = self._drive(_list_gpus_double(
            [dict(_IDLE, free=10 * GB, free_scope="process")], GPU_PROBE_OK))
        self.assertEqual(r.status_code, 200)
        body = r.json()
        # The figure is still reported (best we have), but flagged, never fact.
        self.assertEqual(body["vram_before_bytes"], 10 * GB)
        self.assertIs(body["vram_reading_uncertain"], True)
        self.assertIn("worker process", body["vram_note"])


class TestMediaSwapMessageHonesty(unittest.TestCase):
    """The media VRAM handover is where a user actually SEES this: image/music/
    video generation must not tell the user "VRAM has not dropped yet" on the
    strength of a reading the server has flagged as possibly stale."""

    def _push_lines(self, body):
        from localm import vram as vram_mod
        job = MagicMock()
        lines = []
        job.push.side_effect = lambda d: lines.append(d.get("text", ""))
        resp = MagicMock()
        resp.ok = True
        resp.json.return_value = body
        with patch("localm.selfclient.self_request", return_value=resp):
            ok = vram_mod.unload_chat_for_media(job, "http://x", "image")
        return ok, lines

    def test_uncertain_reading_does_not_claim_vram_has_not_dropped(self):
        ok, lines = self._push_lines({
            "status": "unloaded", "vram_freed": False,
            "vram_before_bytes": 17015111680, "vram_after_bytes": 17015111680,
            "vram_reading_uncertain": True, "vram_note": "stale"})
        self.assertTrue(ok)
        joined = " ".join(lines)
        self.assertNotIn("has not dropped", joined,
                         f"passed the server's flagged-stale reading on to the "
                         f"user as fact: {lines}")
        self.assertIn("could not confirm", joined)

    def test_certain_reading_still_reports_the_real_gb(self):
        ok, lines = self._push_lines({
            "status": "unloaded", "vram_freed": True,
            "vram_before_bytes": 10 * GB, "vram_after_bytes": 14 * GB})
        self.assertIn("freed 4.0 GB of VRAM", " ".join(lines))

    def test_certain_no_drop_still_says_so(self):
        ok, lines = self._push_lines({
            "status": "unloaded", "vram_freed": False,
            "vram_before_bytes": 10 * GB, "vram_after_bytes": 10 * GB})
        self.assertIn("has not dropped", " ".join(lines))

    def test_in_use_is_not_reported_as_unloaded(self):
        """A pinned chat engine (mid-generation) makes unload_all_models()
        return {"status": "in_use", "vram_freed": 0} without raising and without
        a non-2xx, so resp.ok is True and `vram_freed is False` does not match
        either. The image/music/video caller must not read that as "VRAM is
        free" while the chat model is still fully resident."""
        ok, lines = self._push_lines({
            "status": "in_use", "model": "none", "vram_freed": 0,
            "vram_before_bytes": 10 * GB, "vram_after_bytes": 10 * GB,
            "skipped_in_use": ["chat-model"]})
        self.assertFalse(ok, "must report failure so the caller falls back to "
                             "its own conservative swap handling")
        joined = " ".join(lines)
        self.assertNotIn("Chat model unloaded", joined,
                         f"claimed the chat model was unloaded while it was "
                         f"still pinned: {lines}")
        self.assertIn("busy", joined)

    def test_partial_unload_with_pinned_chat_is_not_reported_as_unloaded(self):
        """unload_all_models() reports {"status": "unloaded"} whenever ANYTHING
        was freed (an idle embedding model is enough - released_anything wins
        the status), even when the chat model itself was skipped as in-use. A
        consumer matching only status == "in_use" would read this partial shape
        as chat-model success and load a media model on top of the still-
        resident chat model."""
        ok, lines = self._push_lines({
            "status": "unloaded", "model": "none", "unloaded_models": [],
            "vram_freed": True,
            "vram_before_bytes": 10 * GB, "vram_after_bytes": 11 * GB,
            "skipped_in_use": ["chat-model"]})
        self.assertFalse(ok, "must report failure so the caller falls back to "
                             "its own conservative swap handling")
        joined = " ".join(lines)
        self.assertNotIn("Chat model unloaded", joined,
                         f"claimed the chat model was unloaded while it was "
                         f"still pinned behind a freed bystander: {lines}")
        self.assertIn("chat-model", joined,
                      "the still-resident model must be named honestly")


class TestMediaSwapHonorsPinEndToEnd(_UnloadCase):
    """The test above hand-crafts the {"status": "in_use", ...} body, pinning
    the CONTRACT but not that unload_all_models() actually produces that body
    for a genuinely pinned engine. This drives the REAL
    ``POST /v1/models/unload`` route (via the same TestClient the other endpoint
    tests in this file use) with an engine whose ``active_requests`` is set like
    an in-flight request would, so both halves of the bug are exercised together:
    the real server-side pin check AND unload_chat_for_media's consumption of
    whatever it actually returns."""

    class _Adapter:
        """Wraps the TestClient's httpx response in the requests.Response shape
        unload_chat_for_media actually touches (.ok/.status_code/.json())."""

        def __init__(self, r):
            self._r = r
            self.ok = r.status_code < 400
            self.status_code = r.status_code

        def json(self):
            return self._r.json()

    def test_pinned_engine_is_reported_honestly_not_as_unloaded(self):
        self.engine.active_requests = 1          # a request is mid-generation

        def fake_self_request(method, path, *, json=None, timeout=30, base_url=None,
                              instance_token=None):
            return self._Adapter(self.client.post(f"/v1{path}"))

        from localm import vram as vram_mod
        job = MagicMock()
        lines = []
        job.push.side_effect = lambda d: lines.append(d.get("text", ""))
        with patch("localm.selfclient.self_request", fake_self_request):
            ok = vram_mod.unload_chat_for_media(job, "http://x", "image")

        self.assertFalse(ok, "a pinned chat model was not actually unloaded")
        self.assertTrue(self.engine.loaded,
                        "the pinned engine must survive the media handover's unload")
        joined = " ".join(lines)
        self.assertNotIn("Chat model unloaded", joined,
                         f"claimed success while the real endpoint reported the "
                         f"engine still pinned: {lines}")
        self.assertIn("busy", joined)

    def test_partial_unload_pinned_chat_plus_freed_embedder_end_to_end(self):
        """The partial shape driven through the REAL route: the chat engine is
        pinned (active_requests set like an in-flight request) while a resident
        embedding model frees successfully, so unload_all_models() genuinely
        returns status "unloaded" + skipped_in_use=[engine]. The consumer must
        not read that as chat-model success."""
        self.engine.active_requests = 1          # a request is mid-generation

        def fake_self_request(method, path, *, json=None, timeout=30, base_url=None,
                              instance_token=None):
            return self._Adapter(self.client.post(f"/v1{path}"))

        from localm import vram as vram_mod
        job = MagicMock()
        lines = []
        job.push.side_effect = lambda d: lines.append(d.get("text", ""))
        with patch("localm.inference.embedder.loaded_dim", return_value=384), \
             patch("localm.inference.embedder.active_requests", return_value=0), \
             patch("localm.inference.embedder.reset_embedder", lambda force=True: True), \
             patch("localm.vram.wait_for_vram_release", _impatient_wait), \
             patch("localm.selfclient.self_request", fake_self_request):
            ok = vram_mod.unload_chat_for_media(job, "http://x", "image")

        self.assertFalse(ok, "the chat model was NOT freed; the embedder was")
        self.assertTrue(self.engine.loaded,
                        "the pinned engine must survive the media handover's unload")
        joined = " ".join(lines)
        self.assertNotIn("Chat model unloaded", joined,
                         f"claimed chat-model success off an embedder-only "
                         f"partial unload: {lines}")
        self.assertIn("NOT unloaded", joined,
                      "must hit the partial-unload branch specifically, not a "
                      f"sibling failure path: {lines}")


def _mcp_engine_double(name):
    """An EngineCache stand-in that honors the real Engine's contract.

    A bare MagicMock answers every getattr with a truthy Mock, so an idle
    engine reads as ``active_requests != 0`` and ``unloading is True`` to the
    eviction policy - it is never selected as a victim, the cache logs "no
    resident model could be evicted", and the unload + wait_for_vram_release
    cycle these tests guard never runs. The stub models the attributes the
    policy reads.
    """
    engine = MagicMock(display_name=name)
    engine.active_requests = 0
    engine.unloading = False
    engine.loaded = True
    return engine


class TestMcpModelSwitchHonesty(unittest.TestCase):
    """The MCP server's model switch runs the same before/after cycle.

    It also guards the GPU driver against a TDR: it must WAIT for the previous
    model's native free to land before loading the next one. Those two duties
    pull in opposite directions on a stale probe - seeding the wait from the
    live-only reader turns the wait into a 0-second no-op."""

    def _switch(self, status, scope="device"):
        from localm.plugins.mcpserver.server import EngineCache
        waits = []
        real_wait = wait_for_vram_release

        def _spy_wait(read_free, **kw):
            waits.append(kw.get("before_bytes"))
            kw["timeout_s"] = 0.05
            kw["sleep"] = lambda _s: None
            return real_wait(read_free, **kw)

        logs = []
        # Default DEVICE-scoped: these tests isolate the STALENESS axis (fresh vs
        # stale), so the reading has to be device-global. The process-scoped
        # (blind) case is its own test below.
        gpu = dict(_IDLE, free_scope=scope) if scope else dict(_IDLE)
        cache = EngineCache(default_model="a",
                            engine_factory=_mcp_engine_double)
        cache.get("a")                      # load the first model
        with patch("localm.discover.list_gpus",
                   side_effect=_list_gpus_double([gpu], status)), \
             patch("localm.vram.wait_for_vram_release", _spy_wait), \
             patch("localm.plugins.mcpserver.server._log",
                   side_effect=lambda m: logs.append(m)):
            cache.get("b")                  # switch -> triggers the unload + wait
        return waits, " ".join(logs)

    def test_stale_probe_still_waits_for_the_native_free(self):
        """THE SAFETY GUARD. A probe that merely ran slow must not disable the
        driver-hang wait: before_bytes must still be seeded with a number, because
        wait_for_vram_release short-circuits to a no-op on before_bytes=None."""
        waits, _logs = self._switch(GPU_PROBE_TIMEOUT)
        self.assertEqual(len(waits), 1, "the release wait did not run at all")
        self.assertIsNotNone(
            waits[0],
            "seeded the wait with None on a merely-stale probe, which makes "
            "wait_for_vram_release a 0-second no-op and drops the GPU "
            "driver-hang (TDR) guard")

    def test_stale_probe_does_not_claim_vram_did_not_rise(self):
        _waits, logs = self._switch(GPU_PROBE_TIMEOUT)
        self.assertNotIn("did not rise", logs,
                         f"claimed VRAM did not rise from a stale reading: {logs}")
        self.assertIn("could not confirm", logs,
                      f"a wedged probe must still be surfaced, not silently "
                      f"dropped (rule 5): {logs}")

    def test_live_probe_with_no_rise_still_says_it_did_not_rise(self):
        """The honest negative survives: both ends live AND device-global, genuinely
        no rise. Device-global is what makes the no-rise backable (see the
        process-scoped complement below)."""
        _waits, logs = self._switch(GPU_PROBE_OK)
        self.assertIn("did not rise", logs)

    def test_process_scoped_no_rise_cannot_claim_it_did_not_rise(self):
        """The scope complement: a fresh but PROCESS-scoped reading cannot see
        the previous model's VRAM in its isolated worker, so a no-rise proves
        nothing and this path must not log 'did not rise'."""
        _waits, logs = self._switch(GPU_PROBE_OK, scope="process")
        self.assertNotIn("did not rise", logs,
                         f"claimed VRAM did not rise from a blind reading: {logs}")
        self.assertIn("could not confirm", logs,
                      f"a process-scoped reading must be surfaced as unconfirmable "
                      f"(rule 5), not silently dropped: {logs}")

    def test_stale_before_with_a_recovered_after_still_cannot_claim_no_rise(self):
        """The INVERSE shape: the before-read times out on a cold driver init -
        which is itself what WARMS the driver - so the after-polls answer. With
        'after' live and 'before' a frozen value, their delta means nothing, so
        'did not rise' is unsupportable even though released is False and the
        after-reading is genuine."""
        from localm.plugins.mcpserver.server import EngineCache
        calls = {"n": 0}

        def _cold_then_warm(*, deadline=None, return_status=False):
            calls["n"] += 1
            served = [dict(_IDLE, free=10 * GB)]
            # #1 = the before-read: cold init overruns -> serves last-known-good.
            # #2+ = the after-polls: the driver is warm now and answers.
            status = GPU_PROBE_TIMEOUT if calls["n"] == 1 else GPU_PROBE_OK
            return (served, status) if return_status else served

        logs = []
        cache = EngineCache(default_model="a",
                            engine_factory=_mcp_engine_double)
        cache.get("a")
        with patch("localm.discover.list_gpus", side_effect=_cold_then_warm), \
             patch("localm.vram.wait_for_vram_release", _impatient_wait), \
             patch("localm.plugins.mcpserver.server._log",
                   side_effect=lambda m: logs.append(m)):
            cache.get("b")
        joined = " ".join(logs)
        self.assertGreater(calls["n"], 1, "the after-poll never ran")
        self.assertNotIn("did not rise", joined,
                         f"compared a live 'after' against a frozen 'before' and "
                         f"called the difference a fact: {joined}")
        self.assertIn("could not confirm", joined)


if __name__ == "__main__":
    unittest.main()

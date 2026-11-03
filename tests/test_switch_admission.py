# SPDX-License-Identifier: AGPL-3.0-or-later
"""localm.inference.switch_admission: the pure admission decisions behind
http_server.switch_engine().

Every decision is driven here with plain values and fake engines; nothing is
patched and no event loop runs. tests/test_switch_engine_transition_matrix.py
pins the same decisions end to end through switch_engine().
"""

from types import SimpleNamespace

import pytest

from localm.inference import residency
from localm.inference import switch_admission as sa

GB = 1024 ** 3
MB = 1024 ** 2
NEED = residency.required_vram_bytes(4 * GB)
HEADROOM = residency.DEFAULT_HEADROOM_BYTES


def _budget(name="model-new", *, cap=None, pinned=(), split=False, backend_need=None,
            backend_sized=False, min_need=None):
    return sa.LoadBudget(name=name, vram_required=NEED, headroom=HEADROOM,
                         resident_cap=cap, pinned=frozenset(pinned),
                         check_split_fit=split, backend_need=backend_need,
                         backend_sized=backend_sized, min_need=min_need)


def _probe(free=20 * GB, *, ok=True, process_scoped=False, shortfall=(),
           adaptive=False):
    return sa.VramProbe(free=free, probe_ok=ok, process_scoped=process_scoped,
                        shortfall=list(shortfall), shares_adaptive=adaptive)


def _engine(*, busy=0, unloading=False):
    return SimpleNamespace(active_requests=busy, unloading=unloading)


class _NoLookups(dict):
    """An engines mapping that fails the test if anything reads from it."""

    def get(self, *args, **kwargs):
        raise AssertionError("the victim was looked up")

    def __getitem__(self, key):
        raise AssertionError("the victim was looked up")


class TestLoadBudget:
    def test_needed_bytes_is_the_requirement_plus_headroom(self):
        assert _budget().needed_bytes == NEED + HEADROOM

    def test_a_larger_backend_need_raises_the_bar_but_not_the_split_share(self):
        budget = _budget(backend_need=NEED + HEADROOM + GB)
        assert budget.needed_bytes == NEED + HEADROOM + GB
        assert budget.whole_model_bytes == NEED + HEADROOM + GB
        assert budget.split_share_bytes == NEED + HEADROOM

    def test_a_smaller_backend_need_never_lowers_the_bar(self):
        budget = _budget(backend_need=NEED // 2)
        assert budget.needed_bytes == NEED + HEADROOM
        assert budget.whole_model_bytes == NEED
        assert budget.split_share_bytes == NEED + HEADROOM

    def test_a_backend_sized_need_replaces_the_file_estimate_and_its_headroom(self):
        budget = _budget(backend_need=NEED // 2, backend_sized=True)
        assert budget.needed_bytes == NEED // 2
        assert budget.plan_bytes == NEED // 2
        assert budget.keeps_experts_in_ram is False
        assert budget.split_share_bytes == NEED + HEADROOM

    def test_backend_sized_without_a_figure_keeps_the_file_estimate(self):
        assert _budget(backend_sized=True).needed_bytes == NEED + HEADROOM

    def test_the_plan_bar_is_the_expert_offload_minimum_when_it_is_lower(self):
        budget = _budget(backend_need=20 * GB, backend_sized=True, min_need=3 * GB)
        assert budget.needed_bytes == 20 * GB
        assert budget.plan_bytes == 3 * GB
        assert budget.keeps_experts_in_ram is True

    @pytest.mark.parametrize("min_need", [None, 0, 20 * GB, 21 * GB])
    def test_a_minimum_not_below_the_full_need_changes_nothing(self, min_need):
        budget = _budget(backend_need=20 * GB, backend_sized=True, min_need=min_need)
        assert budget.plan_bytes == 20 * GB
        assert budget.keeps_experts_in_ram is False


class TestExpertsInRamFit:
    MOE = dict(backend_need=20 * GB, backend_sized=True, min_need=3 * GB)

    def test_free_vram_between_the_two_bars_fits(self):
        assert sa.experts_in_ram_fit(_probe(free=3 * GB), _budget(**self.MOE)) is True

    def test_free_vram_below_the_minimum_does_not(self):
        assert sa.experts_in_ram_fit(_probe(free=3 * GB - 1), _budget(**self.MOE)) is False

    def test_a_dense_model_never_fits_this_way(self):
        budget = _budget(backend_need=20 * GB, backend_sized=True)
        assert sa.experts_in_ram_fit(_probe(free=19 * GB), budget) is False

    @pytest.mark.parametrize("probe", [
        _probe(free=None), _probe(ok=False), _probe(process_scoped=True),
        _probe(shortfall=[{"index": 1, "needed": GB, "free": 0}]),
    ], ids=["unmeasurable", "inconclusive", "process-scoped", "split-shortfall"])
    def test_an_untrusted_reading_never_fits(self, probe):
        assert sa.experts_in_ram_fit(probe, _budget(**self.MOE)) is False


class TestVramProbe:
    @pytest.mark.parametrize("ok,free,measurable,cannot_measure,inconclusive", [
        (True, 8 * GB, True, False, False),
        (True, None, False, True, False),
        (False, 8 * GB, True, False, True),
        (False, None, False, False, True),
    ])
    def test_measurable_cannot_measure_and_inconclusive_are_distinct(
            self, ok, free, measurable, cannot_measure, inconclusive):
        probe = _probe(free, ok=ok)
        assert probe.measurable is measurable
        assert probe.cannot_measure is cannot_measure
        assert probe.inconclusive is inconclusive


class TestDecideAdmission:
    def test_a_fitting_load_under_the_cap_is_admitted_without_a_victim_lookup(self):
        decision = sa.decide_admission(_probe(), _budget(), ["model-a"],
                                       _NoLookups(model_a=_engine()))
        assert decision == sa.AdmissionDecision(sa.ADMIT, vram_ok=True, over_cap=False)

    def test_free_vram_above_the_estimate_but_below_the_backend_need_evicts(self):
        free = NEED + HEADROOM + GB
        engines = {"model-a": _engine()}
        decision = sa.decide_admission(
            _probe(free=free), _budget(backend_need=free + MB), ["model-a"], engines)
        assert decision == sa.AdmissionDecision(
            sa.EVICT_IDLE, vram_ok=False, over_cap=False, victim="model-a")

    def test_free_vram_covering_the_backend_need_is_admitted(self):
        free = NEED + HEADROOM + GB
        decision = sa.decide_admission(
            _probe(free=free), _budget(backend_need=free), ["model-a"],
            _NoLookups(model_a=_engine()))
        assert decision == sa.AdmissionDecision(sa.ADMIT, vram_ok=True, over_cap=False)

    def test_the_least_recently_used_idle_model_is_the_victim(self):
        engines = {"model-a": _engine(), "model-b": _engine()}
        decision = sa.decide_admission(_probe(free=GB), _budget(),
                                       ["model-a", "model-b"], engines)
        assert decision == sa.AdmissionDecision(
            sa.EVICT_IDLE, vram_ok=False, over_cap=False, victim="model-a")

    def test_busy_pinned_unloading_and_requested_models_are_never_victims(self):
        engines = {"model-new": _engine(), "busy": _engine(busy=1),
                   "pinned": _engine(), "going": _engine(unloading=True),
                   "idle": _engine()}
        decision = sa.decide_admission(
            _probe(free=GB), _budget(pinned=["pinned"]),
            ["model-new", "busy", "pinned", "going", "idle"], engines)
        assert decision.action == sa.EVICT_IDLE and decision.victim == "idle"

    def test_nothing_evictable_and_short_of_vram_is_exhausted(self):
        decision = sa.decide_admission(_probe(free=GB), _budget(), ["busy"],
                                       {"busy": _engine(busy=1)})
        assert decision == sa.AdmissionDecision(sa.EXHAUSTED, vram_ok=False,
                                                over_cap=False)

    def test_a_cap_miss_with_enough_vram_and_nothing_evictable_loads_over_the_cap(self):
        decision = sa.decide_admission(_probe(), _budget(cap=1, pinned=["model-a"]),
                                       ["model-a"], {"model-a": _engine()})
        assert decision == sa.AdmissionDecision(sa.ADMIT_OVER_CAP, vram_ok=True,
                                                over_cap=True)

    def test_a_cap_miss_evicts_an_idle_model_even_when_vram_fits(self):
        decision = sa.decide_admission(_probe(), _budget(cap=1), ["model-a"],
                                       {"model-a": _engine()})
        assert decision == sa.AdmissionDecision(
            sa.EVICT_IDLE, vram_ok=True, over_cap=True, victim="model-a")

    def test_reloading_a_resident_model_never_exceeds_the_cap(self):
        decision = sa.decide_admission(_probe(), _budget("model-a", cap=1),
                                       ["model-a"], {"model-a": _engine()})
        assert decision.action == sa.ADMIT

    @pytest.mark.parametrize("probe", [
        _probe(ok=False),
        _probe(free=None),
        _probe(process_scoped=True),
        _probe(shortfall=[{"index": 0, "needed": 3000 * MB, "free": 1000 * MB}]),
        _probe(free=NEED + HEADROOM - 1),
    ], ids=["inconclusive", "unmeasurable", "process-scoped", "split-shortfall",
            "one-byte-short"])
    def test_a_reading_that_cannot_prove_the_fit_never_admits(self, probe):
        decision = sa.decide_admission(probe, _budget(), [], {})
        assert decision == sa.AdmissionDecision(sa.EXHAUSTED, vram_ok=False,
                                                over_cap=False)

    def test_exactly_the_requirement_plus_headroom_fits(self):
        decision = sa.decide_admission(_probe(free=NEED + HEADROOM), _budget(), [], {})
        assert decision.action == sa.ADMIT


class TestExhaustedProbeVerdict:
    @pytest.mark.parametrize("probe,retries,verdict", [
        (_probe(free=None), 0, sa.LOAD_BEST_EFFORT),
        (_probe(free=None), 5, sa.LOAD_BEST_EFFORT),
        (_probe(ok=False), 0, sa.RETRY_PROBE),
        (_probe(ok=False), 1, sa.RETRY_PROBE),
        (_probe(ok=False), 2, sa.GIVE_UP_PROBE),
        (_probe(ok=False, free=None), 2, sa.GIVE_UP_PROBE),
        (_probe(free=GB), 0, sa.KEEP_FREEING),
    ])
    def test_verdicts(self, probe, retries, verdict):
        assert sa.exhausted_probe_verdict(probe, retries_used=retries,
                                          max_retries=2) == verdict

    def test_no_retries_configured_gives_up_at_once(self):
        assert sa.exhausted_probe_verdict(_probe(ok=False), retries_used=0,
                                          max_retries=0) == sa.GIVE_UP_PROBE


class TestPinBlocksPeerCooperation:
    def test_true_when_only_a_pin_keeps_a_model_resident(self):
        assert sa.pin_blocks_peer_cooperation(
            _budget(pinned=["model-a"]), ["model-a"], {"model-a": _engine()})

    def test_false_without_pins(self):
        assert not sa.pin_blocks_peer_cooperation(
            _budget(), ["model-a"], {"model-a": _engine()})

    def test_false_when_the_pinned_model_is_busy_anyway(self):
        assert not sa.pin_blocks_peer_cooperation(
            _budget(pinned=["model-a"]), ["model-a"], {"model-a": _engine(busy=1)})


class TestBusyVictimCandidate:
    def test_an_explicit_switch_picks_the_least_recently_used_busy_model(self):
        engines = {"model-a": _engine(busy=1), "model-b": _engine(busy=2)}
        assert sa.busy_victim_candidate(
            _budget(), ["model-a", "model-b"], engines, preempt=True,
            already_attempted=False) == "model-a"

    def test_an_api_routed_load_never_picks_one(self):
        assert sa.busy_victim_candidate(
            _budget(), ["model-a"], {"model-a": _engine(busy=1)}, preempt=False,
            already_attempted=False) is None

    def test_only_once_per_load_attempt(self):
        assert sa.busy_victim_candidate(
            _budget(), ["model-a"], {"model-a": _engine(busy=1)}, preempt=True,
            already_attempted=True) is None

    def test_pinned_and_unloading_models_are_skipped(self):
        engines = {"pinned": _engine(busy=1), "going": _engine(busy=1, unloading=True)}
        assert sa.busy_victim_candidate(
            _budget(pinned=["pinned"]), ["pinned", "going"], engines,
            preempt=True, already_attempted=False) is None


class TestFinalExhaustionVerdict:
    _SHORT = [{"index": 1, "needed": 3000 * MB, "free": 1000 * MB}]

    @pytest.mark.parametrize("preempt,force", [(False, False), (True, False),
                                               (True, True)])
    def test_static_split_shortfall_is_refused_whoever_asks(self, preempt, force):
        probe = _probe(shortfall=self._SHORT, adaptive=False)
        assert sa.final_exhaustion_verdict(
            probe, preempt=preempt, force=force) == sa.REFUSE_SPLIT_SHORTFALL

    @pytest.mark.parametrize("shortfall", [[], _SHORT])
    @pytest.mark.parametrize("preempt,force,verdict", [
        (True, False, sa.CONFIRM_DEGRADED_LOAD),
        (True, True, sa.DEFER_TO_BACKEND),
        (False, False, sa.DEFER_TO_BACKEND),
    ])
    def test_adaptive_or_no_shortfall_asks_an_explicit_switch_else_defers(
            self, shortfall, preempt, force, verdict):
        probe = _probe(shortfall=shortfall, adaptive=True)
        assert sa.final_exhaustion_verdict(probe, preempt=preempt,
                                           force=force) == verdict


class TestEvictableResidents:
    def test_lists_idle_and_busy_models_but_never_the_target_a_pin_or_an_unloading_one(self):
        engines = {"idle": _engine(), "busy": _engine(busy=1),
                   "pinned": _engine(), "going": _engine(unloading=True),
                   "model-new": _engine()}
        lru = ["idle", "pinned", "going", "model-new", "busy", "gone"]

        assert sa.evictable_residents(_budget(pinned=["pinned"]), lru, engines) == [
            "idle", "busy"]


class TestFreeAfterEviction:
    def test_adds_every_hold_to_the_free_reading(self):
        assert sa.free_after_eviction(_probe(free=2 * GB), [3 * GB, GB]) == 6 * GB

    @pytest.mark.parametrize("probe, holds", [
        (_probe(ok=False), [GB]),
        (_probe(free=None), [GB]),
        (_probe(process_scoped=True), [GB]),
        (_probe(shortfall=[{"index": 0, "needed": GB, "free": 0}]), [GB]),
        (_probe(), [GB, 0]),
    ], ids=["inconclusive", "unmeasurable", "process-scoped", "split-shortfall",
            "unknown-hold"])
    def test_cannot_estimate(self, probe, holds):
        assert sa.free_after_eviction(probe, holds) is None


class TestMessages:
    def test_split_shortfall_refusal_names_every_short_device(self):
        assert sa.split_shortfall_refusal("m", [
            {"index": 0, "needed": 3000 * MB, "free": 2048 * MB},
            {"index": 1, "needed": 3000 * MB, "free": 1024 * MB},
        ]) == ("Not enough VRAM on the configured split device(s) to load 'm' "
               "(GPU 0 needs ~3000 MB, 2048 MB free; GPU 1 needs ~3000 MB, "
               "1024 MB free).")

    def test_degraded_load_confirm(self):
        budget = _budget("m", backend_need=4800 * MB, backend_sized=True)
        assert sa.degraded_load_confirm("m", budget, 3000 * MB) == {
            "status": "confirm_required", "model": "m", "reason": "degraded_load",
            "plan_mb": 4800, "full_mb": 4800, "free_mb": 3000, "moe": False,
            "unloaded": [], "unloads_shared": False, "unloads_peers": False,
            "detail": ("'m' does not fit: every layer on the GPU needs about 4800 MB "
                       "of free VRAM, and about 3000 MB is free. Loading it anyway "
                       "runs some of its layers on the CPU, which is slower.")}

    def test_degraded_load_confirm_for_a_moe_model_names_both_figures(self):
        budget = _budget("m", backend_need=25570 * MB, backend_sized=True,
                         min_need=4200 * MB)
        result = sa.degraded_load_confirm("m", budget, 3000 * MB, ["big"], shared=True)
        assert (result["plan_mb"], result["full_mb"], result["moe"]) == (4200, 25570, True)
        assert result["detail"] == (
            "'m' does not fit: even with its experts in system RAM, every layer on "
            "the GPU needs about 4200 MB of free VRAM (about 25570 MB with every "
            "expert on the GPU too), and about 3000 MB would be free after unloading "
            "'big', the shared embedding, reranker and speech models. Loading it "
            "anyway runs some of its layers on the CPU, which is slower.")
        assert "localm config" not in result["detail"]

    def test_busy_victim_confirm(self):
        assert sa.busy_victim_confirm("m", "v", "still in use") == {
            "status": "confirm_required", "model": "m",
            "detail": ("loading 'm' needs to free 'v', which is still in use; "
                       "loading it anyway stops 'v' and cuts off the reply it "
                       "is writing")}

    def test_inconclusive_texts(self):
        assert sa.inconclusive_restart_reason("m", 3) == (
            "GPU probe still inconclusive after 3 attempts loading 'm'")
        assert sa.inconclusive_refusal("m", 3, 4.6) == (
            "Cannot load 'm': tried measuring free VRAM 3 times over about 5s "
            "without a conclusive reading, could not free anything, and "
            "automatic recovery is unavailable. Please file a bug report if "
            "this keeps happening.")
        assert sa.inconclusive_restarting_refusal("m").startswith(
            "Cannot load 'm' right now: the server detected a stuck GPU check")

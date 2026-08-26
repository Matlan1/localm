# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Tests for the shared multi-model residency policy (inference/residency.py).

This module is the single source of truth both serving layers ask "may this
model load alongside the resident ones, and if not who is the safe victim". A
wrong answer in the PERMIT direction costs a native OOM or a driver TDR rather
than a tidy error, so the permit tests below are adversarial: each one takes an
otherwise-fitting load and breaks exactly one precondition.
"""

from types import SimpleNamespace

from localm.inference import residency

GB = 1024 ** 3


def _engine(active=0, unloading=False):
    return SimpleNamespace(active_requests=active, unloading=unloading)


class TestFitsAlongsideResidents:
    def _fits(self, **over):
        kw = dict(free_vram=20 * GB, vram_required=4 * GB, probe_ok=True)
        kw.update(over)
        return residency.fits_alongside_residents(**kw)

    def test_plenty_of_free_vram_admits_with_no_eviction(self):
        assert self._fits() is True

    def test_exactly_required_plus_headroom_admits(self):
        assert self._fits(free_vram=4 * GB + residency.DEFAULT_HEADROOM_BYTES) is True

    def test_one_byte_under_the_headroom_refuses(self):
        """The headroom is a floor, not a suggestion."""
        assert self._fits(
            free_vram=4 * GB + residency.DEFAULT_HEADROOM_BYTES - 1) is False

    def test_required_alone_without_headroom_refuses(self):
        assert self._fits(free_vram=4 * GB) is False

    def test_stale_probe_refuses_even_when_the_number_looks_huge(self):
        """probe_ok=False means the reading was not taken THIS call. A frozen
        last-known-good that happens to read high must never admit a load on top
        of resident peers - that is the direction that OOMs."""
        assert self._fits(free_vram=100 * GB, probe_ok=False) is False

    def test_unmeasurable_box_refuses(self):
        """free=None is 'this box cannot report VRAM', never 'plenty'."""
        assert self._fits(free_vram=None) is False

    def test_split_shortfall_refuses_despite_sufficient_aggregate(self):
        """Aggregate free clears the bar while ONE split device is short: the
        GGUF backend divides by a static ratio with no live per-device check."""
        assert self._fits(shortfall=[{"index": 1, "needed": 3 * GB,
                                      "free": 1 * GB}]) is False

    def test_custom_headroom_is_honored(self):
        assert self._fits(free_vram=6 * GB, headroom=1 * GB) is True
        assert self._fits(free_vram=6 * GB, headroom=3 * GB) is False

    def test_process_scoped_reading_refuses_despite_plenty_of_free_vram(self):
        """Every resident model lives in its OWN isolated worker subprocess, so a
        PROCESS-scoped reading is structurally blind to a resident peer's VRAM and
        can only ever OVER-report free space. Without this guard, exactly this
        shape (a fresh, "sufficient" 15GB/10GB reading) returns True even though
        only ~7GB is genuinely free once an 8GB resident model - invisible to a
        process-scoped probe - is accounted for."""
        assert residency.fits_alongside_residents(
            free_vram=15 * GB, vram_required=10 * GB, probe_ok=True,
            is_process_scoped=True) is False

    def test_process_scoped_refuses_even_when_free_vram_is_enormous(self):
        """Not a stricter threshold - an unconditional early return. A reading
        that would trivially clear any headroom still refuses once it is known to
        be blind to other processes' VRAM."""
        assert self._fits(free_vram=1000 * GB, is_process_scoped=True) is False

    def test_device_scoped_reading_admits_exactly_as_before(self):
        """The PERMIT-only guardrail: an explicit is_process_scoped=False (an
        ordinary device-wide reading) must admit exactly as it does with the
        guard absent - a genuinely trustworthy reading must never be turned into
        a refusal."""
        assert self._fits(is_process_scoped=False) is True

    def test_process_scoped_flag_defaults_to_false(self):
        """A caller that omits the keyword entirely, as every test above this
        one does, must see the same behavior as an explicit
        is_process_scoped=False: the parameter defaults to False."""
        assert self._fits() is True


class TestRequiredBytes:
    def test_applies_the_overhead_factor(self):
        assert residency.required_vram_bytes(10 * GB) == int(10 * GB * 1.2)

    def test_footprint_of_a_file(self, tmp_path):
        f = tmp_path / "m.gguf"
        f.write_bytes(b"x" * 2048)
        assert residency.model_footprint_bytes(f) == 2048

    def test_footprint_of_a_sharded_directory_sums_the_shards(self, tmp_path):
        d = tmp_path / "hf-model"
        (d / "nested").mkdir(parents=True)
        (d / "a.safetensors").write_bytes(b"x" * 100)
        (d / "nested" / "b.safetensors").write_bytes(b"x" * 50)
        assert residency.model_footprint_bytes(d) == 150

    def test_missing_path_falls_back_pessimistically(self, tmp_path):
        """An unreadable model must make the fit check HARDER, not easier."""
        assert residency.model_footprint_bytes(
            tmp_path / "gone") == residency.UNKNOWN_FOOTPRINT_BYTES


class TestPickEvictionVictim:
    def test_picks_least_recently_used(self):
        engines = {"a": _engine(), "b": _engine()}
        assert residency.pick_eviction_victim(["a", "b"], engines) == "a"

    def test_never_evicts_the_requested_model(self):
        engines = {"a": _engine(), "b": _engine()}
        assert residency.pick_eviction_victim(
            ["a", "b"], engines, requested="a") == "b"

    def test_skips_a_busy_engine(self):
        engines = {"a": _engine(active=1), "b": _engine()}
        assert residency.pick_eviction_victim(["a", "b"], engines) == "b"

    def test_skips_an_engine_already_mid_unload(self):
        """Another path keeps its victim listed until the native free lands, so
        selecting it again would double-free one native context."""
        engines = {"a": _engine(unloading=True), "b": _engine()}
        assert residency.pick_eviction_victim(["a", "b"], engines) == "b"

    def test_skips_a_pinned_model(self):
        engines = {"a": _engine(), "b": _engine()}
        assert residency.pick_eviction_victim(
            ["a", "b"], engines, pinned={"a"}) == "b"

    def test_returns_none_when_nothing_is_safe_to_evict(self):
        engines = {"a": _engine(active=2), "b": _engine(unloading=True),
                   "c": _engine()}
        assert residency.pick_eviction_victim(
            ["a", "b", "c"], engines, pinned={"c"}) is None

    def test_ignores_a_name_with_no_engine_behind_it(self):
        engines = {"b": _engine()}
        assert residency.pick_eviction_victim(["ghost", "b"], engines) == "b"


class TestResidentCap:
    def test_default_is_no_cap(self):
        assert residency.resident_cap({}) is None

    def test_reads_a_valid_cap(self):
        assert residency.resident_cap({"max_resident_models": 2}) == 2

    def test_zero_and_negative_are_ignored_not_coerced(self):
        """A typo that silently became 'cap 1' would look like a perf
        regression with nothing in the log to trace it to."""
        assert residency.resident_cap({"max_resident_models": 0}) is None
        assert residency.resident_cap({"max_resident_models": -3}) is None

    def test_non_integer_is_ignored(self):
        assert residency.resident_cap({"max_resident_models": "two"}) is None
        assert residency.resident_cap({"max_resident_models": True}) is None

    def test_exceeds_cap_only_when_admitting_a_new_name(self):
        assert residency.exceeds_resident_cap(["a"], "b", 1) is True
        assert residency.exceeds_resident_cap(["a"], "b", 2) is False
        assert residency.exceeds_resident_cap(["a", "b"], "c", 2) is True

    def test_reloading_a_resident_model_never_exceeds_the_cap(self):
        assert residency.exceeds_resident_cap(["a", "b"], "a", 2) is False

    def test_no_cap_never_exceeds(self):
        assert residency.exceeds_resident_cap(["a", "b", "c"], "d", None) is False


class TestPinnedModels:
    def test_default_is_empty(self):
        assert residency.pinned_model_names({}) == frozenset()
        assert residency.pinned_model_names({"pinned_models": None}) == frozenset()

    def test_reads_names(self):
        assert residency.pinned_model_names(
            {"pinned_models": ["a", "b"]}) == frozenset({"a", "b"})

    def test_non_list_is_ignored(self):
        assert residency.pinned_model_names(
            {"pinned_models": "a,b"}) == frozenset()

    def test_non_string_entries_are_dropped(self):
        assert residency.pinned_model_names(
            {"pinned_models": ["a", 7, None, ""]}) == frozenset({"a"})

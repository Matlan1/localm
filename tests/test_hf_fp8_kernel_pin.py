# SPDX-License-Identifier: AGPL-3.0-or-later
"""The finegrained-fp8 Hub kernel (downloaded Python/GPU code) is requested at
a pinned commit, not at a mutable version ref. A transformers that asks for a
different kernel version loads no native kernel until the pin is moved, and a
Hub cache filled before the pin existed keeps working offline."""

from __future__ import annotations

import re
from types import SimpleNamespace

import pytest

from localm.inference.backends import _hf_worker as w

hub_kernels = pytest.importorskip("transformers.integrations.hub_kernels")
MAPPING = hub_kernels._HUB_KERNEL_MAPPING


@pytest.fixture(autouse=True)
def _restore_mapping(monkeypatch):
    monkeypatch.setitem(MAPPING, "finegrained-fp8", dict(MAPPING["finegrained-fp8"]))


def test_the_pin_is_a_full_commit_and_the_version_it_belongs_to():
    assert re.fullmatch(r"[0-9a-f]{40}", w.FP8_KERNEL_REVISION)
    assert isinstance(w.FP8_KERNEL_VERSION, int)


def test_installed_transformers_still_requests_the_pinned_kernel_version():
    """Fails when a transformers upgrade changes the kernel version it asks for:
    re-read the commit that version points at before moving the pin."""
    entry = MAPPING["finegrained-fp8"]
    assert entry["repo_id"] == w.FP8_KERNEL_REPO
    assert entry.get("version") == w.FP8_KERNEL_VERSION


def test_the_request_can_be_set_to_the_commit_and_back_to_the_version():
    assert w._set_fp8_kernel_request(pinned=True) is None
    assert MAPPING["finegrained-fp8"] == {"repo_id": w.FP8_KERNEL_REPO,
                                          "revision": w.FP8_KERNEL_REVISION}
    assert w._set_fp8_kernel_request(pinned=False) is None
    assert MAPPING["finegrained-fp8"] == {"repo_id": w.FP8_KERNEL_REPO,
                                          "version": w.FP8_KERNEL_VERSION}


def test_a_transformers_that_requests_another_version_is_refused_and_left_alone(monkeypatch):
    other = {"repo_id": w.FP8_KERNEL_REPO, "version": w.FP8_KERNEL_VERSION + 1}
    monkeypatch.setitem(MAPPING, "finegrained-fp8", dict(other))
    reason = w._set_fp8_kernel_request(pinned=True)
    assert reason is not None and "not version" in reason
    assert MAPPING["finegrained-fp8"] == other


def test_another_repository_is_refused(monkeypatch):
    monkeypatch.setitem(MAPPING, "finegrained-fp8",
                        {"repo_id": "someone/else", "version": w.FP8_KERNEL_VERSION})
    assert w._set_fp8_kernel_request(pinned=True) is not None


class TestLoad:
    @pytest.fixture()
    def load(self, monkeypatch):
        from localm.inference.backends import _hf_hub_gate as hub_gate
        state = SimpleNamespace(seen=[], results=[None], cached=False, recorded=[], offline=0)

        def try_kernel():
            state.seen.append(dict(MAPPING["finegrained-fp8"]))
            return state.results.pop(0) if state.results else None

        monkeypatch.setattr(w, "_try_fp8_kernel", try_kernel)
        monkeypatch.setattr(w, "_fp8_kernel_cached", lambda: state.cached)
        monkeypatch.setattr(hub_gate, "set_hub_offline",
                            lambda: setattr(state, "offline", state.offline + 1))
        monkeypatch.setattr(hub_gate, "record_kernel_version_ref",
                            lambda repo, version: state.recorded.append((repo, version)))
        state.run = lambda offline=False: w._load_fp8_kernel(
            "model-dir", SimpleNamespace(offline=offline, native=True))
        return state

    PINNED = {"repo_id": w.FP8_KERNEL_REPO, "revision": w.FP8_KERNEL_REVISION}
    VERSIONED = {"repo_id": w.FP8_KERNEL_REPO, "version": w.FP8_KERNEL_VERSION}

    def test_an_online_load_requests_the_pinned_commit_and_records_the_version(self, load):
        plan = load.run()
        assert load.seen == [self.PINNED] and plan.offline is False
        assert load.recorded == [(w.FP8_KERNEL_REPO, w.FP8_KERNEL_VERSION)]

    def test_an_offline_load_with_the_pinned_commit_cached_requests_it(self, load):
        load.cached = True
        load.run(offline=True)
        assert load.seen == [self.PINNED] and load.recorded == []

    def test_an_offline_load_with_only_an_older_cache_requests_the_cached_version(self, load):
        load.run(offline=True)
        assert load.seen == [self.VERSIONED]

    def test_a_failed_fetch_falls_back_to_a_cache_filled_before_the_pin(self, load):
        load.results = ["OSError: offline", None]
        plan = load.run()
        assert load.seen == [self.PINNED, self.VERSIONED]
        assert plan.offline is True and load.offline == 1 and load.recorded == []

    def test_a_failed_fetch_keeps_the_pin_when_that_commit_is_cached(self, load):
        load.cached = True
        load.results = ["OSError: offline", None]
        load.run()
        assert load.seen == [self.PINNED, self.PINNED]

    def test_a_kernel_that_cannot_load_expands_the_weights(self, load, monkeypatch):
        load.results = ["OSError: offline", "OSError: still offline"]
        plan = load.run()
        assert not getattr(plan, "native", False)
        assert "could not be loaded" in plan.reason

    def test_a_transformers_with_another_kernel_version_loads_no_native_kernel(
            self, load, monkeypatch):
        monkeypatch.setitem(MAPPING, "finegrained-fp8",
                            {"repo_id": w.FP8_KERNEL_REPO, "version": w.FP8_KERNEL_VERSION + 1})
        plan = load.run()
        assert load.seen == [] and not getattr(plan, "native", False)
        assert "could not be pinned" in plan.reason

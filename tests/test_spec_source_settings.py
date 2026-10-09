# SPDX-License-Identifier: AGPL-3.0-or-later
"""The spec_source setting from config to the reply's usage figures."""

import ctypes
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from localm.config import DEFAULT_CONFIG
from localm.inference.backends.gguf import GgufBackend
from localm.inference.backends.llamacpp._drafting import resolve_spec_source
from tests._bare_llama import make_bare_llama


@pytest.mark.parametrize("spec_source,mtp_enabled,expected", [
    (None, False, "off"), (None, True, "mtp"), ("", True, "mtp"),
    ("ngram", True, "ngram"), ("off", True, "off"), ("MTP", False, "mtp"),
])
def test_an_unset_source_follows_mtp_enabled_and_an_explicit_one_wins(
        spec_source, mtp_enabled, expected):
    assert resolve_spec_source(spec_source, mtp_enabled) == expected


def test_an_unknown_source_is_refused():
    with pytest.raises(ValueError, match="spec_source"):
        resolve_spec_source("draft-model", False)


@pytest.mark.parametrize("cfg_source,cfg_mtp,mtp_override,source_override,expected", [
    (None, False, None, None, "off"),
    (None, True, None, None, "mtp"),
    ("ngram", True, None, None, "ngram"),
    ("ngram", False, True, None, "mtp"),
    ("ngram", False, False, None, "off"),
    ("ngram", True, True, "off", "off"),
    ("nonsense", True, None, None, "mtp"),
])
def test_create_backend_resolves_the_source(cfg_source, cfg_mtp, mtp_override,
                                            source_override, expected):
    from localm.inference import engine as engine_mod
    captured = {}

    class _FakeBackend:
        def __init__(self, *a, **kw):
            captured.update(kw)

    cfg = dict(DEFAULT_CONFIG)
    cfg["spec_source"] = cfg_source
    cfg["mtp_enabled"] = cfg_mtp
    with patch.object(engine_mod, "load_config", return_value=cfg), \
         patch("localm.inference.backends.gguf.GgufBackend", _FakeBackend):
        engine_mod.create_backend("model.gguf", mtp_enabled=mtp_override,
                                  spec_source=source_override)

    assert captured["spec_source"] == expected
    assert captured["mtp_enabled"] is (expected == "mtp")


@pytest.mark.parametrize("raw,expected", [(None, None), ("", None), (12, 12),
                                          ("6", 6), ("many", None)])
def test_spec_draft_tokens_from_config(raw, expected):
    from localm.inference.engine import _resolve_spec_draft_tokens
    assert _resolve_spec_draft_tokens({"spec_draft_tokens": raw}, None) == expected
    assert _resolve_spec_draft_tokens({"spec_draft_tokens": raw}, 3) == 3


def test_the_engine_forwards_the_source_override():
    from localm.inference import engine as engine_mod
    seen = {}

    def _fake_create_backend(model_path, **kw):
        seen.update(kw)
        return MagicMock()

    with patch.object(engine_mod, "create_backend", _fake_create_backend):
        engine_mod.Engine("model.gguf", spec_source="ngram", spec_draft_tokens=5)
    assert (seen["spec_source"], seen["spec_draft_tokens"]) == ("ngram", 5)


def test_the_setting_validates_as_a_nullable_choice():
    from localm import settings_schema as ss
    assert ss.validate_update({"spec_source": "ngram"}) == {"spec_source": "ngram"}
    assert ss.validate_update({"spec_source": ""}) == {"spec_source": None}
    with pytest.raises(ValueError):
        ss.validate_update({"spec_source": "draft-model"})
    stored = ss.validate_update({"spec_draft_tokens": "12"})
    assert stored == {"spec_draft_tokens": 12} and type(stored["spec_draft_tokens"]) is int
    assert ss.validate_update({"spec_draft_tokens": ""}) == {"spec_draft_tokens": None}
    with pytest.raises(ValueError):
        ss.validate_update({"spec_draft_tokens": 17})
    assert DEFAULT_CONFIG["spec_source"] is None
    assert DEFAULT_CONFIG["spec_draft_tokens"] is None


@pytest.mark.parametrize("key", ["spec_source", "spec_draft_tokens"])
def test_new_load_keys_reach_the_bug_report_and_the_routing_latch(key):
    from localm.bugreport import _SAFE_CONFIG_KEYS
    from localm.inference.routing_latch import LOAD_CONFIG_KEYS
    assert key in _SAFE_CONFIG_KEYS
    assert key in LOAD_CONFIG_KEYS


def test_the_backend_enables_mtp_only_for_the_mtp_source():
    assert GgufBackend("m.gguf", spec_source="ngram", mtp_enabled=True).mtp_enabled is False
    assert GgufBackend("m.gguf", mtp_enabled=True).spec_source == "mtp"
    assert GgufBackend("m.gguf").spec_source == "off"


# --------------------------------------------------------------------------- #
#  Rollback snapshots                                                         #
# --------------------------------------------------------------------------- #

def _fresh_context_n_rs_seq(llm):
    from localm.inference.backends.llamacpp import llama as llama_mod
    made = []
    with patch.object(llama_mod, "api") as api:
        api.llama_context_default_params.side_effect = lambda: SimpleNamespace(n_rs_seq=0)

        def _init(model, cp):
            made.append(cp)
            return ctypes.c_void_p(7)
        api.llama_init_from_model.side_effect = _init
        llm._tokenizer = MagicMock()
        llm._prefill_fresh_context([], 100)
    llm._ctx_ptr = None
    return made[-1].n_rs_seq


@pytest.mark.parametrize("source,draft,mtp_on,expected", [
    ("ngram", 6, False, 6), ("ngram", 1, False, 2), ("off", 0, False, 0),
])
def test_a_grown_context_keeps_the_snapshots_its_source_needs(source, draft, mtp_on, expected):
    llm = make_bare_llama(_model_ptr=ctypes.c_void_p(1))
    llm._spec_source_name = source
    llm._mtp_enabled = mtp_on
    llm._spec_draft_max = draft
    assert _fresh_context_n_rs_seq(llm) == expected


def test_the_initial_context_is_set_up_by_the_tested_method():
    import inspect

    from localm.inference.backends.llamacpp.llama import LlamaCpp
    src = inspect.getsource(LlamaCpp.__init__)
    assert "self._apply_initial_spec_params(cp, spec_draft_tokens)" in src
    assert "n_rs_seq" not in src.split("_apply_initial_spec_params")[0].split(
        "cp = api.llama_context_default_params()")[-1]


def test_the_vram_charge_counts_the_ngram_snapshots(tmp_path):
    from localm.inference.backends.llamacpp._ngram import (
        NGRAM_RECURRENT_DRAFT_TOKENS_MAX, ngram_rs_seq)
    per_copy = 1000
    b = GgufBackend(str(tmp_path / "m.gguf"), spec_source="ngram")
    with patch("localm.model_manager.gguf.gguf_recurrent_state_bytes",
               return_value=per_copy), \
         patch.object(GgufBackend, "_gguf_parsed_tensor_entries", return_value=None):
        charge = b._recurrent_state_vram_bytes()
    assert charge == per_copy * (1 + ngram_rs_seq(0, NGRAM_RECURRENT_DRAFT_TOKENS_MAX))

    off = GgufBackend(str(tmp_path / "m.gguf"))
    with patch("localm.model_manager.gguf.gguf_recurrent_state_bytes",
               return_value=per_copy), \
         patch.object(GgufBackend, "_gguf_parsed_tensor_entries", return_value=None):
        assert off._recurrent_state_vram_bytes() == per_copy


# --------------------------------------------------------------------------- #
#  Usage figures                                                              #
# --------------------------------------------------------------------------- #

def _ngram_backend():
    b = GgufBackend("m.gguf", spec_source="ngram")
    b._loaded = True
    return b


def _report(**kw):
    base = {"source": "ngram", "status": "ok", "active": False, "call_status": "",
            "skipped": "", "drafted": 0, "accepted": 0, "steps": 0,
            "paused_steps": 0, "draft_max": 8}
    base.update(kw)
    return base


@pytest.mark.parametrize("report,state,reason", [
    (_report(active=True, drafted=10, accepted=7, steps=4), "on", None),
    (_report(), "idle", None),
    (_report(skipped="image"), "off", "image"),
    (_report(status="rewind-unsupported"), "unavailable", "rewind-unsupported"),
    (_report(status="rewind-unsupported", active=True, drafted=4, steps=1), "stopped",
     "rewind-unsupported"),
    (_report(call_status="draft-out-of-step", drafted=2, steps=1), "stopped",
     "draft-out-of-step"),
    (_report(paused_steps=9, steps=3, drafted=3), "paused", "slower-than-plain"),
])
def test_ngram_usage_states(report, state, reason):
    b = _ngram_backend()
    b._record_mtp({"finish_reason": "stop", "speculation": report})
    usage = b.last_speculation_usage
    assert (usage["source"], usage["state"], usage["reason"]) == ("ngram", state, reason)
    assert usage["drafted"] == report["drafted"]
    assert b.last_mtp_usage is None


def test_a_reply_without_figures_reports_idle_and_a_reset_clears_them():
    b = _ngram_backend()
    b._record_mtp({"speculation": _report(active=True, drafted=4, accepted=4, steps=1)})
    b._reset_mtp_call()
    assert b.last_speculation_usage["state"] == "idle"
    b._record_mtp({"speculation": "garbage"})
    assert b.last_speculation_usage["drafted"] == 0


def test_mtp_usage_is_unchanged_and_mirrored_with_its_source():
    b = GgufBackend("m.gguf", mtp_enabled=True)
    b._loaded = True
    b._supports_mtp = True
    b._record_mtp({"mtp_status": "ok:qwen35", "mtp_active": True, "mtp_drafted": 6,
                   "mtp_accepted": 5, "mtp_steps": 3,
                   "speculation": _report(source="mtp")})
    assert b.last_mtp_usage == {"state": "on", "drafted": 6, "accepted": 5,
                                "paused_steps": 0, "reason": None}
    assert b.last_speculation_usage == {"source": "mtp", **b.last_mtp_usage}


def test_no_usage_when_the_source_is_off_or_nothing_is_loaded():
    assert GgufBackend("m.gguf").last_speculation_usage is None
    b = GgufBackend("m.gguf", spec_source="ngram")
    assert b.last_speculation_usage is None


def test_the_server_validates_the_figures_before_emitting_them():
    from localm.inference.http_server import _speculation_usage
    good = SimpleNamespace(speculation_usage=lambda: {
        "source": "ngram", "state": "on", "drafted": 3, "accepted": 2,
        "paused_steps": 0, "reason": None})
    assert _speculation_usage(good).model_dump() == {
        "source": "ngram", "state": "on", "drafted": 3, "accepted": 2,
        "paused_steps": 0, "reason": None}
    bad = SimpleNamespace(speculation_usage=lambda: {"state": "on"})
    assert _speculation_usage(bad) is None
    assert _speculation_usage(SimpleNamespace()) is None


def test_the_worker_hands_the_report_to_the_done_envelope():
    import inspect

    from localm.inference.backends.llamacpp import _runner
    from localm.inference.backends.llamacpp._worker import GgufWorker
    w = GgufWorker.__new__(GgufWorker)
    w._llm = None
    assert w.spec_report is None
    w._llm = SimpleNamespace(speculation_report=lambda: {"source": "ngram"})
    assert w.spec_report == {"source": "ngram"}
    assert '"speculation": worker.spec_report' in inspect.getsource(_runner)


# --------------------------------------------------------------------------- #
#  bench-spec                                                                 #
# --------------------------------------------------------------------------- #

def _spec_arm(rates_off, rates_on, *, usable=True, status=None, counts=(20, 15),
              greedy_on="g", seen=None):
    def _arm(model_path, display, source, gen_tokens, ctx, gpu_layers, draft_tokens=None,
             draft_model=None):
        if seen is not None:
            seen.append((source, draft_tokens))
        if source == "off":
            per = [(rates_off[0], 0, 0)] * 4
            return (rates_off, True, None, None, (0, 0), ["a", "b", "c", "d"], "g", per)
        if not usable:
            return ([], False, status, None, (0, 0), [], "", [])
        per = [(rates_on[0], 0, 0)] * 3 + [(rates_on[0] * 2, counts[0], counts[1])]
        return (rates_on, True, None, None, counts, ["a", "b", "c", "d"], greedy_on, per)
    return _arm


@pytest.mark.parametrize("off,on,phrase", [
    ([50.0], [70.0], "N-gram drafting is 1.40x faster"),
    ([100.0], [60.0], "N-gram drafting is slower"),
    ([100.0], [101.0], "No meaningful difference"),
])
def test_bench_spec_reports_the_measured_verdict(cli_runner, off, on, phrase):
    from localm.cli import models as models_mod
    seen = []
    with patch.object(models_mod, "get_operator_model_info",
                      return_value=("model.gguf", None)), \
         patch.object(models_mod, "_spec_probe_arm", _spec_arm(off, on, seen=seen)):
        res = cli_runner.invoke(models_mod.main,
                                ["bench-spec", "model.gguf", "--rounds", "1",
                                 "--draft-tokens", "6"])
    assert res.exit_code == 0, res.output
    assert phrase in res.output
    assert "Drafts accepted: 15 of 20 (75%)" in res.output
    assert "Output identical to N-gram drafting off: 4 of 4 replies" in res.output
    assert "Greedy replies matched: 1 of 1." in res.output
    assert seen == [("off", None), ("ngram", 6)]
    rewrite = next(line for line in res.output.splitlines() if "rewrite" in line)
    assert "%.2fx" % (2 * on[0] / off[0]) in rewrite and "75%" in rewrite


def test_bench_spec_reports_a_greedy_mismatch(cli_runner):
    from localm.cli import models as models_mod
    with patch.object(models_mod, "get_operator_model_info",
                      return_value=("model.gguf", None)), \
         patch.object(models_mod, "_spec_probe_arm",
                      _spec_arm([50.0], [70.0], greedy_on="different")):
        res = cli_runner.invoke(models_mod.main, ["bench-spec", "model.gguf", "--rounds", "1"])
    assert "Greedy replies matched: 0 of 1." in res.output


def test_bench_spec_stops_when_the_source_cannot_run(cli_runner):
    from localm.cli import models as models_mod
    with patch.object(models_mod, "get_operator_model_info",
                      return_value=("model.gguf", None)), \
         patch.object(models_mod, "_spec_probe_arm",
                      _spec_arm([50.0], [70.0], usable=False, status="rewind-unsupported")):
        res = cli_runner.invoke(models_mod.main, ["bench-spec", "model.gguf"])
    assert res.exit_code == 0, res.output
    assert "cannot run" in res.output and "rewind-unsupported" in res.output
    assert "decode tok/s" not in res.output
    assert "faster" not in res.output


def test_bench_spec_refuses_more_than_three_mtp_draft_tokens(cli_runner):
    from localm.cli import models as models_mod
    res = cli_runner.invoke(models_mod.main,
                            ["bench-spec", "model.gguf", "--source", "mtp", "--draft-tokens", "4"])
    assert res.exit_code != 0
    assert "1-3" in res.output


def test_bench_spec_never_writes_the_setting(cli_runner):
    from localm.cli import models as models_mod
    from localm.config import load_config, save_config
    cfg = load_config()
    cfg["spec_source"] = None
    save_config(cfg)
    with patch.object(models_mod, "get_operator_model_info",
                      return_value=("model.gguf", None)), \
         patch.object(models_mod, "_spec_probe_arm", _spec_arm([50.0], [70.0])):
        res = cli_runner.invoke(models_mod.main, ["bench-spec", "model.gguf", "--rounds", "1"])
    assert res.exit_code == 0, res.output
    assert load_config()["spec_source"] is None


@pytest.mark.parametrize("source,expected", [("ngram", "NgramSource"), ("mtp", "MtpSource"),
                                             ("off", "MtpSource")])
def test_the_model_drives_the_source_it_was_configured_with(source, expected):
    llm = make_bare_llama()
    llm._spec_source_name = source
    llm._spec_draft_max = 5 if source == "ngram" else 0
    llm._source = None
    picked = llm._draft_source()
    assert type(picked).__name__ == expected
    assert llm._draft_source() is picked
    assert llm.speculation_report()["source"] == source
    if source == "ngram":
        assert picked.draft_max == 5

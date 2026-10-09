# SPDX-License-Identifier: AGPL-3.0-or-later
"""The use_mmap setting: config default, schema field, CLI, plumbing into the
GGUF backend, and the note a load reports about it."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from localm import settings_schema as ss
from localm.config import DEFAULT_CONFIG
from localm.inference.backends.gguf import GgufBackend
from localm.inference.mmap_setting import (
    MMAP_FROM_DISK_NOTE, USE_MMAP_MODES, coerce_use_mmap, describe_mmap,
    resolve_use_mmap)


# --------------------------------------------------------------------------- #
#  Config default and schema field                                            #
# --------------------------------------------------------------------------- #

def test_the_default_is_auto():
    assert DEFAULT_CONFIG["use_mmap"] == "auto"


def test_the_schema_field_offers_exactly_the_modes():
    field = next(f for f in ss.CORE_FIELDS if f.key == "use_mmap")
    assert field.widget == ss.Widget.SELECT
    assert field.options == list(USE_MMAP_MODES)
    assert field.applies == ss.Applies.NEXT_LOAD
    assert field.group == "Engine"


@pytest.mark.parametrize("mode", USE_MMAP_MODES)
def test_every_mode_validates_and_round_trips(mode):
    assert ss.validate_update({"use_mmap": mode}) == {"use_mmap": mode}


@pytest.mark.parametrize("bad", ["maybe", "true", "", "ON ", True, 1, ["on"]])
def test_a_value_that_is_not_a_mode_is_rejected(bad):
    with pytest.raises(ValueError, match="use_mmap"):
        ss.validate_update({"use_mmap": bad})


def test_null_is_refused_rather_than_clearing_the_setting():
    with pytest.raises(ValueError, match="a value is required"):
        ss.validate_update({"use_mmap": None})


def test_it_is_in_the_load_fingerprint_keys():
    from localm.inference.routing_latch import LOAD_CONFIG_KEYS
    assert "use_mmap" in LOAD_CONFIG_KEYS


def test_it_is_in_the_bug_report_config_keys():
    from localm.bugreport import _SAFE_CONFIG_KEYS
    assert "use_mmap" in _SAFE_CONFIG_KEYS


# --------------------------------------------------------------------------- #
#  `localm config`                                                            #
# --------------------------------------------------------------------------- #

class TestConfigCli:
    def _stored(self):
        from localm.config import load_config
        return load_config()["use_mmap"]

    @pytest.mark.parametrize("mode", ["on", "off", "auto"])
    def test_a_mode_persists(self, cli_runner, mode):
        from localm.cli import main
        r = cli_runner.invoke(main, ["config", "use_mmap", mode])
        assert r.exit_code == 0, r.output
        assert self._stored() == mode
        assert f"use_mmap = {mode}" in r.output

    def test_a_value_that_is_not_a_mode_is_refused_and_leaves_the_setting(self, cli_runner):
        from localm.cli import main
        assert cli_runner.invoke(main, ["config", "use_mmap", "on"]).exit_code == 0
        r = cli_runner.invoke(main, ["config", "use_mmap", "true"])
        assert r.exit_code != 0
        assert "use_mmap" in r.output
        assert "auto" in r.output and "off" in r.output
        assert self._stored() == "on"


# --------------------------------------------------------------------------- #
#  Mode parsing                                                               #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("raw,expected", [
    ("auto", "auto"), ("on", "on"), ("off", "off"), (" ON ", "on"), ("Off", "off"),
    ("maybe", None), ("", None), (None, None), (True, None), (1, None),
])
def test_coerce_use_mmap(raw, expected):
    assert coerce_use_mmap(raw) == expected


@pytest.mark.parametrize("cfg,expected", [
    ({}, "auto"), ({"use_mmap": None}, "auto"), ({"use_mmap": ""}, "auto"),
    ({"use_mmap": "on"}, "on"), ({"use_mmap": "OFF"}, "off"),
])
def test_resolve_use_mmap_reads_the_config(cfg, expected):
    assert resolve_use_mmap(cfg) == expected


@pytest.mark.parametrize("raw", ["maybe", True, 7])
def test_a_hand_edited_value_is_read_as_auto_and_warned_about(raw):
    with patch("localm.debuglog.logger") as logger:
        assert resolve_use_mmap({"use_mmap": raw}) == "auto"
    assert logger.warning.call_count == 1
    assert "use_mmap" in logger.warning.call_args.args[0]


# --------------------------------------------------------------------------- #
#  Config -> create_backend -> GgufBackend                                    #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("stored,expected", [
    ("auto", "auto"), ("on", "on"), ("off", "off"), ("garbage", "auto"),
])
def test_create_backend_passes_the_setting(stored, expected):
    from localm.inference import engine as engine_mod
    captured = {}

    class _FakeBackend:
        def __init__(self, *a, **kw):
            captured.update(kw)

    cfg = dict(DEFAULT_CONFIG)
    cfg["use_mmap"] = stored
    with patch.object(engine_mod, "load_config", return_value=cfg), \
         patch("localm.inference.backends.gguf.GgufBackend", _FakeBackend):
        engine_mod.create_backend("model.gguf")
    assert captured["use_mmap"] == expected


def test_a_backend_defaults_to_auto_with_no_load_state():
    b = GgufBackend("m.gguf")
    assert b.use_mmap == "auto"
    assert b.effective_use_mmap is None
    assert b.mmap_forced_by_ram is False


@pytest.mark.parametrize("mode", USE_MMAP_MODES)
def test_a_backend_keeps_the_mode_it_was_given(mode):
    assert GgufBackend("m.gguf", use_mmap=mode).use_mmap == mode


def test_a_backend_refuses_a_value_that_is_not_a_mode():
    with pytest.raises(ValueError, match="use_mmap"):
        GgufBackend("m.gguf", use_mmap="maybe")


# --------------------------------------------------------------------------- #
#  The note                                                                   #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("setting,effective,forced,expected", [
    ("auto", True, True, "mmap on: " + MMAP_FROM_DISK_NOTE),
    ("auto", True, False, None),
    ("auto", False, False, None),
    ("auto", None, False, None),
    ("on", True, False, "mmap on"),
    ("on", False, False, "mmap off, although use_mmap is on"),
    ("off", False, False, "mmap off"),
    ("off", True, False, "mmap on, although use_mmap is off"),
    ("on", None, False, None),
    ("off", None, False, None),
])
def test_describe_mmap(setting, effective, forced, expected):
    assert describe_mmap(setting, effective, forced) == expected


def test_the_hint_names_what_happened_and_what_it_costs():
    assert "may not fit in available RAM" in MMAP_FROM_DISK_NOTE
    assert "disk-backed memory" in MMAP_FROM_DISK_NOTE
    assert "first tokens slower" in MMAP_FROM_DISK_NOTE


def _backend(setting="auto", effective=None, forced=False):
    b = GgufBackend("m.gguf", use_mmap=setting)
    b.effective_use_mmap = effective
    b.mmap_forced_by_ram = forced
    return b


def test_the_load_output_prints_the_hint_when_ram_forced_mmap():
    b = _backend("auto", True, True)
    with patch("localm.inference.backends.gguf.console") as console:
        b._print_mmap_note()
    printed = " ".join(str(c.args[0]) for c in console.print.call_args_list)
    assert "disk-backed memory" in printed and "first tokens slower" in printed


@pytest.mark.parametrize("setting,effective", [
    ("auto", None), ("auto", False), ("auto", True),
])
def test_the_load_output_stays_silent_when_there_is_nothing_to_say(setting, effective):
    b = _backend(setting, effective, False)
    with patch("localm.inference.backends.gguf.console") as console:
        b._print_mmap_note()
    console.print.assert_not_called()


@pytest.mark.parametrize("setting", USE_MMAP_MODES)
def test_a_real_load_reports_mmap_once_just_before_model_loaded(tmp_path, setting):
    """GgufBackend._load_native run for real (isolated worker and GPU probe
    stubbed), with no MoE placement requested: the note is reported exactly
    once, as the last step before the "Model loaded" line."""
    model = tmp_path / "model.gguf"
    model.write_bytes(b"\0" * 4096)
    b = GgufBackend(str(model), n_gpu_layers=99, n_ctx=512, use_mmap=setting)
    order = []

    def _printed(*args, **kwargs):
        if args and "Model loaded" in str(args[0]):
            order.append("loaded")

    with patch("localm.discover.list_gpus", return_value=([], "ok")), \
         patch("localm.inference.backends.llamacpp._runner.ModelRunner."
               "spawn_and_load",
               return_value={"n_layers": 8, "kv_bytes_per_token": 0,
                             "supports_images": False}), \
         patch.object(GgufBackend, "_print_mmap_note",
                      lambda self: order.append("note")), \
         patch("localm.inference.backends.gguf.console.print", _printed):
        b._load_native()
    assert order == ["note", "loaded"]


def _engine(backend):
    from localm.inference.engine import Engine
    e = Engine.__new__(Engine)
    e._backend = backend
    return e


def test_engine_mmap_state_carries_setting_effect_and_hint():
    state = _engine(_backend("auto", True, True)).mmap_state
    assert state == {"use_mmap": "auto", "mmap": True, "mmap_from_disk": True,
                     "mmap_note": "mmap on: " + MMAP_FROM_DISK_NOTE}


def test_engine_mmap_state_for_a_default_mapped_load_has_no_note():
    state = _engine(_backend("auto", True, False)).mmap_state
    assert state == {"use_mmap": "auto", "mmap": True, "mmap_from_disk": False}


def test_engine_mmap_state_is_none_until_a_load_reports_it():
    assert _engine(_backend("on", None, False)).mmap_state is None
    assert _engine(SimpleNamespace()).mmap_state is None


def test_mmap_from_disk_needs_mmap_actually_on():
    assert _engine(_backend("auto", False, True)).mmap_state["mmap_from_disk"] is False


@pytest.mark.parametrize("setting", ["on", "off"])
def test_mmap_from_disk_is_an_auto_decision_only(setting):
    assert _engine(_backend(setting, True, True)).mmap_state["mmap_from_disk"] is False


def test_the_load_payload_carries_the_mmap_fields():
    from localm.inference import http_server as hs
    e = SimpleNamespace(gpu_placement={"gpu_layers_offloaded": 4,
                                       "gpu_layers_total": 4, "degraded": False},
                        mmap_state={"use_mmap": "auto", "mmap": True,
                                    "mmap_from_disk": True, "mmap_note": "n"})
    fields = hs._gpu_placement_fields(e)
    assert fields["mmap_from_disk"] is True
    assert fields["gpu_layers_total"] == 4


def test_the_load_payload_is_unchanged_without_mmap_state():
    from localm.inference import http_server as hs
    placement = {"gpu_layers_offloaded": 4, "gpu_layers_total": 4, "degraded": False}
    assert hs._gpu_placement_fields(
        SimpleNamespace(gpu_placement=placement, mmap_state=None)) == placement
    assert hs._gpu_placement_fields(SimpleNamespace(gpu_placement=None)) == {}


def test_the_load_log_line_includes_the_note_only_when_there_is_one():
    from localm.inference import http_server as hs
    base = dict(gpu_placement=None, gpu_sizing=None)
    with_note = SimpleNamespace(**base, mmap_state={"mmap_note": "mmap on: x"})
    without = SimpleNamespace(**base, mmap_state={"mmap": True})
    assert "mmap on: x" in hs._describe_load_placement("m", with_note)
    assert "mmap" not in hs._describe_load_placement("m", without)
    assert "mmap" not in hs._describe_load_placement("m", SimpleNamespace(**base))

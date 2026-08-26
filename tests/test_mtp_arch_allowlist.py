# SPDX-License-Identifier: AGPL-3.0-or-later
"""scripts/check_mtp_arch_allowlist.py: the gate that keeps MTP detection a
CAPABILITY test rather than a METADATA test.

WHY A STATIC GATE EXISTS ALONGSIDE tests/test_mtp_support.py: those tests pin
the detector's answer for the architectures somebody already thought of. They
cannot notice the two ways this defect comes back, because neither changes the
detector's answer for a known input:

  - the llama.cpp runtime pin moves, and which architectures build an MTP graph
    changes with it (glm4moe has no MTP graph at b10375 and does on upstream's
    later default branch), so a frozen allowlist silently describes a runtime
    nobody ships any more;
  - a second call site asks for an MTP context without going through the
    detector at all.

WHAT THESE TESTS PIN, in order of importance:

1. It FIRES. One synthetic case per failure the gate claims to detect. A gate
   that has never been red proves nothing.
2. It does NOT fire on the correct shape, so it cannot be satisfied by
   deleting the feature.
3. The upstream re-derivation parses what upstream actually looks like, driven
   offline from captured snippets, so --refresh is not a mode nobody has run.
4. The real tree passes --gate. That is the recurrence guard, and it is the
   only assertion here a future change can trip.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "check_mtp_arch_allowlist", REPO / "scripts" / "check_mtp_arch_allowlist.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


GOOD_API = '''
MTP_ARCH_SOURCE_TAG = "bTEST"

MTP_GRAPH_ARCHITECTURES = frozenset({"qwen35", "deepseek2"})


def llama_model_mtp_support(model):
    if arch not in MTP_GRAPH_ARCHITECTURES:
        return False, "no-mtp-graph"
    return True, "ok"
'''

GOOD_SETUP = '_PINNED_TAG = "bTEST"\n'

GATED_CALLER = '''
from ._api import llama_model_mtp_support
from ._structs import LLAMA_CONTEXT_TYPE_MTP


def build():
    ok, reason = llama_model_mtp_support(model)
    if ok:
        cp.ctx_type = LLAMA_CONTEXT_TYPE_MTP
'''


def _bench(tmp_path: Path, mod, api_src=GOOD_API, setup_src=GOOD_SETUP,
           caller_src=GATED_CALLER):
    """Point the gate at a throwaway tree and return its --gate exit code."""
    pkg = tmp_path / "localm"
    pkg.mkdir(parents=True, exist_ok=True)
    api = pkg / "_api.py"
    api.write_text(api_src, encoding="utf-8")
    setup = pkg / "setup_llama.py"
    setup.write_text(setup_src, encoding="utf-8")
    (pkg / "caller.py").write_text(caller_src, encoding="utf-8")
    mod.API_PATH = api
    mod.SETUP_PATH = setup
    mod.REPO = tmp_path
    return mod.main(["--gate"])


def test_gate_passes_on_a_correctly_gated_tree(tmp_path):
    """The false-positive direction: the gate must not fire on the right shape."""
    assert _bench(tmp_path, _load()) == 0


def test_gate_fires_when_the_runtime_pin_moves_without_a_rederivation(tmp_path):
    """A pin bump is the way the allowlist rots, and nothing else would notice."""
    assert _bench(tmp_path, _load(), setup_src='_PINNED_TAG = "bNEWER"\n') == 1


def test_gate_fires_when_the_detector_stops_reading_the_allowlist(tmp_path):
    """Dropping that one condition restores the original metadata-only defect."""
    stripped = GOOD_API.replace(
        '    if arch not in MTP_GRAPH_ARCHITECTURES:\n'
        '        return False, "no-mtp-graph"\n', "")
    assert "MTP_GRAPH_ARCHITECTURES" not in stripped.split("def ")[1]
    assert _bench(tmp_path, _load(), api_src=stripped) == 1


def test_gate_fires_on_an_mtp_context_requested_without_the_detector(tmp_path):
    """A second call site added later is the other way the defect returns."""
    ungated = '''
from ._structs import LLAMA_CONTEXT_TYPE_MTP


def build_somewhere_else():
    cp.ctx_type = LLAMA_CONTEXT_TYPE_MTP
'''
    assert _bench(tmp_path, _load(), caller_src=ungated) == 1


def test_gate_fires_on_an_empty_allowlist(tmp_path):
    """Emptying the set silently disables MTP everywhere - that needs saying out loud."""
    emptied = GOOD_API.replace('frozenset({"qwen35", "deepseek2"})', "frozenset()")
    assert _bench(tmp_path, _load(), api_src=emptied) == 1


def test_gate_fires_on_an_architecture_string_that_is_not_one(tmp_path):
    """general.architecture values are lower-case; a C++ enum name here is a mix-up."""
    wrong = GOOD_API.replace('"qwen35", "deepseek2"', '"LLM_ARCH_QWEN35", "deepseek2"')
    assert _bench(tmp_path, _load(), api_src=wrong) == 1


def test_refresh_parses_the_upstream_layout_it_claims_to(monkeypatch):
    """Drive the re-derivation offline over the three upstream shapes it joins.

    The snippets are trimmed from llama.cpp at b10375. Padding keeps each body
    over the length floor _fetch uses to reject an error page.
    """
    mod = _load()
    pad = "// pad\n" * 200
    files = {
        "src/models/models.h": pad + '''
struct llama_model_glm4_moe : public llama_model_base {
    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llama_model_qwen35 : public llama_model_base {
    struct graph_mtp : public llm_graph_context {
        graph_mtp(const llama_model & model, const llm_graph_params & params);
    };
};
''',
        "src/llama-model.cpp": pad + '''
        case LLM_ARCH_GLM4_MOE:
            return new llama_model_glm4_moe(params);
        case LLM_ARCH_QWEN35:
            return new llama_model_qwen35(params);
''',
        "src/llama-arch.cpp": pad + '''
    { LLM_ARCH_GLM4_MOE, "glm4moe" },
    { LLM_ARCH_QWEN35,   "qwen35"  },
''',
    }
    monkeypatch.setattr(mod, "_fetch", lambda tag, path: files[path])

    derived = mod.refresh("bTEST")

    # The whole point: the class that declares graph_mtp is derived, and the one
    # that only carries the metadata key is not.
    assert derived == {"qwen35"}


def test_refresh_refuses_an_upstream_layout_it_no_longer_understands(monkeypatch):
    """An empty parse must raise, not quietly derive an empty allowlist.

    A zero from a parse over an unrecognised file is byte-identical to a zero
    from a file that genuinely declares no MTP architectures.
    """
    mod = _load()
    monkeypatch.setattr(mod, "_fetch", lambda tag, path: "// nothing recognisable\n" * 200)
    try:
        mod.refresh("bTEST")
    except RuntimeError as exc:
        assert "graph_mtp" in str(exc)
    else:
        raise AssertionError("refresh accepted an unparseable upstream layout")


def _bench_api(tmp_path, mod, feeds, default_on):
    """Drive the draft-head-API arm with both inputs forced."""
    mod._runtime_feeds_the_draft_head = lambda: (feeds, "forced")
    mod._mtp_default_enabled = lambda: default_on
    return _bench(tmp_path, mod)


def test_gate_fires_when_the_runtime_gains_the_draft_head_api(tmp_path):
    """The moment a runtime can feed the draft head, the real implementation
    becomes possible and the off-by-default decision has to be revisited. Nothing
    else in the tree would notice that the world changed."""
    assert _bench_api(tmp_path, _load(), feeds=True, default_on=False) == 1


def test_gate_fires_when_mtp_defaults_on_without_the_draft_head_api(tmp_path):
    """The state this default exists to prevent: speculation on by default while
    the head is starved, which costs more per token than it saves."""
    assert _bench_api(tmp_path, _load(), feeds=False, default_on=True) == 1


def test_gate_is_quiet_in_the_state_that_actually_ships(tmp_path):
    """API absent, default off. A gate that fired here would be turned off."""
    assert _bench_api(tmp_path, _load(), feeds=False, default_on=False) == 0


def test_gate_is_quiet_when_the_api_exists_and_mtp_is_on(tmp_path):
    """The other consistent pair, so the check is about AGREEMENT rather than
    about either value on its own."""
    assert _bench_api(tmp_path, _load(), feeds=True, default_on=True) == 0


def test_an_unloadable_runtime_is_not_read_as_an_absent_api(tmp_path):
    """"Could not look" must not collapse into "looked and found nothing" - that
    is the difference between a real absence and a broken probe."""
    assert _bench_api(tmp_path, _load(), feeds=None, default_on=True) == 0
    assert _bench_api(tmp_path, _load(), feeds=None, default_on=False) == 0


def test_the_real_tree_passes_the_gate():
    """The recurrence guard itself, against the shipped source."""
    assert _load().main(["--gate"]) == 0

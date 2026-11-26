# SPDX-License-Identifier: AGPL-3.0-or-later
"""scripts/check_pretokenizer_redos.py --gate: the exit code follows the verdicts.

The sweep itself needs MSVC and the network and is not run here; this pins the
offline decisions the CI pre-flight and the pin bump rely on: a flagged pattern
or an uncovered name turns into a non-zero exit under --gate and into exit 0
without it, and every tokenizer_pre string of llama-vocab.cpp is compared with
the guard's table (``name_problems``), read from made-up sources here.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "check_pretokenizer_redos.py"


@pytest.fixture(scope="module")
def redos():
    spec = importlib.util.spec_from_file_location("check_pretokenizer_redos", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_gate_fails_on_a_flagged_probe_and_only_then(redos):
    flagged = [("pattern", "bait", "QUADRATIC-ish (x4 -> x16)")]
    assert redos._exit_code([], gate=True) == 0
    assert redos._exit_code(flagged, gate=True) == 1
    assert redos._exit_code(flagged, gate=False) == 0, "the default stays a report"
    assert redos._exit_code([], gate=False) == 0


def test_main_wires_the_gate_flag(redos):
    """The flag exists and main() returns _exit_code's verdict rather than a
    literal 0: read from the source, since main() itself needs MSVC to run."""
    src = _SCRIPT.read_text(encoding="utf-8")
    assert 'ap.add_argument("--gate"' in src
    assert "return _exit_code(names + concerning, args.gate)" in src
    assert "\n    return 0\n" not in src.split("def main()", 1)[1].split("def _exit_code", 1)[0]


GPT4O, SAFE, HANDLED = r"[^\\p{L}]?(x)*(y)+|\\s*[\\r\\n]+", r"\\s?\\p{L}+", r"\\p{Han}+"

_UNICODE = """
static std::vector<size_t> unicode_regex_split_custom(const std::string & text) {
    if (regex_expr == "\\\\p{Han}+") { return a(); }
    return {};
}
"""

_VOCAB = """
struct llm_tokenizer_bpe : llm_tokenizer {
    llm_tokenizer_bpe(const llama_vocab & vocab) {
        switch (vocab.get_pre_type()) {
            case LLAMA_VOCAB_PRE_TYPE_GPT4O:
            case LLAMA_VOCAB_PRE_TYPE_MINIMAX_M2:
                regex_exprs = {
                    // original regex from tokenizer.json: "{(?i)}"
                    "@@GPT4O@@",
                };
                break;
            case LLAMA_VOCAB_PRE_TYPE_FALCON:
                regex_exprs = { "@@SAFE@@" };
                byte_encode = false;
                break;
            case LLAMA_VOCAB_PRE_TYPE_KIMI:
                regex_exprs = { "@@HANDLED@@" };
                break;
            default:
                regex_exprs = { "@@SAFE@@", "[0-9]" };
                break;
        }
    }
};
struct llm_tokenizer_bpe_session {};
void load() {
            if (tokenizer_pre.empty()) {
                pre_type = LLAMA_VOCAB_PRE_TYPE_DEFAULT;
            } else if (tokenizer_pre == "default") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_DEFAULT;
            } else if (
                    tokenizer_pre == "gpt-4o" ||
                    tokenizer_pre == "llama4") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_GPT4O;
                clean_spaces = false;
            } else if (tokenizer_pre == "minimax-m2") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_MINIMAX_M2;
            } else if (tokenizer_pre == "falcon") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_FALCON;
            } else if (tokenizer_pre == "kimi") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_KIMI;
            @@MORE@@
            } else {
                throw std::runtime_error(format("unknown pre-tokenizer type: '%s'", tokenizer_pre.c_str()));
            }
}
"""

_GUARDED = {"gpt-4o", "llama4", "minimax-m2"}


def _vocab(more="", gpt4o=GPT4O, safe=SAFE):
    return (_VOCAB.replace("@@GPT4O@@", gpt4o).replace("@@SAFE@@", safe)
            .replace("@@HANDLED@@", HANDLED).replace("@@MORE@@", more))


def _named(name, enum):
    return f'}} else if (tokenizer_pre == "{name}") {{ pre_type = LLAMA_VOCAB_PRE_TYPE_{enum};'


@pytest.fixture
def problems(redos):
    lists = redos.pattern_lists_by_name(_vocab())
    safe = {redos.list_digest(lists[n]) for n in ("falcon", "default")}
    measured = {name: redos.list_digest(lists[name]) for name in _GUARDED}

    def run(vocab, guarded=_GUARDED):
        return redos.name_problems(vocab, _UNICODE, {n: measured[n] for n in guarded}, safe)
    return run


def test_a_source_whose_names_are_all_covered_has_no_problem(redos, problems):
    assert redos.extract_pre_type_enums(_vocab()) == {
        "default": "DEFAULT", "gpt-4o": "GPT4O", "llama4": "GPT4O", "minimax-m2": "MINIMAX_M2",
        "falcon": "FALCON", "kimi": "KIMI"}
    assert problems(_vocab()) == []


def test_a_new_name_for_an_unsafe_pattern_fails(problems):
    found = problems(_vocab(_named("brand-new", "GPT4O")))
    assert len(found) == 1 and "'brand-new'" in found[0] and "gpt-4o" in found[0]


def test_a_name_added_beside_an_unsafe_one_fails(problems):
    vocab = _vocab().replace('tokenizer_pre == "llama4")',
                             'tokenizer_pre == "llama4" ||\n tokenizer_pre == "brand-new")')
    assert ["'brand-new'" in p for p in problems(vocab)] == [True]


def test_a_new_enum_with_an_unsafe_pattern_list_fails(problems):
    vocab = _vocab(_named("brand-new", "BRAND_NEW")).replace(
        "case LLAMA_VOCAB_PRE_TYPE_MINIMAX_M2:",
        "case LLAMA_VOCAB_PRE_TYPE_MINIMAX_M2:\n            case LLAMA_VOCAB_PRE_TYPE_BRAND_NEW:")
    found = problems(vocab)
    assert len(found) == 1 and "'brand-new'" in found[0]


def test_a_name_taken_out_of_the_table_fails(problems):
    found = problems(_vocab(), _GUARDED - {"llama4"})
    assert len(found) == 1 and "'llama4'" in found[0]


def test_a_new_name_for_a_measured_or_a_handled_list_passes(problems):
    assert problems(_vocab(_named("brand-new", "FALCON"))) == []
    assert problems(_vocab(_named("brand-new", "KIMI"))) == []
    assert problems(_vocab(_named("brand-new", "NOT_IN_THE_SWITCH"))) == [], (
        "an enum with no case uses the default list")


def test_a_guarded_list_that_changed_fails(problems):
    found = problems(_vocab(gpt4o=GPT4O + "|x"))
    assert [p.split(" ", 1)[0] for p in found] == ["'gpt-4o'", "'llama4'", "'minimax-m2'"]
    assert all("no longer uses the pattern list" in p for p in found)


def test_a_guarded_name_moved_to_another_list_fails(problems):
    for target in ("FALCON", "KIMI"):
        moved = _vocab().replace(
            'tokenizer_pre == "llama4") {\n                pre_type = LLAMA_VOCAB_PRE_TYPE_GPT4O;',
            'tokenizer_pre == "x") {\n                pre_type = LLAMA_VOCAB_PRE_TYPE_GPT4O;'
            f'\n            }} else if (tokenizer_pre == "llama4") {{ pre_type = '
            f"LLAMA_VOCAB_PRE_TYPE_{target};")
        assert moved != _vocab()
        found = problems(moved)
        assert any(p.startswith("'llama4' no longer uses") for p in found), (target, found)


def test_every_guarded_name_has_the_digest_of_its_measured_list(redos):
    from localm.inference.pretokenizer_guard import UNSAFE_PRE_TYPES
    assert set(redos.GUARDED_LISTS) == set(UNSAFE_PRE_TYPES)
    assert all(len(d) == 16 and int(d, 16) >= 0 for d in redos.GUARDED_LISTS.values())
    assert not set(redos.GUARDED_LISTS.values()) & set(redos.MEASURED_SAFE_LISTS)


def test_a_guarded_name_without_a_digest_matches_no_list(redos, monkeypatch):
    monkeypatch.setattr(redos, "GUARDED_LISTS", {})
    lists = redos._guarded_lists()
    assert lists["gpt-4o"] == ""
    assert len(redos.name_problems(_vocab(), _UNICODE, {"gpt-4o": ""}, set())) >= 1


def test_a_changed_list_is_not_covered_by_its_old_measurement(problems):
    found = problems(_vocab(safe=SAFE + "x"))
    assert len(found) == 2 and all("nothing has measured" in p for p in found)


@pytest.mark.parametrize("broken", [
    lambda v: v.replace("tokenizer_pre.empty()", "pre_names.count(tokenizer_pre)"),
    lambda v: v.replace("llm_tokenizer_bpe(const llama_vocab", "llm_tokenizer_x(const x"),
    lambda v: v.replace("pre_type = LLAMA_VOCAB_PRE_TYPE_KIMI;", "pre_type = kimi();"),
    lambda v: v + "\nvoid other() { regex_exprs = { \"a\" }; }\n",
    lambda v: v.replace('tokenizer_pre == "falcon"', "is_falcon(tokenizer_pre)"),
    lambda v: v.replace('regex_exprs = { "@@"'.replace("@@", HANDLED) + " };", "handled = true;"),
], ids=["chain-gone", "constructor-gone", "name-without-enum", "list-elsewhere", "unnamed-case",
        "case-without-list"])
def test_a_source_of_another_shape_is_unreadable_not_clean(redos, problems, broken):
    with pytest.raises(redos.Unreadable):
        problems(broken(_vocab()))


def test_the_shipped_safe_lists_are_digests(redos):
    assert redos.MEASURED_SAFE_LISTS
    assert all(len(d) == 16 and int(d, 16) >= 0 for d in redos.MEASURED_SAFE_LISTS)


def test_the_names_are_compared_with_the_guard_table(redos):
    from localm.inference.pretokenizer_guard import UNSAFE_PRE_TYPES
    assert redos._guarded_lists() == {n: redos.GUARDED_LISTS[n] for n in UNSAFE_PRE_TYPES}

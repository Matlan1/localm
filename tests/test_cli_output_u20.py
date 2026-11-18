# SPDX-License-Identifier: AGPL-3.0-or-later
"""Command-line output: whole tables, quiet native logs, honest job and
interrupt results, readable benchmark and draft reasons."""

from __future__ import annotations

import io
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner
from rich.table import Table

from localm import debuglog, instances
from localm.cli import main
from localm.console import SafeConsole

_LONG = "D:/models/" + "very-long-folder-name/" * 12 + "model.gguf"


def test_a_piped_table_wider_than_the_piped_width_is_not_cut(monkeypatch):
    monkeypatch.delenv("COLUMNS", raising=False)
    buf = io.StringIO()
    table = Table("Name", "Path")
    table.add_row("n", _LONG)
    SafeConsole(file=buf).print(table)
    out = buf.getvalue()
    assert _LONG in out
    assert "\u2026" not in out


def test_ps_keeps_the_port_and_directory_at_80_columns(monkeypatch):
    monkeypatch.setattr(instances, "snapshot", lambda *a, **k: [
        {"instance_id": "abcd1234ef", "alive": True, "mode": "full",
         "scheme": "http", "host": "127.0.0.1", "port": 8642, "pid": 4321,
         "root_dir": "/proj/" + "x" * 90},
    ])
    monkeypatch.setenv("COLUMNS", "80")
    res = CliRunner().invoke(main, ["ps"])
    assert res.exit_code == 0
    assert "\u2026" not in res.output
    cells = [line.split("\u2502")[-2].strip() for line in res.output.splitlines()
             if line.startswith("\u2502")]
    assert "x" * 90 in "".join(cells)
    assert "8642" in "".join(res.output.split())


def test_list_keeps_a_long_path_on_a_narrow_terminal(monkeypatch, tmp_path):
    from localm.model_manager import registry
    name = "Qwen3.5-9B-The-Defiant-Fable-Uncnr-Heretic-NEO-MAX-MTP-Q6_K"
    f = tmp_path / ("sub" * 30) / (name + ".gguf")
    f.parent.mkdir()
    f.write_bytes(b"GGUF")
    monkeypatch.setattr(registry._mm, "load_registry",
                        lambda: {name: {"path": str(f), "model_type": "llm"}})
    monkeypatch.setenv("COLUMNS", "100")
    res = CliRunner().invoke(main, ["list"])
    assert res.exit_code == 0
    assert "\u2026" not in res.output
    flat = "".join(res.output.split()).replace("\u2502", "")
    assert name in flat


def test_rag_query_runs_the_query_with_native_output_paused(monkeypatch):
    from localm import rag as rag_pkg
    seen = {}

    class _Coll:
        rerank_degrade_reason = None

        def __init__(self, name):
            pass

        def exists(self):
            return True

        def query(self, *a, **k):
            seen["paused"] = debuglog.native_console_is_paused()
            return []

    monkeypatch.setattr(rag_pkg, "Collection", _Coll)
    from localm.rag import rerank as rr
    monkeypatch.setattr(rr, "rerank_plan", lambda enabled: SimpleNamespace(
        note=None, fn=None, candidates=0, min_score=None))
    res = CliRunner().invoke(main, ["rag", "query", "c", "hello", "--no-rerank"])
    assert res.exit_code == 0, res.output
    assert seen["paused"] is True


def test_speak_synthesizes_with_native_output_paused(monkeypatch, tmp_path):
    from localm.inference import speech
    seen = {}

    def _synth(*a, **k):
        seen["paused"] = debuglog.native_console_is_paused()
        return SimpleNamespace(wav=b"RIFF", seconds=1.0, seed=1)

    monkeypatch.setattr(speech, "resolve_speech_model", lambda *a, **k: object())
    monkeypatch.setattr(speech, "voice_reference", lambda *a, **k: b"")
    monkeypatch.setattr(speech, "synthesize", _synth)
    out = tmp_path / "x.wav"
    res = CliRunner().invoke(main, ["speak", "Hello", "-o", str(out)])
    assert res.exit_code == 0, res.output
    assert seen["paused"] is True


def test_job_add_with_a_path_says_to_register_it(tmp_path, monkeypatch):
    from localm.model_manager import registry
    monkeypatch.setattr(registry._mm, "load_registry",
                        lambda: {"other": {"path": "x"}})
    msg = registry.unregistered_model_error("D:/m/a.gguf", from_cli=True)
    assert "localm add" in msg
    assert "only a command-line run" not in msg
    assert "only a command-line run" in registry.unregistered_model_error("D:/m/a.gguf")


def test_draft_status_codes_read_as_sentences():
    from localm.cli.models import _draft_reason
    assert "vocabulary" in _draft_reason("draft-vocab-mismatch")
    assert "slower" in _draft_reason("draft-cannot-pay")
    assert _draft_reason("something-else") == "something-else"


def test_benchmark_times_each_prompt_without_the_prompt_cache(monkeypatch):
    from localm.cli import models as cli_models
    from localm.inference import engine as eng_mod
    calls = []

    class _Engine:
        def __init__(self, *a, **k):
            pass

        def load(self):
            pass

        def unload(self):
            pass

        def count_tokens(self, text):
            return len(text.split())

        def chat_stream(self, messages, **kw):
            calls.append(kw)
            yield "tok"

    monkeypatch.setattr(eng_mod, "Engine", _Engine)
    monkeypatch.setattr(cli_models, "get_operator_model_info",
                        lambda m: ("x.gguf", None))
    res = CliRunner().invoke(main, ["benchmark", "m", "-n", "4", "--prompts", "8,16"])
    assert res.exit_code == 0, res.output
    assert len(calls) == 3
    assert all(c.get("cache_prompt") is False for c in calls)


def test_one_slot_reason_names_speculative_drafting_and_diffusion():
    from localm.inference.backends.llamacpp import llama
    fn = llama.LlamaCpp._one_slot_reason
    base = dict(is_encoder_decoder=False, is_diffusion=False,
                _spec_source_name=llama.SPEC_OFF)
    assert fn(SimpleNamespace(**base)) == ""
    assert "speculative drafting is on" in fn(
        SimpleNamespace(**{**base, "_spec_source_name": "ngram"}))
    assert "diffusion" in fn(SimpleNamespace(**{**base, "is_diffusion": True}))


def test_the_asr_header_is_not_printed_even_when_split_across_tokens(capsys):
    from localm.cli.chat import _ThinkPrinter
    p = _ThinkPrinter()
    for tok in ("lang", "uage Eng", "lish<asr", "_text>The quick", " fox"):
        p.feed(tok)
    p.flush()
    assert capsys.readouterr().out == "The quick fox"


def test_a_reply_that_merely_starts_like_the_header_is_printed(capsys):
    from localm.cli.chat import _ThinkPrinter
    p = _ThinkPrinter()
    p.feed("language is fun")
    p.flush()
    assert capsys.readouterr().out == "language is fun"


def _run_repl(monkeypatch, streams, inputs):
    from localm.cli import chat as chat_mod
    seen = []
    engine = MagicMock()
    it = iter(streams)

    def _chat_stream(messages, **kw):
        seen.append([dict(m) for m in messages])
        return next(it)()

    engine.chat_stream.side_effect = _chat_stream
    engine.display_name = "m"
    engine.count_tokens.return_value = 1
    engine.count_messages_tokens.return_value = 3
    engine.context_capacity.return_value = 4096
    feed = iter(inputs)

    def _input(*a, **k):
        try:
            return next(feed)
        except StopIteration as e:
            raise EOFError() from e

    monkeypatch.setattr(chat_mod.console, "input", _input)
    chat_mod._interactive(engine, None, {})
    return seen


def test_ctrl_c_before_the_first_token_leaves_no_answer_in_the_history(monkeypatch):
    def _cut():
        raise KeyboardInterrupt
        yield

    def _ok():
        yield "fine"

    seen = _run_repl(monkeypatch, [_cut, _ok], ["Write an essay", "second"])
    sent = seen[1]
    assert all("(interrupted)" not in str(m["content"]) for m in sent)
    assert [m["role"] for m in sent] == ["user"]
    assert sent[0]["content"] == "second"


def test_a_reply_cut_by_ctrl_c_is_kept_with_a_marker(monkeypatch):
    def _cut():
        yield "half an ans"
        raise KeyboardInterrupt

    def _ok():
        yield "fine"

    seen = _run_repl(monkeypatch, [_cut, _ok], ["a question", "second"])
    reply = [m for m in seen[1] if m["role"] == "assistant"][0]["content"]
    assert reply.startswith("half an ans")
    assert "cut off" in reply


def test_help_shows_the_save_argument(monkeypatch, capsys):
    seen = _run_repl(monkeypatch, [], ["/help"])
    assert seen == []
    assert "/save [file]" in capsys.readouterr().out


def test_a_coder_job_without_a_server_says_to_start_one(monkeypatch):
    import requests
    from localm.plugins.builtin.jobs import runner
    from localm.plugins.coder import agent as agent_mod

    class _Agent:
        def __init__(self, *a, **k):
            pass

        def run_task(self, prompt):
            raise requests.exceptions.ConnectionError("refused")

    monkeypatch.setattr(agent_mod, "Agent", _Agent)
    monkeypatch.setattr(runner, "_coder_backend", lambda job: object())
    from localm.plugins.builtin.jobs.store import Job
    job = Job(name="c", task_kind="coder", prompt="x", schedule_kind="interval",
              schedule=60, cwd=".")
    with pytest.raises(RuntimeError, match="localm serve"):
        runner._run_coder(job)

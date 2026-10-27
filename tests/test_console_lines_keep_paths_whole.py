# SPDX-License-Identifier: AGPL-3.0-or-later
"""Console lines that carry a path or a progress bar stay readable at the
terminal width: the chat REPL's save failure names the OS reason and the plain
path, the saved-bug-report line keeps its path in one piece, and the hashing
bar fits on one line."""

from __future__ import annotations

import io
import re
from pathlib import Path

import pytest
from rich.console import Console

def _narrow(width=40):
    buf = io.StringIO()
    return Console(file=buf, width=width, force_terminal=False, color_system=None), buf


def test_repl_save_failure_names_the_os_reason_and_the_plain_path(tmp_path, capsys):
    from localm.cli.chat import _save_chat
    target = str(tmp_path / "no-such-folder" / "chat.json")

    _save_chat([{"role": "user", "content": "hi"}], target)

    out = " ".join(capsys.readouterr().out.split())
    assert f"Save failed: No such file or directory: {target}" in out
    assert "[Errno" not in out
    assert not Path(target).exists()


def test_repl_save_failure_for_a_non_os_error_keeps_its_message(tmp_path, capsys):
    from localm.cli.chat import _save_chat

    _save_chat([{"role": "user", "content": object()}], str(tmp_path / "chat.json"))

    out = capsys.readouterr().out
    assert "Save failed:" in out
    assert "not JSON serializable" in out


def _long_report(tmp_path):
    report = (tmp_path / "a-long-folder-name-for-wrapping" / "another-long-folder-name"
              / "bug-report-20261022-154950.md")
    report.parent.mkdir(parents=True)
    report.write_text("# report", encoding="utf-8")
    return report


def test_report_failure_prints_the_saved_path_in_one_piece(tmp_path, monkeypatch):
    import localm.bugreport as br
    from localm.bugreport import consent
    report = _long_report(tmp_path)
    console, buf = _narrow()
    monkeypatch.setattr(consent, "console", console)
    monkeypatch.setattr(br, "_diagnostics_allowed", lambda: True)
    monkeypatch.setattr(br, "save_report", lambda text: report)
    monkeypatch.setattr(consent, "build_report", lambda *a, **k: "text")
    monkeypatch.setattr(consent, "offer_to_send", lambda *a, **k: None)

    consent.report_failure(summary="boom", interactive=False, assume_yes=True)

    assert str(report) in buf.getvalue()


def test_bug_report_command_prints_the_saved_path_in_one_piece(cli_runner, tmp_path,
                                                               monkeypatch):
    from localm import bugreport
    from localm.cli import maintenance
    report = _long_report(tmp_path)
    console, buf = _narrow()
    monkeypatch.setattr(maintenance, "console", console)
    monkeypatch.setattr(bugreport, "stdin_is_interactive", lambda: False)
    monkeypatch.setattr(bugreport, "live_server_hang_trace", lambda *a, **k: "")
    monkeypatch.setattr(bugreport, "save_user_report", lambda *a, **k: report)
    monkeypatch.setattr(bugreport, "offer_to_send", lambda *a, **k: None)

    result = cli_runner.invoke(maintenance.bug_report_cmd, ["-w", "it crashed"])

    assert result.exit_code == 0, result.output
    assert str(report) in buf.getvalue()


@pytest.mark.parametrize("width", [60, 80, 100, 120])
def test_hash_bar_label_fits_beside_the_other_columns(width):
    from localm.model_manager.gguf import (_HASH_BAR_RESERVED_COLUMNS,
                                           _hash_bar_label)
    name = "mmproj-Qwen3-VL-8B-Instruct-f16.gguf"

    label = _hash_bar_label(name, "to check the vision projector", width)

    assert len(label) <= max(20, width - _HASH_BAR_RESERVED_COLUMNS)
    assert label.startswith("Hashing ")


def test_hash_bar_label_keeps_the_purpose_when_it_fits():
    from localm.model_manager.gguf import _hash_bar_label

    assert (_hash_bar_label("a.gguf", "for duplicate detection", 200)
            == "Hashing a.gguf for duplicate detection")


def test_hash_bar_for_a_projector_renders_on_one_line_at_100_columns(tmp_path, monkeypatch):
    import localm.model_manager as mm
    from localm.model_manager import gguf
    from rich.progress import Progress, TaskID

    seen = {}

    class Spy(Progress):
        instances: list = []

        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            Spy.instances.append(self)

    def fake_sha(path, progress=None):
        prog = Spy.instances[0]
        prog.update(TaskID(0), total=1_200_000_000, completed=500_000_000)
        probe = Console(file=io.StringIO(), width=100, force_terminal=False,
                        color_system=None)
        with probe.capture() as cap:
            probe.print(prog.get_renderable())
        seen["lines"] = cap.get().splitlines()
        return "digest"

    f = tmp_path / "mmproj-Qwen3-VL-8B-Instruct-f16.gguf"
    f.write_bytes(b"x" * 16)
    wide = Console(file=io.StringIO(), width=100, force_terminal=False, color_system=None)
    monkeypatch.setattr(gguf, "Progress", Spy)
    monkeypatch.setattr(gguf, "console", wide)
    monkeypatch.setattr(mm, "_HASH_PROGRESS_MIN_BYTES", -1)
    monkeypatch.setattr(mm, "_sha256_file", fake_sha)

    digest = gguf._hash_with_progress(f, purpose="to check the vision projector")

    assert digest == "digest"
    assert len(seen["lines"]) == 1, seen["lines"]
    line = seen["lines"][0]
    assert len(line) <= 100
    assert re.search(r"GB\b", line), line


class _SpyEngine:
    """An engine whose stream reports whether native console output was paused
    while it ran."""

    def __init__(self):
        self.paused_during = []

    def chat_stream(self, messages, **kwargs):
        from localm import debuglog
        self.paused_during.append(debuglog._native_console_is_paused())
        yield "hello"

    def count_tokens(self, text):
        return 1


def test_repl_reply_stream_runs_with_native_console_output_paused():
    from localm import debuglog
    from localm.cli.chat import _stream_once
    engine = _SpyEngine()

    reply = _stream_once(engine, [{"role": "user", "content": "hi"}])

    assert reply == "hello"
    assert engine.paused_during == [True]
    assert not debuglog._native_console_is_paused()


def test_repl_compaction_summary_runs_with_native_console_output_paused():
    from localm import debuglog
    from localm.cli.chat import _collect_summary
    engine = _SpyEngine()

    assert _collect_summary(engine, [], 64) == "hello"
    assert engine.paused_during == [True]
    assert not debuglog._native_console_is_paused()


def test_every_chat_stream_call_in_the_repl_is_inside_a_native_console_pause():
    import ast
    import localm.cli.chat as chat_mod

    tree = ast.parse(Path(chat_mod.__file__).read_text(encoding="utf-8"))
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}

    def inside_pause(node):
        while node in parents:
            node = parents[node]
            if isinstance(node, ast.With) and any(
                    isinstance(item.context_expr, ast.Call)
                    and getattr(item.context_expr.func, "id", "") == "native_console_paused"
                    for item in node.items):
                return True
        return False

    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "chat_stream"]

    assert calls
    assert [n.lineno for n in calls if not inside_pause(n)] == []

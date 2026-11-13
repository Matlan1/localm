# SPDX-License-Identifier: AGPL-3.0-or-later
"""`localm status`, `localm stop` and `localm benchmark` say what is true: an
API-only server is not called old, an idle activity list says what it covers,
two servers for one folder are named, and the benchmark counts tokens."""

import importlib

import pytest
from click.testing import CliRunner

from localm import instances
from localm.cli import main

models_mod = importlib.import_module("localm.cli.models")


def _register(home, root, iid, started):
    return instances.register_instance(
        home, instance_id=iid, port=8000 + int(started[-2:]), host="127.0.0.1",
        root_dir=root, mode="full", token="t-" + iid, scheme="http", started=started)


def _entry(iid, mode="full", started="2026-01-01T00:00:00"):
    return {"instance_id": iid, "scheme": "http", "host": "127.0.0.1", "port": 8642,
            "mode": mode, "pid": 4321, "version": "0.3.4", "token": "t", "started": started,
            "root_dir": "/proj"}


def test_live_entries_are_oldest_first_and_find_attachable_takes_the_oldest(tmp_path):
    root = str(tmp_path / "proj")
    _register(tmp_path, root, "zzzz0000", "2026-01-01T00:00:05")
    _register(tmp_path, root, "aaaa0000", "2026-01-01T00:00:09")
    _register(tmp_path, root, "mmmm0000", "2026-01-01T00:00:01")

    live = instances.live_entries(tmp_path, root, probe=lambda e: True)

    assert [e["instance_id"] for e in live] == ["mmmm0000", "zzzz0000", "aaaa0000"]
    assert instances.find_attachable(tmp_path, root, probe=lambda e: True)["instance_id"] \
        == "mmmm0000"


def test_live_entries_skip_an_entry_that_does_not_answer(tmp_path):
    root = str(tmp_path / "proj")
    _register(tmp_path, root, "aaaa0000", "2026-01-01T00:00:01")
    _register(tmp_path, root, "bbbb0000", "2026-01-01T00:00:02")

    live = instances.live_entries(tmp_path, root, probe=lambda e: e["instance_id"] == "bbbb0000")

    assert [e["instance_id"] for e in live] == ["bbbb0000"]


@pytest.mark.parametrize("mode,expected", [
    ("api", "An API-only server (localm serve) does not report activity"),
    ("full", "it predates this feature"),
])
def test_a_server_without_the_activity_route_is_described_by_its_kind(
        monkeypatch, mode, expected):
    monkeypatch.setattr(instances, "find_attachable", lambda *a, **k: _entry("abcd1234", mode))
    monkeypatch.setattr(instances, "live_entries", lambda *a, **k: [])
    monkeypatch.setattr(models_mod, "read_activity", lambda *a, **k: ("unsupported", 404))

    res = CliRunner().invoke(main, ["status"])

    assert res.exit_code == 0, res.output
    assert expected in " ".join(res.output.split())


def test_an_idle_activity_list_says_a_chat_reply_is_not_listed(monkeypatch):
    monkeypatch.setattr(instances, "find_attachable", lambda *a, **k: _entry("abcd1234"))
    monkeypatch.setattr(instances, "live_entries", lambda *a, **k: [])
    monkeypatch.setattr(models_mod, "read_activity",
                        lambda *a, **k: ("ok", {"now": 1.0, "operations": []}))

    out = " ".join(CliRunner().invoke(main, ["status"]).output.split())

    assert "No background operation running right now" in out
    assert "a chat reply or a scheduled job is not listed here" in out
    assert "Nothing running" not in out


def test_status_names_a_second_server_for_the_same_folder(monkeypatch):
    first, second = _entry("aaaa1111"), _entry("bbbb2222")
    monkeypatch.setattr(instances, "find_attachable", lambda *a, **k: first)
    monkeypatch.setattr(instances, "live_entries", lambda *a, **k: [first, second])
    monkeypatch.setattr(models_mod, "read_activity",
                        lambda *a, **k: ("ok", {"now": 1.0, "operations": []}))

    out = " ".join(CliRunner().invoke(main, ["status"]).output.split())

    assert "2 servers serve this directory" in out
    assert "commands here use aaaa1111" in out
    assert "bbbb2222" in out


def test_stop_with_no_id_refuses_when_two_servers_serve_the_folder(monkeypatch):
    first, second = _entry("aaaa1111"), _entry("bbbb2222")
    monkeypatch.setattr(instances, "reap_stale", lambda *a, **k: [])
    monkeypatch.setattr(instances, "list_entries", lambda *a, **k: [first, second])
    monkeypatch.setattr(instances, "find_attachable", lambda *a, **k: first)
    monkeypatch.setattr(instances, "live_entries", lambda *a, **k: [first, second])
    posted = []
    monkeypatch.setattr("requests.post", lambda *a, **k: posted.append(a) or None)

    res = CliRunner().invoke(main, ["stop"])

    assert res.exit_code == 1, res.output
    assert "2 servers serve" in res.output
    assert "aaaa1111" in res.output and "bbbb2222" in res.output
    assert posted == []


@pytest.mark.parametrize("prompts", ["", " , ", "0", "64,-5"])
def test_benchmark_refuses_no_or_non_positive_sizes_before_loading(monkeypatch, prompts):
    monkeypatch.setattr(models_mod, "get_operator_model_info", lambda m: ("/x.gguf", None))
    built = []
    monkeypatch.setattr("localm.inference.engine.Engine",
                        lambda *a, **k: built.append(a) or pytest.fail("loaded"))

    res = CliRunner().invoke(main, ["benchmark", "m", "--prompts", prompts])

    assert res.exit_code == 1, res.output
    assert "--prompts needs one or more prompt sizes above 0" in res.output
    assert built == []


class _FakeEngine:
    def __init__(self, *a, **k):
        pass

    def load(self):
        pass

    def unload(self):
        pass

    def count_tokens(self, text):
        return len(text)

    def chat_stream(self, messages, **k):
        yield "abc"
        yield "de"


def test_benchmark_counts_the_reply_in_tokens_not_stream_pieces(monkeypatch):
    monkeypatch.setattr(models_mod, "get_operator_model_info", lambda m: ("/x.gguf", None))
    monkeypatch.setattr("localm.inference.engine.Engine", _FakeEngine)
    from rich.table import Table
    rows = []
    real_add_row = Table.add_row
    monkeypatch.setattr(Table, "add_row",
                        lambda self, *cells, **k: rows.append(cells) or real_add_row(self, *cells, **k))

    res = CliRunner().invoke(main, ["benchmark", "m", "--prompts", "8"])

    assert res.exit_code == 0, res.output
    assert len(rows) == 1
    assert rows[0][1] == "5"

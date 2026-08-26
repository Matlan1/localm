# SPDX-License-Identifier: AGPL-3.0-or-later
"""Inference/coder behaviour:

  - effective_mode('coder', cwd) honors .localcoder/config.toml
  - detect_model_family matches substrings / resolved repo ids
  - the coder aborts on repeated identical (scaffold) responses
"""

from unittest.mock import patch



def test_effective_mode_coder_reads_project_toml(tmp_path, monkeypatch):
    from localm.audit import effective_mode, SessionMode
    monkeypatch.delenv("LOCALM_MODE", raising=False)
    proj = tmp_path / "proj"
    (proj / ".localcoder").mkdir(parents=True)
    (proj / ".localcoder" / "config.toml").write_text('mode = "log"\n', encoding="utf-8")
    # A project-local .localcoder/config.toml mode is honored when cwd is passed.
    assert effective_mode("coder", cwd=proj) == SessionMode.LOG


def test_effective_mode_coder_toml_absent_is_safe(tmp_path, monkeypatch):
    from localm.audit import effective_mode, SessionMode
    monkeypatch.delenv("LOCALM_MODE", raising=False)
    # No toml -> falls through to config/privacy without crashing.
    assert isinstance(effective_mode("coder", cwd=tmp_path), SessionMode)



def test_detect_family_substring_and_repo_id():
    from localm.plugins.coder.prompts import detect_model_family
    assert detect_model_family("gemma4-4b") == "gemma"
    # an opaque alias enriched with its registry source is still classified
    assert detect_model_family("mycoder hf:google/gemma-4-4b") == "gemma"
    assert detect_model_family("hf:microsoft/phi-4") == "small"
    assert detect_model_family("llama-3.1-8b") == "default"



def _make_agent(tmp_path, backend=None, **kwargs):
    from localm.plugins.coder.agent import Agent
    from unittest.mock import MagicMock
    if backend is None:
        backend = MagicMock()
        backend.model_id = "test-model"
        backend.native_tools = False
    with patch("localm.plugins.coder.agent.ProjectMap") as MockPM, \
         patch("localm.plugins.coder.agent.make_audit_log"), \
         patch("localm.plugins.coder.agent.load_memory", return_value=""):
        MockPM.build.return_value.file_count.return_value = 0
        return Agent(backend=backend, cwd=tmp_path, **kwargs)


def test_repeated_response_breaker_aborts(tmp_path):
    agent = _make_agent(tmp_path, max_turns=12, auto_approve=True, self_verify=False)
    # The model emits the SAME tool-call response every turn (a stuck scaffold);
    # the loop continues (the call succeeds) but makes no real progress.
    fixed = '<tool_call>\n{"name": "list_dir", "args": {"path": "."}}\n</tool_call>'
    with patch.object(agent, "_call_llm", return_value=fixed):
        result = agent.run_task("do the thing")
    # WHICH breaker tripped, asserted on STATE rather than on the message text.
    # _repeat_response_count is incremented in exactly ONE place (the
    # repeated-scaffold breaker in agent/loop.py) and run_task does not reset
    # it, so after the call it records that THIS breaker fired. loop.py has
    # three messages beginning "[circuit breaker:", so the prose check below
    # cannot tell them apart. Compared against the constant, not a literal.
    from localm.plugins.coder.agent.constants import _REPEAT_RESPONSE_ABORT
    assert agent._repeat_response_count >= _REPEAT_RESPONSE_ABORT - 1
    # A breaker message reached the user. "circuit breaker" is the prefix all
    # three share, so this says something was reported, not which.
    assert "circuit breaker" in result.lower()
    # It stopped well before the turn ceiling.
    assert agent._turns < 12


def test_family_enrichment_uses_registry_source(tmp_path, monkeypatch):
    # An opaque alias whose registry source is a gemma repo classifies as gemma:
    # the enrichment reads the registry entry's "source", not the (path,hint) tuple.
    import localm.model_manager as mm
    monkeypatch.setattr(
        mm, "load_registry",
        lambda: {"mycoder": {"source": "hf:google/gemma-4-4b", "path": "/x"}})
    from unittest.mock import MagicMock
    from localm.plugins.coder.prompts import detect_model_family
    backend = MagicMock()
    backend.model_id = "mycoder"
    backend.native_tools = False
    agent = _make_agent(tmp_path, backend=backend)
    assert "gemma" in agent._family_id.lower(), agent._family_id
    assert detect_model_family(agent._family_id) == "gemma"

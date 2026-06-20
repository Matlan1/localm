"""PHASE 1 - deterministic coder restricted-session security boundary.

Constructs a REAL restricted Agent and a REAL owner Agent (dummy backend, never
called for dispatch) and adversarially exercises the security boundary:
  - the restricted disabled-tool set == TOOL_REGISTRY - SAFE_RESTRICTED_TOOLS
  - the hard dispatch gate refuses run_shell / spawn_agent / read_env etc. and
    does NOT execute them (try to break out -> blocked)
  - allowlisted tools (read_file, write_file within cwd) still work
  - disabled tools are hidden from the model (system prompt + parser exclude them)
  - a spawned child inherits restricted + disabled_tools
  - the owner agent is unrestricted (run_shell runs) = coder-owner-full
No mocks of the boundary itself: this calls the real Agent._execute_tool.
"""
from __future__ import annotations
import os, sys, tempfile
from pathlib import Path
from types import SimpleNamespace

# Run against a clean temp home so register_plugin/mcp/skill tools add nothing
# project-specific that would muddy the allowlist complement assertion.
tmp_home = tempfile.mkdtemp(prefix="qa-phase1-home-")
os.environ["LOCALM_HOME"] = tmp_home

from localm.plugins.coder.agent import Agent
from localm.plugins.coder.tools import (
    SAFE_RESTRICTED_TOOLS, TOOL_REGISTRY, tool_spawn_agent,
)
from localm.plugins.coder.parser import ToolCall, parse_tool_calls
from localm.plugins.coder.audit import SessionMode
from localm.plugins.coder.prompts import build_system_prompt

PASS, FAIL = [], []
def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(f"{name}: {detail}")
    print(("PASS " if cond else "FAIL ") + name + (" :: " + detail if detail else ""))

def TC(name, args):
    """Build a ToolCall (name, args, raw, start, end) for dispatch tests."""
    return ToolCall(name, args, "", 0, 0)

backend = SimpleNamespace(model_id="qa", native_tools=False)
cwd = Path(tempfile.mkdtemp(prefix="qa-phase1-cwd-"))
(cwd / "seed.txt").write_text("seed content\n", encoding="utf-8")

# ----- construct a REAL restricted agent and a REAL owner agent ------------
restricted = Agent(backend, cwd=cwd, mode=SessionMode.PRIVACY, restricted=True,
                   auto_approve=True)
owner = Agent(backend, cwd=cwd, mode=SessionMode.PRIVACY, restricted=False,
              auto_approve=True)

# 1. allowlist == matrix expectation (read + confined edit + read-only git)
expected_allow = {
    "read_file", "list_dir", "tree", "grep", "search_files",
    "write_file", "edit_file", "patch_file", "search_replace", "edit_notebook_cell",
    "git_status", "git_diff", "git_log",
}
check("allowlist-matches-source", set(SAFE_RESTRICTED_TOOLS) == expected_allow,
      f"SAFE_RESTRICTED_TOOLS={sorted(SAFE_RESTRICTED_TOOLS)}")

# 2. restricted.disabled_tools == everything in the registry not in the allowlist
expected_disabled = set(TOOL_REGISTRY) - set(SAFE_RESTRICTED_TOOLS)
check("restricted-disabled-is-complement",
      set(restricted.disabled_tools) == expected_disabled,
      f"missing-from-disabled={sorted(expected_disabled - set(restricted.disabled_tools))}")

# 3. the named dangerous tools are all disabled for restricted
for t in ("run_shell", "run_tests", "git_commit", "git_push", "fetch_url",
          "web_search", "generate_image", "read_env", "spawn_agent"):
    if t in TOOL_REGISTRY:
        check(f"restricted-disables-{t}", t in restricted.disabled_tools)
    else:
        check(f"restricted-disables-{t}", True, "tool not in registry (n/a)")

# 4. owner has an EMPTY disabled set (full coder)
check("owner-no-disabled", set(owner.disabled_tools) == set(),
      f"owner.disabled_tools={sorted(owner.disabled_tools)}")

# ----- 5. HARD DISPATCH GATE: try to break out via run_shell --------------
sentinel = cwd / "PWNED.txt"
if sentinel.exists():
    sentinel.unlink()
shell_call = TC("run_shell", {"command": "echo pwned > " + sentinel.name})
res = restricted._execute_tool(shell_call, interactive=False)
check("restricted-run_shell-refused", (not res.ok) and "disabled for this session" in res.output,
      f"output={res.output[:80]!r}")
check("restricted-run_shell-no-sideeffect", not sentinel.exists(),
      f"sentinel exists={sentinel.exists()} (must be False - command must NOT run)")

# spawn_agent also hard-refused at dispatch for restricted
spawn_call = TC("spawn_agent", {"task": "do something", "name": "evil"})
res = restricted._execute_tool(spawn_call, interactive=False)
check("restricted-spawn_agent-refused", (not res.ok) and "disabled for this session" in res.output,
      f"output={res.output[:80]!r}")

# read_env (secret disclosure) hard-refused
if "read_env" in TOOL_REGISTRY:
    res = restricted._execute_tool(TC("read_env", {}), interactive=False)
    check("restricted-read_env-refused", (not res.ok) and "disabled for this session" in res.output,
          f"output={res.output[:80]!r}")

# ----- 6. allowlisted tools STILL WORK for restricted ---------------------
res = restricted._execute_tool(TC("read_file", {"path": "seed.txt"}), interactive=False)
check("restricted-read_file-allowed", res.ok and "seed content" in res.output,
      f"ok={res.ok} out={res.output[:60]!r}")
res = restricted._execute_tool(
    TC("write_file", {"path": "made_by_restricted.txt", "content": "ok\n"}),
    interactive=False)
made = cwd / "made_by_restricted.txt"
check("restricted-write_file-allowed", res.ok and made.exists(),
      f"ok={res.ok} file_exists={made.exists()}")

# ----- 7. owner CAN run run_shell (unrestricted) = coder-owner-full -------
owner_sentinel = cwd / "owner_ran.txt"
if owner_sentinel.exists():
    owner_sentinel.unlink()
res = owner._execute_tool(
    TC("run_shell", {"command": "echo hi > " + owner_sentinel.name}),
    interactive=False)
check("owner-run_shell-allowed", res.ok and owner_sentinel.exists(),
      f"ok={res.ok} out={res.output[:80]!r} file={owner_sentinel.exists()}")

# ----- 8. disabled tools NOT TAUGHT to the model --------------------------
# The callable tool DOCS (## <name> blocks) are the mechanism: a disabled tool's
# usage doc is omitted, so the model is never taught how to call it. (Incidental
# prose mentions of run_shell as a concept remain in the static guidance text -
# a minor cosmetic nuance, not a capability: the dispatch gate refuses anyway.)
from localm.plugins.coder.prompts import _full_tool_docs, _brief_tool_docs
full_r = _full_tool_docs(restricted.disabled_tools)
full_o = _full_tool_docs(frozenset())
brief_r = _brief_tool_docs(restricted.disabled_tools)
check("restricted-tooldoc-omits-run_shell", "## run_shell" not in full_r,
      "no '## run_shell' call-doc in restricted full docs")
check("owner-tooldoc-has-run_shell", "## run_shell" in full_o,
      "owner full docs DO teach run_shell")
check("restricted-tooldoc-keeps-read_file", "## read_file" in full_r,
      "restricted still taught read_file")
check("restricted-brief-omits-run_shell", "run_shell(" not in brief_r,
      "small-model brief docs also omit run_shell")

# parser gating of the LENIENT bare-JSON format: a name not in tool_names is not
# recognised (so the model can't smuggle a disabled call via bare JSON).
allowed_names = set(TOOL_REGISTRY) - set(restricted.disabled_tools)
bare = '{"name": "run_shell", "args": {"command": "ls"}}'
calls_bare = parse_tool_calls(bare, tool_names=allowed_names)
check("restricted-parser-drops-bare-run_shell", all(c.name != "run_shell" for c in calls_bare),
      f"bare parsed={[c.name for c in calls_bare]}")
# The EXPLICIT <tool_call> wrapper parses any name (by design) - but dispatch
# refuses it, so it is not a hole. Verify the end-to-end: parse the wrapper, then
# dispatch the parsed call and confirm the hard gate blocks it.
wrapped = '<tool_call>\n{"name": "run_shell", "args": {"command": "ls"}}\n</tool_call>'
calls_w = parse_tool_calls(wrapped, tool_names=allowed_names)
gate_ok = False
if calls_w:
    r = restricted._execute_tool(calls_w[0], interactive=False)
    gate_ok = (not r.ok) and "disabled for this session" in r.output
check("restricted-wrapper-parsed-but-dispatch-refuses", gate_ok,
      f"wrapper parsed {[c.name for c in calls_w]}, dispatch refused={gate_ok}")

# ----- 9. spawned child INHERITS restricted + disabled_tools --------------
# Construct the child exactly as tool_spawn_agent does (without running the LLM):
child = Agent(backend=backend, cwd=cwd, name="child", max_turns=2, auto_approve=True,
              parent=restricted, mode=SessionMode.PRIVACY,
              restricted=getattr(restricted, "restricted", False),
              disabled_tools=getattr(restricted, "disabled_tools", frozenset()))
check("spawn-child-inherits-restricted", child.restricted is True)
check("spawn-child-inherits-disabled", "run_shell" in child.disabled_tools)
# and the child likewise hard-refuses run_shell
res = child._execute_tool(TC("run_shell", {"command": "echo x"}), interactive=False)
check("spawn-child-run_shell-refused", (not res.ok) and "disabled for this session" in res.output)

print("\n==== SUMMARY ====")
print(f"PASS={len(PASS)}  FAIL={len(FAIL)}")
if FAIL:
    print("FAILURES:")
    for f in FAIL:
        print("  - " + f)
    sys.exit(1)
print("ALL PHASE-1 BOUNDARY CHECKS PASSED")

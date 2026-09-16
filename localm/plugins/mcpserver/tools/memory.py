# SPDX-License-Identifier: AGPL-3.0-or-later
"""MCP tools over the owner's durable chat memory: ``memory_recall`` and
``memory_append``.

Both handlers re-check the memory plugin's availability and the privacy gate
on every call. The plugin probe and the embedder resolver are read from the
server module at call time, so they resolve to whatever that module currently
binds.
"""

from __future__ import annotations

from typing import Dict

from .. import server as _srv
from ..server import _quiet_stdout, _text_result


def _memory_writes_allowed() -> bool:
    """The privacy gate for chat memory, resolved fresh on every call.

    The surface is "chat", not "server": these tools read and write the SAME
    namespace the chat routes use (agent "chat"), so gating them on a different
    surface would answer to a different mode knob than the store they touch."""
    from localm.memory.gating import writes_allowed
    return writes_allowed("chat")


def _memory_store():
    """The owner's chat-memory store: the same namespace the /api/memory routes
    and `localm memory` open. The four values are load-bearing and deliberately
    not parameterised - a different agent, scope_key or root would read an empty
    store and report that localm has learned nothing about the user."""
    from localm import memory as _mem
    from localm.config import home_dir
    return _mem.open_store(None, "chat", "", root=home_dir() / "memory")


# Default facts per memory_recall call; the store caps any request at K_CAP.
_MEMORY_RECALL_DEFAULT_K = 6

_MEMORY_OFF_MSG = (
    "The memory plugin is not active, so localm has no chat memory to read or "
    "write. Install and enable it with: localm plugin install memory")

_MEMORY_PRIVACY_MSG = (
    "Refused: localm is in privacy mode, where chat memory is fully off - no "
    "recall and no writes. This surface has no privacy-mode opt-in, unlike the "
    "in-process chat recall knob. Set mode/chat_mode to 'log' or 'full' to use it.")

_MEMORY_WRITE_OFF_MSG = (
    "Refused: memory_append is not enabled for this MCP server. Writing to your "
    "personal memory from an external client is opt-in and separate from enabling "
    "the memory plugin. Relaunch the server with: localm mcp --memory-write")


def build(*, enable_memory_write: bool) -> Dict[str, dict]:
    """``memory_recall`` and ``memory_append``. *enable_memory_write* is the
    server's write opt-in; ``memory_append`` refuses every call when it is
    False."""
    def memory_recall(args: dict) -> dict:
        """Read the owner's durable chat memory. Never writes."""
        # Re-checked here, not only in build_tools: a stale client tool list, a
        # call by name, or a later refactor that drops the build-time gate must
        # still not reach the store.
        if not _srv._memory_available():
            return _text_result(_MEMORY_OFF_MSG, is_error=True)
        if not _memory_writes_allowed():
            return _text_result(_MEMORY_PRIVACY_MSG, is_error=True)
        query = (args.get("query") or "").strip()
        if not query:
            return _text_result("'query' is required", is_error=True)
        from localm.memory import render_memories
        from localm.memory.store import K_CAP, MAX_TEXT_LEN
        try:
            k = int(args.get("limit") or _MEMORY_RECALL_DEFAULT_K)
        except (TypeError, ValueError):
            k = _MEMORY_RECALL_DEFAULT_K
        k = max(1, min(k, K_CAP))
        store = _memory_store()
        # reinforce=False: a read from an external client must not bump
        # last_used/uses. That is a write, and it would let a client's query
        # pattern reshape which of the owner's facts survive prune().
        with _quiet_stdout():
            records = store.recall(query, k=k, embed_fn=_srv._memory_embed_fn(),
                                   reinforce=False)
        if not records:
            return _text_result(
                f"No remembered facts matched {query!r}. localm has "
                f"{len(store.all())} fact(s) stored.")
        # render_memories neutralises every line and fences the block as
        # data-not-instructions, so an instruction-shaped memory cannot become an
        # instruction in the calling client's prompt. Its default max_chars is the
        # CHAT prompt-injection budget, far below what an explicit request for k
        # facts needs, so the budget is sized to the request here and the block is
        # still bounded (K_CAP * MAX_TEXT_LEN plus fence/bullet overhead).
        block = render_memories(records, max_chars=k * (MAX_TEXT_LEN + 4) + 128)
        rendered = sum(1 for ln in block.splitlines() if ln.startswith("- "))
        if rendered < len(records):
            # Never let the block cap drop facts silently: a short answer would
            # otherwise read as "that is all localm remembers".
            block += (f"\n\n[{len(records) - rendered} more matching fact(s) were "
                      f"not shown: the block size limit was reached.]")
        return _text_result(block)

    def memory_append(args: dict) -> dict:
        """Offer a fact for the owner's durable chat memory.

        Never writes a TRUSTED record and never overwrites one: a fact that
        contradicts an existing user-typed memory becomes a pending correction
        the owner accepts or rejects."""
        if not _srv._memory_available():
            return _text_result(_MEMORY_OFF_MSG, is_error=True)
        # Unreachable while this tool is registered only when the flag is set:
        # the flag comes from argv and cannot change during the process. It
        # guards a later refactor that registers the tool unconditionally and
        # lets the handler decide. It is NOT the live gate - the plugin probe
        # and the privacy check below are, and each re-resolves per call.
        if not enable_memory_write:
            return _text_result(_MEMORY_WRITE_OFF_MSG, is_error=True)
        if not _memory_writes_allowed():
            return _text_result(_MEMORY_PRIVACY_MSG, is_error=True)
        text = (args.get("text") or "").strip()
        if not text:
            return _text_result("'text' is required", is_error=True)

        from localm.memory import N_MAX, MemoryRecord, PendingCorrection
        from localm.memory.consolidate import (MATCH_THRESHOLD, SYNTH_IMP_CAP,
                                               _nearest)
        from localm.memory.store import TRUSTED_SOURCES
        store = _memory_store()
        existing = store.all()

        # A supersession of a TRUSTED (user/import) fact is PROPOSED, never
        # applied: an MCP client is not the user typing, so it may not rewrite
        # what the user typed. Lexical ratio only - no model call, and when the
        # answer is unclear the conservative direction (propose) is already
        # correct because the owner sees it either way.
        idx, ratio = _nearest(text, existing)
        if idx >= 0 and ratio >= MATCH_THRESHOLD and \
                existing[idx].source in TRUSTED_SOURCES:
            target = existing[idx]
            with _quiet_stdout():
                added = store.propose_corrections([PendingCorrection(
                    target_id=target.id, action="update", proposed_text=text,
                    target_text=target.text, confidence=ratio)])
            if not added:
                return _text_result(
                    "Already proposed: this correction is pending review, or was "
                    "previously rejected. Nothing was changed.")
            return _text_result(
                "Queued for review, NOT saved: this contradicts a fact you "
                f"entered yourself ({target.text!r}). localm never overwrites "
                "your own memories from an external client. Accept or reject it "
                "in Settings > Memory, or with `localm memory corrections`.")

        if len(existing) >= N_MAX:
            return _text_result(
                f"Memory is at its {N_MAX}-record cap; delete a fact before "
                "adding another.", is_error=True)
        # source="synth", never "user": a trusted record is exempt from recall
        # decay and from prune() eviction, so a client that could mint one could
        # pin a fact the owner never asserted.
        with _quiet_stdout():
            rec = store.add(
                MemoryRecord(text=text, kind="semantic", source="synth",
                             importance=SYNTH_IMP_CAP, meta={"via": "mcp"}),
                embed_fn=_srv._memory_embed_fn())
        return _text_result(f"Remembered (unverified, id {rec.id}): {rec.text}")

    return {
        "memory_recall": {
            "description": (
                "Search what localm durably remembers about this user (their "
                "stated preferences, projects, tools, recurring goals) and return "
                "the matching facts. Read-only. Use it to carry context between "
                "agents instead of asking the user to repeat themselves."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string",
                              "description": "What to recall facts about"},
                    "limit": {"type": "integer",
                              "description": "Max facts to return "
                                             "(default 6, capped at 32)"},
                },
                "required": ["query"],
            },
            "annotations": {"readOnlyHint": True, "title": "Recall memory"},
            "handler": memory_recall,
        },
        "memory_append": {
            "description": (
                "Offer one durable fact about this user for localm's memory. "
                "Stored as UNVERIFIED (machine-asserted), never as a "
                "user-typed fact. A fact that contradicts something the user "
                "entered themselves is queued for their review instead of "
                "being saved, and never overwrites it."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "text": {"type": "string",
                             "description": "One durable fact about the user"},
                },
                "required": ["text"],
            },
            "annotations": {"title": "Remember a fact"},
            "handler": memory_append,
        },
    }

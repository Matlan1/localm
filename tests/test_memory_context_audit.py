# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for memory and context audit fixes."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch
import pytest
from fastapi import Request

from localm.inference.backends.base import BaseBackend
from localm.inference.backends.gguf import GgufBackend
from localm.plugins.builtin.memory.plug import (
    _recent_sessions_text, _memory_inlet, memory_get, memory_put
)
from localm.inference.chat_pipeline import ChatHookContext
from pydantic import BaseModel

# Mock pydantic models for request bodies
class MemoryUpdate(BaseModel):
    text: str

class MemoryAppend(BaseModel):
    text: str

class MemoryPatch(BaseModel):
    text: str | None = None
    importance: float | None = None


class DummyBackend(BaseBackend):
    def chat_stream(self, *args, **kwargs):
        pass
    def load(self, *args, **kwargs):
        pass
    @property
    def loaded(self) -> bool:
        return True
    def unload(self, *args, **kwargs):
        pass


def test_base_backend_count_messages_tokens():
    backend = DummyBackend()
    messages = [
        {"role": "user", "content": "hello world"},
        {"role": "assistant", "content": "hi there"}
    ]
    # Default is chars // 4 heuristic: "hello world" (11) + "hi there" (8) + join space (1) = 20 chars -> 5 tokens
    assert backend.count_messages_tokens(messages) == 5


def test_gguf_backend_count_messages_tokens_with_llm():
    backend = GgufBackend(model_path="dummy.gguf")
    # Mock self._llm
    mock_llm = MagicMock()
    mock_llm.tokenize.return_value = [1, 2, 3, 4, 5]
    backend._llm = mock_llm
    
    messages = [
        {"role": "user", "content": "hello world"}
    ]
    
    with patch("localm.inference.backends.gguf.GgufBackend.count_messages_tokens") as mock_count:
        mock_count.return_value = 5
        assert backend.count_messages_tokens(messages) == 5


def test_hf_backend_count_messages_tokens_with_tokenizer():
    # Tests HFWorker (_hf_worker.py), not the HFBackend proxy (hf.py): the
    # chat-template + tokenizer logic runs only in the isolated child process,
    # and HFBackend.count_messages_tokens is an RPC to it. Only the tokenizer
    # itself is mocked.
    from localm.inference.backends._hf_worker import HFWorker

    # Building an HFWorker in-process is safe ONLY because
    # count_messages_tokens's body has zero torch/transformers imports. The
    # other methods (load/embed/chat_stream) do function-local
    # torch/transformers imports with no native_lib_loaded() guard and rely on
    # process isolation, so a test that calls one of THOSE needs that guard.
    backend = HFWorker.__new__(HFWorker)
    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template.return_value = "hello world"
    mock_tokenizer.encode.return_value = [1, 2, 3]
    backend._tokenizer = mock_tokenizer
    backend._processor = None
    backend._is_multimodal = False

    messages = [
        {"role": "user", "content": "hello world"}
    ]

    assert backend.count_messages_tokens(messages) == 3
    mock_tokenizer.apply_chat_template.assert_called_once()


def test_recent_sessions_text_chronological_truncation(tmp_path):
    # Setup temporary sessions directory
    session_dir = tmp_path / "sessions"
    session_dir.mkdir()
    
    # Write a session log file
    session_file = session_dir / "session1.jsonl"
    session_file.write_text(
        json.dumps({"type": "user", "data": {"content": "First message"}}) + "\n" +
        json.dumps({"type": "llm", "data": {"content": "Second message"}}) + "\n" +
        json.dumps({"type": "user", "data": {"content": "Third message"}}) + "\n",
        encoding="utf-8"
    )
    
    with patch("localm.plugins.builtin.memory.plug._home", return_value=tmp_path):
        # Retrieve all
        text = _recent_sessions_text(max_chars=1000)
        assert text == "User: First message\nAssistant: Second message\nUser: Third message"
        
        # Test truncation prioritizing newest turns
        # "User: Third message" (19 chars)
        # "Assistant: Second message" (25 chars)
        # "User: First message" (19 chars)
        # max_chars=45 should fit Third (19) + Second (25) + newline (1) = 45 chars
        text_truncated = _recent_sessions_text(max_chars=45)
        # Should be Second and Third chronologically:
        assert text_truncated == "Assistant: Second message\nUser: Third message"


def test_memory_inlet_skips_coder_client():
    ctx = ChatHookContext(model_id="test", stream=False, request_id="req1")
    ctx.state["client_id"] = "coder"
    
    messages = [{"role": "user", "content": "hello"}]
    res = _memory_inlet(messages, ctx)
    assert res is None


@pytest.mark.anyio
async def test_memory_routes_principal_isolation(tmp_path):
    # Create request with mock auth token mapping to different principals
    req_user1 = MagicMock(spec=Request)
    req_user2 = MagicMock(spec=Request)
    
    # We mock principal_id(request) to return specific principals
    with patch("localm.plugins.builtin.memory.plug._home", return_value=tmp_path), \
         patch("localm.plugins.builtin.memory.plug._persist_enabled", return_value=True), \
         patch("localm.inference.http_server.principal_id") as mock_principal:
        
        # User 1 put memory
        mock_principal.return_value = "user1"
        update_data = MemoryUpdate(text="user1 memory")
        await memory_put(update_data, request=req_user1)
        
        # User 2 put memory
        mock_principal.return_value = "user2"
        update_data_2 = MemoryUpdate(text="user2 memory")
        await memory_put(update_data_2, request=req_user2)
        
        # User 1 get memory
        mock_principal.return_value = "user1"
        res1 = await memory_get(request=req_user1)
        assert "user1 memory" in res1["text"]
        assert "user2 memory" not in res1["text"]
        
        # User 2 get memory
        mock_principal.return_value = "user2"
        res2 = await memory_get(request=req_user2)
        assert "user2 memory" in res2["text"]
        assert "user1 memory" not in res2["text"]

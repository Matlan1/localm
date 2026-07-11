# SPDX-License-Identifier: AGPL-3.0-or-later
"""Adversarial /v1 request-validation hardening.

Three break-it findings, each a clean-rejection or info-leak guard:

  BUG-4  max_tokens <= 0 was neither rejected nor honored: it collided with the
         internal "<=0 == unlimited" sentinel, so a client asking for 0/short
         tokens got a full, unbounded generation (silent-wrong + soft DoS). Now a
         request-level max_tokens must be >= 1 (a clean 422), matching OpenAI.

  BUG-5  temperature / top_p / repeat_penalty accepted NaN/Infinity (stdlib json
         parses the bare tokens; pydantic's default allow_inf_nan admitted them),
         feeding a non-finite value straight into the native sampler. Now non-finite
         values are rejected with a clean 422.

  BUG-3  GET /v1/models/{id} for an entry with an empty path ran
         Path("").rglob("*") -> a walk of the server CWD (aggregate-size info leak
         + filesystem-walk DoS). The size branch is now guarded on a real path.
"""

import os
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from localm.inference.http_server import create_app


def _client():
    """A client over a fully-working engine stub: a REJECTED request 422s at
    validation before the handler (engine untouched); an ACCEPTED request reaches
    generation and returns 200 with a trivial reply."""
    os.environ.pop("LOCALM_API_KEY", None)
    engine = MagicMock()
    engine.display_name = "m"
    engine.active_requests = 0
    engine.last_finish_reason = "stop"
    engine.count_messages_tokens.return_value = 3
    engine.count_tokens.return_value = 1
    engine.context_capacity.return_value = None
    engine.validate_grammar.return_value = None
    type(engine).loaded = property(lambda self: True)
    engine.chat_stream.side_effect = lambda messages, **kw: iter(["ok"])
    return TestClient(create_app(engine), raise_server_exceptions=True)


def _chat(client, **extra):
    body = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    body.update(extra)
    return client.post("/v1/chat/completions", json=body)


# --------------------------------------------------------------------------- #
# BUG-4: max_tokens <= 0
# --------------------------------------------------------------------------- #

def test_max_tokens_zero_rejected_chat():
    r = _chat(_client(), max_tokens=0)
    assert r.status_code == 422, r.text


def test_max_tokens_negative_rejected_chat():
    r = _chat(_client(), max_tokens=-5)
    assert r.status_code == 422, r.text


def test_max_tokens_one_is_allowed():
    """The boundary value 1 is valid (only <= 0 is rejected): it passes validation
    and generates normally (guards against over-rejection)."""
    r = _chat(_client(), max_tokens=1, stream=False)
    assert r.status_code == 200, r.text


def test_max_tokens_zero_rejected_completions():
    client = _client()
    r = client.post("/v1/completions", json={"model": "m", "prompt": "hi", "max_tokens": 0})
    assert r.status_code == 422, r.text


# --------------------------------------------------------------------------- #
# BUG-5: NaN / Infinity numeric params
# --------------------------------------------------------------------------- #

def _chat_raw(client, raw_json: str):
    return client.post("/v1/chat/completions", content=raw_json,
                       headers={"content-type": "application/json"})


def test_nan_temperature_rejected():
    r = _chat_raw(_client(),
                  '{"model":"m","messages":[{"role":"user","content":"hi"}],'
                  '"temperature":NaN,"max_tokens":4}')
    assert r.status_code == 422, r.text


def test_infinity_top_p_rejected():
    r = _chat_raw(_client(),
                  '{"model":"m","messages":[{"role":"user","content":"hi"}],'
                  '"top_p":Infinity,"max_tokens":4}')
    assert r.status_code == 422, r.text


def test_infinity_repeat_penalty_rejected():
    r = _chat_raw(_client(),
                  '{"model":"m","messages":[{"role":"user","content":"hi"}],'
                  '"repeat_penalty":-Infinity,"max_tokens":4}')
    assert r.status_code == 422, r.text


def test_nan_int_field_is_422_not_500():
    """A non-finite value in an INT field (top_k) fails validation; the 422 error
    body includes the offending `input`, which Starlette's JSONResponse cannot
    serialize (allow_nan=False) - a live 500 before the safe validation handler.
    Must be a clean 422, not a 500."""
    r = _chat_raw(_client(),
                  '{"model":"m","messages":[{"role":"user","content":"hi"}],'
                  '"top_k":NaN,"max_tokens":4}')
    assert r.status_code == 422, r.text


def test_nan_seed_is_422_not_500():
    r = _chat_raw(_client(),
                  '{"model":"m","messages":[{"role":"user","content":"hi"}],'
                  '"seed":NaN,"max_tokens":4}')
    assert r.status_code == 422, r.text


def test_finite_out_of_range_still_accepted():
    """A large but FINITE value is not the target of this guard (llama tolerates it
    and the server did not crash); only non-finite values are rejected. So a finite
    temperature must be accepted, not a 422."""
    r = _chat(_client(), temperature=1000.0, max_tokens=1)
    assert r.status_code == 200, r.text


# --------------------------------------------------------------------------- #
# BUG-3: GET /v1/models/{id} with an empty path must not walk the CWD
# --------------------------------------------------------------------------- #

def test_model_detail_empty_path_does_not_walk_cwd():
    engine = MagicMock()
    engine.display_name = "m"
    type(engine).loaded = property(lambda self: True)
    os.environ.pop("LOCALM_API_KEY", None)
    client = TestClient(create_app(engine), raise_server_exceptions=True)
    fake_registry = {"m": {"path": "", "source": "startup", "model_type": "llm"}}
    with patch("localm.config.load_registry", return_value=fake_registry):
        r = client.get("/v1/models/m")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["path"] == "", body
    # The vulnerable code did Path("").rglob("*") -> a CWD walk that returns a real
    # aggregate size; the guard leaves size_bytes None for an empty path.
    assert body["size_bytes"] is None, \
        "an empty registry path must not be stat/walked (server-CWD walk + size leak)"

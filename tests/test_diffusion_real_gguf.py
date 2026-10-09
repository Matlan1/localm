# SPDX-License-Identifier: AGPL-3.0-or-later
"""REAL end-to-end test of a diffusion language model through the GGUF backend.

No mocks: a real LLaDA-MoE GGUF (IQ1_S, about 1.7 GB, about 1B parameters
active) is loaded in the isolated worker and answers through chat_stream, the
path ``localm run``, the GUI chat and ``/v1/chat/completions`` all take. A
1-bit quant writes poor text, so the assertions are about the mechanics: the
reply arrives whole after the denoising status, a second request is answered
independently, a cancelled request leaves the worker serving, and a grammar is
refused.

@integration: needs the native runtime (``localm setup-llama``), a GPU-sized
download on first run, and minutes of compute. Skips when the runtime is not
provisioned or the file cannot be fetched; once both are present, any failure is
real.
"""

from __future__ import annotations

import pytest

from tests._real_gguf import fetch_gguf, require_native_runtime

pytestmark = [pytest.mark.integration, pytest.mark.real_gguf]

_REPO = "mradermacher/LLaDA-MoE-7B-A1B-Instruct-i1-GGUF"
_FILE = "LLaDA-MoE-7B-A1B-Instruct.i1-IQ1_S.gguf"
_MESSAGES = [{"role": "user", "content": "Name three colours."}]


@pytest.fixture(scope="module")
def diffusion_backend():
    require_native_runtime()
    path = fetch_gguf(_REPO, _FILE)

    from localm.inference.backends.gguf import GgufBackend
    be = GgufBackend(path, n_ctx=1024)
    assert be.is_diffusion is True
    be.load()
    yield be
    be.unload()


def _ask(backend, statuses, **kw):
    return "".join(backend.chat_stream(
        _MESSAGES, max_tokens=64, temperature=0.0, seed=3,
        on_status=statuses.append, **kw))


def test_reply_arrives_after_denoising_status(diffusion_backend):
    statuses = []
    out = _ask(diffusion_backend, statuses)
    assert statuses and statuses[0] == "Denoising reply (0%)..."
    percents = [int(s.split("(")[1].split("%")[0]) for s in statuses]
    assert percents == sorted(set(percents)) and percents[-1] >= 50
    assert diffusion_backend.last_finish_reason in ("stop", "length")
    assert "<|mdm_mask|>" not in out
    assert diffusion_backend.loaded


def test_same_seed_repeats_and_requests_are_independent(diffusion_backend):
    first = _ask(diffusion_backend, [])
    second = _ask(diffusion_backend, [])
    assert first == second


def test_cancel_mid_run_leaves_the_worker_serving(diffusion_backend):
    import threading

    from localm.inference.backends.base import stream_stop_check
    expected = _ask(diffusion_backend, [])
    pid = diffusion_backend._runner._proc.pid
    stop = threading.Event()
    statuses = []

    def stop_at_30(s):
        statuses.append(s)
        if s == "Denoising reply (30%)...":
            stop.set()
    with stream_stop_check(stop.is_set):
        out = "".join(diffusion_backend.chat_stream(
            _MESSAGES, max_tokens=64, temperature=0.0, seed=3, on_status=stop_at_30))
    assert out == ""
    assert statuses[-1] == "Denoising reply (30%)..."
    assert diffusion_backend.loaded
    assert diffusion_backend._runner._proc.pid == pid
    assert _ask(diffusion_backend, []) == expected


def test_grammar_is_refused(diffusion_backend):
    from localm.inference.backends.base import GrammarUnsupportedError
    with pytest.raises(GrammarUnsupportedError):
        diffusion_backend.validate_grammar('root ::= "a"')
    with pytest.raises(GrammarUnsupportedError):
        list(diffusion_backend.chat_stream(_MESSAGES, grammar='root ::= "a"'))
    assert diffusion_backend.loaded

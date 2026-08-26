# SPDX-License-Identifier: AGPL-3.0-or-later
"""REAL mid-stream cancel test for the HuggingFace backend's isolated worker.

No mocks: a tiny ungated causal LM (sshleifer/tiny-gpt2) is loaded for real
through HFBackend, then a live generation is cancelled mid-stream exactly the
way a client disconnect does (gen.close()). The property: the worker process
must stay alive and keep serving the SAME os process id, not respawn and reload
- see _hf_runner.py's module docstring for the mechanism (a
StoppingCriteria-based cooperative cancel relayed over ctrl_q).

Marked @integration so the default `pytest -m "not integration"` skips it (it
downloads a few MB on first run). A mock here would prove nothing: the property
is that the worker's actual OS process identity survives a real cancel, which no
fake child process can demonstrate.
"""

from __future__ import annotations

import json
import shutil
import time

import pytest

pytestmark = pytest.mark.integration

_MODEL = "sshleifer/tiny-gpt2"
# tiny-gpt2 ships no chat template; this trivial one (concatenate message
# contents) lets chat_stream's apply_chat_template run.
_MINIMAL_CHAT_TEMPLATE = "{% for m in messages %}{{ m['content'] }}\n{% endfor %}"


@pytest.fixture(scope="module")
def hf_backend(tmp_path_factory):
    # exc_type=ImportError: transformers' own internal tokenizers version-gate, or
    # a version-clashed torch build, can raise a plain ImportError rather than a
    # ModuleNotFoundError, which is all pytest 9.1's importorskip catches by
    # default.
    pytest.importorskip("torch", exc_type=ImportError)
    pytest.importorskip("transformers", exc_type=ImportError)
    from huggingface_hub import snapshot_download

    try:
        local_dir = snapshot_download(_MODEL)
    except Exception as e:                       # offline / hub unreachable
        pytest.skip(f"could not fetch {_MODEL}: {e}")

    from localm.inference.backends.hf import HFBackend

    # Inject the chat_template into tokenizer_config.json, into a COPY of the
    # snapshot, never the shared HF hub cache directory snapshot_download returned.
    model_dir = tmp_path_factory.mktemp("tiny_gpt2_cancel")
    shutil.copytree(local_dir, model_dir, dirs_exist_ok=True)
    config_path = model_dir / "tokenizer_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["chat_template"] = _MINIMAL_CHAT_TEMPLATE
    config_path.write_text(json.dumps(config), encoding="utf-8")

    be = HFBackend(str(model_dir), device="cpu")
    be.load()
    yield be
    be.unload()


def test_midstream_close_keeps_the_same_worker_process_alive(hf_backend):
    """Close the REAL generator chain mid-generation and prove the worker is
    NOT killed and NOT respawned: the same OS process id must still be
    alive, and a follow-up request must be served by it, not by a fresh
    reload. A chat_stream that calls shutdown(grace=0) on disconnect leaves the
    worker dead and a new pid on the follow-up request."""
    be = hf_backend
    pid_before = be._runner._proc.pid

    gen = be.chat_stream(
        [{"role": "user", "content": "Count from 1 to 100."}],
        max_tokens=500, temperature=0.0,
    )
    first = next(gen)                        # prefill + first token: generation in flight
    assert isinstance(first, str)

    t0 = time.monotonic()
    gen.close()                              # what a real client disconnect does
    elapsed = time.monotonic() - t0
    # A cooperative cancel confirms within one drain poll of the child stopping,
    # bounded well under the 5s drain timeout. The pid/liveness checks below are
    # the proof, not this timing.
    assert elapsed < 5.0, f"cancel took {elapsed:.1f}s - did not confirm promptly"

    assert be._runner.is_alive(), (
        "the worker process was killed by a mid-stream cancel - cooperative "
        "cancellation regressed to the old kill-based behavior")
    assert be._runner._proc.pid == pid_before, (
        "a new worker process was spawned after cancellation - the cancel "
        "fell back to a full kill+respawn instead of confirming cooperatively")
    assert be.loaded, "the backend must still report loaded with a live worker"

    # A fresh generation on the SAME loaded worker must run, not need a reload.
    out = "".join(be.chat_stream(
        [{"role": "user", "content": "Say hi."}],
        max_tokens=8, temperature=0.0,
    ))
    assert out.strip(), "follow-up generation produced nothing"
    assert be._runner._proc.pid == pid_before, (
        "the follow-up request was served by a different process")


def test_midstream_close_does_not_disable_a_later_generation_after_close(hf_backend):
    """A second, independent cancel/close on the same backend must behave
    the same way - the cancel path is repeatable, not a one-shot fix-up that
    only works the first time (e.g. a stale cancel_event that never gets
    cleared would fire immediately on the NEXT stream and cut it to zero
    tokens)."""
    be = hf_backend
    pid_before = be._runner._proc.pid

    gen = be.chat_stream(
        [{"role": "user", "content": "Tell me a long story."}],
        max_tokens=500, temperature=0.0,
    )
    next(gen)
    gen.close()
    assert be._runner.is_alive()
    assert be._runner._proc.pid == pid_before

    # The NEXT stream must not be pre-empted by a leftover cancel signal.
    out = list(be.chat_stream(
        [{"role": "user", "content": "Say a short word."}],
        max_tokens=10, temperature=0.0,
    ))
    assert len(out) > 0, (
        "a stale cancel_event from the PRIOR stream fired immediately on "
        "this one - it was not cleared at the top of the new chat_stream "
        "dispatch")

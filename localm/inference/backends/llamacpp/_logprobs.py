# SPDX-License-Identifier: AGPL-3.0-or-later
"""Log probabilities of sampled tokens, read from the context's logits.

:class:`LogprobScorer` measures, for the logits row a token was just sampled
from, that token's log probability and the ``n_top`` most likely tokens with
theirs. The distribution is the model's own: a softmax over the raw logits,
before temperature, top-k/top-p/min-p, penalties or a grammar.

The softmax and the top-N selection run natively. The row is copied into a
reusable candidate array and a private ``dist`` + ``top_k`` sampler chain is
applied to it: ``dist`` fills every candidate's normalised ``p`` and ``top_k``
moves the best candidates to the front. The log normaliser follows from the
best candidate, ``logit - ln(p)``, so no Python loop runs over the vocabulary.
"""

from __future__ import annotations

import array
import ctypes
import math
import threading
from typing import Any, Callable, Iterator, Optional

from ._structs import LlamaTokenData, LlamaTokenDataArray

# The most alternatives one token may report (OpenAI's ``top_logprobs`` cap).
MAX_TOP_LOGPROBS = 20
# The logprob reported for a token whose probability underflows to zero.
FLOOR_LOGPROB = -9999.0

_TEMPLATES: dict[int, Any] = {}
_TEMPLATES_LOCK = threading.Lock()


class ScoredToken(int):
    """A sampled token id with its ``logprob`` and ``top``, the most likely
    tokens of the same row as ``(token_id, logprob)`` pairs, best first."""

    logprob: float
    top: tuple

    def __new__(cls, token: int, logprob: float, top: tuple) -> ScoredToken:
        obj = super().__new__(cls, token)
        obj.logprob = logprob
        obj.top = top
        return obj


def clamp_logprob(value: float) -> float:
    """*value* as a reportable log probability: at most 0, and
    :data:`FLOOR_LOGPROB` when it is not finite or lies below the floor."""
    if not math.isfinite(value) or value < FLOOR_LOGPROB:
        return FLOOR_LOGPROB
    return min(value, 0.0)


def _id_template(n_vocab: int):
    """A candidate array of *n_vocab* entries whose ids run 0..n_vocab-1, shared
    read-only by every scorer of that vocabulary size."""
    with _TEMPLATES_LOCK:
        template = _TEMPLATES.get(n_vocab)
        if template is None:
            template = (LlamaTokenData * n_vocab)()
            ids = memoryview(template).cast("B").cast("i")
            ids[0::3] = memoryview(array.array("i", range(n_vocab)))
            _TEMPLATES[n_vocab] = template
        return template


class LogprobScorer:
    """Scores sampled tokens against their logits row, for one generation.

    *api* is the bound llama.cpp module (``_api``), *n_vocab* the model's
    vocabulary size and *n_top* how many alternatives each token reports
    (0 to :data:`MAX_TOP_LOGPROBS`). Holds a native sampler chain: call
    :meth:`close` when the generation ends. Not thread-safe.

    A row holding a NaN or +inf logit has no probabilities; its token is
    reported at :data:`FLOOR_LOGPROB` with no alternatives, and the first such
    row of a scorer is logged at WARNING."""

    def __init__(self, api: Any, n_vocab: int, n_top: int) -> None:
        if not 0 <= n_top <= MAX_TOP_LOGPROBS:
            raise ValueError(f"n_top must be 0..{MAX_TOP_LOGPROBS}, got {n_top}")
        if n_vocab < 1:
            raise ValueError(f"n_vocab must be positive, got {n_vocab}")
        self._api = api
        self._n_vocab = n_vocab
        self._n_top = n_top
        self._k = max(1, n_top)
        self._warned = False
        self._chain: Optional[Any] = None
        params = api.llama_sampler_chain_default_params()
        params.no_perf = True
        chain = api.llama_sampler_chain_init(params)
        self._chain = chain
        try:
            api.llama_sampler_chain_add(chain, api.llama_sampler_init_dist(0))
            api.llama_sampler_chain_add(chain, api.llama_sampler_init_top_k(self._k))
            self._template = _id_template(n_vocab)
            self._cand = (LlamaTokenData * n_vocab)()
            self._cand_ptr = ctypes.cast(self._cand, ctypes.POINTER(LlamaTokenData))
            self._cand_floats = memoryview(self._cand).cast("B").cast("f")
            self._nbytes = ctypes.sizeof(self._cand)
            self._row_t = ctypes.c_float * n_vocab
            self._array = LlamaTokenDataArray()
            self._array_ref = ctypes.byref(self._array)
        except BaseException:
            self.close()
            raise

    @property
    def n_top(self) -> int:
        return self._n_top

    def score(self, ctx: Any, idx: int, token: int) -> ScoredToken:
        """*token*, sampled from output row *idx* of *ctx*, with its log
        probability and the row's ``n_top`` most likely tokens. Call before the
        next decode on *ctx* overwrites the row."""
        chain = self._chain
        if chain is None:
            raise RuntimeError("the logprob scorer is closed")
        api = self._api
        row = api.llama_get_logits_ith(ctx, idx)
        address = ctypes.cast(row, ctypes.c_void_p).value
        if not address:
            raise RuntimeError(f"llama.cpp returned no logits for output row {idx}")
        logits = self._row_t.from_address(address)
        ctypes.memmove(self._cand, self._template, self._nbytes)
        self._cand_floats[1::3] = memoryview(logits).cast("B").cast("f")
        arr = self._array
        arr.data = self._cand_ptr
        arr.size = self._n_vocab
        arr.selected = -1
        arr.sorted = False
        api.llama_sampler_apply(chain, self._array_ref)
        size = min(int(arr.size), self._k)
        data_address = ctypes.cast(arr.data, ctypes.c_void_p).value
        if size < 1 or not data_address:
            raise RuntimeError("the logprob sampler chain left no candidates")
        view = (LlamaTokenData * size).from_address(data_address)
        best = sorted(((float(view[i].logit), float(view[i].p), int(view[i].id))
                       for i in range(size)), key=lambda c: c[0], reverse=True)
        top_logit, top_p, _ = best[0]
        if not (top_p > 0.0 and math.isfinite(top_p) and math.isfinite(top_logit)):
            if not self._warned:
                self._warned = True
                from localm.debuglog import logger
                logger.warning(
                    "logprobs: output row %d holds a logit that is not a finite number "
                    "(best logit %r, p %r); its token is reported at %s with no "
                    "alternatives, as is any later such row of this reply",
                    idx, top_logit, top_p, FLOOR_LOGPROB)
            return ScoredToken(int(token), FLOOR_LOGPROB, ())
        lse = top_logit - math.log(top_p)
        logprob = clamp_logprob(float(logits[int(token)]) - lse)
        top = tuple((tid, clamp_logprob(logit - lse))
                    for logit, _p, tid in best[:self._n_top])
        return ScoredToken(int(token), logprob, top)

    def close(self) -> None:
        """Free the native sampler chain. Idempotent."""
        chain, self._chain = self._chain, None
        if chain is not None:
            self._api.llama_sampler_free(chain)


def tap_records(tokens: Iterator[int], piece: Callable[[int], bytes],
                sink: list) -> Iterator[int]:
    """Pass *tokens* through, appending ``(token_bytes, logprob,
    ((alt_bytes, alt_logprob), ...))`` to *sink* for each one before it is
    yielded. *piece* maps a token id to its bytes. Raises RuntimeError for a
    token that is not a :class:`ScoredToken`."""
    for token in tokens:
        if not isinstance(token, ScoredToken):
            raise RuntimeError("logprobs were requested but this generation path "
                               "did not score its tokens")
        sink.append((piece(token), token.logprob,
                     tuple((piece(tid), lp) for tid, lp in token.top)))
        yield token

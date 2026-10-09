# SPDX-License-Identifier: AGPL-3.0-or-later
"""Query / document pair prompts for reranker models.

A reranker scores one (query, document) pair per sequence. The prompt is built
the way llama.cpp's ``llama-server`` builds it for ``/v1/rerank``:

- A model whose GGUF carries a ``tokenizer.chat_template.rerank`` entry (the
  Qwen3 rerankers) gets that template with ``{query}`` and ``{document}``
  replaced, tokenised with special tokens parsed and none added.
- Any other model (BERT and XLM-R cross-encoders) gets the explicit token
  sequence ``[BOS] query [EOS] [SEP] document [EOS]``, each marker present only
  when the vocabulary asks for it, with query and document tokenised as plain
  text.

A pair that does not fit the context window is cut in the DOCUMENT; a query that
leaves no room for any document is refused.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Optional

from localm.inference.backends.base import RerankInputError

# (text, add_special, parse_special) -> token ids.
Tokenize = Callable[[str, bool, bool], list[int]]

# A document is cut to this many characters per window token before it is
# tokenised, so a document far larger than the window is never tokenised whole.
_CHARS_PER_WINDOW_TOKEN = 32

_TEMPLATE_FIELD_RE = re.compile(r"\{query\}|\{document\}")
_TEMPLATE_SPLIT_RE = re.compile(r"(\{query\}|\{document\})")


@dataclass(frozen=True)
class VocabSpecials:
    """The special tokens and add-flags a vocabulary declares."""
    bos: int
    eos: int
    sep: int
    add_bos: bool
    add_eos: bool
    add_sep: bool

    @classmethod
    def read(cls, api, vocab) -> VocabSpecials:
        """Read the specials of *vocab* through the native binding *api*."""
        return cls(
            bos=int(api.llama_vocab_bos(vocab)),
            eos=int(api.llama_vocab_eos(vocab)),
            sep=int(api.llama_vocab_sep(vocab)),
            add_bos=bool(api.llama_vocab_get_add_bos(vocab)),
            add_eos=bool(api.llama_vocab_get_add_eos(vocab)),
            add_sep=bool(api.llama_vocab_get_add_sep(vocab)))


@dataclass(frozen=True)
class PairTokens:
    """One tokenised pair and whether its document was cut to fit."""
    tokens: list[int]
    truncated: bool


def fill_template(template: str, query: str, document: str) -> str:
    """*template* with ``{query}`` and ``{document}`` replaced in a single pass,
    so a query that itself contains ``{document}`` is left as written."""
    return _TEMPLATE_FIELD_RE.sub(
        lambda m: query if m.group(0) == "{query}" else document, template)


def build_pair(tokenize: Tokenize, specials: VocabSpecials,
               template: Optional[str], window: int,
               query: str, document: str) -> PairTokens:
    """Tokenise one (query, document) pair into at most *window* tokens.

    *template* is the model's ``rerank`` chat template, or None for the
    explicit special-token layout. Raises :class:`RerankInputError` when the
    query alone leaves no room for the document."""
    if len(query) > window * _CHARS_PER_WINDOW_TOKEN:
        raise RerankInputError(
            f"the query is {len(query)} characters, far beyond this model's "
            f"{window}-token window; shorten the query")
    if template is not None:
        return _build_templated(tokenize, template, window, query, document)
    return _build_plain(tokenize, specials, window, query, document)


def _clip_chars(document: str, window: int) -> str:
    return document[: window * _CHARS_PER_WINDOW_TOKEN]


def _query_too_long(n_tokens: int, window: int) -> RerankInputError:
    return RerankInputError(
        f"the query takes {n_tokens} tokens, which leaves no room for a "
        f"document in this model's {window}-token window; shorten the query")


def _build_plain(tokenize: Tokenize, sp: VocabSpecials, window: int,
                 query: str, document: str) -> PairTokens:
    eos = sp.eos if sp.eos >= 0 else sp.sep
    head: list[int] = []
    if sp.add_bos and sp.bos >= 0:
        head.append(sp.bos)
    head.extend(tokenize(query, False, False))
    if sp.add_eos and eos >= 0:
        head.append(eos)
    if sp.add_sep and sp.sep >= 0:
        head.append(sp.sep)
    tail = [eos] if sp.add_eos and eos >= 0 else []
    room = window - len(head) - len(tail)
    if room < 1:
        raise _query_too_long(len(head) + len(tail), window)
    clipped = _clip_chars(document, window)
    body = tokenize(clipped, False, False)
    truncated = len(clipped) < len(document)
    if len(body) > room:
        body = body[:room]
        truncated = True
    return PairTokens(head + body + tail, truncated)


def _special_free(tokenize: Tokenize, text: str) -> bool:
    """True when *text* tokenises the same whether special-token strings are
    parsed or not, i.e. it contains none."""
    return not text or tokenize(text, False, True) == tokenize(text, False, False)


def _build_templated(tokenize: Tokenize, template: str, window: int,
                     query: str, document: str) -> PairTokens:
    clipped = _clip_chars(document, window)
    pre_cut = len(clipped) < len(document)
    if not (_special_free(tokenize, query) and _special_free(tokenize, clipped)):
        return _build_templated_isolated(tokenize, template, window, query,
                                         clipped, pre_cut)
    tokens = tokenize(fill_template(template, query, clipped), False, True)
    if len(tokens) <= window:
        return PairTokens(tokens, pre_cut)
    base = tokenize(fill_template(template, query, ""), False, True)
    if len(base) > window:
        raise _query_too_long(len(base), window)
    fits, over = 0, len(clipped)
    while over - fits > 1:
        mid = (fits + over) // 2
        probe = tokenize(fill_template(template, query, clipped[:mid]), False, True)
        if len(probe) <= window:
            fits = mid
        else:
            over = mid
    if fits == 0 and clipped:
        raise _query_too_long(len(base), window)
    return PairTokens(
        tokenize(fill_template(template, query, clipped[:fits]), False, True), True)


def _build_templated_isolated(tokenize: Tokenize, template: str, window: int,
                              query: str, clipped: str, pre_cut: bool) -> PairTokens:
    """A templated pair whose query or document contains a special-token string
    (``<|im_end|>`` and the like): the template's own text is tokenised with
    special tokens parsed, the query and document as plain text, so they cannot
    produce control tokens that steer the score."""
    parts = []
    for part in _TEMPLATE_SPLIT_RE.split(template):
        if part == "{query}":
            parts.append(tokenize(query, False, False))
        elif part == "{document}":
            parts.append(None)
        elif part:
            parts.append(tokenize(part, False, True))
    fixed = sum(len(p) for p in parts if p is not None)
    n_document = sum(1 for p in parts if p is None)
    if n_document == 0:
        return PairTokens([t for p in parts for t in p], pre_cut)
    room = (window - fixed) // n_document
    if room < 1:
        raise _query_too_long(fixed, window)
    body = tokenize(clipped, False, False)
    truncated = pre_cut or len(body) > room
    body = body[:room]
    return PairTokens([t for p in parts for t in (body if p is None else p)], truncated)

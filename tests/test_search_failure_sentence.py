# SPDX-License-Identifier: AGPL-3.0-or-later
"""A failed search's sentence ends with exactly one period, whether or not the
cause it carries already ends with one."""

from __future__ import annotations

import pytest

from localm import netpolicy
from localm.web_retrieval.retrieve import search_failure_text


class _Provider:
    def __init__(self, name):
        self.name = name
        self.endpoint = "https://search.example/q"


@pytest.mark.parametrize("name", ["duckduckgo", "searxng"])
@pytest.mark.parametrize("message", [
    "the reply is larger than 2000000 bytes; refusing to read it.",
    "the reply is larger than 2000000 bytes; refusing to read it",
])
def test_the_sentence_ends_with_one_period(name, message):
    text = search_failure_text(netpolicy.NetworkPolicyError(message), _Provider(name))
    assert text.endswith("refusing to read it.")
    assert not text.endswith("..")

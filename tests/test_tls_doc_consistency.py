# SPDX-License-Identifier: AGPL-3.0-or-later
"""Bind docs/tls.md's CA-lifecycle claims to localm/tls.py.

``_leaf_is_reusable`` forces a regenerate on a CA that has reached its own
expiry, so the doc must not promise that "The CA is reused even then, so a
device you trusted once stays trusted".

These tests read the CONSTANTS out of localm.tls rather than restating the
numbers, so changing a validity window in the code fails here.
"""

import re
from pathlib import Path

from localm import tls

_DOC = Path(__file__).resolve().parents[1] / "docs" / "tls.md"


def _doc() -> str:
    """The doc with every run of whitespace collapsed to one space.

    The prose is hard-wrapped at ~80 columns, so a sentence-level assertion
    against the raw text would be decided by where the line break falls: a
    claim wrapped as "The CA is\\n  reused even then" contains no matching
    substring.
    """
    return re.sub(r"\s+", " ", _DOC.read_text(encoding="utf-8"))


def test_renew_margin_stated_in_the_doc_matches_the_code():
    """The "regenerated as it nears expiry (about N days before)" number is
    _RENEW_MARGIN_DAYS, not a number someone typed once."""
    stated = re.search(r"about (\d+) days before", _doc())
    assert stated, "docs/tls.md no longer states the renewal margin in days"
    assert int(stated.group(1)) == tls._RENEW_MARGIN_DAYS


def test_ca_lifetime_stated_in_the_doc_matches_the_code():
    """Same for the CA's own validity window, which is what decides when a
    device has to trust a new CA."""
    stated = re.search(r"issued for about (\d+) years", _doc())
    assert stated, "docs/tls.md no longer states the CA lifetime in years"
    assert int(stated.group(1)) == round(tls._CA_DAYS / 365)


def test_doc_does_not_promise_the_ca_is_always_reused():
    """Asserts the ABSENCE of the retired claim, rather than the presence of a
    particular rewording.
    """
    assert "The CA is reused even then" not in _doc()


def test_doc_states_the_ca_replacement_consequence():
    """A replaced CA means every device repeats the trust step. That is the
    user-visible half, and it is the half the old wording denied."""
    doc = _doc()
    assert "replaced only when it cannot be reused" in doc
    assert "repeat the trust step" in doc


def test_doc_documents_deleting_the_tls_dir_as_recovery():
    """The docs name deleting <LOCALM_HOME>/tls/ as the recovery step."""
    doc = _doc()
    assert "<LOCALM_HOME>/tls/" in doc
    assert re.search(r"delete\s+`?<LOCALM_HOME>/tls/`?", doc)

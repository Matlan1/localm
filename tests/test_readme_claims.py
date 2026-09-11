# SPDX-License-Identifier: AGPL-3.0-or-later
"""Binds README.md/docs/network.md/docs/privacy.md claims to the code and to
each other, mirroring tests/test_tls_doc_consistency.py's whitespace-collapse
approach for hard-wrapped prose and tests/test_readme_license.py's real-file
reads.

Three overclaims this pins:

1. README.md said Multi-Token Prediction "just engages when it can" - it is
   off by default (config.DEFAULT_CONFIG["mtp_enabled"]) and needs
   `localm bench-mtp` plus a Settings toggle.
2. README.md and docs/privacy.md both said "every [other] outbound request"
   goes through the network policy - several fixed-destination requests
   (bug-report upload, ComfyUI, embedding/Whisper downloads, the update
   check) deliberately do not call netpolicy.check_url at all.
3. docs/privacy.md's privacy-mode table implied privacy mode writes
   NOTHING - four kinds of explicit user data (scheduled job results,
   RAG collections, the prompt library, GUI uploads) are written in every
   session mode, privacy included.
"""

import re
from pathlib import Path

from localm.config import DEFAULT_CONFIG

_ROOT = Path(__file__).resolve().parents[1]
_README = _ROOT / "README.md"
_NETWORK_DOC = _ROOT / "docs" / "network.md"
_PRIVACY_DOC = _ROOT / "docs" / "privacy.md"


def _collapsed(path: Path) -> str:
    """The doc with every run of whitespace collapsed to one space, so a
    hard-wrapped sentence assertion is not decided by where the line breaks."""
    return re.sub(r"\s+", " ", path.read_text(encoding="utf-8"))


def _raw(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
#  1. Multi-Token Prediction is off by default                                #
# --------------------------------------------------------------------------- #

def test_mtp_is_off_by_default_in_code():
    """The claim this test binds the README to."""
    assert DEFAULT_CONFIG["mtp_enabled"] is False


def test_readme_does_not_claim_mtp_just_engages_with_no_setup():
    assert "it just engages when it can" not in _raw(_README)


def test_readme_mtp_claim_says_off_by_default_and_points_at_bench_mtp():
    doc = _collapsed(_README)
    idx = doc.find("Multi-Token Prediction")
    assert idx != -1, "README no longer mentions Multi-Token Prediction at all"
    window = doc[idx:idx + 400]
    assert "off by default" in window
    assert "bench-mtp" in window


# --------------------------------------------------------------------------- #
#  2. The network policy does not cover every outbound request                #
# --------------------------------------------------------------------------- #

def test_readme_does_not_claim_every_outbound_request_is_governed():
    assert "every outbound request" not in _raw(_README)


def test_privacy_doc_does_not_claim_every_other_outbound_request_is_governed():
    assert "every other outbound request" not in _raw(_PRIVACY_DOC)


def test_network_doc_names_the_ungoverned_paths():
    doc = _collapsed(_NETWORK_DOC)
    idx = doc.find("What the policy does NOT govern")
    assert idx != -1, "docs/network.md lost its 'What the policy does NOT govern' section"
    section = doc[idx:idx + 2000]
    assert "bug-report" in section.lower()
    assert "comfyui" in section.lower()


# --------------------------------------------------------------------------- #
#  3. Privacy mode does not stop every write to disk                          #
# --------------------------------------------------------------------------- #

def test_privacy_doc_documents_what_it_does_not_cover():
    """Searches for the ### HEADING specifically, not just the phrase: the
    mode table's row for "privacy" forward-references this section by name
    ("see 'What privacy mode does not cover' below"), so a bare substring
    search finds that cross-reference first and reads a 1500-char window
    starting inside the table instead of the section it points at."""
    doc = _collapsed(_PRIVACY_DOC)
    idx = doc.find("### What privacy mode does not cover")
    assert idx != -1, (
        "docs/privacy.md lost its 'What privacy mode does not cover' section")
    section = doc[idx:idx + 1500]
    assert "jobs/results" in section
    assert "rag" in section.lower()
    assert "prompts.json" in section
    assert "uploads" in section.lower()

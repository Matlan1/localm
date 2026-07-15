# SPDX-License-Identifier: AGPL-3.0-or-later
"""REG-460: safe .env config-doc TEMPLATES must be indexable; real .env* must not.

is_secret_index_name() blanket-matched `low.startswith(".env.")`, so a folder
index silently dropped .env.example / .env.template / .env.sample - the
by-convention, committed-to-git, placeholder-only documentation of which config
vars a project needs. That is genuinely useful context for a RAG index over a
repo, and it was lost with no warning.

The fix is an ALLOWLIST, not a loosened prefix match, because the safe/unsafe
split here is by NAME convention and is fail-safe only in one direction:

  SAFE (committed by convention, placeholders only):
      .env.example  .env.template  .env.sample  .env.dist
  SECRET (gitignored by convention, holds REAL values):
      .env  .env.local  .env.production  .env.development  .env.test
      .env.production.local  ... and anything else .env.*

REG-460 as filed also asked for `.env.local` to be treated as safe. That is
WRONG and is deliberately NOT implemented: `.env.local` is the canonical name
for LOCAL SECRET OVERRIDES. The upstream github/gitignore Node.gitignore is
explicit - it ignores `.env` and `.env.*` and un-ignores ONLY `!.env.example` -
and the Next.js docs describe `.env.local` as the file for values "you don't
want to commit, like sensitive credentials". Un-blocking it would index real
credentials into a searchable, model-visible store: the exact AUDIT-MED-18 leak
this filter exists to prevent. An allowlist keeps any name we have not
positively vetted on the SECRET side.
"""

from __future__ import annotations

import pytest

from localm.rag.extract import is_secret_index_name

SAFE_TEMPLATES = [
    ".env.example",
    ".env.template",
    ".env.sample",
    ".env.dist",
]

# Real config files that carry real values. `.env.local` is in here on purpose.
SECRET_ENV_FILES = [
    ".env",
    ".env.local",
    ".env.production",
    ".env.development",
    ".env.test",
    ".env.production.local",
    ".env.development.local",
]


class TestSafeEnvTemplatesAreIndexable:
    @pytest.mark.parametrize("name", SAFE_TEMPLATES)
    def test_template_is_not_treated_as_secret(self, name):
        assert is_secret_index_name(name) is False

    @pytest.mark.parametrize("name", [n.upper() for n in SAFE_TEMPLATES])
    def test_template_match_is_case_insensitive(self, name):
        assert is_secret_index_name(name) is False


class TestRealEnvFilesStayBlocked:
    """NEGATIVE CASE. Without these, 'fixing' REG-460 by weakening the prefix
    match to `not startswith('.env.')` would pass the tests above while
    re-opening the AUDIT-MED-18 secret leak."""

    @pytest.mark.parametrize("name", SECRET_ENV_FILES)
    def test_real_env_file_is_still_secret(self, name):
        assert is_secret_index_name(name) is True

    @pytest.mark.parametrize("name", [n.upper() for n in SECRET_ENV_FILES])
    def test_real_env_file_match_is_case_insensitive(self, name):
        assert is_secret_index_name(name) is True

    def test_an_unvetted_env_variant_defaults_to_secret(self):
        # Fail-safe: a name nobody has vetted is NOT waved through.
        assert is_secret_index_name(".env.some-new-thing") is True

    def test_env_local_is_secret_even_though_reg460_asked_otherwise(self):
        # Guards the deliberate deviation from the filed report, so a future
        # reader does not "fix" it back. See this module's docstring.
        assert is_secret_index_name(".env.local") is True

    @pytest.mark.parametrize("name", ["id_rsa", ".netrc", ".envrc", ".pgpass"])
    def test_other_secret_names_unaffected(self, name):
        assert is_secret_index_name(name) is True

    @pytest.mark.parametrize("name", ["notes.txt", "README.md", "env.py",
                                      "environment.yml"])
    def test_ordinary_files_unaffected(self, name):
        assert is_secret_index_name(name) is False

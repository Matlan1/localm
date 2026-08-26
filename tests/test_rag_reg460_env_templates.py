# SPDX-License-Identifier: AGPL-3.0-or-later
"""Safe .env config-doc TEMPLATES must be indexable; real .env* must not.

is_secret_index_name() uses an ALLOWLIST rather than a prefix match, because the
safe/unsafe split here is by NAME convention and is fail-safe in one direction
only:

  SAFE (committed by convention, placeholders only):
      .env.example  .env.template  .env.sample  .env.dist
  SECRET (gitignored by convention, holds REAL values):
      .env  .env.local  .env.production  .env.development  .env.test
      .env.production.local  ... and anything else .env.*

`.env.local` sits on the SECRET side: it is the canonical name for local secret
overrides. Any name that has not been positively vetted stays on the SECRET side.
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

# Real config files that carry real values, including `.env.local`.
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
    """Real .env files, which hold real values, stay blocked."""

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
        assert is_secret_index_name(".env.local") is True

    @pytest.mark.parametrize("name", ["id_rsa", ".netrc", ".envrc", ".pgpass"])
    def test_other_secret_names_unaffected(self, name):
        assert is_secret_index_name(name) is True

    @pytest.mark.parametrize("name", ["notes.txt", "README.md", "env.py",
                                      "environment.yml"])
    def test_ordinary_files_unaffected(self, name):
        assert is_secret_index_name(name) is False

# SPDX-License-Identifier: AGPL-3.0-or-later
"""Allow ``python -m localm`` as an alternative to the ``localm`` script."""

from localm._venvguard import require_venv

if __name__ == "__main__":
    require_venv()

from localm.cli import main  # noqa: E402  (imports click; must follow the guard)

if __name__ == "__main__":
    main()

# SPDX-License-Identifier: AGPL-3.0-or-later
"""Allow ``python -m localm`` as an alternative to the ``localm`` script."""

from localm._venvguard import require_venv
from localm.cli import main

if __name__ == "__main__":
    require_venv()
    main()

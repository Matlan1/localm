# SPDX-License-Identifier: AGPL-3.0-or-later
"""Allow ``python -m localm`` as an alternative to the ``localm`` script."""

from localm._srcguard import require_own_source
from localm._venvguard import require_venv
from localm.cli import main

if __name__ == "__main__":
    # ``-m`` does put cwd on sys.path, so this normally resolves to the checkout
    # the caller is standing in and the guard is silent. It still earns its place
    # for the subdirectory case: from ``<checkout>/docs`` the cwd entry holds no
    # localm package, so resolution falls through to the editable install's own
    # checkout while the caller believes they are running the tree they are in.
    require_own_source()
    require_venv()
    main()

# SPDX-License-Identifier: AGPL-3.0-or-later
"""Allow ``python -m localm`` as an alternative to the ``localm`` script."""

import sys
from localm.cli import main

if __name__ == "__main__":
    if sys.prefix == sys.base_prefix:
        sys.exit("Error: localm must be run from its virtual environment. Use '.venv\\Scripts\\localm' or activate the venv first.")
    main()

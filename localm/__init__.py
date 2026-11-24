# SPDX-License-Identifier: AGPL-3.0-or-later
"""The first act of importing localm is to record this process (:mod:`localm.procs`).
Reading the record of an update comes after it and before anything else of localm: an
updater writes its record and then lists the processes, so one side sees the other."""
__version__ = "0.3.4b4"

from localm import procs as _procs

_procs.enter()

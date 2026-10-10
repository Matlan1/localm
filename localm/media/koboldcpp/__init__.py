# SPDX-License-Identifier: AGPL-3.0-or-later
"""Native music generation (ACE-Step 1.5) through a managed KoboldCpp runtime.

``runtime`` installs and locates the pinned KoboldCpp release, ``server`` runs it
as a loopback-only subprocess for one model set, ``models`` resolves and pulls the
ACE-Step model files, ``_proc`` ties the process to localm's lifetime, and ``cli``
provides ``localm setup-music``.
"""

# SPDX-License-Identifier: AGPL-3.0-or-later
"""Browser automation for localm: a headless Chromium the model can drive.

``netgate`` decides whether one request the browser is about to make may
proceed. Every decision defers to ``localm.netpolicy``, which stays the single
authority on network reach; this package never keeps a second allowlist.
"""

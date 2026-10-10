# SPDX-License-Identifier: AGPL-3.0-or-later
"""Native image (and video) generation through stable-diffusion.cpp.

``runtime`` installs and locates the upstream prebuilt runtime, ``runner``
drives it in an isolated worker process, ``_binding`` and ``_child`` run only
inside that worker, and ``cli`` provides ``localm setup-sdcpp``.
"""

# SPDX-License-Identifier: AGPL-3.0-or-later
"""
localm.media - shared media-generation plumbing.

The generic ComfyUI client (server reachability, model preflight, VRAM
handoff, queue / poll / download transport, output containment) lives in
``localm.media.comfy_client`` and is used by the image, video, and music
generators. It carries no medium-specific workflow logic: each generator keeps
its own workflow shaping and calls these shared helpers for transport.
"""

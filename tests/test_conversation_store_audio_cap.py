# SPDX-License-Identifier: AGPL-3.0-or-later
"""The conversation store accepts a conversation holding the largest audio clip
the composer allows."""

from localm.plugins.builtin.chat import plug

AUDIO_MAX_BYTES = 50_000_000


def test_cap_fits_the_largest_attachable_clip_as_base64():
    base64_len = (AUDIO_MAX_BYTES + 2) // 3 * 4
    assert plug._CONV_MAX_BYTES > base64_len + 1_000_000

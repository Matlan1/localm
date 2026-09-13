# SPDX-License-Identifier: AGPL-3.0-or-later
"""Regenerate the metadata-bearing media fixtures next to this script.

Each file is written with PyAV the way ComfyUI's save nodes write it
(``AudioSaveHelper.save_audio`` for FLAC/MP3/Opus, ``VideoFromComponents.save_to``
for MP4 with ``movflags=use_metadata_tags+faststart``): container-level
``prompt`` and ``workflow`` tags holding a JSON workflow whose lyrics and tags
contain ``MARKER``. The strip tests assert that marker is gone afterwards.

Run only when regenerating: ``python tests/fixtures/media/gen_prompt_fixtures.py``
(needs ``av`` and ``numpy``, which the test suite itself does not).
"""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import av
import numpy as np

HERE = Path(__file__).parent
MARKER = "SECRET-LYRICS-MARKER-7Q4M"
PROMPT = {
    "3": {"class_type": "TextEncodeAceStepAudio",
          "inputs": {"tags": "synthwave, 80s", "lyrics": f"[verse] {MARKER} ohh ohh"}},
    "8": {"class_type": "SaveAudio", "inputs": {"filename_prefix": "audio/localm"}},
}
TAGS = {
    "prompt": json.dumps(PROMPT),
    "workflow": json.dumps({"nodes": [{"id": 3, "widgets_values": [f"tags {MARKER}"]}]}),
}


def audio(fmt: str, codec: str) -> bytes:
    rate = 8000
    t = np.arange(rate // 4, dtype=np.float32) / rate
    wave = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32).reshape(1, -1)
    buf = BytesIO()
    container = av.open(buf, mode="w", format=fmt)
    for key, value in TAGS.items():
        container.metadata[key] = value
    stream = container.add_stream(codec, rate=rate, layout="mono")
    frame = av.AudioFrame.from_ndarray(wave, format="flt", layout="mono")
    frame.sample_rate = rate
    frame.pts = 0
    container.mux(stream.encode(frame))
    container.mux(stream.encode(None))
    container.close()
    buf.seek(0)
    return buf.getbuffer().tobytes()


def video(path: Path) -> None:
    container = av.open(str(path), mode="w", format="mp4",
                        options={"movflags": "use_metadata_tags+faststart"})
    for key, value in TAGS.items():
        container.metadata[key] = value
    stream = container.add_stream("h264", rate=24)
    stream.width, stream.height, stream.pix_fmt = 32, 32, "yuv420p"
    for i in range(4):
        image = np.full((32, 32, 3), (i * 60) % 255, dtype=np.uint8)
        for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
            container.mux(packet)
    for packet in stream.encode(None):
        container.mux(packet)
    container.close()


if __name__ == "__main__":
    (HERE / "prompt.flac").write_bytes(audio("flac", "flac"))
    (HERE / "prompt.mp3").write_bytes(audio("mp3", "libmp3lame"))
    (HERE / "prompt.opus").write_bytes(audio("opus", "libopus"))
    video(HERE / "prompt.mp4")
    for p in sorted(HERE.glob("prompt.*")):
        data = p.read_bytes()
        print(f"{p.name}: {len(data)} bytes, marker x{data.count(MARKER.encode())}")

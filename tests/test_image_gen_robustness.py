# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for image generation robustness fixes (comfy.py + coder tool)."""

import struct
import zlib
from unittest.mock import MagicMock, patch


from localm.image_gen import comfy


def _minimal_png(width: int, height: int) -> bytes:
    """Build a tiny valid PNG header with the given dimensions."""
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    ihdr = (struct.pack(">I", len(ihdr_data)) + b"IHDR" + ihdr_data
            + struct.pack(">I", zlib.crc32(b"IHDR" + ihdr_data)))
    return sig + ihdr


class TestImageDimensions:
    def test_png_dimensions_read_correctly(self, tmp_path):
        """Regression: Path.read_bytes(32) raised TypeError and silently
        forced every img2img run to 1024x1024."""
        p = tmp_path / "img.png"
        p.write_bytes(_minimal_png(640, 480))
        assert comfy._image_dimensions(p) == (640, 480)

    def test_unreadable_file_falls_back(self, tmp_path):
        p = tmp_path / "junk.png"
        p.write_bytes(b"not an image")
        assert comfy._image_dimensions(p) == (1024, 1024)


class TestComfyAlive:
    def test_alive_when_endpoint_responds(self):
        with patch.object(comfy.urllib.request, "urlopen") as mock_open:
            mock_open.return_value.__enter__ = MagicMock()
            mock_open.return_value.__exit__ = MagicMock(return_value=False)
            assert comfy._comfy_alive("http://127.0.0.1:8188") is True

    def test_dead_when_connection_refused(self):
        with patch.object(
            comfy.urllib.request, "urlopen",
            side_effect=OSError("refused"),
        ):
            assert comfy._comfy_alive("http://127.0.0.1:8188") is False


class TestHistoryExecutionError:
    """I2: surface a ComfyUI node crash from /history instead of the generic
    'no output found' that just sends the user to read the ComfyUI console."""

    def test_surfaces_node_crash(self):
        entry = {
            "outputs": {},
            "status": {
                "status_str": "error",
                "completed": False,
                "messages": [
                    ["execution_start", {}],
                    ["execution_error", {
                        "node_type": "SaveAudio",
                        "exception_message":
                            "'function' object has no attribute '__func__'",
                    }],
                ],
            },
        }
        msg = comfy.history_execution_error(entry)
        assert msg is not None
        assert "__func__" in msg
        assert "SaveAudio" in msg

    def test_status_error_without_detail_still_flags(self):
        entry = {"outputs": {}, "status": {"status_str": "error", "messages": []}}
        assert comfy.history_execution_error(entry) is not None

    def test_none_on_success(self):
        ok_entry = {
            "outputs": {"9": {"images": [{"filename": "x.png"}]}},
            "status": {"status_str": "success", "completed": True, "messages": []},
        }
        assert comfy.history_execution_error(ok_entry) is None

    def test_none_when_no_status(self):
        # Older ComfyUI entries without a status block are treated as no-error.
        assert comfy.history_execution_error({"outputs": {}}) is None


class TestFailFastBeforeUnload:
    def test_dead_comfy_errors_without_unloading_llm(self, tmp_path):
        """A dead image server must not cost the user an LLM unload+reload."""
        unload_spy = MagicMock()
        with patch.object(comfy, "_comfy_alive", return_value=False), \
             patch.object(comfy, "_localm_unload", unload_spy):
            ok, msg = comfy.generate_image("a cat", tmp_path / "out.png")
        assert ok is False
        assert "not reachable" in msg
        assert "FLUX_API_URL" in msg          # actionable hint present
        unload_spy.assert_not_called()


class TestToolConfinement:
    def test_output_path_escape_rejected(self, tmp_path):
        from localm.plugins.coder.tools import tool_generate_image
        with patch("localm.image_gen.comfy.generate_image") as gen:
            res = tool_generate_image(
                tmp_path, prompt="x", output_path="../../escape.png")
        assert not res.ok
        gen.assert_not_called()

    def test_input_image_escape_rejected(self, tmp_path):
        from localm.plugins.coder.tools import tool_generate_image
        with patch("localm.image_gen.comfy.generate_image") as gen:
            res = tool_generate_image(
                tmp_path, prompt="x",
                output_path="ok.png",
                input_image="../../../secret.png")
        assert not res.ok
        gen.assert_not_called()

    def test_seed_forwarded(self, tmp_path):
        from localm.plugins.coder.tools import tool_generate_image
        with patch("localm.plugins.coder.tools.os.environ", {"FLUX_API_URL": "http://x"}):
            with patch("localm.image_gen.comfy.generate_image",
                       return_value=(True, "Image saved to out.png (seed 42)")) as gen:
                res = tool_generate_image(tmp_path, prompt="x", seed=42)
        assert res.ok
        assert gen.call_args.kwargs["seed"] == 42


class TestSidecarContent:
    def test_sidecar_written_next_to_output(self, tmp_path):
        """Drive generate_image fully mocked to the save step and verify
        the sidecar JSON lands as <output>.png.json with the seed."""
        import json

        out = tmp_path / "art.png"

        responses = {
            "/system_stats": b"{}",
            # Preflight fetches /object_info; an empty map makes it a graceful
            # no-op (nothing to validate), so the happy path is unaffected.
            "/object_info": b"{}",
            "/prompt": b'{"prompt_id": "p1"}',
            "/history/p1": json.dumps({
                "p1": {"outputs": {"9": {"images": [
                    {"filename": "f.png", "subfolder": "", "type": "output"}
                ]}}}
            }).encode(),
            "/view": _minimal_png(8, 8),
        }

        def fake_urlopen(req, timeout=None):
            url = req if isinstance(req, str) else req.full_url
            for key, body in responses.items():
                if key in url:
                    m = MagicMock()
                    m.read.return_value = body
                    m.__enter__ = lambda s=m: s
                    m.__exit__ = MagicMock(return_value=False)
                    return m
            raise AssertionError(f"unexpected url {url}")

        with patch.object(comfy.urllib.request, "urlopen", side_effect=fake_urlopen), \
             patch.object(comfy, "_localm_unload"):
            ok, msg = comfy.generate_image(
                "a fox", out, seed=1234, guidance=3.0)

        assert ok, msg
        assert "seed 1234" in msg
        sidecar = json.loads((tmp_path / "art.png.json").read_text(encoding="utf-8"))
        assert sidecar["seed"] == 1234
        assert sidecar["prompt"] == "a fox"
        assert sidecar["guidance"] == 3.0

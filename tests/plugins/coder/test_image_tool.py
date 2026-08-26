# SPDX-License-Identifier: AGPL-3.0-or-later
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import json
import urllib.error

# Import the tool function and class
from localm.plugins.coder.tools import tool_generate_image

class TestFluxImageTool(unittest.TestCase):
    def setUp(self):
        self.cwd = Path(__file__).parent
        self.output_path = "test_output.png"
        self.abs_output_path = self.cwd / self.output_path
        # Cleanup output if it exists
        if self.abs_output_path.exists():
            self.abs_output_path.unlink()

    def tearDown(self):
        # Cleanup output if it exists
        if self.abs_output_path.exists():
            self.abs_output_path.unlink()

    # Stub the /object_info preflight query so the positional urlopen sequence
    # below (system_stats -> prompt -> history -> view) is unaffected.
    # preflight_models calls comfy_object_info as a bare global in comfy_client,
    # so the stub targets comfy_client.
    @patch("localm.media.comfy_client.comfy_object_info", return_value=None)
    # comfy_client routes ComfyUI calls through its own _comfy_urlopen, never
    # the top-level urllib.request.urlopen; that is the seam to patch.
    @patch("localm.media.comfy_client._comfy_urlopen")
    @patch("urllib.request.Request")
    def test_generate_image_success(self, mock_request_cls, mock_urlopen, mock_obj_info):
        # 1. Setup the mock responses
        # First call: /prompt endpoint -> return prompt_id
        mock_prompt_response = MagicMock()
        mock_prompt_response.__enter__.return_value = mock_prompt_response
        mock_prompt_response.read.return_value = json.dumps({"prompt_id": "mock_prompt_abc_123"}).encode("utf-8")
        
        # Second call: /history/mock_prompt_abc_123 -> return history outputs
        mock_history_response = MagicMock()
        mock_history_response.__enter__.return_value = mock_history_response
        history_data = {
            "mock_prompt_abc_123": {
                "outputs": {
                    "9": {
                        "images": [
                            {"filename": "flux_mock_image.png", "subfolder": "", "type": "output"}
                        ]
                    }
                }
            }
        }
        mock_history_response.read.return_value = json.dumps(history_data).encode("utf-8")
        
        # Third call: /view endpoint -> return image bytes
        mock_view_response = MagicMock()
        mock_view_response.__enter__.return_value = mock_view_response
        mock_view_response.read.return_value = b"MOCK_PNG_IMAGE_BYTES"

        # Zeroth call: /system_stats reachability probe
        mock_alive_response = MagicMock()
        mock_alive_response.__enter__.return_value = mock_alive_response
        mock_alive_response.read.return_value = b"{}"

        # Make urlopen return the mocks sequentially
        mock_urlopen.side_effect = [
            mock_alive_response,   # for GET /system_stats (fail-fast probe)
            mock_prompt_response,  # for POST /prompt
            mock_history_response, # for GET /history/mock_prompt_abc_123
            mock_view_response     # for GET /view?filename=...
        ]

        # 2. Execute the tool
        result = tool_generate_image(self.cwd, "A photorealistic cat coding", self.output_path)

        # 3. Assertions
        self.assertTrue(result.ok)
        self.assertIn("Image saved to", result.output)
        self.assertIn("seed", result.output)   # reproducibility hint
        self.assertTrue(self.abs_output_path.exists())
        self.assertEqual(self.abs_output_path.read_bytes(), b"MOCK_PNG_IMAGE_BYTES")
        # Sidecar JSON saved next to the image
        sidecar = self.abs_output_path.with_suffix(".png.json")
        self.assertTrue(sidecar.exists())
        sidecar.unlink()

    @patch("localm.image_gen.comfy.generate_image")
    def test_privacy_mode_suppresses_sidecar(self, mock_gen):
        """In privacy mode the prompt sidecar must not be written."""
        mock_gen.return_value = (True, "Image saved to x")
        tool_generate_image(self.cwd, "p", self.output_path, _privacy=True)
        _, kwargs = mock_gen.call_args
        self.assertIs(kwargs.get("write_sidecar"), False)

    @patch("localm.image_gen.comfy.generate_image")
    def test_privacy_mode_deletes_comfy_output_copy(self, mock_gen):
        """Privacy mode also deletes ComfyUI's own on-disk output copy, which
        embeds the full prompt/workflow as PNG metadata, not just the
        sidecar."""
        mock_gen.return_value = (True, "Image saved to x")
        tool_generate_image(self.cwd, "p", self.output_path, _privacy=True)
        _, kwargs = mock_gen.call_args
        self.assertIs(kwargs.get("delete_outputs"), True)

    @patch("localm.image_gen.comfy.generate_image")
    def test_uses_default_api_url_not_a_hardcoded_default(self, mock_gen):
        """The tool resolves its target through default_api_url(), which carries
        the localm-managed-instance routing, rather than reading FLUX_API_URL
        with a hardcoded loopback fallback. default_api_url is mocked to a
        distinct URL and that value is asserted to reach generate_image()."""
        mock_gen.return_value = (True, "Image saved to x")
        with patch("localm.media.comfy_client.default_api_url",
                   return_value="http://127.0.0.1:8189"):
            tool_generate_image(self.cwd, "p", self.output_path)
        _, kwargs = mock_gen.call_args
        self.assertEqual(kwargs.get("api_url"), "http://127.0.0.1:8189")

    @patch("localm.image_gen.comfy.generate_image")
    def test_default_mode_keeps_sidecar(self, mock_gen):
        mock_gen.return_value = (True, "Image saved to x")
        tool_generate_image(self.cwd, "p", self.output_path)
        _, kwargs = mock_gen.call_args
        self.assertIs(kwargs.get("write_sidecar"), True)

    @patch("localm.image_gen.comfy.generate_image")
    def test_default_mode_keeps_output_copy(self, mock_gen):
        mock_gen.return_value = (True, "Image saved to x")
        tool_generate_image(self.cwd, "p", self.output_path)
        _, kwargs = mock_gen.call_args
        self.assertFalse(kwargs.get("delete_outputs"))

    @patch("localm.image_gen.comfy.ensure_comfy", return_value=(True, "up"))
    def test_lora_name_traversal_rejected(self, mock_ensure):
        """This tool calls localm.image_gen.comfy.generate_image DIRECTLY and
        never goes through the HTTP image route's plug.py._validate_lora_name,
        so a malicious lora_name is caught by the same check inside
        _build_image_workflow (comfy.is_safe_lora_name). ensure_comfy is stubbed
        only to reach that check; nothing past it (preflight, submit) runs, which
        the absence of the output file confirms."""
        result = tool_generate_image(
            self.cwd, "a photorealistic cat", self.output_path,
            lora_name="../../secrets.safetensors")
        self.assertFalse(result.ok)
        self.assertIn("Invalid LoRA name", result.output)
        self.assertFalse(self.abs_output_path.exists())

    @patch("localm.media.comfy_client._comfy_urlopen")
    def test_generate_image_connection_failure(self, mock_urlopen):
        # Force a connection refusal
        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

        # Execute the tool
        result = tool_generate_image(self.cwd, "A photorealistic cat coding", self.output_path)

        # Assertions
        self.assertFalse(result.ok)
        self.assertIn("not reachable", result.output)
        self.assertIn("FLUX_API_URL", result.output)   # actionable hint
        self.assertFalse(self.abs_output_path.exists())

def test_agent_injects_privacy_for_generate_image(tmp_path):
    """The coder agent injects _privacy=True for generate_image in privacy
    mode, so the tool suppresses the prompt sidecar."""
    from localm.plugins.coder.agent import Agent
    from localm.audit import SessionMode, NullAuditLog
    from localm.plugins.coder.parser import ToolCall
    from localm.plugins.coder import tools as T

    backend = MagicMock()
    backend.model_id = "test-model"
    backend.last_usage = {}
    with patch("localm.plugins.coder.agent.make_audit_log", return_value=NullAuditLog()), \
         patch("localm.plugins.coder.agent.load_memory", return_value=""), \
         patch("localm.plugins.coder.agent.ProjectMap") as mock_pm:
        mock_pm.build.return_value.file_count.return_value = 0
        agent = Agent(backend=backend, cwd=tmp_path, mode=SessionMode.PRIVACY)

    captured = {}

    def spy(cwd, **kwargs):
        captured.update(kwargs)
        return T.ToolResult.success("ok")

    call = ToolCall(name="generate_image", args={"prompt": "p"}, raw="", start=0, end=0)
    with patch.object(T.TOOL_REGISTRY["generate_image"], "fn", spy):
        agent._execute_tool(call, interactive=False)
    assert captured.get("_privacy") is True


def test_repl_generate_image_privacy_no_sidecar(tmp_path, monkeypatch):
    """REPL /generate-image suppresses the sidecar in privacy mode and passes
    the resolved api_url, not the 8188 default."""
    from localm import cli
    from localm.audit import SessionMode

    monkeypatch.setattr(cli, "HOME_DIR", tmp_path)
    engine = MagicMock()
    calls = {}

    def fake_gen(prompt, out, **kwargs):
        calls.update(kwargs)
        return (True, "ok")

    with patch("localm.image_gen.comfy.generate_image", fake_gen), \
         patch("localm.media.comfy_client._comfy_alive", return_value=True), \
         patch("localm.image_gen.comfy.default_api_url", return_value="http://127.0.0.1:9999"), \
         patch("localm.image_gen.comfy.free_comfy_vram"), \
         patch("localm.audit.effective_mode", return_value=SessionMode.PRIVACY):
        cli._handle_command("/generate-image a cat", [], {}, engine=engine)

    assert calls.get("write_sidecar") is False
    assert calls.get("api_url") == "http://127.0.0.1:9999"
    # Privacy mode deletes ComfyUI's own on-disk output copy, not just the sidecar.
    assert calls.get("delete_outputs") is True


def test_repl_generate_image_logmode_keeps_sidecar(tmp_path, monkeypatch):
    from localm import cli
    from localm.audit import SessionMode

    monkeypatch.setattr(cli, "HOME_DIR", tmp_path)
    engine = MagicMock()
    calls = {}

    def fake_gen(prompt, out, **kwargs):
        calls.update(kwargs)
        return (True, "ok")

    with patch("localm.image_gen.comfy.generate_image", fake_gen), \
         patch("localm.media.comfy_client._comfy_alive", return_value=True), \
         patch("localm.image_gen.comfy.default_api_url", return_value="http://127.0.0.1:8188"), \
         patch("localm.image_gen.comfy.free_comfy_vram"), \
         patch("localm.audit.effective_mode", return_value=SessionMode.LOG):
        cli._handle_command("/generate-image a cat", [], {}, engine=engine)

    assert calls.get("write_sidecar") is True
    assert not calls.get("delete_outputs")


if __name__ == "__main__":
    unittest.main()

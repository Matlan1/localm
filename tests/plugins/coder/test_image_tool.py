import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import json
import urllib.error

# Import the tool function and class
from localm.plugins.coder.tools import tool_generate_image, ToolResult

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

    @patch("urllib.request.urlopen")
    @patch("urllib.request.Request")
    def test_generate_image_success(self, mock_request_cls, mock_urlopen):
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

        # Make urlopen return the three mocks sequentially
        mock_urlopen.side_effect = [
            mock_prompt_response,  # for POST /prompt
            mock_history_response, # for GET /history/mock_prompt_abc_123
            mock_view_response     # for GET /view?filename=...
        ]

        # 2. Execute the tool
        result = tool_generate_image(self.cwd, "A photorealistic cat coding", self.output_path)

        # 3. Assertions
        self.assertTrue(result.ok)
        self.assertIn("Image saved to", result.output)
        self.assertTrue(self.abs_output_path.exists())
        self.assertEqual(self.abs_output_path.read_bytes(), b"MOCK_PNG_IMAGE_BYTES")

    @patch("urllib.request.urlopen")
    def test_generate_image_connection_failure(self, mock_urlopen):
        # Force a connection refusal
        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

        # Execute the tool
        result = tool_generate_image(self.cwd, "A photorealistic cat coding", self.output_path)

        # Assertions
        self.assertFalse(result.ok)
        self.assertIn("Could not connect to ComfyUI", result.output)
        self.assertFalse(self.abs_output_path.exists())

if __name__ == "__main__":
    unittest.main()

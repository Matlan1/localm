# Implementation Plan - FLUX Image Generation Tool for localllm-coder

Add a `generate_image` tool to `localllm-coder` that allows the offline LLM to dynamically generate high-quality images via a local ComfyUI instance running FLUX.

---

## User Review Required

> [!IMPORTANT]
> This tool requires a running local ComfyUI instance (default: `http://127.0.0.1:8188`) with the `ComfyUI-GGUF` node and the FLUX GGUF weights loaded. 
> To generate the image, the tool will submit a preset workflow JSON payload to ComfyUI's `/prompt` endpoint, poll for status, and download the finished image.

---

## Resolved Design Decisions

*   **API Configuration**: The tool will check the `FLUX_API_URL` environment variable, defaulting to `http://127.0.0.1:8188` (local ComfyUI API endpoint).
*   **Missing Server Handling**: Because ComfyUI is not yet installed on your machine, the tool will catch connection errors and output a clear, user-friendly message redirecting to your local setup guide. We will verify the implementation using mock unit tests.

---

## Proposed Changes

### localllm-coder Agent

---

#### [MODIFY] [tools.py](file:///d:/projects/localllm-coder/localcoder/core/tools.py)
*   Implement `tool_generate_image(cwd: Path, prompt: str, output_path: str = "output.png")`:
    *   Load a default ComfyUI workflow JSON.
    *   Inject the `prompt` and output file details.
    *   Perform a `POST` request to `http://127.0.0.1:8188/prompt` to queue the job.
    *   Poll the `/history` endpoint until the prompt is processed.
    *   Download the output image and save it to the specified `output_path` relative to `cwd`.
    *   Return a `ToolResult` containing success status and paths.
*   Register the tool in `TOOL_REGISTRY`.

#### [MODIFY] [prompts.py](file:///d:/projects/localllm-coder/localcoder/prompts.py)
*   Add the `generate_image` documentation to `tool_docs` so the LLM understands the tool schemas and rules.
*   Provide a clear instruction example.

#### [NEW] [flux_workflow.json](file:///d:/projects/localllm-coder/localcoder/core/flux_workflow.json)
*   Create a default API-format workflow JSON describing the node connections (Load GGUF UNET, Load VAE, Load CLIPs, KSampler, VAE Decode, Save Image) used by ComfyUI to execute FLUX.

---

## Verification Plan

### Automated Tests
Since ComfyUI is external, we will implement mock unit tests:
*   Create a mock test in `localllm-coder` that patches `urllib.request.urlopen` to simulate ComfyUI’s API response queue, status polling, and image delivery.
*   Verify that `tool_generate_image` handles network timeouts and returns appropriate error messages if ComfyUI is offline.

### Manual Verification
*   Launch ComfyUI locally.
*   Run `localcoder` and prompt the agent: *"generate a photorealistic image of a retro computer terminal and save it to test_retro.png"*.
*   Verify that the agent correctly calls the `generate_image` tool and that the file is successfully created in the workspace.

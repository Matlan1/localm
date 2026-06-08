# Walkthrough - FLUX Integration in localllm-coder

We have implemented a local image generation tool in `localllm-coder` that interfaces with a local ComfyUI FLUX instance. The agent's offline LLM can now call this tool directly when asked to generate images.

## Changes Made

### Core Tooling & Prompt Configuration
*   **[NEW] [flux_workflow.json](file:///d:/projects/localllm-coder/localcoder/core/flux_workflow.json)**: Created a standard API-format workflow JSON mapping for GGUF-based FLUX generation (KSampler, EmptyLatentImage, DualCLIPLoaderGGUF, VAEDecode, SaveImage, etc.).
*   **[MODIFY] [tools.py](file:///d:/projects/localllm-coder/localcoder/core/tools.py)**:
    *   Implemented `tool_generate_image(cwd, prompt, output_path)` which loads the template workflow, injects the user prompt, queues it in ComfyUI (utilizing the `FLUX_API_URL` environment variable, defaulting to `http://127.0.0.1:8188`), polls the execution history, and downloads the output file.
    *   If ComfyUI is offline, the tool fails gracefully and outputs a descriptive link pointing the user to their local setup guide.
    *   Registered the tool in `TOOL_REGISTRY`.
*   **[MODIFY] [prompts.py](file:///d:/projects/localllm-coder/localcoder/prompts.py)**: Added the `generate_image` tool description and usage examples to the `tool_docs` section in the system prompt.

### Testing & Verification
*   **[NEW] [test_image_tool.py](file:///d:/projects/localllm-coder/tests/test_image_tool.py)**: Created a test suite under the `tests/` directory leveraging `unittest` and `unittest.mock` to verify prompt queueing, polling, image downloads, and offline failure redirection.

---

## Verification Results

We executed the unit tests inside the workspace:

```bash
python -m unittest tests/test_image_tool.py
```

### Output
```text
..
----------------------------------------------------------------------
Ran 2 tests in 0.031s

OK
```
*   `test_generate_image_success`: Confirmed that prompt injection works and the file is correctly created on success.
*   `test_generate_image_connection_failure`: Confirmed that connection refusal triggers a graceful response referencing the local setup guide.

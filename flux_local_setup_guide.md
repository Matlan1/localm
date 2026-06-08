# FLUX Local Image Generation Setup Guide

> [!IMPORTANT]
> **FULL REVIEW REQUIRED:** The tool integration has been fully coded and verified via unit testing. However, manual validation is required once a local ComfyUI server is installed and running.

This document serves as a reference report and hand-off guide for setting up high-quality local image generation using **FLUX** on a system with **16 GB VRAM** (Resizable BAR enabled).

---

## User Questions & Recommendations (For Verification)

*   **Question:** What are the currently available local high-quality image generation options?
    *   **Recommendation:** The current state-of-the-art options are the **FLUX.2** family (by Black Forest Labs) and **Stable Diffusion 3.5**. FLUX.2 is recommended for photorealism and prompt adherence, while Stable Diffusion 3.5 is best for a mature ecosystem of fine-tunes (LoRAs) and community nodes.
*   **Question:** Can FLUX run 100% offline?
    *   **Recommendation:** **Yes.** Once the model weights and text encoders are downloaded locally, inference is run entirely on local hardware without sending any data over the internet.
*   **Question:** Does it come with a safety filter, or do we have to implement it ourselves?
    *   **Recommendation:** The base weights of FLUX **do not contain a safety filter**. If safety filtering is required, it must be implemented at the pipeline level. *(Note: Implementation of a safety filter is currently **paused indefinitely** and will be handled later).*
*   **Question:** Which size should we choose for a 16 GB VRAM card (speed is not a major issue as long as it takes under 15-30 mins)?
    *   **Recommendation:** Choose **FLUX [dev]** quantized to **GGUF (Q8_0 or Q6_K)**. Instead of taking minutes, a quantized FLUX [dev] will generate an image in **30 to 60 seconds** on a 16 GB card, preserving the full quality of the 32B model. Couple this with a quantized **T5XXL text encoder** to keep VRAM usage low during prompt compilation.

---

## Technical Specifications for 16 GB VRAM

### Recommended Model Configuration
| File / Component | Recommended File Version | Purpose | Target Folder (ComfyUI) |
| :--- | :--- | :--- | :--- |
| **UNET Model** | `flux1-dev-Q8_0.gguf` or `flux1-dev-Q6_K.gguf` | Core image diffusion weights | `/models/unet/` |
| **CLIP / Text** | `t5-v1_1-xxl-encoder-Q8_0.gguf` | Quantized T5 encoder for prompt parsing | `/models/clip/` |
| **VAE** | `ae.safetensors` | Autoencoder to decode latents into final images | `/models/vae/` |

### Recommended Software Interfaces
1.  **ComfyUI** (with the [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) extension installed).
2.  **Fooocus** or **Stable Diffusion WebUI Forge** (for a more direct user interface that hides node complexity).
3.  **Stability Matrix** (to manage local installations of these interfaces in one place).

---

## Action Plan for the Next Agent / Next Step

To pick up this task and execute the local installation, run these steps:

1.  **Setup Environment:**
    *   Install **Stability Matrix** or clone **ComfyUI** locally.
    *   Run ComfyUI and open the manager to install `ComfyUI-GGUF`.
2.  **Download Assets (Hugging Face / Model Registries):**
    *   Retrieve the GGUF weights of `flux1-dev` (e.g., from `city96/FLUX.1-dev-gguf` on Hugging Face).
    *   Retrieve quantized T5 encoder weights.
    *   Retrieve standard FLUX VAE (`ae.safetensors`).
3.  **Load Workflow:**
    *   Create or load a basic FLUX GGUF workflow using `Unet Loader (GGUF)` instead of the standard Unet Loader.
4.  **Optionally Add Safety Checker (Paused Indefinitely):**
    *   This task is currently deferred. If implemented later, a secondary node checkpoint can be added after the VAE Decode step to flag/replace unsafe outputs.

---

## Uncensored Generation & Community Fine-tunes

*   **No Active Filters**: The base GGUF weights of FLUX and the ComfyUI/Forge frontends contain no built-in, active censorship code or runtime prompt blocks.
*   **Aesthetics & Anatomy Correcting**: If you want to improve prompt adherence for unrestricted content or optimize anatomy rendering, you can stack community-trained **LoRAs** or download alternative **Checkpoints** from repositories like **Civitai**.
*   **Custom Safety**: Any safety or moderation filtering is entirely up to you and can be handled via pipeline nodes in ComfyUI (currently paused indefinitely).

---

## Integration with localm & localllm-coder (Implemented — Full Review Required)

> [!IMPORTANT]
> The integration tool has been fully implemented and verified using mock unit tests. A full review and manual run are required once you have completed the local ComfyUI setup.

### Goal
To connect FLUX to `localllm-coder` so that the agent's LLM can invoke image generation when prompted by the user (e.g., *"make an image of X"*), rather than using a manual workflow.

### 1. Agent Tool Design
*   **Tool**: The `generate_image(prompt, output_path)` tool has been successfully added to `TOOL_REGISTRY` in [tools.py](file:///d:/projects/localllm-coder/localcoder/core/tools.py).
*   **Prompting**: Added documentation to `tool_docs` in [prompts.py](file:///d:/projects/localllm-coder/localcoder/prompts.py) so the LLM understands when and how to call this tool.
*   **Workflow Template**: Built a default API configuration file [flux_workflow.json](file:///d:/projects/localllm-coder/localcoder/core/flux_workflow.json) containing the node routing maps for GGUF-based FLUX generation.

### 2. Execution Backend
*   **Default Connection**: The tool queries ComfyUI's API (default: `http://127.0.0.1:8188`) or reads the `FLUX_API_URL` environment variable if configured.
*   **Offline Failure Redirection**: In the event that the server is unreachable, the tool gracefully reports a connection failure and redirects the user to their local setup guide.
*   **Mock Verification**: Verified the API call logic, JSON prompt modifications, and response handling using unit tests in [test_image_tool.py](file:///d:/projects/localllm-coder/tests/test_image_tool.py).

---

## Dynamic VRAM Swapping (Gemma vs. FLUX on 16GB VRAM)

To run both Gemma (via `localm`) and FLUX (via ComfyUI) on a single 16 GB VRAM card without slow system memory offloading:
1. **Dynamic Unloading**: The `localm` ctypes engine allows calling `.unload()`, which calls `llama_free_model` and `llama_free` to immediately release all LLM GPU memory back to the OS.
2. **Seamless Tool Lifecycle**:
   * When the agent triggers `generate_image`, it sends a `POST /v1/models/unload` to `localm` to free VRAM.
   * ComfyUI loads FLUX and generates the image at maximum speed.
   * On the next turn, the agent requests the next chat completion.
   * `localm` detects that the model is unloaded and automatically reloads it before streaming the response (~2-5 seconds overhead).


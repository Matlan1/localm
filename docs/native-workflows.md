# Native workflows

A native workflow is a saved preset for the built-in (native) image, video and music
generators: the models of a job (the main model, its text encoders and VAE, and LoRAs with
their strengths) and the generation settings. The Images, Video and Music pages show a
**Workflow** card while the native backend is active (the ComfyUI workflow card takes its place
when ComfyUI runs).

## Using the card

- **Apply** fills the page's model picker and settings from the selected workflow. A model you
  have registered is used by its name, or by the registered model with the same file name. Every
  model you do not have yet is offered for download, one dialog per file, with its repository,
  file and size; the download is checked against the workflow's sha256 when it has one. A model
  you skip is offered again when you press Generate. While the workflow's main model is still not
  downloaded, Generate stops with a message instead of running another model, unless you pick a
  main model yourself.
- **Save current as...** stores the page's current model choice and settings under a new name.
  A model localm knows a public download for is saved with that download, so the workflow works
  on another machine too.
- **Rename**, **Delete** and **Export** act on saved workflows. Built-in workflows cannot be
  renamed or deleted; save a copy instead.
- **Import...** takes a workflow exported here or a ComfyUI workflow (see below).

Built-in starter workflows: SD-Turbo, FLUX.1 schnell and FLUX.1 dev (Q4_K_S GGUF with CLIP-L,
T5-XXL fp8 and the FLUX VAE), Wan 2.1 text to video 1.3B, and the ACE-Step 1.5 default set.

Saved workflows are JSON files in `workflows/<image|video|music>/native/` under the localm data
directory.

## Importing a ComfyUI workflow

Both ComfyUI formats import: **Save (API format)** and the regular **Save**. These nodes map onto
the native workflow: `UNETLoader`, `UnetLoaderGGUF`, `CheckpointLoaderSimple`, `DualCLIPLoader`,
`TripleCLIPLoader`, `CLIPLoader`, `CLIPVisionLoader`, `VAELoader`, `LoraLoader`,
`LoraLoaderModelOnly`, `KSampler`, `KSamplerAdvanced`, `EmptyLatentImage`, `EmptySD3LatentImage`,
`EmptyHunyuanLatentVideo`, `FluxGuidance`, `CLIPTextEncode` (the prompts connected to the
sampler), `ModelSamplingSD3` (flow shift for video, shift for music), and for music
`EmptyAceStepLatentAudio`, `TextEncodeAceStepAudio` and `TextEncodeAceStepAudio15`. Output and
plumbing nodes (`VAEDecode`, `SaveImage`, `CreateVideo`, notes, reroutes, ...) need nothing.

After an import the card shows a report: which nodes were mapped, every node with no native
equivalent (the result can differ from ComfyUI), and every value that was not kept (the seed,
a sampler or scheduler name stable-diffusion.cpp does not have, a denoise below 1, a second
sampler, a LoRA's separate CLIP strength). A model file gets its download from localm's own
list of files when it is on it; any other one can be searched for on Hugging Face by its exact
file name when you apply the workflow (nothing is searched until you click Search).

The native music backend runs ACE-Step 1.5 GGUF parts, so a ComfyUI music workflow keeps its
settings and prompts but not its model files.

## File format

```json
{
  "format": "localm-native-workflow",
  "version": 1,
  "media": "image",
  "name": "FLUX.1 schnell",
  "description": "",
  "models": {
    "model": {"name": "flux1-schnell-Q4_K_S", "file": "flux1-schnell-Q4_K_S.gguf",
              "source": {"repo": "city96/FLUX.1-schnell-gguf",
                         "file": "flux1-schnell-Q4_K_S.gguf",
                         "sha256": "4fd16477...", "size_bytes": 6783943712,
                         "model_type": "diffusion-unet"}},
    "vae": {"name": "ae"}
  },
  "loras": [{"name": "my-style", "strength": 0.8}],
  "params": {"steps": 4, "cfg": 1.0, "width": 1024, "height": 1024}
}
```

- `models` slots: `model`, `clip_l`, `clip_g`, `t5xxl`, `llm`, `vae` (image);
  `model`, `t5xxl`, `llm`, `clip_vision`, `vae` (video); `dit`, `text_encoder`, `vae`, `lm`
  (music, no LoRAs). A slot left out is picked automatically.
- `source` is optional: a Hugging Face repository and a `.safetensors`, `.sft` or `.gguf` file
  in it, an optional lowercase `sha256` and size, and the registry type the download gets.
- `params`: `steps`, `cfg`, `sample_method`, `scheduler`, `width`, `height`, `prompt`,
  `negative_prompt` for image and video; plus `guidance` and `clip_skip` (image), `seconds`,
  `fps` and `flow_shift` (video). Music takes `steps`, `cfg`, `shift`, `duration_seconds`, `tags`
  and `lyrics`.

## API

Each media plugin serves `/api/<image|video|music>/native-workflows`: `GET` lists them,
`GET /{id}` returns one, `POST {"workflow": {...}}` saves one (`"id"` replaces a saved one),
`POST /{id}/rename {"name"}`, `DELETE /{id}`, `POST /import {"data": {...}, "name"}` imports
a native or ComfyUI workflow and returns the report, and `POST /{id}/resolve` returns the
`native_models`, `loras` and `params` a generate request sends plus the `missing` models with
their downloads. `POST /api/media/<kind>/preflight {"native_workflow": "<id>"}` lists the
workflow's missing models the way the generate pre-check does. `POST /api/video` takes
`flow_shift` for the native backend.

# localm-llama-runtime

The native llama.cpp inference binaries, packaged as a wheel so **localm carries
its own runtime inside its venv** instead of depending on a folder elsewhere on
disk.

`localm_llama_runtime/lib/` holds `llama.dll` + `ggml-*.dll` (and, for a GPU
prebuilt, the matched ROCm/CUDA runtime DLLs and the `llama-cli`/`llama-server`
executables). These binaries are **never committed** — they are large,
GPU/platform-specific, and license-encumbered — so the directory ships empty.

## Provisioning

```
localm setup-llama                         # download the default prebuilt
localm setup-llama --from <build-dir>      # copy from your own llama.cpp build
localm setup-llama --url <zip-url>         # a different prebuilt
```

`setup-llama` extracts the binaries here and installs this wheel editable, so
adding or replacing binaries later needs no rebuild — just drop files in `lib/`.

## How localm finds it

`localm.inference.backends.llamacpp._loader` resolves the binary directory in
order: `LLAMA_CPP_LIB` env → `binary_dir` config → **this wheel** → (deprecated
external dirs). It also adds the venv's `_rocm_sdk_*/bin` directories to the DLL
search path, so a build that bundles only `llama.dll` + `ggml-*.dll` still finds
its ROCm runtime (`amdhip64`, `rocm_kpack`, `rocblas`, …) from the `rocm-sdk`
wheels already in the venv.

## Where the gfx1030 binaries come from

The default prebuilt is the lemonade-sdk gfx103X (RDNA2) Windows ROCm build.
For a from-source build see `rocm-canary-forge/windows-native`
(`build_llamacpp_gfx1030.bat`, Ninja + AMD clang-cl, `-DGGML_HIP=ON
-DGPU_TARGETS=gfx1030`, compiled against the venv's rocm-sdk).

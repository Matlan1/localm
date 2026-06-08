# GPU Setup

## AMD (ROCm / HIP)

localm uses the native `llama.dll` HIP build for AMD GPUs. No Python GPU packages are needed for GGUF inference — only the prebuilt DLLs.

### gfx1030 (RX 6000 series — Navi21)

Required DLLs (load order matters):

```
ggml.dll
ggml-base.dll
ggml-cpu.dll
ggml-hip.dll        ← GPU compute via ROCm HIP
llama.dll
```

The DLL directory is auto-detected. To use a custom location:

```bash
set LLAMA_CPP_LIB=C:\path\to\llama.dll
localm run mymodel --prompt "hello"
```

Or set it permanently in your shell profile.

### Building from source (optional)

If you want to build `llama.dll` yourself from the llama.cpp source:

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1030 -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j8
```

Output: `build\bin\Release\llama.dll` and sibling `ggml*.dll` files.

### Verify GPU is being used

Run with a 13B+ model and watch VRAM usage in Task Manager or `rocm-smi`.  
localm prints at startup:

```
ggml_cuda_init: found 1 ROCm devices (Total VRAM: 16368 MiB):
  Device 0: AMD Radeon RX 6900 XT, gfx1030
```

To limit GPU layers (partial offload):
```bash
localm run mymodel --gpu-layers 20 --prompt "hello"
```

## NVIDIA (CUDA)

Use a CUDA-enabled `llama.dll` build (replace `ggml-hip.dll` with `ggml-cuda.dll` in the dependency chain). Set `LLAMA_CPP_LIB` to point to it. Everything else is identical.

## CPU-only

Drop the HIP/CUDA DLL. The DLL load order becomes:

```
ggml.dll → ggml-base.dll → ggml-cpu.dll → llama.dll
```

Set `--gpu-layers 0` to disable GPU offloading:

```bash
localm run mymodel --gpu-layers 0 --prompt "hello"
```

## HuggingFace Transformers (AMD ROCm)

For HF-format models (not GGUF), localm uses PyTorch + Transformers. On AMD Windows:

```bash
uv tool install -e ".[gpu]"
```

This pulls PyTorch built against ROCm 7.13.0 from AMD's wheel index. Requires Python 3.12.

GPU is selected automatically if available. Override with:

```bash
localm run owner/model-name --device cuda   # HIP maps to "cuda" in PyTorch
localm run owner/model-name --device cpu
```

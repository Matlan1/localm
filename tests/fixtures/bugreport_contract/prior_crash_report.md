# localm bug report: localm server crashed - native fault captured: Windows fatal exception: access violation

## What I was doing
<!-- Please describe what you ran and what you expected. -->

## What happened
localm server crashed - native fault captured: Windows fatal exception: access violation

Reason: a native crash was caught by the fault handler; see the captured trace below for the exact fault and thread/frame. The process exited with 3221225477 (0xC0000005, access violation).

## App state
- Model: contract-model-Q4_K_M (loaded), backend ContractBackend, ctx<=8192
- Session mode: log
- Debug logging: off
- Enabled plugins: chat, gui, coder

## Environment
- localm: 0.0.0+contract
- Python: {{PYTHON}}
- OS: TestOS-1.0-contract
- Arch: testarch
- GPU vendor(s): nvidia
- Recommended backend: cuda
- Detected via: contract-probe
- NVIDIA GPU: Contract GPU 9000
- NVIDIA driver: 999.99
- Driver CUDA capability: 12.8
- GPU compute capability: 8.9
- Selected CUDA line: cu12
- NVIDIA GPUs (nvidia-smi order): 0: Contract GPU 9000 (8.0 of 16.0 GB free)
- GPUs: 0: Contract GPU 9000 (8.0 of 16.0 GB free via nvidia-smi), 1: not detected (4.0 GB total, free not detected)
- Native runtime provisioned: True
- Native runtime backend: cuda
- Native runtime build: b1234
- Native runtime pinned to: b1234
- Native libraries: ggml-cuda.dll, libmtmd.so.1, llama.dll

## Configuration (safe subset)
- binary_dir: C:\Users\<redacted>\localm\bin
- n_ctx: 8192
- n_gpu_layers: 99
- spec_source: ngram
- port: 8765
- require_auth: yes
- mode: log
- comfy_workdir: ~/ComfyUI
- comfy_api_url: http://<redacted>@alice-desktop.local:8188/?api_key=<redacted>
- net_search_url: https://search.example.org/search?q=x&token=<redacted>
- coder_reviewer: http://reviewer.example.net/v1?key=<redacted>&model=m

## Dependencies
- localm: 0.0.0+contract
- fastapi: 0.0.1
- uvicorn: 0.0.2

## Native fault trace
```
Windows fatal exception: code 0x8001010d

Windows fatal exception: access violation

Current thread 0x0000beef (most recent call first):
  File "C:\Users\<redacted>\Documents\localm\notes.txt", line 99 in decode
  X-Api-Key: <redacted>
```

## Process exit
- Exit code: 3221225477 (0xC0000005, access violation)
- Watched for: 30s

## Recent log (tail)
```
... (1 debug record(s) withheld - chat content is never included in a bug report) ...
2026-10-09 11:59:00,001 INFO     localm: server start on alice-desktop.local:8765
2026-10-09 11:59:01,300 INFO     localm.http: GET /api/stats 200 3ms  (repeated 3x, 2026-10-09 11:59:01 .. 2026-10-09 11:59:01)
2026-10-09 11:59:02,003 WARNING  localm.net: GET http://<redacted>@alice-desktop.local/?api_key=<redacted> failed for ~/models/private-model.gguf
2026-10-09 11:59:03,004 ERROR    localm.engine: load failed
Traceback (most recent call last):
  File "C:\Users\<redacted>\Documents\localm\notes.txt", line 7, in load
RuntimeError: Authorization: <redacted> for bob.builder@example.com
llama_context: constructing llama_co
```

## Server hang trace (event-loop stall)
The hang watchdog captured every thread's stack when the server froze (the top of the main thread is the blocking call):
```
Thread 0x0002 (most recent call first):
  File "C:\Users\<redacted>\Documents\localm\notes.txt", line 40 in _run_once
  Authorization: <redacted>
```

## Recent activity (in-memory log)
```
12:00:00 INFO localm: loaded ~/models/private-model.gguf
12:00:01 WARNING localm: GET http://<redacted>@alice-desktop.local/?sig=<redacted> failed
12:00:02 INFO localm: reviewer bob.builder@example.com at \\FILESERVER01\share\report.txt
12:00:03 ERROR localm: header X-Api-Key: <redacted> key <redacted>
```

---
Sent to the localm maintainer ({{MAINTAINER_EMAIL}}). You can edit anything above before sending.

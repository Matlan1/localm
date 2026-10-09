# localm bug report: model load failed for ~/models/private-model.gguf on alice-desktop.local token=<redacted>

## What I was doing
<!-- Please describe what you ran and what you expected. -->

## What happened
model load failed for ~/models/private-model.gguf on alice-desktop.local token=<redacted>

Reason: backend said api_key=<redacted> reading C:\Users\<redacted>\Documents\localm\notes.txt; ask <redacted-email>

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
- Operation: chat
- Backend (effective): cuda
- Backend (requested): auto
- Fetched CUDA runtime bundle: True

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
- coder_reviewer: http://reviewer.example.net/v1?key=<redacted>&model=m&cc=<redacted-email>

## Dependencies
- localm: 0.0.0+contract
- fastapi: 0.0.1
- uvicorn: 0.0.2

## Error detail
```
RuntimeError: POST http://<redacted>@alice-desktop.local:8188/prompt?token=<redacted>&mode=fast failed: Authorization: <redacted> key <redacted> at /home/<redacted>/.cache/localm/model.gguf
```

## Native fault trace
```
Windows fatal exception: access violation

Current thread 0x00001a2b (most recent call first):
  File "C:\Users\<redacted>\Documents\localm\notes.txt", line 12 in load
  File "/Users/<redacted>/Library/Logs/localm.log", line 3 in <module>
env OPENAI_API_KEY=<redacted>

```

## Process exit
- Exit code: 3221225477 (0xC0000005, access violation)
- Watched for: 12.5s

## Recent log (tail)
```
2026-10-09 12:00:00,000 WARNING  localm: upstream \\FILESERVER01\share\report.txt rejected X-Api-Key: <redacted>
payload {"api_key": <redacted>, "has_token": false}
state {"path": "C:\\Users\\<redacted>\\AppData\\Roaming\\localm\\state.json"}
SECRET_KEY=<redacted> key=visible monkey=visible
```

## Server hang trace (event-loop stall)
The hang watchdog captured every thread's stack when the server froze (the top of the main thread is the blocking call):
```
Thread 0x0001 (most recent call first):
  File "~/models/private-model.gguf", line 5 in wait
  token=<redacted>

```

## Recent activity (in-memory log)
```
11:59:59 INFO localm: before restart token=<redacted>
--- RESTART ---
12:00:00 INFO localm: loaded ~/models/private-model.gguf
12:00:01 WARNING localm: GET http://<redacted>@alice-desktop.local/?sig=<redacted> failed
12:00:02 INFO localm: reviewer <redacted-email> at \\FILESERVER01\share\report.txt
12:00:03 ERROR localm: header X-Api-Key: <redacted> key <redacted>
```

## Browser / client
- User agent: Mozilla/5.0 (contract)
- Page: http://alice-desktop.local:8765/#/chat?api_key=<redacted>
- Viewport: 1280x720
- GUI build: contract-build

Recent browser console errors:
```
fetch failed {"api_key":<redacted>,"has_token":false}
Error: <redacted> at /Users/<redacted>/Library/Logs/localm.log
user <redacted-email> SECRET_KEY=<redacted> Authorization: <redacted>
```

---
Sent to the localm maintainer ({{MAINTAINER_EMAIL}}). You can edit anything above before sending.

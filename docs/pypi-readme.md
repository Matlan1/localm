# LocaLM

Run large language models on your own machine. Offline, private, and yours.

LocaLM downloads and runs LLMs locally: GGUF models through llama.cpp and
HuggingFace models through transformers, on AMD, NVIDIA, Intel, Apple Silicon
or CPU. It ships a chat GUI, an OpenAI-compatible API, a coding agent, RAG
over your own documents, and an MCP server, with everything off by default and
nothing leaving your machine.

## Install

```bash
pip install localm
localm setup-llama
```

`setup-llama` detects your GPU and provisions the matching llama.cpp runtime.
It needs no vendor toolkit: NVIDIA gets a self-contained CUDA build, AMD on
Windows a bundled ROCm build, Intel and toolkit-less AMD a Vulkan build, Apple
Silicon Metal, and anything else CPU.

Then pull a model and talk to it:

```bash
localm pull unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M
localm run Qwen3-4B-Instruct-2507
```

Or open the graphical interface:

```bash
localm gui
```

Python 3.12 is required. `localm doctor` reports what is installed and what is
missing at any point.

## Where data lives

Set `LOCALM_HOME` to choose where models, chats and settings are stored. Left
unset, it defaults to a directory inside the Python environment you installed
into (not a per-user directory); `localm info` prints the path actually in
use.

## The self-contained installer

The pip package installs LocaLM into an environment you already manage. The
installer on GitHub instead provisions its own Python and its own private
environment, adds a desktop launcher and a native app window, and walks you
through choosing plugins. If you would rather have that, or you are not on
Python 3.12, use it:

<https://github.com/Matlan1/localm>

## Documentation, issues and source

<https://github.com/Matlan1/localm>

AGPL-3.0-or-later.

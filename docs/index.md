# LocaLM documentation

LocaLM runs large language models on your own machine: GGUF models through
llama.cpp and HuggingFace models through transformers, on AMD, NVIDIA, Intel,
Apple Silicon or CPU. It ships a chat GUI, an OpenAI-compatible API, a coding
agent, RAG over your own documents and an MCP server, with every online
feature off by default. The only thing it does online without being asked is
a periodic update check, which you can turn off too ([network](network.md)).

## Where to start

- **Install and set up:** [Linux and macOS](linux-setup.md), [GPU setup](gpu-setup.md),
  [the web GUI](gui.md).
- **Use it:** [CLI reference](cli.md), [knowledge and RAG](rag.md),
  [memory](memory.md), [scheduled jobs](jobs.md), [privacy modes](privacy.md).
- **Connect other software:** [HTTP Server API](server-api.md),
  [API reference](api-reference.md), [MCP support](mcp.md), [TLS and reverse proxies](tls.md).
- **Extend or contribute:** [plugins](plugins.md), [architecture](architecture.md).

The source, releases and issue tracker are on
[GitHub](https://github.com/Matlan1/localm).

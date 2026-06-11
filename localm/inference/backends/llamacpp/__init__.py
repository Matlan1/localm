"""
Pure-Python ctypes wrapper around llama.dll — our own llama-cpp-python.

Usage
-----
    from localm.inference.backends.llamacpp import LlamaCpp

    llm = LlamaCpp("path/to/model.gguf", n_gpu_layers=99, n_ctx=4096)
    for chunk in llm.create_chat_completion(messages, stream=True):
        print(chunk["choices"][0]["delta"].get("content", ""), end="", flush=True)
    llm.close()
"""

from .llama import LlamaCpp

__all__ = ["LlamaCpp"]

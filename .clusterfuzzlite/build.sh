#!/bin/bash -eu
pip3 install --require-hashes --no-deps -r "$SRC/localm/.clusterfuzzlite/requirements.txt"
export PYTHONPATH="$SRC/localm"
compile_python_fuzzer "$SRC/localm/.clusterfuzzlite/fuzz_gguf_readers.py"

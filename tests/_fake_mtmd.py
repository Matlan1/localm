# SPDX-License-Identifier: AGPL-3.0-or-later
"""Pure-Python stand-ins for the mtmd library and the llama KV cache.

FakeKV holds one entry per KV position. FakeMtmdLib implements the mtmd
functions MtmdContext calls, and its image decode writes into a FakeKV.
FakeLlamaApi implements the llama functions LlamaCpp's image prefill calls, and
its llama_decode writes into the same FakeKV. A test can therefore compare the
cache a reusing prefill leaves with the cache a from-scratch prefill leaves.

FakeKV.write refuses a write that does not start right after the last held
position, so a prefill that decodes at a wrong position fails loudly.
"""
import ctypes
import zlib
from types import SimpleNamespace

from localm.inference.backends.llamacpp import mtmd as lmtmd

MARKER = "<__media__>"
_TEXT, _IMAGE, _AUDIO = 0, 1, 2


def word_token(word: str) -> int:
    """Stable token id for one whitespace-separated word."""
    return zlib.crc32(word.encode("utf-8")) % 50000


def solid_image(value: int, w: int = 4, h: int = 4):
    """A (w, h, rgb) image whose every byte is *value*."""
    return (w, h, bytes([value]) * (w * h * 3))


class FakeKV:
    def __init__(self):
        self.cells = []

    def write(self, pos, entries):
        assert pos == len(self.cells), (
            f"decode at position {pos} while the cache holds {len(self.cells)}")
        self.cells.extend(entries)

    def truncate(self, p0):
        del self.cells[p0:]

    def clear(self):
        self.cells.clear()


class FakeMtmdLib:
    """mtmd as MtmdContext sees it. Text between markers becomes one text chunk
    of word tokens; each image becomes *slices* media chunks of *image_tokens*
    tokens and *image_pos* positions, carrying the bitmap's id. Each audio
    bitmap becomes *audio_segments* audio chunks of the same size."""

    def __init__(self, kv: FakeKV, *, image_tokens=4, image_pos=None, slices=1,
                 n_embd=3):
        self.kv = kv
        self.image_tokens = image_tokens
        self.image_pos = image_tokens if image_pos is None else image_pos
        self.slices = slices
        self.audio_segments = 1
        self.n_embd = n_embd
        self.localm_has_audio_api = True
        self.audio_bitmaps = []          # (n_samples, pcm bytes) per audio bitmap
        self.rc_tokenize = 0
        self.fail_encode = False
        self.fail_decode = False
        self.pos_skew = 0
        self.encode_calls = 0
        self.image_decodes = []          # (content, n_past) per image decode
        self.image_n_batch = []          # n_batch per image decode
        self.bitmaps = {}
        self.bitmaps_freed = 0
        self.chunk_lists = {}
        self.chunk_lists_freed = 0
        self.chunks = {}
        self._next = 0x1000
        self._out = None

    def _handle(self):
        self._next += 1
        return self._next

    def mtmd_bitmap_init(self, w, h, rgb):
        handle = self._handle()
        self.bitmaps[handle] = {"w": w, "h": h, "rgb": bytes(rgb), "id": b""}
        return handle

    def mtmd_bitmap_init_from_audio(self, n_samples, samples):
        handle = self._handle()
        pcm = bytes(samples)
        self.bitmaps[handle] = {"audio": True, "n": n_samples, "rgb": pcm, "id": b"",
                                "w": n_samples, "h": 1}
        self.audio_bitmaps.append((n_samples, pcm))
        return handle

    def mtmd_bitmap_set_id(self, bmp, id_bytes):
        self.bitmaps[bmp]["id"] = id_bytes

    def mtmd_bitmap_free(self, bmp):
        self.bitmaps_freed += 1

    def mtmd_input_chunks_init(self):
        handle = self._handle()
        self.chunk_lists[handle] = []
        return handle

    def mtmd_input_chunks_free(self, handle):
        self.chunk_lists_freed += 1

    def _add_chunk(self, lst, **chunk):
        handle = self._handle()
        self.chunks[handle] = chunk
        self.chunk_lists[lst].append(handle)

    def mtmd_tokenize(self, ctx, lst, itext_addr, arr, n_bitmaps):
        if self.rc_tokenize:
            return self.rc_tokenize
        itext = ctypes.cast(itext_addr, ctypes.POINTER(lmtmd._MtmdInputTextV2)).contents
        text = itext.text.decode("utf-8")
        parts = text.split(MARKER)
        if len(parts) - 1 != n_bitmaps:
            return 1
        for i, part in enumerate(parts):
            words = part.split()
            if words:
                tokens = (ctypes.c_int32 * len(words))(*(word_token(w) for w in words))
                self._add_chunk(lst, type=_TEXT, tokens=tokens)
            if i < n_bitmaps:
                bmp = self.bitmaps[arr[i]]
                if bmp.get("audio"):
                    for s in range(self.audio_segments):
                        content = (zlib.crc32(bmp["rgb"]), bmp["w"], bmp["h"], s)
                        self._add_chunk(lst, type=_AUDIO, id=bmp["id"], content=content)
                    continue
                for s in range(self.slices):
                    content = (zlib.crc32(bmp["rgb"]), bmp["w"], bmp["h"], s)
                    self._add_chunk(lst, type=_IMAGE, id=bmp["id"], content=content)
        return 0

    def mtmd_input_chunks_size(self, lst):
        return len(self.chunk_lists[lst])

    def mtmd_input_chunks_get(self, lst, i):
        return self.chunk_lists[lst][i]

    def mtmd_input_chunk_get_type(self, handle):
        return self.chunks[handle]["type"]

    def mtmd_input_chunk_get_tokens_text(self, handle, n_ref):
        tokens = self.chunks[handle]["tokens"]
        n_ref._obj.value = len(tokens)
        return tokens

    def mtmd_input_chunk_get_n_tokens(self, handle):
        chunk = self.chunks[handle]
        return len(chunk["tokens"]) if chunk["type"] == _TEXT else self.image_tokens

    def mtmd_input_chunk_get_n_pos(self, handle):
        chunk = self.chunks[handle]
        return len(chunk["tokens"]) if chunk["type"] == _TEXT else self.image_pos

    def mtmd_input_chunk_get_id(self, handle):
        return self.chunks[handle].get("id") or None

    def _embedding_of(self, content):
        base = float(content[0] % 997)
        return [base + 0.25 * j for j in range(self.image_tokens * self.n_embd)]

    def mtmd_encode_chunk(self, ctx, handle):
        self.encode_calls += 1
        if self.fail_encode:
            return 1
        values = self._embedding_of(self.chunks[handle]["content"])
        self._out = (ctypes.c_float * len(values))(*values)
        return 0

    def mtmd_get_output_embd(self, ctx):
        return ctypes.addressof(self._out) if self._out is not None else None

    def mtmd_helper_decode_image_chunk(self, ctx, lctx, handle, embd_addr, n_past,
                                       seq_id, n_batch, new_n_past_ref, cb, user_data):
        if self.fail_decode:
            return 1
        content = self.chunks[handle]["content"]
        n = self.image_tokens * self.n_embd
        embd = tuple(ctypes.cast(embd_addr, ctypes.POINTER(ctypes.c_float))[:n])
        self.image_decodes.append((content, n_past))
        self.image_n_batch.append(n_batch)
        self.kv.write(n_past, [("image", content, embd, i) for i in range(self.image_pos)])
        new_n_past_ref._obj.value = n_past + self.image_pos + self.pos_skew
        return 0

    def mtmd_free(self, ctx):
        pass


def make_mtmd_context(lib: FakeMtmdLib, *, on_gpu=True, vision=True, audio_rate=0):
    """A real MtmdContext with its native __init__ bypassed, driving *lib*.
    *audio_rate* non-zero makes the projector take audio at that rate."""
    ctx = lmtmd.MtmdContext.__new__(lmtmd.MtmdContext)
    ctx._m = lib
    ctx._ctx = 0x77
    ctx.on_gpu = on_gpu
    ctx.supports_vision = vision
    ctx.supports_audio = bool(audio_rate)
    ctx.audio_sample_rate = audio_rate
    ctx.marker = MARKER
    ctx._input_text_class = lmtmd._MtmdInputTextV2
    ctx._n_embd_inp = lib.n_embd
    ctx._open = lambda use_gpu: 0x78
    return ctx


class FakeLlamaApi:
    """The llama functions LlamaCpp's image prefill and context rebuild call."""

    def __init__(self, kv: FakeKV, *, n_ctx=4096, mrope=False):
        self.kv = kv
        self.n_ctx = n_ctx
        self.mrope = mrope
        self.seq_rm_ok = True
        self.decode_rc = 0
        self.decoded = []                # (start position, tokens) per text batch
        self.seq_rm_calls = []
        self.clears = 0
        self.inits = []
        self.freed = []

    def llama_n_ctx(self, ctx):
        return self.n_ctx

    def llama_get_memory(self, ctx):
        return 0x99

    def has_memory_api(self):
        return True

    def has_model_meta_api(self):
        return True

    def llama_model_meta_val_str(self, model, key):
        if key == "general.architecture":
            return "qwen2vl" if self.mrope else "llama"
        return None

    def llama_model_rope_type(self, model):
        return 8 if self.mrope else 0

    def llama_memory_seq_rm(self, mem, seq_id, p0, p1):
        self.seq_rm_calls.append(p0)
        if not self.seq_rm_ok:
            return False
        self.kv.truncate(p0)
        return True

    def llama_memory_clear(self, mem, data):
        self.clears += 1
        self.kv.clear()

    def llama_decode(self, ctx, batch):
        if self.decode_rc:
            return self.decode_rc
        self.kv.write(batch.start, [("text", t) for t in batch.tokens])
        self.decoded.append((batch.start, list(batch.tokens)))
        return 0

    def llama_batch_free(self, batch):
        pass

    def llama_context_default_params(self):
        return SimpleNamespace()

    def llama_init_from_model(self, model, params):
        self.kv.clear()
        self.inits.append(params.n_ctx)
        self.n_ctx = params.n_ctx
        return 0x500 + len(self.inits)

    def llama_free(self, ctx):
        self.freed.append(ctx)

    def llama_free_model(self, model):
        self.freed.append(model)


def fake_vision_prompt(*, text_tokens=(1, 2, 3), image_tokens=0):
    """A real MtmdPrompt holding one text chunk of *text_tokens* and, when
    *image_tokens* is non-zero, one keyed image chunk of that many tokens before
    it. Its handles are fake and its release does nothing."""
    chunks = []
    if image_tokens:
        chunks.append(lmtmd.MtmdChunk(0x5001, None, ("img", 0, image_tokens, image_tokens),
                                      image_tokens, image_tokens))
    chunks.append(lmtmd.MtmdChunk(0x5002, tuple(text_tokens), None,
                                  len(text_tokens), len(text_tokens)))
    return lmtmd.MtmdPrompt(chunks, release=lambda: None)


def fake_create_batch(tokens, start_pos, logits_at_last_only=True):
    """Stand-in for LlamaCpp._create_batch that FakeLlamaApi.llama_decode reads."""
    return SimpleNamespace(tokens=list(tokens), start=start_pos, n_tokens=len(tokens))

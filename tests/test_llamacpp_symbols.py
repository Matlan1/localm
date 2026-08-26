# SPDX-License-Identifier: AGPL-3.0-or-later
"""localm/inference/backends/llamacpp/_symbols.py: resolving a llama.cpp function
that is exported with C++ linkage rather than as `extern "C"`.

WHY THIS EXISTS. `getattr(lib, "llama_set_embeddings_nextn")` raises AttributeError
for the MTP draft-head API, because those functions are declared in llama.cpp's
internal header without `extern "C"` and are therefore exported mangled. Reading
that AttributeError as "this runtime cannot do MTP" is an error this repo actually
made and wrote down as a conclusion, when the shipped library exports all five and
every one of them can be called.

WHAT THESE TESTS PIN, in order:

1. A mangled name is FOUND, in both compiler schemes, since a Windows-only match
   would silently drop the feature on Linux and macOS.
2. An `extern "C"` name still wins outright, so promoting the API upstream cannot
   regress into the slower path.
3. It can report ABSENCE. A resolver that finds everything is as useless as one
   that finds nothing, and absence is the answer that turns MTP off.
4. The identifier boundary holds: `llama_get_embeddings_nextn` must not resolve to
   `llama_get_embeddings_nextn_ith`, which is a different function.
"""

from __future__ import annotations

import struct

from localm.inference.backends.llamacpp import _symbols


# Real exports read from the shipped llama.dll: MSVC mangling for the C++-linkage
# staging API, plain names for the extern "C" surface.
MSVC = [
    "llama_model_n_embd",
    "llama_model_n_layer_nextn",
    "?llama_set_embeddings_nextn@@YAXPEAUllama_context@@_N1@Z",
    "?llama_get_embeddings_nextn@@YAPEAMPEAUllama_context@@@Z",
    "?llama_get_embeddings_nextn_ith@@YAPEAMPEAUllama_context@@H@Z",
    "?llama_get_ctx_other@@YAPEAUllama_context@@PEAU1@@Z",
]

# The same functions as a GCC/Clang build would export them.
ITANIUM = [
    "llama_model_n_embd",
    "_Z26llama_set_embeddings_nextnP13llama_contextbb",
    "_Z26llama_get_embeddings_nextnP13llama_context",
    "_Z30llama_get_embeddings_nextn_ithP13llama_contexti",
]


def _fake_image(tmp_path, names, tag="img"):
    """An image whose export scan is stubbed to *names*."""
    path = tmp_path / (tag + ".bin")
    path.write_bytes(b"not a real binary")
    _symbols._cache[("names", str(path))] = list(names)
    return path


def test_a_msvc_mangled_export_is_found(tmp_path):
    img = _fake_image(tmp_path, MSVC, "msvc")
    assert (_symbols.mangled_name(img, "llama_set_embeddings_nextn")
            == "?llama_set_embeddings_nextn@@YAXPEAUllama_context@@_N1@Z")


def test_an_itanium_mangled_export_is_found(tmp_path):
    """Hardcoding the MSVC spelling would drop MTP on every non-Windows build."""
    img = _fake_image(tmp_path, ITANIUM, "gcc")
    assert (_symbols.mangled_name(img, "llama_set_embeddings_nextn")
            == "_Z26llama_set_embeddings_nextnP13llama_contextbb")


def test_a_plain_extern_c_name_wins_over_any_mangled_one(tmp_path):
    """If upstream promotes the API, the direct name must be preferred."""
    img = _fake_image(tmp_path, MSVC + ["llama_set_embeddings_nextn"], "promoted")
    assert _symbols.mangled_name(img, "llama_set_embeddings_nextn") == \
        "llama_set_embeddings_nextn"


def test_absence_is_reported_as_absence(tmp_path):
    """The resolver must be able to say no - that answer is what turns MTP off."""
    img = _fake_image(tmp_path, MSVC, "absent")
    assert _symbols.mangled_name(img, "llama_model_has_mtp") is None
    assert _symbols.mangled_name(img, "zzz_not_a_symbol") is None


def test_a_prefix_does_not_match_a_longer_function(tmp_path):
    """llama_get_embeddings_nextn and ..._ith are different functions with
    different signatures; matching the shorter name against the longer symbol
    would call the wrong one with the wrong arguments."""
    img = _fake_image(tmp_path, MSVC, "boundary")
    assert (_symbols.mangled_name(img, "llama_get_embeddings_nextn")
            == "?llama_get_embeddings_nextn@@YAPEAMPEAUllama_context@@@Z")
    assert (_symbols.mangled_name(img, "llama_get_embeddings_nextn_ith")
            == "?llama_get_embeddings_nextn_ith@@YAPEAMPEAUllama_context@@H@Z")


def test_an_unreadable_image_yields_no_names(tmp_path):
    """A missing or unparseable file is "could not look", reported as no symbols
    rather than raising into a caller that is only asking a capability question."""
    assert _symbols.exported_names(tmp_path / "does-not-exist.dll") == []
    junk = tmp_path / "junk.dll"
    junk.write_bytes(b"\x00" * 512)
    assert _symbols.exported_names(junk) == []


def test_the_pe_parser_reads_a_real_export_table(tmp_path):
    """The parser is exercised against a genuine PE export directory, built here,
    so a change to it cannot pass on stubbed names alone."""
    names = [b"llama_alpha", b"?llama_beta@@YAXXZ"]
    # one section holding the export directory, name RVAs and name strings
    sec_rva, sec_off = 0x1000, 0x400
    blob = bytearray()
    name_offsets = []
    for n in names:
        name_offsets.append(len(blob))
        blob += n + b"\x00"
    dll_name_off = len(blob)
    blob += b"test.dll\x00"
    addr_of_names_off = len(blob)
    for off in name_offsets:
        blob += struct.pack("<I", sec_rva + 0x100 + off)
    edir_off = len(blob)
    edir = bytearray(40)
    struct.pack_into("<I", edir, 12, sec_rva + 0x100 + dll_name_off)   # Name
    struct.pack_into("<I", edir, 20, len(names))                        # NumberOfFunctions
    struct.pack_into("<I", edir, 24, len(names))                        # NumberOfNames
    struct.pack_into("<I", edir, 32, sec_rva + 0x100 + addr_of_names_off)
    body = bytearray(0x100) + blob + edir
    struct.pack_into("<I", body, 0, 0)  # keep the 0x100 pad explicit

    pe_off = 0x80
    data = bytearray(sec_off + len(body))
    data[0:2] = b"MZ"
    struct.pack_into("<I", data, 0x3C, pe_off)
    data[pe_off:pe_off + 4] = b"PE\x00\x00"
    struct.pack_into("<H", data, pe_off + 6, 1)      # NumberOfSections
    struct.pack_into("<H", data, pe_off + 20, 240)   # SizeOfOptionalHeader
    struct.pack_into("<H", data, pe_off + 24, 0x20B)  # PE32+
    struct.pack_into("<II", data, pe_off + 24 + 112, sec_rva + 0x100 + edir_off, 40)
    sh = pe_off + 24 + 240
    struct.pack_into("<IIII", data, sh + 8, len(body), sec_rva, len(body), sec_off)
    data[sec_off:sec_off + len(body)] = body

    img = tmp_path / "real.dll"
    img.write_bytes(bytes(data))
    _symbols._cache.pop(("names", str(img)), None)
    found = _symbols.exported_names(img)
    assert found == ["llama_alpha", "?llama_beta@@YAXXZ"], found
    assert _symbols.mangled_name(img, "llama_beta") == "?llama_beta@@YAXXZ"

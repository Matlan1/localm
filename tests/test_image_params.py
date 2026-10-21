# SPDX-License-Identifier: AGPL-3.0-or-later
"""localm.media.image_params: PNG text chunks, A1111 / ComfyUI / localm parsing."""

import io
import json
import struct
import zlib

import pytest

from localm.media import image_params as ip


def _chunk(ctype: bytes, body: bytes) -> bytes:
    return struct.pack(">I", len(body)) + ctype + body + struct.pack(
        ">I", zlib.crc32(ctype + body) & 0xFFFFFFFF)


def _png(chunks: list) -> bytes:
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    raw = zlib.compress(b"\x00\xff\x00\x00")
    return (ip.PNG_SIGNATURE + _chunk(b"IHDR", ihdr) + b"".join(chunks)
            + _chunk(b"IDAT", raw) + _chunk(b"IEND", b""))


def _text(key: str, value: str) -> bytes:
    return _chunk(b"tEXt", key.encode("latin-1") + b"\x00" + value.encode("latin-1"))


def _pil_png(entries) -> bytes:
    from PIL import Image
    from PIL.PngImagePlugin import PngInfo
    info = PngInfo()
    for k, v in entries:
        info.add_text(k, v)
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), "red").save(buf, format="PNG", pnginfo=info)
    return buf.getvalue()


RECORD = {
    "prompt": "a red fox, \u00e9t\u00e9 \u2603", "negative_prompt": "blurry", "seed": 42,
    "cfg": 7.0, "guidance": 3.5, "steps": 20, "width": 512, "height": 768,
    "backend": "native", "model": "sd-turbo",
    "components": {"vae": "ae.safetensors"},
    "loras": [{"name": "style-a", "strength": 0.8}],
    "input_image": "C:/Users/someone/private/in.png", "runtime": "x", "elapsed_seconds": 3.1,
}


class TestRoundTrip:
    def test_localm_chunk_round_trips_through_a_real_png(self):
        data = _pil_png(ip.png_text_entries(RECORD))
        got = ip.params_from_bytes(data)
        assert got["source"] == "localm"
        assert got["params"]["prompt"] == RECORD["prompt"]
        assert got["params"]["seed"] == 42
        assert got["params"]["loras"] == [{"name": "style-a", "strength": 0.8}]
        assert got["params"]["components"] == {"vae": "ae.safetensors"}

    def test_embedded_record_carries_no_local_path_or_timing(self):
        entries = dict(ip.png_text_entries(RECORD))
        blob = json.loads(entries["localm"])
        assert "input_image" not in blob and "runtime" not in blob
        assert "someone" not in entries["localm"] + entries["parameters"]

    def test_a1111_text_is_readable_by_the_a1111_parser(self):
        parsed = ip.parse_a1111(ip.a1111_text(RECORD))
        assert parsed["prompt"] == RECORD["prompt"]
        assert parsed["negative_prompt"] == "blurry"
        assert (parsed["width"], parsed["height"], parsed["seed"], parsed["steps"]) == (
            512, 768, 42, 20)
        assert parsed["cfg"] == 7.0 and parsed["model"] == "sd-turbo"

    def test_sidecar_json_is_read_and_unknown_keys_dropped(self):
        blob = dict(RECORD, evil="x" * 10, created="now")
        got = ip.params_from_bytes(json.dumps(blob).encode())
        assert got["source"] == "sidecar"
        assert "evil" not in got["params"] and "input_image" not in got["params"]
        assert got["params"]["seed"] == 42


A1111_SAMPLE = (
    "masterpiece, 1girl, <lora:detail:0.6> smiling\n"
    "Negative prompt: lowres, bad hands,\nextra fingers\n"
    'Steps: 30, Sampler: DPM++ 2M Karras, CFG scale: 6.5, Seed: 123456789, Size: 640x896, '
    'Model hash: abcd1234, Model: dreamshaper_8, Denoising strength: 0.45, '
    'Lora hashes: "detail: 11aa22bb, other: 33cc44dd", Version: v1.9.0')


class TestA1111:
    def test_real_world_sample(self):
        p = ip.parse_a1111(A1111_SAMPLE)
        assert p["prompt"] == "masterpiece, 1girl,  smiling"
        assert p["negative_prompt"] == "lowres, bad hands,\nextra fingers"
        assert p["steps"] == 30 and p["seed"] == 123456789
        assert (p["width"], p["height"]) == (640, 896)
        assert p["cfg"] == 6.5 and p["denoise"] == 0.45
        assert p["model"] == "dreamshaper_8" and p["sampler"] == "DPM++ 2M Karras"
        assert p["loras"] == [{"name": "detail", "strength": 0.6}]

    def test_without_negative_prompt(self):
        p = ip.parse_a1111("a cat\nSteps: 8, Seed: 1, Size: 64x64")
        assert p["prompt"] == "a cat" and p["negative_prompt"] == ""

    def test_prompt_containing_the_word_steps_is_not_the_settings_line(self):
        p = ip.parse_a1111("Steps: of a staircase\nSteps: 9, Seed: 2, Size: 64x64")
        assert p["prompt"] == "Steps: of a staircase" and p["steps"] == 9

    def test_no_settings_line_is_refused(self):
        with pytest.raises(ip.ImageParamsError):
            ip.parse_a1111("just a prompt\nNegative prompt: x")

    def test_clip_skip_is_imported(self):
        p = ip.parse_a1111("a cat\nSteps: 8, Seed: 1, Size: 64x64, Clip skip: 2")
        assert p["clip_skip"] == 2

    @pytest.mark.parametrize("value", ["0", "13", "-1", "20"])
    def test_a_clip_skip_outside_the_supported_range_is_absent(self, value):
        assert "clip_skip" not in ip.parse_a1111("a cat" + chr(10) + "Steps: 8, Clip skip: " + value)

    def test_a_clip_skip_that_is_not_a_number_is_absent(self):
        assert "clip_skip" not in ip.parse_a1111("a cat\nSteps: 8, Clip skip: lots")

    def test_hires_settings_are_reported_as_not_imported(self):
        line = ("Steps: 20, Seed: 3, Size: 512x512, Denoising strength: 0.5, "
                "Hires upscale: 2, Hires steps: 10, Hires upscaler: Latent")
        got = ip.params_from_bytes(_png([_text("parameters", "a cat\n" + line)]))
        assert got["source"] == "a1111"
        assert got["unsupported"] == ["Hires fix"]
        assert "Not imported: Hires fix." in got["notes"]
        assert "hires" not in got["params"]

    def test_an_image_without_hires_reports_nothing_unsupported(self):
        got = ip.params_from_bytes(_png([_text("parameters", A1111_SAMPLE)]))
        assert "unsupported" not in got and got["notes"] == []

    def test_png_with_only_a_parameters_chunk(self):
        got = ip.params_from_bytes(_png([_text("parameters", A1111_SAMPLE)]))
        assert got["source"] == "a1111" and got["params"]["seed"] == 123456789


COMFY_GRAPH = {
    "3": {"class_type": "KSampler", "inputs": {
        "seed": 777, "steps": 25, "cfg": 5.5, "sampler_name": "euler", "scheduler": "normal",
        "denoise": 1.0, "model": ["4", 0], "positive": ["6", 0], "negative": ["7", 0],
        "latent_image": ["5", 0]}},
    "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "sd_xl_base_1.0.safetensors"}},
    "5": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1344, "batch_size": 1}},
    "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "a lighthouse", "clip": ["4", 1]}},
    "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "text, watermark", "clip": ["4", 1]}},
    "9": {"class_type": "LoraLoader", "inputs": {"lora_name": "xl_detail.safetensors",
                                                  "strength_model": 0.7, "strength_clip": 0.7}},
}


class TestComfyUI:
    def test_graph_is_resolved_through_links(self):
        p = ip.parse_comfy_prompt(COMFY_GRAPH)
        assert p["prompt"] == "a lighthouse" and p["negative_prompt"] == "text, watermark"
        assert (p["seed"], p["steps"], p["cfg"]) == (777, 25, 5.5)
        assert (p["width"], p["height"]) == (1024, 1344)
        assert p["model"] == "sd_xl_base_1.0.safetensors"
        assert p["sampler"] == "euler" and p["scheduler"] == "normal"
        assert p["loras"] == [{"name": "xl_detail.safetensors", "strength": 0.7}]

    def test_flux_guidance_is_followed_to_the_text(self):
        g = json.loads(json.dumps(COMFY_GRAPH))
        g["8"] = {"class_type": "FluxGuidance", "inputs": {"guidance": 3.5, "conditioning": ["6", 0]}}
        g["3"]["inputs"]["positive"] = ["8", 0]
        p = ip.parse_comfy_prompt(g)
        assert p["prompt"] == "a lighthouse" and p["guidance"] == 3.5

    def test_cyclic_links_terminate(self):
        g = {"3": {"class_type": "KSampler", "inputs": {"positive": ["8", 0], "seed": 1}},
             "8": {"class_type": "ConditioningCombine", "inputs": {"conditioning": ["9", 0]}},
             "9": {"class_type": "ConditioningCombine", "inputs": {"conditioning": ["8", 0]}}}
        assert "prompt" not in ip.parse_comfy_prompt(g)

    def test_linked_numeric_inputs_are_not_read_as_numbers(self):
        g = json.loads(json.dumps(COMFY_GRAPH))
        g["3"]["inputs"]["seed"] = ["11", 0]
        assert "seed" not in ip.parse_comfy_prompt(g)

    def test_no_sampler_is_refused(self):
        with pytest.raises(ip.ImageParamsError):
            ip.parse_comfy_prompt({"1": {"class_type": "SaveImage", "inputs": {}}})

    def test_png_with_prompt_and_workflow_chunks(self):
        data = _png([_text("prompt", json.dumps(COMFY_GRAPH)), _text("workflow", "{}")])
        got = ip.params_from_bytes(data)
        assert got["source"] == "comfyui" and got["params"]["seed"] == 777


class TestHostileInput:
    def test_not_a_png_is_refused(self):
        with pytest.raises(ip.ImageParamsError, match="not a PNG"):
            ip.params_from_bytes(b"GIF89a....")

    def test_truncated_png_is_refused(self):
        with pytest.raises(ip.ImageParamsError, match="truncated"):
            ip.read_png_text(ip.PNG_SIGNATURE + struct.pack(">I", 1 << 30) + b"tEXt" + b"abc")

    def test_png_without_parameters_says_so(self):
        with pytest.raises(ip.ImageParamsError, match="no generation parameters"):
            ip.params_from_bytes(_png([_text("Software", "gimp")]))

    def test_compressed_text_bomb_is_skipped_not_inflated(self):
        bomb = zlib.compress(b"A" * (ip.MAX_TEXT_BYTES * 4))
        data = _png([_chunk(b"zTXt", b"parameters\x00\x00" + bomb)])
        assert ip.read_png_text(data) == {}

    def test_itxt_bomb_is_skipped(self):
        bomb = zlib.compress(b"A" * (ip.MAX_TEXT_BYTES * 4))
        body = b"parameters\x00\x01\x00\x00\x00" + bomb
        assert ip.read_png_text(_png([_chunk(b"iTXt", body)])) == {}

    def test_zlib_and_itxt_text_are_decoded(self):
        z = _chunk(b"zTXt", b"k1\x00\x00" + zlib.compress(b"hello"))
        i = _chunk(b"iTXt", b"k2" + b"\x00\x00\x00en\x00\x00" + "h\u00e9llo".encode())
        assert ip.read_png_text(_png([z, i])) == {"k1": "hello", "k2": "h\u00e9llo"}

    def test_malformed_text_chunk_is_skipped(self):
        data = _png([_chunk(b"tEXt", b"no-terminator"), _text("good", "yes")])
        assert ip.read_png_text(data) == {"good": "yes"}

    def test_oversize_body_is_refused(self, monkeypatch):
        monkeypatch.setattr(ip, "MAX_IMPORT_BYTES", 10)
        with pytest.raises(ip.ImageParamsError, match="too large"):
            ip.params_from_bytes(b"x" * 11)

    def test_invalid_json_sidecar_is_refused(self):
        with pytest.raises(ip.ImageParamsError, match="not valid JSON"):
            ip.params_from_bytes(b"{not json")

    def test_deeply_nested_json_is_refused_not_a_crash(self):
        with pytest.raises(ip.ImageParamsError):
            ip.params_from_bytes(b'{"a":[' * 100000)
        with pytest.raises(ip.ImageParamsError):
            ip.params_from_bytes(b"{" + b'"a":{' * 100000)

    def test_localm_record_types_are_checked(self):
        bad = {"prompt": 5, "seed": "7", "cfg": float("inf"), "steps": True,
               "loras": [{"name": 3}, "x"], "components": {"vae": 4}}
        got = ip.params_from_bytes(json.dumps(bad).encode())
        assert got["params"] == {"components": {}}

    def test_long_strings_are_capped(self):
        got = ip.params_from_bytes(json.dumps({"prompt": "x" * 100000}).encode())
        assert len(got["params"]["prompt"]) == 32000

    def test_a_bad_localm_chunk_falls_back_to_the_a1111_one(self):
        data = _png([_text("localm", "{broken"), _text("parameters", A1111_SAMPLE)])
        got = ip.params_from_bytes(data)
        assert got["source"] == "a1111" and got["notes"]


FLUX_GRAPH = {
    "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "a neon street", "clip": ["11", 0]}},
    "10": {"class_type": "FluxGuidance", "inputs": {"guidance": 3.5, "conditioning": ["6", 0]}},
    "13": {"class_type": "BasicGuider", "inputs": {"model": ["12", 0], "conditioning": ["10", 0]}},
    "14": {"class_type": "RandomNoise", "inputs": {"noise_seed": 42}},
    "15": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
    "16": {"class_type": "BasicScheduler", "inputs": {"model": ["12", 0], "scheduler": "simple",
                                                      "steps": 20, "denoise": 1.0}},
    "17": {"class_type": "EmptySD3LatentImage", "inputs": {"width": 1024, "height": 768, "batch_size": 1}},
    "12": {"class_type": "UNETLoader", "inputs": {"unet_name": "flux1-dev.safetensors"}},
    "25": {"class_type": "SamplerCustomAdvanced", "inputs": {
        "noise": ["14", 0], "guider": ["13", 0], "sampler": ["15", 0], "sigmas": ["16", 0],
        "latent_image": ["17", 0]}},
}


class TestComfyCustomSamplers:
    def test_stock_flux_graph_resolves_through_guider_noise_and_sigmas(self):
        p = ip.parse_comfy_prompt(FLUX_GRAPH)
        assert p["prompt"] == "a neon street" and p["guidance"] == 3.5
        assert (p["seed"], p["steps"]) == (42, 20)
        assert p["sampler"] == "euler" and p["scheduler"] == "simple"
        assert (p["width"], p["height"]) == (1024, 768)
        assert p["model"] == "flux1-dev.safetensors"

    def test_a_graph_with_no_prompt_text_says_so_and_has_no_prompt_key(self):
        g = json.loads(json.dumps(FLUX_GRAPH))
        g["25"]["inputs"]["guider"] = ["99", 0]
        got = ip.params_from_bytes(_png([_text("prompt", json.dumps(g))]))
        assert "prompt" not in got["params"] and "negative_prompt" not in got["params"]
        assert any("No prompt text" in n for n in got["notes"])

    def test_sampler_custom_reads_its_direct_inputs(self):
        g = {"1": {"class_type": "SamplerCustom", "inputs": {
                "positive": ["2", 0], "negative": ["3", 0], "cfg": 4.5, "noise_seed": 8,
                "sampler": ["4", 0], "sigmas": ["5", 0], "latent_image": ["6", 0]}},
             "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "pos"}},
             "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "neg"}},
             "4": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "dpmpp_2m"}},
             "5": {"class_type": "BasicScheduler", "inputs": {"scheduler": "karras", "steps": 12, "denoise": 1}},
             "6": {"class_type": "EmptyLatentImage", "inputs": {"width": 64, "height": 64}}}
        p = ip.parse_comfy_prompt(g)
        assert (p["prompt"], p["negative_prompt"], p["cfg"], p["seed"]) == ("pos", "neg", 4.5, 8)
        assert (p["steps"], p["sampler"], p["scheduler"]) == (12, "dpmpp_2m", "karras")


class TestBoundedWork:
    def _timed(self, fn):
        import time
        t0 = time.perf_counter()
        try:
            fn()
        except ip.ImageParamsError:
            pass
        return time.perf_counter() - t0

    def test_a_lora_tag_with_a_long_unclosed_weight_is_linear(self):
        text = "<lora:a:" + "1" * 200000 + "\nSteps: 20"
        assert self._timed(lambda: ip.parse_a1111(text)) < 1.0

    def test_a_settings_line_of_colonless_words_is_linear(self):
        text = "p\nSteps: 20, " + "a " * 400000
        assert self._timed(lambda: ip.parse_a1111(text)) < 1.0

    def test_many_colons_and_quotes_in_the_settings_line_are_linear(self):
        text = "p\nSteps: 20, " + 'k: "' * 100000 + ", " + ("a: b, " * 100000)
        assert self._timed(lambda: ip.parse_a1111(text)) < 1.0

    def test_a_parameters_text_over_the_cap_is_refused(self):
        with pytest.raises(ip.ImageParamsError, match="too long"):
            ip.parse_a1111("x" * (ip.MAX_PARAMETERS_CHARS + 1))

    def test_total_decoded_text_is_capped_across_many_chunks(self):
        zeros = zlib.compress(b"\x00" * (ip.MAX_TEXT_BYTES - 10))
        chunks = [_chunk(b"zTXt", f"k{i}".encode() + b"\x00\x00" + zeros) for i in range(40)]
        got = ip.read_png_text(_png(chunks))
        assert sum(len(v) for v in got.values()) <= ip.MAX_TOTAL_TEXT_BYTES
        assert 0 < len(got) < 40

    def test_only_wanted_keywords_are_decoded(self):
        zeros = zlib.compress(b"\x00" * 1000)
        data = _png([_chunk(b"zTXt", b"junk\x00\x00" + zeros), _text("parameters", "x")])
        assert ip.read_png_text(data, ip.WANTED_KEYWORDS) == {"parameters": "x"}

    def test_a_duplicate_keyword_is_not_decoded_again(self):
        zeros = zlib.compress(b"\x00" * (ip.MAX_TEXT_BYTES - 10))
        chunks = [_text("parameters", "first")] + [
            _chunk(b"zTXt", b"parameters\x00\x00" + zeros) for _ in range(30)]
        assert ip.read_png_text(_png(chunks)) == {"parameters": "first"}


class TestNumbersStayFinite:
    def test_huge_or_non_finite_numbers_are_dropped_not_raised(self):
        blob = (b'{"prompt":"x","seed":' + b"9" * 300 + b',"steps":1e999,"cfg":Infinity,'
                b'"loras":[{"name":"a","strength":' + b"1" + b"0" * 400 + b"},"
                b'{"name":"b","strength":NaN},{"name":"c","strength":Infinity}]}')
        got = ip.params_from_bytes(blob)["params"]
        assert "seed" not in got and "steps" not in got and "cfg" not in got
        assert got["loras"] == [{"name": "a", "strength": 1.0}, {"name": "b", "strength": 1.0},
                                {"name": "c", "strength": 1.0}]
        json.dumps(got, allow_nan=False)

    def test_a1111_and_comfy_values_pass_the_same_filter(self):
        got = ip.params_from_bytes(_png([_text("parameters", "p\nSteps: 20, Seed: " + "9" * 300)]))
        assert "seed" not in got["params"] and got["params"]["steps"] == 20


class TestImportNotes:
    def test_a_comfy_graph_without_a_sampler_names_that_not_invalid_json(self):
        data = _png([_text("prompt", json.dumps({"1": {"class_type": "SaveImage", "inputs": {}}}))])
        with pytest.raises(ip.ImageParamsError) as e:
            ip.params_from_bytes(data)
        assert "no sampler node" in str(e.value) and "not valid JSON" not in str(e.value)

    def test_invalid_comfy_json_says_so(self):
        with pytest.raises(ip.ImageParamsError, match="not valid JSON"):
            ip.params_from_bytes(_png([_text("prompt", "{broken")]))


def test_a_localm_sidecar_brings_its_hires_settings_back_with_the_sampler():
    import json

    from localm.media.image_params import params_from_bytes
    blob = {"prompt": "p", "sample_method": "euler_a", "scheduler": "karras",
            "hires": {"upscaler": "model", "model": "real-esrgan", "scale": 1.5, "steps": 4,
                      "denoise": 0.4}}
    got = params_from_bytes(json.dumps(blob).encode())
    assert got["source"] == "sidecar"
    assert got["params"]["hires"] == blob["hires"]
    assert (got["params"]["sample_method"], got["params"]["scheduler"]) == ("euler_a", "karras")


def test_a_hires_record_with_a_bad_part_is_dropped_whole():
    import json

    from localm.media.image_params import params_from_bytes
    good = {"upscaler": "latent", "model": None, "scale": 2.0, "steps": 0, "denoise": 0.7}
    bad = [dict(good, upscaler="nope"), dict(good, scale=99), dict(good, scale="2"),
           dict(good, steps=-1), dict(good, denoise=0), dict(good, upscaler="model", model=None),
           dict(good, denoise=float("nan")), "latent", ["latent"]]
    for hires in bad:
        raw = json.dumps({"prompt": "p", "hires": hires}, allow_nan=True).encode()
        assert "hires" not in params_from_bytes(raw)["params"], hires
    ok = params_from_bytes(json.dumps({"prompt": "p", "hires": good}).encode())["params"]
    assert ok["hires"] == good

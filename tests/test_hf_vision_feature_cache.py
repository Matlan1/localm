# SPDX-License-Identifier: AGPL-3.0-or-later
"""The HF backend's vision feature cache (``_VisionFeatureCache`` in
``_hf_worker.py``).

Real transformers code throughout: a tiny LLaVA (CLIP vision tower + Llama text
model) is built from a config with seeded random weights and a real
``LlavaProcessor`` (CLIP image processor + an in-memory word-level tokenizer),
so nothing is downloaded and ``HFWorker.chat_stream`` runs its real image path
on CPU. The vision tower's forward calls are counted with a forward pre-hook.
"""

from __future__ import annotations

import base64
import io
from typing import List

import pytest

torch = pytest.importorskip("torch", exc_type=ImportError)
transformers = pytest.importorskip("transformers", exc_type=ImportError)
Image = pytest.importorskip("PIL.Image")

from localm.inference.backends import _hf_worker as hfmod  # noqa: E402

_WORDS = ["user", "assistant", ":", "what", "is", "this", "and", "now", "the",
          "red", "blue", "green", "circle", "square", "picture", "two", "more", "?"]
_SPECIALS = ["<pad>", "<unk>", "<s>", "</s>", "<image>"]
_CHAT_TEMPLATE = (
    "{% for m in messages %}{{ m['role'] }} : "
    "{% if m['content'] is string %}{{ m['content'] }}"
    "{% else %}{% for p in m['content'] %}"
    "{% if p['type'] == 'image' %}<image>{% else %}{{ p['text'] }}{% endif %} "
    "{% endfor %}{% endif %}\n{% endfor %}"
    "{% if add_generation_prompt %}assistant :{% endif %}"
)
_IMAGE_SIZE = 32
_PATCH = 8


@pytest.fixture(autouse=True)
def _skip_if_native_runtime_already_loaded():
    # A fresh torch import after llama.cpp's native runtime loaded in the same
    # process is the known DLL-identity conflict, not this file's subject.
    from localm.inference.backends.llamacpp import _loader
    if _loader.native_lib_loaded():
        pytest.skip("llama.cpp's native runtime is already loaded in this process")


def _tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    vocab = {tok: i for i, tok in enumerate(_SPECIALS + _WORDS)}
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    fast = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tok, unk_token="<unk>", pad_token="<pad>",
        bos_token="<s>", eos_token="</s>", extra_special_tokens={"image_token": "<image>"})
    fast.chat_template = _CHAT_TEMPLATE
    return fast


def _processor(tokenizer):
    image_processor = transformers.CLIPImageProcessor(
        size={"shortest_edge": _IMAGE_SIZE},
        crop_size={"height": _IMAGE_SIZE, "width": _IMAGE_SIZE})
    return transformers.LlavaProcessor(
        image_processor=image_processor, tokenizer=tokenizer,
        patch_size=_PATCH, vision_feature_select_strategy="default",
        num_additional_image_tokens=1, image_token="<image>",
        chat_template=_CHAT_TEMPLATE)


def _model(tokenizer):
    torch.manual_seed(1234)
    config = transformers.LlavaConfig(
        vision_config=transformers.CLIPVisionConfig(
            hidden_size=32, intermediate_size=48, num_hidden_layers=2,
            num_attention_heads=4, image_size=_IMAGE_SIZE, patch_size=_PATCH),
        text_config=transformers.LlamaConfig(
            vocab_size=len(_SPECIALS) + len(_WORDS), hidden_size=32,
            intermediate_size=48, num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=4, max_position_embeddings=512,
            bos_token_id=2, eos_token_id=3, pad_token_id=0),
        image_token_id=tokenizer.convert_tokens_to_ids("<image>"),
        vision_feature_layer=-1, vision_feature_select_strategy="default")
    model = transformers.LlavaForConditionalGeneration(config).eval()
    model.generation_config.eos_token_id = 3
    model.generation_config.pad_token_id = 0
    return model


class _VisionCalls:
    """Counts the vision tower's forward calls and their batch sizes."""

    def __init__(self, model):
        self.batches: List[int] = []
        model.model.vision_tower.register_forward_pre_hook(self._pre, with_kwargs=True)

    def _pre(self, _module, args, kwargs):
        pixel_values = args[0] if args else kwargs["pixel_values"]
        self.batches.append(int(pixel_values.shape[0]))

    def take(self) -> List[int]:
        out, self.batches = self.batches, []
        return out


def _worker(model, processor, *, cache=True):
    worker = hfmod.HFWorker.__new__(hfmod.HFWorker)
    worker._model = model
    worker._processor = processor
    worker._tokenizer = processor.tokenizer
    worker._is_multimodal = True
    worker._supports_image = True
    worker._supports_audio = False
    worker._loaded = True
    worker.context_capacity = 512
    worker.last_finish_reason = "stop"
    worker._vision_cache = hfmod._install_vision_feature_cache(model) if cache else None
    return worker


def _image(color, size=(48, 40), dot=None):
    image = Image.new("RGB", size, color)
    if dot is not None:
        image.putpixel(dot, (1, 2, 3))
    return image


def _url(image) -> str:
    buf = io.BytesIO()
    image.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _user(text, *images):
    return {"role": "user", "content": [{"type": "text", "text": text}] + [
        {"type": "image_url", "image_url": {"url": _url(im)}} for im in images]}


def _run(worker, messages, status=None):
    return "".join(worker.chat_stream(
        messages, max_tokens=6, temperature=0.0, repeat_penalty=1.0,
        on_status=status.append if status is not None else None))


@pytest.fixture(scope="module")
def tiny():
    tokenizer = _tokenizer()
    return _model(tokenizer), _processor(tokenizer)


@pytest.fixture
def setup(tiny):
    model, processor = tiny
    if "get_image_features" in vars(model.model):
        del model.model.get_image_features
    calls = _VisionCalls(model)
    yield model, processor, calls
    model.model.vision_tower._forward_pre_hooks.clear()
    if "get_image_features" in vars(model.model):
        del model.model.get_image_features


def _get_features(model, processor, image):
    inputs = processor(text="<image>", images=[image], return_tensors="pt")
    return model.model.get_image_features(
        inputs["pixel_values"], vision_feature_layer=-1,
        vision_feature_select_strategy="default", return_dict=True)


def _conversation(worker, turns):
    """Run *turns* (user messages) as one growing conversation; return the
    replies."""
    messages, replies = [], []
    for user in turns:
        messages.append(user)
        reply = _run(worker, messages)
        replies.append(reply)
        messages.append({"role": "assistant", "content": reply})
    return replies


class TestFollowUpTurns:
    def test_a_follow_up_turn_does_not_run_the_vision_tower(self, setup):
        model, processor, calls = setup
        worker = _worker(model, processor)
        red = _image("red")
        messages = [_user("what is this ?", red)]
        _run(worker, messages)
        assert calls.take() == [1]
        messages += [{"role": "assistant", "content": "the circle"},
                     {"role": "user", "content": "and now ?"}]
        status: List[str] = []
        _run(worker, messages, status)
        assert calls.take() == []
        assert "Encoding image..." not in status
        assert status[0] == "Processing prompt..."

    def test_a_new_image_encodes_only_that_image(self, setup):
        model, processor, calls = setup
        worker = _worker(model, processor)
        red, blue = _image("red"), _image("blue")
        messages = [_user("what is this ?", red)]
        _run(worker, messages)
        calls.take()
        messages += [{"role": "assistant", "content": "red"},
                     _user("and this ?", blue)]
        status: List[str] = []
        _run(worker, messages, status)
        assert calls.take() == [1]
        assert status[0] == "Encoding image..."
        assert worker._vision_cache.last_encoded == 1

    def test_a_changed_pixel_is_a_miss(self, setup):
        model, processor, calls = setup
        worker = _worker(model, processor)
        _run(worker, [_user("what is this ?", _image("red"))])
        calls.take()
        _run(worker, [_user("what is this ?", _image("red", dot=(3, 4)))])
        assert calls.take() == [1]

    def test_the_same_image_twice_in_one_request_is_encoded_once(self, setup):
        model, processor, calls = setup
        worker = _worker(model, processor)
        red = _image("red")
        _run(worker, [_user("two picture", red, red)])
        assert calls.take() == [1]

    def test_replies_match_encoding_every_image_on_every_turn(self, setup):
        model, processor, calls = setup
        red, blue, green = _image("red"), _image("blue", (40, 64)), _image("green", (64, 30))
        turns = [
            _user("what is this ?", red),
            {"role": "user", "content": "and now ?"},
            _user("and this ?", blue),
            {"role": "user", "content": "the circle ?"},
            _user("two more", green, red),
        ]
        cached = _conversation(_worker(model, processor), turns)
        cached_calls = calls.take()

        uncached_worker = _worker(model, processor)
        messages, uncached = [], []
        for user in turns:
            messages.append(user)
            reply = _run(uncached_worker, messages)
            uncached_worker._vision_cache.clear()
            uncached.append(reply)
            messages.append({"role": "assistant", "content": reply})
        uncached_calls = calls.take()

        assert cached == uncached
        assert any(cached)
        assert cached_calls == [1, 1, 1]
        assert uncached_calls == [1, 1, 1, 1, 1, 1, 1, 1, 1]

    def test_cached_features_equal_a_fresh_encode(self, setup):
        model, processor, _calls = setup
        worker = _worker(model, processor)
        red = _image("red")
        cache = worker._vision_cache
        inputs = processor(text="<image>", images=[red], return_tensors="pt")
        with torch.no_grad():
            fresh = type(model.model).get_image_features(
                model.model, inputs["pixel_values"], vision_feature_layer=-1,
                vision_feature_select_strategy="default", return_dict=True).pooler_output
        _run(worker, [_user("what is this ?", red)])
        (stored,) = cache._features.values()
        assert torch.equal(torch.cat(list(stored["pooler_output"])), torch.cat(list(fresh)))


class TestBounds:
    def test_images_no_longer_in_the_request_are_dropped(self, setup):
        model, processor, _calls = setup
        worker = _worker(model, processor)
        red, blue = _image("red"), _image("blue")
        _run(worker, [_user("two picture", red, blue)])
        assert len(worker._vision_cache) == 2
        _run(worker, [_user("what is this ?", blue)])
        assert len(worker._vision_cache) == 1
        assert hfmod._image_content_key(blue) in worker._vision_cache

    def test_a_request_without_images_empties_the_cache(self, setup):
        model, processor, _calls = setup
        worker = _worker(model, processor)
        _run(worker, [_user("what is this ?", _image("red"))])
        assert len(worker._vision_cache) == 1
        _run(worker, [{"role": "user", "content": "what is this ?"}])
        assert len(worker._vision_cache) == 0

    def test_the_cache_holds_at_most_max_images(self, setup):
        model, processor, calls = setup
        worker = _worker(model, processor)
        worker._vision_cache._max_images = 2
        images = [_image(c) for c in ("red", "blue", "green")]
        _run(worker, [_user("two more", *images)])
        assert calls.take() == [1, 1, 1]
        assert len(worker._vision_cache) == 2
        assert hfmod._image_content_key(images[0]) not in worker._vision_cache

    def test_unload_empties_the_cache_and_the_next_load_encodes_again(self, setup, monkeypatch):
        model, processor, calls = setup
        worker = _worker(model, processor)
        red = _image("red")
        _run(worker, [_user("what is this ?", red)])
        cache = worker._vision_cache
        assert len(cache) == 1
        worker.unload()
        assert len(cache) == 0
        assert worker._vision_cache is None
        assert "get_image_features" not in vars(model.model)
        reloaded = _worker(model, processor)
        calls.take()
        _run(reloaded, [_user("what is this ?", red)])
        assert calls.take() == [1]


class TestRefusedRequests:
    def test_a_refused_request_leaves_nothing_armed(self, setup, monkeypatch):
        from localm.inference.backends.base import ContextCapacityExceededError
        model, processor, calls = setup
        worker = _worker(model, processor)
        red, blue = _image("red"), _image("blue")
        worker.context_capacity = 4
        with pytest.raises(ContextCapacityExceededError):
            _run(worker, [_user("what is this ?", red)])
        assert worker._vision_cache._armed is None
        assert calls.take() == []

        worker.context_capacity = 512
        fed = []
        model.model.vision_tower.register_forward_pre_hook(
            lambda _m, args, kwargs: fed.append(
                (args[0] if args else kwargs["pixel_values"]).clone()),
            with_kwargs=True)

        def _refuse(*_a, **_k):
            raise RuntimeError("per-image processing failed")

        monkeypatch.setattr(worker, "_single_image_inputs", _refuse)
        _run(worker, [_user("what is this ?", blue)])
        blue_pixels = processor(text="<image>", images=[blue],
                                return_tensors="pt")["pixel_values"]
        assert len(fed) == 1
        assert torch.equal(fed[0], blue_pixels)


class TestPassThrough:
    def test_an_unarmed_call_runs_the_original_method(self, setup):
        model, processor, calls = setup
        worker = _worker(model, processor)
        inputs = processor(text="<image>", images=[_image("red")], return_tensors="pt")
        with torch.no_grad():
            for _ in range(2):
                model.model.get_image_features(
                    inputs["pixel_values"], vision_feature_layer=-1,
                    vision_feature_select_strategy="default", return_dict=True)
        assert calls.take() == [1, 1]
        assert len(worker._vision_cache) == 0

    def test_a_call_after_a_request_runs_the_original_method(self, setup):
        model, processor, calls = setup
        worker = _worker(model, processor)
        red = _image("red")
        _run(worker, [_user("what is this ?", red)])
        calls.take()
        inputs = processor(text="<image>", images=[red], return_tensors="pt")
        with torch.no_grad():
            model.model.get_image_features(
                inputs["pixel_values"], vision_feature_layer=-1,
                vision_feature_select_strategy="default", return_dict=True)
        assert calls.take() == [1]

    def test_a_call_with_other_pixel_values_runs_the_original_method(self, setup):
        model, processor, calls = setup
        worker = _worker(model, processor)
        cache = worker._vision_cache
        red = _image("red")
        _run(worker, [_user("what is this ?", red)])
        calls.take()
        cache.arm([hfmod._image_content_key(red)], {}, (9, 9, 9, 9))
        with torch.no_grad():
            _get_features(model, processor, red)
        assert calls.take() == [1]

    def test_an_armed_request_serves_exactly_one_call(self, setup):
        model, processor, calls = setup
        worker = _worker(model, processor)
        cache = worker._vision_cache
        red = _image("red")
        _run(worker, [_user("what is this ?", red)])
        calls.take()
        pixel_shape = tuple(processor(text="<image>", images=[red],
                                      return_tensors="pt")["pixel_values"].shape)
        cache.arm([hfmod._image_content_key(red)], {}, pixel_shape)
        with torch.no_grad():
            served = _get_features(model, processor, red)
            again = _get_features(model, processor, red)
        assert calls.take() == [1]
        assert torch.equal(torch.cat(list(served.pooler_output)),
                           torch.cat(list(again.pooler_output)))

    def test_missing_lists_each_uncached_key_once(self, setup):
        model, processor, _calls = setup
        cache = _worker(model, processor)._vision_cache
        assert cache.missing(["a", "b", "a"]) == ["a", "b"]

    def test_a_worker_without_a_cache_encodes_every_turn(self, setup):
        model, processor, calls = setup
        worker = _worker(model, processor, cache=False)
        messages = [_user("what is this ?", _image("red"))]
        _run(worker, messages)
        messages += [{"role": "assistant", "content": "red"},
                     {"role": "user", "content": "and now ?"}]
        _run(worker, messages)
        assert calls.take() == [1, 1]

    def test_only_listed_model_classes_get_a_cache(self, setup, monkeypatch):
        model, _processor, _calls = setup
        assert type(model.model).__name__ in hfmod._VISION_CACHE_MODEL_CLASSES
        text_only = transformers.LlamaForCausalLM(model.config.text_config)
        assert hfmod._install_vision_feature_cache(text_only) is None
        monkeypatch.setattr(hfmod, "_VISION_CACHE_MODEL_CLASSES", frozenset())
        assert hfmod._install_vision_feature_cache(model) is None
        assert "get_image_features" not in vars(model.model)


class TestJoinImageFeatures:
    def test_tensor_outputs_are_concatenated_in_request_order(self):
        a = {"pooler_output": torch.full((1, 2, 3), 1.0)}
        b = {"pooler_output": torch.full((2, 2, 3), 2.0)}
        joined = hfmod._join_image_features([b, a])
        assert joined["pooler_output"].shape == (3, 2, 3)
        assert joined["pooler_output"][:, 0, 0].tolist() == [2.0, 2.0, 1.0]
        assert "deepstack_features" not in joined

    def test_tuple_outputs_are_joined_into_one_tuple(self):
        a = {"pooler_output": (torch.zeros(4, 3),)}
        b = {"pooler_output": (torch.ones(2, 3), torch.ones(5, 3))}
        joined = hfmod._join_image_features([a, b])
        assert [t.shape[0] for t in joined["pooler_output"]] == [4, 2, 5]

    def test_deepstack_layers_are_joined_layer_by_layer(self):
        a = {"pooler_output": (torch.zeros(2, 3),),
             "deepstack_features": [torch.full((2, 3), 1.0), torch.full((2, 3), 2.0)]}
        b = {"pooler_output": (torch.zeros(1, 3),),
             "deepstack_features": [torch.full((1, 3), 3.0), torch.full((1, 3), 4.0)]}
        joined = hfmod._join_image_features([a, b])
        layers = joined["deepstack_features"]
        assert len(layers) == 2
        assert layers[0][:, 0].tolist() == [1.0, 1.0, 3.0]
        assert layers[1][:, 0].tolist() == [2.0, 2.0, 4.0]

    def test_deepstack_is_left_out_when_an_entry_lacks_it(self):
        a = {"pooler_output": (torch.zeros(2, 3),), "deepstack_features": [torch.zeros(2, 3)]}
        b = {"pooler_output": (torch.zeros(1, 3),)}
        assert "deepstack_features" not in hfmod._join_image_features([a, b])

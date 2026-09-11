# SPDX-License-Identifier: AGPL-3.0-or-later
"""HFWorker.chat_stream's audio call shape.

transformers processors take ``audio=`` (a raw waveform or list of them) plus
a separate ``sampling_rate=``, never the ``(waveform, rate)`` tuple shape
``localm.inference.media.decode_audio`` returns, and never under the name
``audios``. These tests drive the real decode-and-call path with a recording
fake processor, so a defect in the kwarg shape shows up as a wrong recorded
call rather than a wrong assertion about intent.
"""

import base64
import io

import pytest

np = pytest.importorskip("numpy")
sf = pytest.importorskip("soundfile")

from localm.inference.backends._hf_worker import HFWorker  # noqa: E402
from localm.inference.backends.base import UnsupportedInputError  # noqa: E402


def _audio_message(rate=8000, duration_s=0.1):
    samples = np.zeros(int(rate * duration_s), dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, samples, rate, format="WAV")
    data = base64.b64encode(buf.getvalue()).decode("ascii")
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": "listen to this"},
            {"type": "input_audio", "input_audio": {"data": data, "format": "wav"}},
        ],
    }


class _StubDevice:
    pass


class _StubModel:
    device = _StubDevice()


def _worker_with_processor(processor):
    worker = HFWorker("does-not-matter", device="cpu")
    worker._model = _StubModel()
    worker._processor = processor
    worker._tokenizer = object()
    worker._is_multimodal = True
    worker._supports_image = False
    worker._supports_audio = True
    worker._loaded = True
    return worker


class _StopAfterCall(Exception):
    pass


class _RecordingProcessor:
    """A processor with no readable expected sample rate (neither
    feature_extractor nor audio_processor is set), so the call-shape fix must
    proceed to call it rather than refuse up front."""

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, *args, **kwargs):
        return "PROMPT"

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        raise _StopAfterCall()


class TestAudioCallShape:
    def test_audio_passed_as_audio_kwarg_with_sampling_rate(self):
        worker = _worker_with_processor(_RecordingProcessor())
        messages = [_audio_message(rate=8000)]

        recorded = None
        try:
            next(worker.chat_stream(messages))
        except _StopAfterCall:
            recorded = worker._processor.calls[-1] if worker._processor.calls else None

        assert recorded is not None, "the processor was never called"
        assert "audio" in recorded
        assert "audios" not in recorded
        assert "sampling_rate" in recorded
        assert recorded["sampling_rate"] == 8000
        value = recorded["audio"]
        if isinstance(value, list):
            assert all(isinstance(v, np.ndarray) for v in value)
            assert not any(isinstance(v, tuple) for v in value)
        else:
            assert isinstance(value, np.ndarray)


class TestAudioProcessorSignatureIsPinned:
    def test_qwen2audio_call_signature_uses_audio_not_audios(self):
        transformers = pytest.importorskip("transformers")
        import inspect
        params = list(
            inspect.signature(transformers.Qwen2AudioProcessor.__call__).parameters)
        assert "audio" in params
        assert "audios" not in params


class _FakeAudioExtractor:
    def __init__(self, rate):
        self.sampling_rate = rate


class _MismatchAwareProcessor:
    """Exposes the model's expected rate via *attr* (``feature_extractor`` or
    ``audio_processor``, the two real attribute shapes) and records whether
    it was ever called."""

    def __init__(self, rate, attr):
        setattr(self, attr, _FakeAudioExtractor(rate))
        self.called = False

    def apply_chat_template(self, *args, **kwargs):
        return "PROMPT"

    def __call__(self, **kwargs):
        self.called = True
        return kwargs


class TestAudioSampleRateMismatch:
    @pytest.mark.parametrize("attr", ["feature_extractor", "audio_processor"])
    def test_mismatch_is_refused_before_reaching_the_processor(self, attr):
        processor = _MismatchAwareProcessor(16000, attr)
        worker = _worker_with_processor(processor)
        messages = [_audio_message(rate=8000)]

        exc = None
        try:
            next(worker.chat_stream(messages))
        except UnsupportedInputError as e:
            exc = e

        assert processor.called is False, (
            "the processor must not be called at all once the rate "
            "mismatch is known in advance"
        )
        assert exc is not None, "no UnsupportedInputError was raised"
        assert "8000" in str(exc)
        assert "16000" in str(exc)


class _UnreadableRateProcessor:
    """Exposes an audio attribute with no ``sampling_rate`` of its own, and
    rejects the call the way a real feature extractor rejects a bad rate."""

    def __init__(self):
        self.feature_extractor = object()

    def apply_chat_template(self, *args, **kwargs):
        return "PROMPT"

    def __call__(self, **kwargs):
        raise ValueError("Sampling rate mismatch: expected 16000, got 8000")


class TestAudioUnreadableExpectedRate:
    def test_unreadable_rate_value_error_becomes_unsupported_input(self):
        worker = _worker_with_processor(_UnreadableRateProcessor())
        messages = [_audio_message(rate=8000)]

        exc = None
        try:
            next(worker.chat_stream(messages))
        except UnsupportedInputError as e:
            exc = e
        except ValueError:
            pytest.fail(
                "a raw ValueError escaped chat_stream instead of being "
                "converted to UnsupportedInputError - this is the arm that "
                "kills the worker and evicts the model"
            )

        assert exc is not None, "no UnsupportedInputError was raised"

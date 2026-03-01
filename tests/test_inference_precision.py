import torch
import time

from src.models.data_models import Config
import src.core.yourmt3_transcriber as yourmt3_transcriber


class _DummyModel:
    def inference(self, x, _task_tokens):
        return torch.zeros((x.shape[0], 1, 4), dtype=torch.float32)


def test_inference_uses_full_precision_without_autocast(monkeypatch):
    monkeypatch.setattr(yourmt3_transcriber, "get_device", lambda *_: "cuda:0")
    transcriber = yourmt3_transcriber.YourMT3Transcriber(Config())

    autocast_calls = []

    class _DummyAutocast:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_autocast(*args, **kwargs):
        autocast_calls.append((args, kwargs))
        return _DummyAutocast()

    monkeypatch.setattr(torch, "autocast", fake_autocast)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False, raising=False)
    now = {"t": 0.0}
    monkeypatch.setattr(
        time,
        "time",
        lambda: (now.__setitem__("t", now["t"] + 0.1) or now["t"]),
    )

    audio_segments = torch.zeros((2, 1, 32), dtype=torch.float32)
    preds = transcriber._inference_with_oom_retry(
        _DummyModel(),
        bsz=2,
        audio_segments=audio_segments,
    )

    assert len(preds) == 1
    assert preds[0].shape == (2, 1, 4)
    assert autocast_calls == []

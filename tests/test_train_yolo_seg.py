import train_yolo_seg


def test_train_calls_YOLO_train_with_explicit_device(monkeypatch):
    calls = {}

    class _FakeYOLO:
        def __init__(self, base_model):
            calls["base_model"] = base_model

        def train(self, **kwargs):
            calls["train_kwargs"] = kwargs

    monkeypatch.setattr(train_yolo_seg, "YOLO", _FakeYOLO)

    train_yolo_seg.train("dataset.yaml", base_model="yolov8n-seg.pt", epochs=5, device="cpu")

    assert calls["base_model"] == "yolov8n-seg.pt"
    assert calls["train_kwargs"] == {"data": "dataset.yaml", "epochs": 5, "single_cls": True, "device": "cpu"}


def test_train_auto_detects_device_when_not_given(monkeypatch):
    calls = {}

    class _FakeYOLO:
        def __init__(self, base_model):
            pass

        def train(self, **kwargs):
            calls["train_kwargs"] = kwargs

    monkeypatch.setattr(train_yolo_seg, "YOLO", _FakeYOLO)
    monkeypatch.setattr(train_yolo_seg, "_auto_device", lambda: "mps")

    train_yolo_seg.train("dataset.yaml", epochs=5)

    assert calls["train_kwargs"]["device"] == "mps"


def test_auto_device_prefers_cuda_then_mps_then_cpu(monkeypatch):
    monkeypatch.setattr(train_yolo_seg.torch.cuda, "is_available", lambda: True)
    assert train_yolo_seg._auto_device() == 0

    monkeypatch.setattr(train_yolo_seg.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(train_yolo_seg.torch.backends.mps, "is_available", lambda: True)
    assert train_yolo_seg._auto_device() == "mps"

    monkeypatch.setattr(train_yolo_seg.torch.backends.mps, "is_available", lambda: False)
    assert train_yolo_seg._auto_device() == "cpu"

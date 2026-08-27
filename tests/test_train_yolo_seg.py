import train_yolo_seg


def test_train_calls_YOLO_train_with_expected_args(monkeypatch):
    calls = {}

    class _FakeYOLO:
        def __init__(self, base_model):
            calls["base_model"] = base_model

        def train(self, **kwargs):
            calls["train_kwargs"] = kwargs

    monkeypatch.setattr(train_yolo_seg, "YOLO", _FakeYOLO)

    train_yolo_seg.train("dataset.yaml", base_model="yolov8n-seg.pt", epochs=5)

    assert calls["base_model"] == "yolov8n-seg.pt"
    assert calls["train_kwargs"] == {"data": "dataset.yaml", "epochs": 5, "single_cls": True}

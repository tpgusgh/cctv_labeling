import cv2
import numpy as np
import pytest

import slot_classifier

CROP_SIZE = 60


def _uniform_crop(path, gray_value):
    # ponytail: uniform floor-colored patch stands in for a real "empty slot"
    # crop -- low edge density, low saturation, matches a high confidence.
    img = np.full((CROP_SIZE, CROP_SIZE, 3), gray_value, dtype=np.uint8)
    cv2.imwrite(str(path), img)


def _noisy_crop(path, seed):
    # ponytail: random noise stands in for a real false-positive crop (sign/
    # structure/reflection) -- high edge density, high color variance, low
    # confidence.
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, size=(CROP_SIZE, CROP_SIZE, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)


def _make_labels(tmp_path, n=10):
    labels = []
    for i in range(n):
        accept_path = tmp_path / f"accept_{i}.png"
        _uniform_crop(accept_path, gray_value=180)
        labels.append({"crop_path": str(accept_path), "confidence": 0.9, "decision": "accept"})

        reject_path = tmp_path / f"reject_{i}.png"
        _noisy_crop(reject_path, seed=i)
        labels.append({"crop_path": str(reject_path), "confidence": 0.3, "decision": "reject"})
    return labels


def test_train_raises_without_both_classes():
    labels = [{"crop_path": "x.png", "confidence": 0.9, "decision": "accept"}]
    with pytest.raises(ValueError):
        slot_classifier.train(labels)


def test_train_raises_with_no_labels():
    with pytest.raises(ValueError):
        slot_classifier.train([])


def test_trained_classifier_separates_held_out_accept_and_reject_examples(tmp_path):
    labels = _make_labels(tmp_path)
    model = slot_classifier.train(labels)

    held_out_accept = tmp_path / "held_out_accept.png"
    _uniform_crop(held_out_accept, gray_value=185)
    held_out_reject = tmp_path / "held_out_reject.png"
    _noisy_crop(held_out_reject, seed=999)

    accept_features = slot_classifier.extract_features(cv2.imread(str(held_out_accept)), 0.9)
    reject_features = slot_classifier.extract_features(cv2.imread(str(held_out_reject)), 0.3)

    assert model.predict([accept_features])[0] == 1
    assert model.predict([reject_features])[0] == 0


def test_save_and_load_round_trip(tmp_path):
    labels = _make_labels(tmp_path)
    model = slot_classifier.train(labels)

    model_path = tmp_path / "model.joblib"
    slot_classifier.save(model, model_path)
    loaded = slot_classifier.load(model_path)

    crop = cv2.imread(str(tmp_path / "accept_0.png"))
    features = slot_classifier.extract_features(crop, 0.9)
    assert loaded.predict([features])[0] == model.predict([features])[0]

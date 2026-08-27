import cv2
import joblib
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HIST_BINS = 8


def crop_polygon(image_bgr, polygon, pad=6):
    """Bounding-box crop of a candidate polygon, padded a few px for context."""
    pts = np.asarray(polygon, dtype=np.float64)
    h, w = image_bgr.shape[:2]
    x0 = max(int(pts[:, 0].min()) - pad, 0)
    y0 = max(int(pts[:, 1].min()) - pad, 0)
    x1 = min(int(pts[:, 0].max()) + pad, w)
    y1 = min(int(pts[:, 1].max()) + pad, h)
    return image_bgr[y0:y1, x0:x1]


def extract_features(crop_bgr, confidence):
    """Hand-picked features for a candidate crop: geometric confidence plus
    cheap grayscale/HSV/edge summary stats -- no CNN, so it trains instantly
    and doesn't overfit on the few dozen labels a review round produces.
    """
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.mean(edges > 0))

    gray_hist = cv2.calcHist([gray], [0], None, [HIST_BINS], [0, 256]).flatten()
    gray_hist = gray_hist / (gray_hist.sum() + 1e-6)
    hue_hist = cv2.calcHist([hsv], [0], None, [HIST_BINS], [0, 180]).flatten()
    hue_hist = hue_hist / (hue_hist.sum() + 1e-6)
    sat_mean = float(hsv[:, :, 1].mean() / 255.0)

    return np.concatenate([[float(confidence), edge_density, sat_mean], gray_hist, hue_hist]).astype(np.float64)


def train(labels):
    """Fit StandardScaler + LogisticRegression on human accept/reject labels.

    Raises ValueError if there aren't enough labels, or only one class is
    present -- callers should not save a model trained on a single class.
    """
    X, y = [], []
    for record in labels:
        crop = cv2.imread(record["crop_path"])
        if crop is None:
            continue
        X.append(extract_features(crop, record["confidence"]))
        y.append(1 if record["decision"] == "accept" else 0)

    if not X:
        raise ValueError("no usable labeled crops to train on")
    if len(set(y)) < 2:
        raise ValueError("need both accept and reject examples to train a classifier")

    pipeline = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))])
    pipeline.fit(np.array(X), np.array(y))
    return pipeline


def save(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load(path):
    return joblib.load(path)

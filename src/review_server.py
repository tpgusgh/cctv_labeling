import argparse
import json
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs

import cv2
import numpy as np

import review_store

CROPS_DIR = review_store.CROPS_DIR
FRAME_SIZE = 640  # this project's cameras are all 640x640 (see generate_config.py defaults)


def _render_page(candidate):
    body = "<h1>남은 후보 없음 (전부 리뷰 완료)</h1>" if candidate is None else f"""
<h1>후보 리뷰</h1>
<p>카메라: {candidate['camera_id']} | 신뢰도: {candidate['confidence']}</p>
<img src="/crops/{candidate['id']}.png" style="max-width:480px;border:1px solid #333">
<p>
<button onclick="decide('accept')">승인 (진짜 슬롯)</button>
<button onclick="decide('reject')">거부 (슬롯 아님)</button>
</p>
<script>
function decide(decision) {{
  fetch('/decide', {{
    method: 'POST',
    headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify({{id: '{candidate["id"]}', decision: decision}})
  }}).then(() => location.reload());
}}
</script>"""
    return f"<html><body>{body}<p><a href=\"/history\">리뷰 기록 보기</a> | <a href=\"/missed\">미탐 슬롯 표시</a></p></body></html>"


def _next_unreviewed(candidates, labels):
    by_id = {c["id"]: c for c in candidates}
    ids = review_store.unreviewed_ids(list(by_id.keys()), labels)
    return by_id[ids[0]] if ids else None


def _render_history(labels):
    if not labels:
        return "<html><body><h1>리뷰 기록 없음</h1><p><a href=\"/\">리뷰로 돌아가기</a></p></body></html>"

    cards = []
    for label in sorted(labels, key=lambda l: l.get("ts", ""), reverse=True):
        dim = "opacity:0.35;" if label["decision"] == "reject" else ""
        badge = "거부됨" if label["decision"] == "reject" else "승인됨"
        cards.append(f"""
<div style="display:inline-block;margin:8px;text-align:center;{dim}">
  <img src="/crops/{label['id']}.png" style="max-width:200px;display:block;border:1px solid #333">
  <div>{label['camera_id']} | {badge} | 신뢰도 {label['confidence']}</div>
  <button onclick="undo('{label['id']}')">되돌리기</button>
</div>""")

    return f"""<html><body>
<h1>리뷰 기록 ({len(labels)}개)</h1>
<p><a href="/">리뷰로 돌아가기</a> | <a href="/missed">미탐 슬롯 표시</a></p>
{''.join(cards)}
<script>
function undo(id) {{
  fetch('/undo', {{
    method: 'POST',
    headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify({{id: id}})
  }}).then(() => location.reload());
}}
</script>
</body></html>"""


def _camera_ids(candidates):
    return sorted({c["camera_id"] for c in candidates})


def _reference_image_path(camera_id, candidates):
    match = next((c for c in candidates if c["camera_id"] == camera_id), None)
    return match["image_path"] if match else None


def _render_missed_page(camera_id, cameras, missed_count):
    if camera_id is None:
        return "<html><body><h1>후보 없음 -- 먼저 카메라 생성/리뷰부터</h1></body></html>"

    idx = cameras.index(camera_id)
    prev_cam = cameras[idx - 1] if idx > 0 else None
    next_cam = cameras[idx + 1] if idx + 1 < len(cameras) else None
    nav = " | ".join(filter(None, [
        f'<a href="/missed?camera={prev_cam}">이전</a>' if prev_cam else None,
        f"{camera_id} ({idx + 1}/{len(cameras)}) -- 이 카메라에 저장된 미탐 표시: {missed_count}개",
        f'<a href="/missed?camera={next_cam}">다음</a>' if next_cam else None,
    ]))

    return f"""<html><body>
<h1>미탐 슬롯 표시 (학습 데이터로만 쌓임, 바로 config에 안 들어감)</h1>
<p>{nav}</p>
<p>하늘색 = 승인된/미리뷰 후보. 회색 = 리뷰에서 거부된 후보. 주황색 = 지금까지
표시한 미탐. 화면 4곳을 순서대로 터치/클릭하면 점이 이어지면서 네모가
그려지고 자동 저장됨.</p>
<div style="position:relative;width:{FRAME_SIZE}px;height:{FRAME_SIZE}px">
  <img id="frame" src="/frame/{camera_id}.png?t={missed_count}" width="{FRAME_SIZE}" height="{FRAME_SIZE}"
       style="position:absolute;top:0;left:0">
  <canvas id="draw" width="{FRAME_SIZE}" height="{FRAME_SIZE}" style="position:absolute;top:0;left:0;cursor:crosshair"></canvas>
</div>
<p><button onclick="undoLast()">되돌리기 (마지막 점/박스 취소)</button></p>
<p><a href="/">리뷰로 돌아가기</a> | <a href="/history">리뷰 기록</a></p>
<script>
const canvas = document.getElementById('draw');
const ctx = canvas.getContext('2d');
let points = [];

function redraw() {{
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = 'orange';
  ctx.fillStyle = 'orange';
  ctx.lineWidth = 2;
  points.forEach(([x, y]) => {{
    ctx.beginPath();
    ctx.arc(x, y, 4, 0, Math.PI * 2);
    ctx.fill();
  }});
  if (points.length > 1) {{
    ctx.beginPath();
    ctx.moveTo(points[0][0], points[0][1]);
    for (let i = 1; i < points.length; i++) ctx.lineTo(points[i][0], points[i][1]);
    ctx.stroke();
  }}
}}

canvas.addEventListener('click', (e) => {{
  points.push([e.offsetX, e.offsetY]);
  redraw();
  if (points.length < 4) return;

  const polygon = points;
  points = [];
  fetch('/missed/save', {{
    method: 'POST',
    headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify({{camera_id: '{camera_id}', polygon: polygon}})
  }}).then(() => location.reload());
}});

function undoLast() {{
  if (points.length > 0) {{
    points.pop();
    redraw();
    return;
  }}
  fetch('/missed/undo-last', {{
    method: 'POST',
    headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify({{camera_id: '{camera_id}'}})
  }}).then(() => location.reload());
}}
</script>
</body></html>"""


class ReviewHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if self.path.startswith("/crops/"):
            self._serve_crop(self.path[len("/crops/"):])
            return
        if parsed.path.startswith("/frame/") and parsed.path.endswith(".png"):
            self._serve_frame(parsed.path[len("/frame/"):-len(".png")])
            return
        if parsed.path == "/":
            candidates = review_store.load_candidates()
            labels = review_store.load_labels()
            self._send_html(_render_page(_next_unreviewed(candidates, labels)))
            return
        if parsed.path == "/history":
            self._send_html(_render_history(review_store.load_labels()))
            return
        if parsed.path == "/missed":
            self._serve_missed_page(parse_qs(parsed.query))
            return
        self.send_error(404)

    def _serve_missed_page(self, query):
        candidates = review_store.load_candidates()
        cameras = _camera_ids(candidates)
        camera_id = query.get("camera", [None])[0] or (cameras[0] if cameras else None)
        if camera_id not in cameras:
            self.send_error(404, "unknown camera")
            return
        missed = [m for m in review_store.load_missed_annotations() if m["camera_id"] == camera_id]
        self._send_html(_render_missed_page(camera_id, cameras, len(missed)))

    def _serve_frame(self, camera_id):
        candidates = [c for c in review_store.load_candidates() if c["camera_id"] == camera_id]
        image_path = _reference_image_path(camera_id, candidates)
        if image_path is None:
            self.send_error(404, "no reference image for this camera")
            return
        image = cv2.imread(image_path)
        if image is None:
            self.send_error(404, "reference image missing on disk")
            return

        decisions = {label["id"]: label["decision"] for label in review_store.load_labels()}
        for c in candidates:
            pts = [[int(x), int(y)] for x, y in c["polygon"]]
            color = (150, 150, 150) if decisions.get(c["id"]) == "reject" else (255, 200, 0)
            cv2.polylines(image, [np.array(pts)], True, color, 2)
        missed = [m for m in review_store.load_missed_annotations() if m["camera_id"] == camera_id]
        for m in missed:
            pts = [[int(x), int(y)] for x, y in m["polygon"]]
            cv2.polylines(image, [np.array(pts)], True, (0, 165, 255), 2)

        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            self.send_error(500, "could not encode frame")
            return
        data = encoded.tobytes()
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_html(self, page):
        body = page.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_crop(self, filename):
        path = CROPS_DIR / filename
        if ".." in filename or not path.is_file():
            self.send_error(404)
            return
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        if self.path == "/decide":
            self._handle_decide()
            return
        if self.path == "/undo":
            self._handle_undo()
            return
        if self.path == "/missed/save":
            self._handle_missed_save()
            return
        if self.path == "/missed/undo-last":
            self._handle_missed_undo_last()
            return
        self.send_error(404)

    def _read_json_body(self):
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length))

    def _send_ok(self):
        body = b'{"ok": true}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_decide(self):
        payload = self._read_json_body()

        candidates = {c["id"]: c for c in review_store.load_candidates()}
        candidate = candidates.get(payload["id"])
        if candidate is None:
            self.send_error(400, "unknown candidate id")
            return

        record = dict(candidate)
        record["decision"] = payload["decision"]
        record["ts"] = datetime.now(timezone.utc).isoformat()
        review_store.append_decision(record)
        self._send_ok()

    def _handle_undo(self):
        payload = self._read_json_body()
        review_store.remove_decision(payload["id"])
        self._send_ok()

    def _handle_missed_save(self):
        payload = self._read_json_body()
        camera_id = payload["camera_id"]
        polygon = payload["polygon"]

        candidates = [c for c in review_store.load_candidates() if c["camera_id"] == camera_id]
        image_path = _reference_image_path(camera_id, candidates)
        if image_path is None:
            self.send_error(400, "unknown camera_id")
            return

        record = {
            "id": review_store.candidate_id(camera_id, polygon),
            "camera_id": camera_id,
            "image_path": image_path,
            "polygon": polygon,
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        review_store.append_missed_annotation(record)
        self._send_ok()

    def _handle_missed_undo_last(self):
        payload = self._read_json_body()
        camera_id = payload["camera_id"]
        missed = [m for m in review_store.load_missed_annotations() if m["camera_id"] == camera_id]
        if missed:
            latest = max(missed, key=lambda m: m.get("ts", ""))
            review_store.remove_missed_annotation(latest["id"])
        self._send_ok()

    def log_message(self, fmt, *args):
        pass


def build_parser():
    parser = argparse.ArgumentParser(description="Local review UI for slot detection candidates.")
    parser.add_argument("--port", type=int, default=8765)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    server = HTTPServer(("", args.port), ReviewHandler)
    print(f"review server on http://localhost:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()

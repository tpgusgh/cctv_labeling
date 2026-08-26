# Sub-project 1: Core Geometry + Rendering Pipeline — Design

## Status
Approved for implementation planning.

## Context

Full system goal (see original requirements, not reproduced here): batch-label
CCTV parking-space images with occlusion-aware label placement, accounting for
lens and perspective distortion. The full system is too large for one spec, so
it is decomposed into sub-projects, built in order:

1. **Core geometry + rendering pipeline** (this doc)
2. Vehicle detection integration
3. Candidate generation + scoring engine
4. Label/no_label classifier
5. Batch processor + review/log pipeline
6. GUI (PySide6)

This sub-project proves the geometric core: given one raw CCTV frame, a camera
config, a parking-slot polygon, and a fixed label position, produce a
correctly-warped composited PNG. No ML, no batching, no candidate scoring yet.

### Ground truth from actual project data

The requirements doc's assumptions did not match the real sample data. What
was actually found in `label/` and `no_label/`:

- Cameras are **circular fisheye** (ceiling-mounted, mechanical parking tower),
  not mildly-angled wide CCTV. Raw frames are 640x640 JPG, black outside the
  circular valid region.
- `no_label/<camera_id>/*.jpg` — 23 folders, each one **camera** (not one
  parking slot). Folder names (`P1_B1_1_1` etc.) are camera IDs. Each camera's
  fisheye view covers several parking slots at once (confirmed: one reference
  screenshot showed 8 `parkingLocations-*` polygons in a single frame). Files
  inside are raw timestamped frames from that camera — this is the batch
  processing target dataset, not a good/bad training pair set.
- `label/*.png` — 18 screenshots from an unrelated external tool. Their
  polygon shapes are a reference for "ambiguous-angle" slot-detection edge
  cases; their text overlays (`parkingLocations-978: 0.714`) are not part of
  our output format and are ignored.
- No label graphic asset (PNG template) exists or will be supplied. The
  system draws the label shape itself (see Renderer below) rather than
  compositing an external image file.
- Final output format is PNG, per original spec (batch-processed images saved
  to `output/`).

## Approach

**Rectify → composite → re-distort.** Undistort the raw fisheye frame into a
flat rectified image first. Do all perspective/label-placement math in that
flat space (ordinary rectangle/homography geometry). Composite the label
there. Then forward-warp (re-distort) the composited rectified image back
into the original fisheye pixel layout for final output.

This was chosen over two alternatives:
- Warping only the label's 4 corners through homography + distortion
  (matches what the external reference tool appears to do, given the mostly
  straight-edged polygons in `label/` screenshots) — simpler, but produces
  straight label edges where a true fisheye mapping would curve them,
  especially near the image periphery.
- Sampling many boundary points per label edge and mapping each through the
  full distortion model without ever building a full rectified image —
  correct, but pushes distortion math into every renderer call instead of
  two image-level operations.

Rectify-first wins because it satisfies the requirement to treat perspective
distortion and lens distortion as **separate stages** (undistort/re-distort =
lens stage, homography = perspective stage) using two image-level OpenCV
operations, and because every later sub-project (occlusion math, candidate
generation) gets to work in ordinary rectangle space instead of warped fisheye
space.

## Components

All under `src/`.

### `calibration.py`
Per-camera fisheye distortion model.

- MVP model: single-parameter equidistant radial model, center = image
  center (justified by the circular fisheye crop), radius = detected circle
  boundary. `ponytail: single scalar distortion coefficient, fit by
  least-squares against user-clicked collinear points on a real straight
  line (e.g. a parking line) in the raw frame — upgrade to full
  cv2.fisheye.calibrate() + checkerboard captures if that data ever becomes
  available.`
- `fit(raw_image, clicked_line_points) -> CalibrationModel`
- `CalibrationModel.undistort_image(raw) -> rectified`
- `CalibrationModel.undistort_points(pts) -> pts` (raw px → rectified px)
- `CalibrationModel.distort_points(pts) -> pts` (rectified px → raw px, used
  for the final re-distort remap)
- Model persisted as JSON per camera (`config/<camera_id>.json`, `calibration`
  key).

### `parking_slot.py`
Per-camera parking-slot polygon config.

- User clicks slot corners on the **raw** frame (what they actually see).
  Points are converted through `undistort_points` and stored as the
  canonical polygon in **rectified pixel space** — all downstream geometry
  uses rectified space only.
- Load/save `config/<camera_id>.json` (`slots: [{id, polygon_rectified}]`).

### `perspective.py`
Homography between normalized parking-plane coordinates (`(0,0)`–`(1,1)`) and
a slot's rectified-pixel quad.

- `plane_to_pixel(slot_polygon_rectified) -> homography` via
  `cv2.getPerspectiveTransform`.
- Used to place the label's normalized-space shape into rectified pixel
  coordinates.

### `renderer.py`
Draws the label directly — no external image asset.

- Label spec (from config): shape (`rect` | `rounded_rect`), size in
  normalized parking-plane units, fill color, alpha, border, optional text.
- `render_label(rectified_image, homography, candidate_point, label_spec) ->
  rectified_image_with_label` — computes the label's corner points in
  normalized space, maps through the homography to rectified pixels, draws
  with `cv2.fillPoly`/`cv2.polylines`, alpha-blends onto the rectified image.

### Re-distort step
Reuses `CalibrationModel.distort_points` to build a remap
(`cv2.remap` with an inverse-mapped grid) from rectified space back to raw
fisheye layout. Output has the same dimensions as the input raw frame.

### `main.py` (this sub-project's entry point)
CLI test harness: load camera config + one raw frame + one candidate point,
run the full pipeline, write output PNG. This is scaffolding for manual
verification during this sub-project only — the real batch entry point comes
in sub-project 5.

## Data flow

```
raw.jpg (fisheye, from no_label/<camera_id>/)
  -> CalibrationModel.undistort_image
rectified.png (flat, debug-only intermediate)
  -> perspective.plane_to_pixel(slot polygon)
  -> renderer.render_label(candidate_point, label_spec)
composited_rectified.png
  -> remap via CalibrationModel.distort_points (inverse grid)
final.png (same size as raw.jpg)
```

## Config schema (per camera, `config/<camera_id>.json`)

```json
{
  "camera_id": "P1_B1_1_1",
  "image_width": 640,
  "image_height": 640,
  "calibration": { "model": "equidistant_1param", "center": [320, 320], "radius": 310, "k": 0.42 },
  "slots": [
    { "id": "P1_B1_1_1-A", "polygon_rectified": [[100,300],[350,280],[380,650],[80,680]] }
  ],
  "label_spec": { "shape": "rounded_rect", "width": 0.6, "height": 0.25, "color": [30,180,90], "alpha": 0.75, "text": null }
}
```

## Error handling

No review/log pipeline exists yet (added in sub-project 5). For this
sub-project: missing calibration, missing slot config, or a candidate point
outside the rectified image bounds raises an exception with a clear message.
Not swallowed, not defaulted.

## Testing

Ponytail rule: non-trivial logic gets one runnable self-check, not a test
framework.

- `calibration.py`: `demo()` under `if __name__ == "__main__"` — asserts
  `distort_points(undistort_points(p)) ≈ p` (round-trip identity) for a
  handful of sample points across the image.
- `renderer.py` / pipeline: one `test_pipeline.py` — runs the full
  raw→final pipeline on one real sample frame from `no_label/`, asserts (a)
  output dimensions match input, (b) pixels inside the label's raw-space
  bounding box differ from the original raw frame (something was actually
  drawn), (c) pixels far outside the slot polygon are unchanged (no leakage).

## Out of scope (future sub-projects)

Vehicle detection, occlusion scoring, candidate generation, label/no_label
classifier, batch processing, review classification, logging, GUI. This
sub-project only proves single-image geometry + rendering correctness.

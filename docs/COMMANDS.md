# 명령어 모음

전부 저장소 루트(`cctv_labeling/`)에서 실행. `.venv/bin/python` 대신
`source .venv/bin/activate` 해뒀으면 `python`만 써도 됨.

## 설치 / 테스트

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

.venv/bin/pytest -v          # 전체 테스트 (실제 no_label/ 샘플 필요)
```

## 1. 카메라 슬롯 후보 생성 (config + 리뷰 후보)

카메라 원본 프레임 폴더 하나 → 슬롯 폴리곤 자동 탐지 → `config/<camera_id>.json`
저장 + 리뷰용 crop(`review/crops/`)/후보 목록(`review/candidates.jsonl`) 저장.

```bash
.venv/bin/python src/generate_config.py \
  --camera-id P1_B1_1_9 \
  --frames-dir no_label/P1_B1_1_9 \
  --output config/P1_B1_1_9.json
```

여러 카메라 한번에 (예: 전체):

```bash
for cam in no_label/*/; do
  cam_id=$(basename "$cam")
  .venv/bin/python src/generate_config.py \
    --camera-id "$cam_id" \
    --frames-dir "$cam" \
    --output "config/${cam_id}.json"
done
```

학습된 분류기로 오탐 걸러가며 재탐지하고 싶으면 `--model` 추가
(아래 3번에서 학습한 모델):

```bash
.venv/bin/python src/generate_config.py \
  --camera-id P1_B1_1_9 \
  --frames-dir no_label/P1_B1_1_9 \
  --output config/P1_B1_1_9.json \
  --model models/slot_classifier.joblib
```

카메라별 렌즈 파라미터가 다르면 `--cx/--cy/--f/--radius`로 넘김
(기본값은 이 프로젝트 샘플의 실측값 `f≈204, radius≈320`).

## 2. 후보 승인/거부 (리뷰 서버)

`review/candidates.jsonl`에 쌓인 후보 중 아직 안 본 것을 웹 화면으로
하나씩 승인/거부. 결정은 `review/labels.jsonl`에 쌓임.

```bash
.venv/bin/python src/review_server.py --port 8765
```

브라우저에서 `http://localhost:8765` 접속. "남은 후보 없음" 뜨면 그
시점까지 만들어진 후보를 다 리뷰한 것 — 1번으로 카메라 더 돌리면 다시 생김.

잘못 눌렀을 때: `http://localhost:8765/history`에서 지금까지 결정 전부
보임(거부는 흐리게 표시) — "되돌리기" 누르면 다시 안 본 상태로 돌아가서
리뷰 화면에 다시 뜸. (수동으로 하려면 `review/labels.jsonl`에서 해당 id
줄 지워도 동일)

### 미탐(놓친 슬롯) 표시

`http://localhost:8765/missed`에서 카메라 원본 프레임 전체를 보면서
마우스로 드래그해 놓친 슬롯 위치에 박스를 그릴 수 있음(하늘색 = 이미
탐지된 후보, 주황색 = 지금까지 표시한 미탐). **이 박스는 바로 config에
안 들어감** — `review/missed.jsonl`에 학습 데이터로만 쌓임. 나중에
데이터 충분해지면 이 위치들로 별도 탐지기를 학습시키는 게 목표(아직
학습 파이프라인 자체는 없음, 데이터 수집 단계).

## 3. 분류기 재학습

쌓인 승인/거부 라벨로 분류기 학습 → `models/slot_classifier.joblib` 저장.

```bash
.venv/bin/python src/train_from_reviews.py
```

`--labels`/`--output`으로 경로 바꿀 수 있음. 라벨이 아예 없거나 승인/거부
한쪽밖에 없으면 학습 안 하고 이유만 출력하고 종료.

## 4. 재학습 루프

1 → 2 → 3 반복. 3에서 학습한 모델을 1번 `--model`로 넣어서 다시 탐지하면
분류기가 걸러낸 오탐이 줄어든 상태로 새 후보가 생김 (같은 카메라+폴리곤이면
후보 id가 같아서 이미 리뷰한 건 2번 화면에 다시 안 뜸).

## 5. 라벨 합성 (실제 이미지에 라벨 그리기)

`config/<camera_id>.json`이 있어야 함 (1번으로 생성).

수동 위치 지정:

```bash
.venv/bin/python src/main.py \
  --config config/P1_B1_1_9.json \
  --image no_label/P1_B1_1_9/20260820_115033.jpg \
  --slot-id A \
  --candidate-u 0.5 \
  --candidate-v 0.5 \
  --output output/demo.png
```

슬롯 하나 자동(고정 규칙, 앞유리 탐지 아님 — `HANDOFF.md` 참고):

```bash
.venv/bin/python src/main.py \
  --config config/P1_B1_1_9.json \
  --image no_label/P1_B1_1_9/20260820_115033.jpg \
  --slot-id A \
  --auto \
  --output output/demo_auto.png
```

config 안 모든 슬롯 한번에:

```bash
.venv/bin/python src/main.py \
  --config config/P1_B1_1_21.json \
  --image no_label/P1_B1_1_21/20260820_115029.jpg \
  --auto-all \
  --output output/demo_all.png
```

## 6. 카메라 폴더 전체 배치 처리

```bash
.venv/bin/python src/batch_processor.py \
  --config config/P1_B1_1_9.json \
  --input-dir no_label/P1_B1_1_9 \
  --output-dir output/P1_B1_1_9 \
  --review-dir review/P1_B1_1_9 \
  --log output/P1_B1_1_9/log.json
```

성공은 `output-dir`, 슬롯 중 하나라도 review/error면 `review-dir`로 감.
`log.json`에 이미지별 상태 전부 기록.

## 7. YOLOv8-seg 학습 (딥러닝 파이프라인)

리뷰에서 쌓인 승인/미탐 라벨(`review/labels.jsonl` accept + `review/missed.jsonl`)로
YOLOv8-seg 모델을 파인튜닝하는 경로. 웹앱(`web/backend/jobs.py`)은
`models/yolov8_seg_slots_v6.pt`가 존재하면 자동으로 로드해서 classical
CV(Hough+연결영역)와 **앙상블**로 탐지함 (없으면 classical CV만). 두 탐지기가
독립적으로 같은 위치를 잡으면(`agreement_count >= 2`) 자동 승인됨.

### 7-1. 데이터셋 export

```bash
.venv/bin/python src/export_yolo_dataset.py --output /tmp/yolo_dataset
```

카메라 고정 전제 활용: 폴리곤 1세트를 그 카메라의 모든 원본 프레임에
복제해서 학습 이미지로 씀. 카메라 단위로 train/val 분리(`--val-every`로
비율 조절, 기본 5 — 5개 중 1개꼴로 val). `--no-label-dir`/`--labels`/
`--missed`로 경로 바꿀 수 있음. 라벨 있는 카메라 2개 미만이면 에러.

### 7-2. 학습

```bash
.venv/bin/python src/train_yolo_seg.py \
  --data /tmp/yolo_dataset/dataset.yaml \
  --epochs 100 \
  --output models/yolov8_seg_slots.pt
```

최초 실행 시 ultralytics 서버에서 `yolov8n-seg.pt`(COCO 사전학습) 자동
다운로드 — 이때만 네트워크 필요, CCTV 사진 자체는 어디로도 안 나감.
`--base-model`/`--epochs`로 조절.

### 7-3. 학습된 모델로 탐지

```bash
.venv/bin/python src/generate_config.py \
  --camera-id P1_B1_1_9 \
  --frames-dir no_label/P1_B1_1_9 \
  --output config/P1_B1_1_9.json \
  --yolo-model models/yolov8_seg_slots.pt
```

`--yolo-model` 주면 이 모델 + classical CV 앙상블로 탐지 (중복은 IoU/중심점
기준으로 합쳐지고 신뢰도 높은 쪽이 남음). `--auto-accept-agreement` 주면
두 탐지기가 동시에 잡은 후보는 리뷰 없이 자동 승인.

### 7-4. 거부 이력 자동 억제

사람이 거부한 위치는 `generate_config`가 기억함: 같은 카메라에서 리뷰
로그(`review/labels.jsonl`)에 거부로 기록된 지역과 겹치는 탐지는 config에
안 들어감 (같은 지역에 승인 기록도 있으면 — 중복 정리 케이스 — 살아남음).
웹에서 Shift+× 로 슬롯을 삭제해도 같은 거부 기록이 쌓여서, 재탐지 때 그
위치가 되살아나지 않음. 반대로 웹에서 직접 그린 슬롯은 승인으로 기록되어
억제로부터 보호되고 다음 재학습의 정답 데이터가 됨.

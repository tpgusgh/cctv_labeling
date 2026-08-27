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

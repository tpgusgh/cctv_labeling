# 인수인계 — 시작하기 전에 이거 먼저

이 프로젝트를 처음 받았으면 이 파일부터 읽으세요. 5분이면 끝남.

## 이게 뭐하는 프로그램인가

CCTV(어안렌즈) 주차장 사진을 넣으면 주차 슬롯을 자동으로 찾아서 라벨(색깔
테두리)을 씌워주는 웹앱. 사람이 하는 일은 사진 업로드 + 결과 확인/수정뿐,
슬롯 좌표를 손으로 입력하는 일은 없음.

## 지금 당장 실행하기

```bash
cd cctv_labeling
python3 -m venv .venv          # 최초 1회
.venv/bin/pip install -r requirements.txt   # 최초 1회
./run_web.sh
```

브라우저에서 `http://localhost:5050` 열면 끝. 사진(또는 카메라 폴더) 업로드
→ 자동 탐지/라벨링 → 결과 확인 및 수정 → 다운로드.

## 지금 적용된 모델

- 탐지: `models/yolov8_seg_slots_production.pt` (YOLOv8-seg, classical CV와
  앙상블) — 웹앱이 항상 이 **고정된 파일명**을 읽음. 모델이 바뀌어도 코드
  수정 불필요, 이 파일만 교체하면 됨.
- 분류기(보조): `models/slot_classifier.joblib`
- `web/backend/jobs.py`의 `YOLO_MODEL_PATH`/`MODEL_PATH`가 이 두 경로를
  가리킴 — 다른 이름으로 바꾸고 싶으면 여기만 고치면 됨.

## 성능을 더 올리고 싶으면 (재학습)

웹에서 하는 수정(오탐 삭제, 슬롯 추가, 위치 조정)이 전부 학습 데이터로
자동 축적됨. 수정 기록이 100건쯤 쌓이면:

```bash
.venv/bin/python src/retrain_yolo.py
```

데이터 추출 → 파인튜닝(1시간 안팎) → 기존 모델과 자동 성능 비교 →
**더 좋을 때만** `models/yolov8_seg_slots_production.pt` 교체. 지든
이기든 모든 체크포인트는 `models/archive/`에 남아서 롤백 가능. 성능이
정체됐다면(측정으로 확인 가능) 다음 도약은 새로운 카메라/시설 데이터가
생겼을 때임 — 지금 갖고 있는 데이터 양에선 모델 크기를 키우거나 증강을
늘려도 오히려 나빠지는 걸 실측으로 확인함(자세한 건 git 로그의 관련
커밋 메시지 참고).

모델만 따로 평가하려면:

```bash
.venv/bin/python src/evaluate_seg_model.py models/yolov8_seg_slots_production.pt
```

## 더 자세히 알고 싶으면

- **`README.md`** (프로젝트 루트) — 전체 구조, 웹앱 사용법, 슬롯 자동 탐지
  원리
- **`docs/COMMANDS.md`** — CLI 명령어 전체 목록 (탐지/리뷰/재학습/배치처리)
- **`HANDOFF.md`** (프로젝트 루트) — 개발 과정에서 뭘 시도했고 뭐가 왜
  실패했는지 (같은 시행착오 반복 방지용)

## 알아둬야 할 것

- `no_label/`, `label/`, `review/`, `web_uploads/`, `config/`는 실제 CCTV
  데이터라 `.gitignore`에 있음 — git에는 안 올라감, 별도로 전달받아야 함.
  `config/`(카메라별 슬롯 정의, 81개, 460KB)만은 이 폴더의
  `config_snapshot/`에 스냅샷으로 같이 넣어뒀음 — 받는 사람은 이걸
  프로젝트 루트의 `config/`에 그대로 복사하면 됨:
  ```bash
  cp handoff/config_snapshot/*.json config/
  ```
  (이후 계속 수정되는 최신 버전이 아니라 인계 시점 스냅샷이므로, 시간이
  지나면 실제 `config/`가 더 최신일 수 있음.)
- 결과 페이지에서 사람이 지운 슬롯(Shift+×)은 그 카메라 재탐지 시 영구적으로
  다시 안 잡힘 (억제 로직, `generate_config.py` 참고) — "왜 지웠는데 또
  나오지" 걱정 없음.
- 실 데이터가 있어야 테스트가 통과함: `.venv/bin/pytest -v`

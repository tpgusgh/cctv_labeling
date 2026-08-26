# 서브프로젝트 2: 앞유리 탐지 기반 라벨 위치 자동 결정 — 설계

## 상태
구현 계획 단계 진행 승인됨.

## 배경

서브프로젝트 1(핵심 기하/렌더링, 어안렌즈 로컬 gnomonic 재투영)까지 완료됨 —
사람이 슬롯 polygon과 후보 위치를 직접 지정하면 라벨 1개를 정확히 합성함.

이번 서브프로젝트는 원래 계획했던 "차량 detection(YOLO)"이 아니라, 논의를
거쳐 실제 필요로 재정의됨:

- COCO 사전학습 YOLO는 이 카메라의 top-down(천장 직하) 앵글에서 차량을 전혀
  인식 못함(spike로 확인: bottle/clock/train/refrigerator 등 무관한 클래스만
  나옴) — 학습 데이터도 없어 커스텀 학습은 범위 밖.
- 실제 필요한 신호는 차량 앞유리(윈드쉴드)의 검은 영역 — top-down에서 봐도
  주변 바닥/차체와 대비가 뚜렷해 고전적 CV(어두운 영역 threshold)로 탐지
  가능함을 spike로 확인.
- 라벨 배치 목적 자체가 바뀜: "차량에 가려지지 않는 곳"이 아니라 **"차량
  앞유리를 절반 이상 포함하는 곳"**. 앞유리가 이웃 차량에 가려서 안 보이면
  그 슬롯은 라벨을 아예 붙이지 않음(스킵) — 빈 슬롯도 같은 규칙(탐지되는
  앞유리 자체가 없으므로)으로 자연스럽게 스킵됨.

## 핵심 통찰

서브프로젝트 1의 좌표계 설계(슬롯 polygon → homography → 정규화 0~1 평면)가
이미 "라벨이 슬롯의 실제 모서리 방향을 따라간다"는 요구사항을 별도 작업 없이
만족시킴 — 정규화 u/v축 자체가 슬롯의 실제 두 변 방향과 일치하도록
`perspective.plane_to_pixel_homography`가 구성되어 있기 때문. 따라서 이번
서브프로젝트는 "라벨을 그릴 위치(정규화 좌표) + 크기를 앞유리 탐지 결과로부터
자동 계산"하는 것에 집중하면 되고, 렌더링 자체(`renderer.py`,
`perspective.py`)는 손댈 필요가 없음(단, 정규화→픽셀의 역방향 매핑 함수 하나
추가 필요).

## 접근 방식

1. **앞유리(어두운 영역) 탐지** — raw 이미지에서 카메라의 원형 바닥 영역만
   마스킹(배경 선반/구조물 노이즈 제거) 후, 어두운 픽셀 threshold + 윤곽선
   추출로 후보 blob 목록 생성. Spike로 검증: 배경 마스킹 없이도 실제 차량
   앞유리 blob이 잡히지만 노이즈가 많음 — 원형 마스킹으로 대부분(선반 등
   화면 바깥 링 영역)이 사라짐.
2. **슬롯 배정** — 탐지된 blob 중 중심점이 해당 슬롯의 `polygon_raw` 내부에
   있는 것을 그 슬롯의 앞유리로 판정. 내부에 있는 blob이 하나도 없으면(다른
   슬롯 쪽에 붙어있거나, 애초에 차가 없거나) 그 슬롯은 스킵 — 빈 슬롯과
   "이웃 차량에 가려짐"을 하나의 규칙으로 함께 처리.
3. **라벨 후보 위치/크기 계산** — 배정된 blob의 bounding box를
   `LocalView.raw_to_local` → `perspective.pixel_to_plane_points`(신규, 역방향
   매핑)로 정규화 슬롯 평면 좌표로 변환. 그 bbox를 마진(예: 1.3배)만큼 키운
   범위를 라벨의 `candidate_point`/`width`/`height`로 사용 — blob 중심에
   라벨을 앉히고 blob bbox보다 넉넉하게 키우면 최소 50%(사용자 요구사항)를
   항상 넉넉히 초과 충족함.
4. **합성** — 계산된 candidate_point/width/height로 기존
   `renderer.render_label` 그대로 호출, 이후 파이프라인은 서브프로젝트 1과
   동일.

## 컴포넌트 변경

### `perspective.py` (기존 파일에 함수 추가)
- `pixel_to_plane_points(homography, pixel_points) -> np.ndarray[N,2]` —
  `plane_points_to_pixel`의 역방향(`cv2.perspectiveTransform` +
  `np.linalg.inv(homography)`).

### `windshield.py` (신규)
- `WindshieldBlob` (dataclass): `contour`, `bbox (x,y,w,h)`, `centroid (x,y)`,
  `area` — 전부 raw 픽셀 좌표.
- `detect_windshields(raw_image, calibration) -> list[WindshieldBlob]` — 원형
  바닥 마스크(`calibration.cx/cy/radius`) 적용 후 어두운 영역 threshold +
  `cv2.findContours`, 면적 범위로 필터링.
- `ponytail: DARK_THRESHOLD/MIN_BLOB_AREA/MAX_BLOB_AREA는 spike로 확인한
  값을 하드코딩(고정 상수)으로 시작 — 실제 다양한 조명/차종 데이터가
  쌓이면 적응형(예: 이미지별 히스토그램 기반) threshold로 교체.`

### `candidate.py` (신규)
- `point_in_polygon(point, polygon) -> bool` (`cv2.pointPolygonTest` 래핑).
- `find_slot_windshield(slot_polygon_raw, blobs) -> WindshieldBlob | None`.
- `compute_label_candidate(view, homography, blob, coverage_margin=1.3) ->
  (candidate_point, width, height)`.

### `pipeline.py`
- 기존 `run()`의 슬롯 조회/`LocalView` 생성/검증/homography 계산 부분을
  `_find_slot(config, slot_id)` + `_prepare_slot_view(config, slot) -> (view,
  homography)`로 추출(리팩터링, 동작 변화 없음) — `run()`과 신규 `run_auto()`가
  공유.
- 신규 `run_auto(config_path, raw_image_path, slot_id, output_path)` — 앞유리
  탐지 → 슬롯 배정 → (배정 없으면 `None` 반환, 라벨 안 그림, 파일도 안 씀) →
  후보 위치/크기 계산 → `render_label` → `unrectify_into` → 저장.

### `main.py`
- `--auto` 플래그 추가: 켜지면 `--candidate-u`/`--candidate-v` 없이
  `run_auto()` 호출. `run_auto()`가 `None`을 반환(스킵)하면 CLI는 출력 파일을
  만들지 않고 그 사실을 stdout에 명확히 알림(에러 아님 — 정상적인 "라벨링
  대상 아님" 결과).

## 에러 처리

- 슬롯 없음/이미지 못 읽음/후보점 범위 밖/파일 쓰기 실패 — 기존과 동일하게
  명확한 예외.
- "앞유리 안 보임(배정 실패)"은 예외가 아니라 **정상적인 스킵 결과**
  (`None` 반환) — 이건 에러 상황이 아니라 이 서브프로젝트가 의도적으로
  다루는 케이스이므로 구분해야 함.

## 테스트

- `perspective.py`: `pixel_to_plane_points`가 `plane_points_to_pixel`과
  라운드트립.
- `windshield.py`: 실제 차량이 보이는 샘플 이미지(`no_label/P1_B1_1_21/...`)로
  blob이 최소 1개 이상 탐지되는지, synthetic 이미지로 원형 마스크 바깥의
  어두운 영역은 탐지 안 되는지.
- `candidate.py`: `point_in_polygon` 참/거짓 케이스, `find_slot_windshield`가
  polygon 내부 blob만 고르는지, `compute_label_candidate`가 만든 영역이
  blob bbox를 완전히 포함하는지(마진 1.3배이므로 100% 커버 — 요구사항
  "50% 이상"을 넉넉히 충족함을 강한 assertion으로 증명).
- `pipeline.py`: `run_auto()`가 (a) 실제 차량 보이는 슬롯에서 라벨 생성 (b)
  차량 없는/배정 안 되는 슬롯에서 `None` 반환 + 파일 안 씀, 둘 다 검증.

## 범위 밖

- 이웃 슬롯 영역 침범 방지(라벨이 옆 칸 쪽으로 넘어가는 것)는 이번엔 명시적
  체크를 넣지 않음 — blob 크기 기반 산정이라 실질적으로 차량 크기 근처로
  자연스럽게 제한되므로 당장은 괜찮다고 판단, 필요해지면 이후 과제.
- 미래 차량 점유 예측(빈 슬롯이 향후 어떻게 채워질지), 여러 이미지 통합
  안정 위치 계산, GUI 노출 — 전부 이후 서브프로젝트.
- 고정 크기 local patch(`pipeline.py`의 `DEFAULT_PATCH_SIZE=(300,300)`,
  `DEFAULT_LOCAL_F=300.0`)가 사용 가능한 슬롯 polygon 크기의 상한을 만듦 —
  이 카메라의 일반적인 차량 위치 기준 raw 픽셀로 약 142x142 정도로 측정됨(최종
  리뷰 시점 측정). 실제 주차 슬롯은 이보다 클 수 있음 — 문제가 되면
  `DEFAULT_LOCAL_F`가 조정 지점(값을 낮추면 patch의 각도 커버리지가 넓어짐).

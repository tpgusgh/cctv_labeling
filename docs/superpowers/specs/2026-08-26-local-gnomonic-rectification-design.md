# 서브프로젝트 1 수정: 슬롯별 로컬 gnomonic 재투영 — 설계

## 상태
구현 계획 단계 진행 승인됨.

## 배경

서브프로젝트 1(핵심 기하/렌더링 파이프라인, PR #1)의 최종 리뷰에서 "서브프로젝트 2
시작 전 실제 카메라 FOV 측정 필요"가 권고사항으로 남았음. 실측 결과(아래) 현재
구현이 이 카메라에서 구조적으로 성립하지 않음을 확인, PR #1과 같은 브랜치에서
바로 수정.

### 실측 결과

`no_label/P1_B1_1_9/20260820_115033.jpg`(640x640) 기준, 이미지 중심(320,320)에서
8방향으로 실제(검은색 아닌) 콘텐츠가 존재하는 최대 반경 측정:

```
0°: 319   45°: 329   90°: 319   135°: 329
180°: 320  225°: 329  270°: 320  315°: 329
```

모든 방향에서 이미지 half-width(320px)에 근접 — 원형 fisheye 유효 영역이 프레임
전체를 거의 채움. Equidistant 모델(`theta = r/f`)로 역산하면, 콘텐츠 경계에서
`theta ≈ 90°`가 되려면 `f ≈ 320/(π/2) ≈ 204`. 즉 이 렌즈는 거의 180° FOV이고,
콘텐츠 경계 부근에서 이미 `theta`가 90°에 도달하거나 넘어감.

기존 서브프로젝트 1 구현(`undistort_image`/`redistort_image`)은 카메라의 실제
광학 중심을 기준으로 **프레임 전체를 한 번에** rectilinear(`tan(theta)`) 평면으로
펼침. `tan()`은 `theta→90°`에서 발산하므로, 캔버스 크기를 아무리 키워도 콘텐츠
경계 부근(실측상 대부분의 실제 콘텐츠 영역)을 유한한 rectified 캔버스로 표현할
수 없음 — 최종 리뷰가 우려한 최악의 시나리오가 실측으로 확인됨.

## 접근 방식

**슬롯별 로컬 gnomonic(tangent-plane) 재투영.** 프레임 전체를 하나의 좌표계로
펴는 대신, 주차슬롯마다 "그 슬롯 방향을 정면으로 바라보는 가상 카메라"를 하나씩
만든다. 슬롯 중심을 기준으로 한 국소적 각도 편차는 슬롯 크기가 작으므로 항상
작게 유지되고(±20~30° 수준), 따라서 로컬 원점에서는 `tan()` 발산이 원천적으로
발생하지 않는다.

절차:

1. Raw 픽셀 → 3D 단위 광선 (equidistant 모델: `theta=r/f`, `phi=atan2(dy,dx)`,
   `ray=(sinθcosφ, sinθsinφ, cosθ)`)
2. 슬롯 중심 광선이 `(0,0,1)`이 되도록 회전 행렬 `R` 계산 (Rodrigues 공식)
3. 임의의 raw 픽셀 → 광선 → `R` 회전 → 로컬 rectilinear(pinhole) 투영 →
   로컬 패치 픽셀 좌표. 역방향(로컬 → raw)은 `R⁻¹ = Rᵀ` 사용.
4. 이 로컬 패치 공간에서 homography 계산 + 라벨 렌더링은 **기존
   `perspective.py`/`renderer.py`를 그대로 재사용** (둘 다 "어떤 rectified 공간이
   주어지든" 동작하도록 이미 설계되어 있어 변경 불필요)
5. 합성된 로컬 패치를 raw 픽셀 공간의 해당 위치(작은 bounding box)에만
   역변환해서 원본 raw 이미지의 복사본에 합성. 슬롯 영역 바깥은 원본과 완전히
   동일 — 서브프로젝트 1이 겪은 "프레임의 47%가 검게 지워짐" 문제가 설계상
   원천적으로 사라짐 (전체 프레임을 다시 왜곡하는 단계 자체가 없어짐).

## 왜 이 방식인가

대안으로 "슬롯 주변 bounding box만 잘라서 기존 전역 모델로 rectify"도 고려했으나
기각: `tan()`의 국소 배율(미분값)도 `theta→90°`에서 함께 발산하므로, 슬롯이
발산 구간 근처에 있으면 bounding box를 아무리 좁혀도 실세계 크기의 슬롯을
표현하는 데 필요한 로컬 픽셀 수가 여전히 무한대로 발산함. 좌표계 원점 자체를
슬롯 방향으로 옮기는 것(gnomonic 재투영)만이 발산을 근본적으로 제거함.

## 컴포넌트 변경

### `calibration.py`
- `CalibrationModel.pixel_to_ray(points) -> np.ndarray[N,3]`, `ray_to_pixel(rays) -> np.ndarray[N,2]`
  — 기존 2D `undistort_points`/`distort_points`의 3D 버전. 기존 2D 함수는
  점 단위 라운드트립 테스트 등에서 계속 유효하므로 유지(삭제하지 않음).
- 신규 `LocalView` 클래스:
  - `LocalView.centered_on(calibration, center_raw_point, patch_size, local_f) -> LocalView`
  - `.raw_to_local(raw_points) -> local pixel coords`
  - `.local_to_raw(local_points) -> raw pixel coords`
  - `.rectify(raw_image) -> local rectified patch` (patch_size 크기 이미지, remap 기반)
  - `.unrectify_into(local_patch, raw_image) -> raw_image 복사본에 패치만 합성한 결과`
- `ponytail: patch_size와 local_f는 고정 상수(예: 300x300, 반각 25°)로 시작 —
  실제 슬롯의 각크기 데이터가 쌓이면 슬롯별 동적 산정으로 교체.`

### `parking_slot.py`
- `SlotConfig.slots`의 각 항목이 `polygon_rectified`(전역 rectified 좌표) 대신
  `polygon_raw`(raw 클릭 좌표 그대로)를 저장. 전역 rectified 평면 자체가 더 이상
  존재하지 않으므로 저장 시점 좌표 변환이 불필요해짐 — `raw_clicks_to_slot_polygon`
  함수 및 그 안의 `undistort_points` 호출 제거.
- `CalibrationModel`을 계속 참조/저장(로컬 뷰 생성 시 필요).

### `pipeline.py`
- `run()`을 재작성: 슬롯의 `polygon_raw` 중심점으로 `LocalView` 생성 →
  `view.rectify(raw)`로 로컬 패치 생성 → `view.raw_to_local(polygon_raw)`로
  로컬 좌표 slot polygon 계산 → `plane_to_pixel_homography`(변경 없음) →
  `render_label`(변경 없음, 범위 초과 시 기존 `ValueError` 그대로 유효) →
  `view.unrectify_into(composited_local, raw)` → `cv2.imwrite` (반환값 체크,
  변경 없음).

### `perspective.py`, `renderer.py`, `main.py`
변경 없음. `main.py`는 `pipeline.run()`과 같은 시그니처를 그대로 호출하므로
CLI 인터페이스도 변경 없음 — 단, 테스트 픽스처가 새 config 스키마
(`polygon_raw`)를 쓰도록 업데이트 필요.

## 데이터 흐름

```
raw.jpg
  -> slot["polygon_raw"] 중심점 계산
  -> LocalView.centered_on(calibration, center, patch_size, local_f)
  -> view.rectify(raw) -> local_patch (예: 300x300)
  -> view.raw_to_local(polygon_raw) -> polygon_local
  -> plane_to_pixel_homography(polygon_local) -> homography
  -> render_label(local_patch, homography, candidate_point, label_spec) -> composited_local
  -> view.unrectify_into(composited_local, raw) -> final (raw와 동일 크기,
     슬롯 영역 바깥은 raw와 픽셀 단위로 동일)
  -> cv2.imwrite
```

## Config 스키마 변경 (카메라별, `config/<camera_id>.json`)

```json
{
  "camera_id": "P1_B1_1_1",
  "image_width": 640,
  "image_height": 640,
  "calibration": { "model": "equidistant_1param", "center": [320, 320], "f": 204.0, "radius": 320.0 },
  "slots": [
    { "id": "P1_B1_1_1-A", "polygon_raw": [[250,180],[400,190],[410,300],[240,290]] }
  ],
  "label_spec": { "shape": "rect", "width": 0.6, "height": 0.25, "color": [30,180,90], "alpha": 0.75, "text": null }
}
```

`polygon_rectified` → `polygon_raw`로 키 이름도 변경(의미가 완전히 달라졌으므로
같은 키 이름을 재사용해 혼동을 남기지 않음). 기존 PR #1 브랜치에는 아직 실제
운영 데이터가 없으므로(모두 테스트용 synthetic 좌표) 하위 호환 마이그레이션은
불필요.

## 에러 처리

기존과 동일한 원칙 유지: calibration 없음/슬롯 없음/후보점 범위 밖은 명확한
예외. `render_label`의 기존 범위 체크(서브프로젝트 1 최종 리뷰 fix)는 로컬
패치 좌표계에서 그대로 유효하므로 변경 없이 재사용됨.

## 테스트

- `calibration.py`: `pixel_to_ray`/`ray_to_pixel` 라운드트립(3D 버전),
  회전 정렬 헬퍼가 다양한 입력(광선이 이미 `(0,0,1)`인 경우, 정반대인 `(0,0,-1)`인
  경우 포함)에서 올바른 회전을 만드는지, `LocalView.raw_to_local`/`local_to_raw`
  라운드트립.
- **핵심 회귀 테스트**: 실측으로 확인된 콘텐츠 경계 부근(반경 ≈300px, 기존
  전역 모델에서 이미 검게 지워지던 영역)에 슬롯을 배치하고 전체 파이프라인을
  돌려서, 라벨이 정상적으로 합성되고 결과가 검은색으로 뭉개지지 않음을 증명 —
  이번 수정의 존재 이유를 직접 검증.
- `parking_slot.py`: 새 스키마(`polygon_raw`) 기준 저장/로드 라운드트립.
- `pipeline.py`: 기존 테스트를 새 스키마로 갱신 + 위 핵심 회귀 테스트를
  end-to-end로 재현(주변부 슬롯).

## 범위 밖

`patch_size`/`local_f`를 슬롯 실제 각크기 기반으로 동적 산정하는 것,
GUI에서의 로컬 뷰 미리보기, 서브프로젝트 2 이후의 모든 항목. 이번 수정은
서브프로젝트 1의 기하학적 정확성을 실제 카메라 FOV에서도 성립하도록 고치는
것에 한정.

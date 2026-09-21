# 기여 및 품질 관리 가이드 (Contributing Guidelines)

TDC-Studio 프로젝트에 기여하거나 새로운 코드를 작성할 때 준수해야 할 품질 관리 원칙과 4대 체크리스트입니다.

---

## 1. 프로젝트 4대 핵심 관리 체크리스트

모든 개발 및 연구 작업은 다음 4대 체크리스트를 준수해야 합니다:

1. **설치 및 환경:** 가상환경 및 런타임은 반드시 **Python 3.11** 기반의 `uv sync --extra dev`로 관리합니다.
2. **품질 검증 (커밋 전 필수):** `uv run ruff check` 및 `uv run pytest -v` 통과를 의무화합니다.
3. **학습 실행:** 로컬에서는 `--dry-run`으로 인터페이스만 확인하고, 실제 무거운 학습은 `tdc-studio remote run --gpu a100`으로 클라우드에 위임합니다.
4. **실험 기록:** `configs/tracking/`에 W&B 프로젝트명을 명확히 분리하여 Run 및 Artifact를 추적합니다.

---

## 2. 단위 테스트 원칙: Zero Heavy Training

> [!CAUTION]
> `tests/` 폴더 내부의 테스트 코드는 **절대로 실제 대용량 데이터셋 다운로드나 긴 에포크 학습 루프를 돌리지 않아야 합니다.**

* **Synthetic Mock 활용:** 실제 TDC 서버 호출 대신 `tests/conftest.py`에 정의된 3~5개의 합성 SMILES 데이터프레임(`dummy_smiles_df`)을 활용합니다.
* **인터페이스 무결성 검증에 집중:**
  * 텐서 차원(Shape) 정합성
  * 커스텀 Collator의 Batch 결합 여부
  * 손실 함수(Loss) 정상 계산 여부 (NaN 발생 여부 확인)
  * Registry 데코레이터 등록 및 `build()` 팩토리 동작
  * FastAPI 엔드포인트 응답 상태 코드 (200 OK)

---

## 3. 커밋(Commit) 및 PR 전 필수 검증 절차

새로운 코드를 커밋하기 전, 로컬 터미널에서 다음 두 명령어를 반드시 실행하여 에러가 없음을 확인하세요:

```bash
# 1. 정적 분석 및 린트 검사 (자동 포맷팅 및 정렬 포함)
uv run ruff check --fix tdc_studio tests
uv run ruff format tdc_studio tests

# 2. 전체 단위 테스트 실행 (15초 이내 통과 필수)
uv run pytest -v
```

위 두 검증을 통과한 후에만 Git 커밋 및 푸시를 진행합니다.

# 05. 모델 서빙 및 컨테이너 배포 가이드 (Serving & Deployment)

TDC-Studio는 학습된 모델을 손쉽게 프로덕션 환경에 배포할 수 있도록 **FastAPI 기반의 경량 추론 마이크로서비스**와 **Docker 배포 파이프라인**을 제공합니다.

---

## 1. 서빙 아키텍처: `InferencePipeline`

딥러닝 모델 파일(ONNX / TorchScript)은 C++ 순수 텐서만 처리할 수 있으므로, 원시 화학식(SMILES)을 그래프나 텐서로 변환하는 RDKit 전처리가 결합되어야 합니다.

```
[클라이언트 요청] ──► [FastAPI /predict]
                            │
                            ▼ (ThreadPool 비동기 오프로딩)
                     [InferencePipeline]
                      ├── 1. CPU 전처리: SMILES ➔ RDKit ➔ PyG Data / Collate
                      ├── 2. 텐서 모델 추론: PyTorch / ONNX Runtime
                      └── 3. 후처리: Float 점수 / 단위 포맷팅
                            │
                            ▼
[200 OK 응답] ◄─── [Pydantic Response]
```

* **이벤트 루프 보호:** 무거운 RDKit CPU 연산 및 텐서 추론은 `starlette.concurrency.run_in_threadpool`을 통해 백그라운드 스레드 풀로 격리되므로, FastAPI의 I/O 처리량이 저하되지 않습니다.

---

## 2. API 규격 (API Specification)

### (1) 헬스체크 및 준비성 검증 (`GET /healthz`)
* **응답 예시 (200 OK):**
  ```json
  {
    "status": "healthy",
    "model_loaded": true
  }
  ```

### (2) 분자 물성 추론 (`POST /predict`)
* **단일 분자 예측 (ADMET) 요청 바디:**
  ```json
  {
    "smiles": [
      "CC(=O)OC1=CC=CC=C1C(=O)O",
      "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
    ]
  }
  ```
* **약물-표적 결합력 (DTA) 요청 바디:**
  ```json
  {
    "smiles": ["CC(=O)O"],
    "target_sequences": ["MKWVTFISLLLLFSSAYSRG"]
  }
  ```
* **응답 바디 (200 OK):**
  ```json
  {
    "predictions": [1.45, 0.82],
    "unit": "score",
    "model_name": "TDC-Studio-Model"
  }
  ```

---

## 3. 로컬 서빙 API 구동

```bash
# 개발 모드로 API 서버 실행
uv run tdc-studio serve --host 0.0.0.0 --port 8000

# 대화형 Swagger API 문서 확인
# 브라우저에서 http://localhost:8000/docs 접속
```

---

## 4. Docker 컨테이너 프로덕션 배포

### (1) Docker Compose 구동
```bash
# 서빙 컨테이너 빌드 및 백그라운드 구동
docker compose -f deploy/docker-compose.yml up -d
```

### (2) 컨테이너 상태 및 헬스체크 확인
```bash
docker ps
# 헬스체크 테스트
curl -f http://localhost:8000/healthz
```

### (3) 이미지 최적화 포인트
* **UV 멀티스테이지 빌드:** 무거운 pip 대신 `uv` 바이너리를 사용하여 이미지 빌드 속도를 대폭 단축.
* **시스템 종속성 포함:** Linux 환경에서 RDKit 구동에 필수적인 `libgomp1` (OpenMP) 라이브러리를 기본 설치.

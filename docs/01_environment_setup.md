# 01. 개발 환경 설정 및 설치 가이드 (Environment Setup)

이 문서는 `TDC-Studio`의 개발 및 실험 환경을 안정적이고 빠르게 구축하기 위한 가이드입니다.

---

## 1. 런타임 표준: Python 3.11 고정 이유

바이오/화학 인포매틱스 생태계(`PyTDC`, `RDKit`, `PyTorch Geometric`, `onnxruntime`)의 실측 의존성 분석 결과, **Python 3.11이 모든 핵심 라이브러리의 컴파일된 C-extension 바이너리 휠(Wheel)을 제공하는 유일한 Golden Standard**입니다.

### PyPI 의존성 실측 비교표

| 패키지 | 요구 사항 및 현황 | Python 3.10 | Python 3.11 | Python 3.12 | Python 3.13 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`PyTDC` (1.1.15)** | `tiledbsoma 1.11.4`, `numpy<2.0.0` 필수 | O | **O** | X (바이너리 부재) | X (빌드 실패) |
| **`tiledbsoma`** | 사전 빌드 Wheel: `cp38` ~ `cp311` | O | **O** | X (소스 빌드 에러) | X |
| **`rdkit`** | TDC 종속: `<2024.3.1` (cp38~cp312) | O | **O** | O | X (3.13 휠 미지원) |
| **`onnxruntime`** | 최신 버전(1.30.0+)은 `requires-python >= 3.11` | X | **O** | O | O |
| **종합 호환성** | **전체 파이프라인 무결성** | 부분 지원 | **완벽 호환 (권장)** | 빌드 실패 | 빌드 실패 |

> [!IMPORTANT]
> 따라서 로컬 개발, 가상환경, 도커 서빙 런타임 모두 **Python 3.11**을 기준으로 동작하도록 설계되었습니다.

---

## 2. 로컬 가상환경 설치 (UV 기반)

Rust 기반 초고속 패키지 관리자인 `uv`를 통해 수 초 이내에 가상환경을 구축합니다.

### 2.1 uv 설치
* **macOS / Linux:**
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
* **Windows (PowerShell):**
  ```powershell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```

### 2.2 프로젝트 환경 동기화
```bash
# 저장소 복제
git clone https://github.com/brightonmoon/tdc-studio.git
cd tdc-studio

# Python 3.11 가상환경 생성 및 의존성 동기화
uv sync --extra dev
```

### 2.3 환경 정상 동작 검증
실제 다운로드나 모델 학습 없이 시스템 인터페이스가 완벽한지 테스트합니다:
```bash
uv run pytest -v
```

---

## 3. 클라우드 GPU 러너 설정 (Google Colab CLI)

로컬 머신에 고성능 GPU가 없어도 Google Colab의 클라우드 GPU(T4, L4, A100)를 터미널에서 제어할 수 있습니다.

### 3.1 Colab CLI 도구 설치
* **공식 패키지:** [`google-colab-cli`](https://github.com/googlecolab/google-colab-cli)
```bash
uv tool install google-colab-cli
# 또는
pip install google-colab-cli
```

### 3.2 계정 로그인 및 인증
터미널에서 Google 계정으로 1회 인증합니다:
```bash
colab login
```

### 3.3 Windows 환경 사용 시 주의점
`google-colab-cli`는 현재 Linux 및 macOS를 기본 지원합니다. Windows 사용자의 경우 다음 두 가지 방법 중 하나를 선택합니다:
1. **WSL2 (Windows Subsystem for Linux)** 내부에서 `colab` 명령어를 실행 (권장).
2. TDC-Studio의 노트북 생성 명령어로 일회성 Jupyter Notebook을 추출하여 브라우저에서 실행:
   ```bash
   uv run tdc-studio remote export-notebook --output run_colab.ipynb
   ```

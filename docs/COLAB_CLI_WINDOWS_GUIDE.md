# Google Colab CLI Windows Integration & Multi-Account Guide

이 문서는 Windows 환경에서 [google-colab-cli](https://github.com/googlecolab/google-colab-cli)를 원활하게 구동하고, GPU 쿼터 소진 시 계정을 손쉽게 전환하며, 인자 주입(Argument Injection)을 통해 원격 실행을 수행할 수 있도록 **TDC-Studio 프레임워크에 이식된 기능**을 설명합니다.

---

## 1. 배경 및 해결된 문제

### 1.1. 공식 CLI의 Windows 환경 제약
공식 `google-colab-cli`는 Linux/macOS 중심으로 설계되어 Windows 환경에서 다음과 같은 문제가 발생합니다:
- **`termios` / `tty` 부재**: `colab console` 및 터미널 초기화 시 Unix 전용 모듈(`termios`, `signal.SIGWINCH`) 호출로 인한 크래시.
- **`colab exec` 인자 전달 불가**: 원격 커널에 로컬 파이썬 파일을 실행할 때 CLI argument(`sys.argv`)를 직접 주입하지 못함.
- **단일 계정 고정**: OAuth 인증 토큰 경로가 `~/.config/colab-cli/token.json`으로 고정되어 있어, 무료 티어 GPU 할당량 초과 시 계정 전환이 번거로움.

### 1.2. TDC-Studio의 3중 이식 레이어
이를 해결하기 위해 본 프로젝트에서는 3개 계층으로 기능을 완전 이식하였습니다:
1. **Python 코어 프레임워크 API**: [`ColabAccountManager`](file:///C:/Users/xps/orca/workspaces/tdc-studio/tdc_studio/remote/colab_account.py), [`ColabRunner`](file:///C:/Users/xps/orca/workspaces/tdc-studio/tdc_studio/remote/colab_runner.py)
2. **프로젝트 CLI**: `tdc-studio remote accounts`, `tdc-studio remote switch`, `tdc-studio remote run/exec`
3. **PowerShell 스크립트 도구**: [`scripts/colab_switch.ps1`](file:///C:/Users/xps/orca/workspaces/tdc-studio/scripts/colab_switch.ps1), [`scripts/colab_exec.ps1`](file:///C:/Users/xps/orca/workspaces/tdc-studio/scripts/colab_exec.ps1)

---

## 2. 계정 전환 메커니즘 (Multi-Account Switching)

### 2.1. 동작 원리
- Google OAuth 인증이 완료되면 `C:\Users\<user>\.config\colab-cli\token.json`에 자격 증명이 저장됩니다.
- 본 도구는 등록된 계정들의 토큰을 `token_<name>.json` 형태로 보관하며, `token.json`의 MD5 해시를 비교하여 현재 활성 계정을 실시간 감지합니다.
- 계정 전환 시 대상 계정 토큰을 `token.json`으로 원자적 복사(Atomic copy)하여 즉시 활성화합니다.

### 2.2. GPU Quota 초과 시 자동 순환 (Auto-Rotation)
무료 티어 또는 단일 계정에서 GPU를 과도하게 사용하면 다음과 같은 오류가 발생합니다:
- `TooManyAssignmentsError: Failed to issue request POST ... Precondition Failed`
- `ColabRequestError: ... variant=GPU&accelerator=T4: Precondition Failed`

`ColabRunner`는 이러한 오류 문자열을 정규식으로 감지하여, 대기 중인 다음 계정으로 자동 스위칭한 뒤 작업을 1회 자동 재시도(`--auto-switch`)합니다.

---

## 3. 사용 방법

### 방법 A. TDC-Studio CLI 사용 (권장)

#### 계정 목록 및 활성 상태 확인
```bash
uv run tdc-studio remote accounts
```

#### 계정 전환
```bash
uv run tdc-studio remote switch use munhyoungdo@gmail.com
```

#### 현재 로그인 세션을 새 이름으로 저장
```bash
uv run tdc-studio remote switch save main
```

#### 신규 Google 계정 브라우저 인증 및 등록
```bash
uv run tdc-studio remote switch new sub1
```

#### 계정 삭제
```bash
uv run tdc-studio remote switch delete sub1
```

#### 원격 Cloud GPU 작업 실행 (자동 계정 회전 포함)
```bash
# 기본 계정으로 실행하며 Quota 에러 시 자동 다음 계정으로 전환
uv run tdc-studio remote run --gpu t4 --command "tdc-studio train --config configs/config.yaml"

# 특정 계정을 지정하여 실행
uv run tdc-studio remote run --gpu a100 --account munhyoungdo@gmail.com --command "tdc-studio train"
```

#### 활성 세션에 인자 주입 원격 실행 (`exec`)
```bash
uv run tdc-studio remote exec -s <SESSION_ID> -f train.py --arg "--epochs" --arg "10"
```

---

### 방법 B. PowerShell 스크립트 사용

PowerShell 터미널에서 단독으로 빠르게 계정 및 원격 실행을 제어할 수 있습니다:

```powershell
# 1. 계정 목록 확인
.\scripts\colab_switch.ps1 list

# 2. 특정 계정 활성화
.\scripts\colab_switch.ps1 use munhyoungdo@gmail.com

# 3. 현재 로그인 저장
.\scripts\colab_switch.ps1 save main

# 4. 새 계정 OAuth 인증 등록
.\scripts\colab_switch.ps1 new sub_account_2

# 5. colab exec 인자 주입 실행
.\scripts\colab_exec.ps1 <SESSION_ID> deploy/colab_runner_job.py "tdc-studio train"
```

---

### 방법 C. Python 코드 레벨 연동

코드 내에서 프로그래밍 방식으로 계정을 관리하고 작업을 오케스트레이션할 수 있습니다:

```python
from tdc_studio.remote import ColabAccountManager, ColabRunner

# 1. 계정 매니저 활용
mgr = ColabAccountManager()
active = mgr.get_active_account()
print(f"Active Account: {active}")

# 계정 전환
mgr.use_account("munhyoungdo@gmail.com")

# 2. GPU 러너 구동 (Quota 에러 시 자동 계정 회전)
runner = ColabRunner(
    gpu_type="T4",
    account="munhyoungdo@gmail.com",
    auto_switch_on_quota=True,  # TooManyAssignmentsError 발생 시 다른 계정으로 자동 전환 및 재시도
)

# Colab GPU 작업 디스패치
exit_code = runner.run_remote_job("tdc-studio train --config configs/config.yaml")
```

---

## 4. 파일 구성 안내

| 파일 경로 | 설명 |
|-----------|------|
| [`tdc_studio/remote/colab_account.py`](file:///C:/Users/xps/orca/workspaces/tdc-studio/tdc_studio/remote/colab_account.py) | 다중 계정 토큰 해시 비교, 스위칭, 신규 인증, 자동 순환 코어 모듈 |
| [`tdc_studio/remote/colab_runner.py`](file:///C:/Users/xps/orca/workspaces/tdc-studio/tdc_studio/remote/colab_runner.py) | Windows UTF-8 프로세스 제어, Quota 에러 탐지 & 재시도, `sys.argv` 주입 래퍼 |
| [`scripts/colab_switch.ps1`](file:///C:/Users/xps/orca/workspaces/tdc-studio/scripts/colab_switch.ps1) | 계정 전환 전용 PowerShell 스크립트 |
| [`scripts/colab_exec.ps1`](file:///C:/Users/xps/orca/workspaces/tdc-studio/scripts/colab_exec.ps1) | `colab exec` 인자 주입용 PowerShell 스크립트 |
| [`tests/test_colab_account.py`](file:///C:/Users/xps/orca/workspaces/tdc-studio/tests/test_colab_account.py) | 계정 관리자 단위 테스트 |
| [`tests/test_remote_runner.py`](file:///C:/Users/xps/orca/workspaces/tdc-studio/tests/test_remote_runner.py) | 원격 실행, 인자 주입 및 에러 탐지 단위 테스트 |

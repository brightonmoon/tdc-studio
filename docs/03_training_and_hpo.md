# 03. 학습 및 하이퍼파라미터 최적화 운영 가이드 (Training & HPO)

TDC-Studio는 로컬 개발 머신의 자원 부담을 없애고 클라우드 GPU의 유연성을 극대화하기 위해 **"3단계 표준 실험 파이프라인"**을 운영합니다.

```
[1단계: 로컬 사전 검증] ────► [2단계: 클라우드 GPU 위임] ────► [3단계: W&B 모니터링]
  `--dry-run` 1초 점검           Google Colab CLI (A100)           실시간 메트릭 및 아티팩트
```

---

## 1단계. 로컬 1-Step 무결성 사전 검증 (Dry-run)

실제 수십 분~수 시간이 걸리는 학습을 클라우드에 올리기 전에, 설정 파일(YAML)의 문법 오류, 텐서 차원 불일치, 데이터 로더 정합성을 1초 만에 로컬에서 점검합니다.

### (1) 단일 모델 학습 드라이런
```bash
uv run tdc-studio train --config configs/config.yaml --dry-run
```
* **동작:** 1 에포크, 1 배치만 Forward 및 Backward를 수행하여 모델 파라미터 업데이트가 정상인지 확인하고 즉시 종료합니다.

### (2) Optuna HPO 드라이런
```bash
uv run tdc-studio tune --config configs/config.yaml --n-trials 2 --dry-run
```
* **동작:** 2개의 가상 Trial을 1 배치 단위로 수행하여 YAML Search Space 파싱 및 Pruner 인터페이스가 유효한지 확인합니다.

---

## 2단계. Google Colab CLI를 통한 클라우드 GPU 학습

로컬 검증이 완료되면, [`google-colab-cli`](https://github.com/googlecolab/google-colab-cli)를 통해 클라우드 GPU 인스턴스를 일회성(Ephemeral)으로 띄워 학습을 실행합니다.

### (1) 단일 모델 베이스라인 학습 실행
```bash
# T4 또는 A100 GPU에서 30 에포크 학습 디스패치
uv run tdc-studio remote run --gpu a100 --command "tdc-studio train --config configs/config.yaml --epochs 30"
```

### (2) 대규모 Optuna HPO 탐색 실행
```bash
# A100 GPU에서 50개 Trial의 병렬/순차 탐색 수행
uv run tdc-studio remote run --gpu a100 --command "tdc-studio tune --config configs/config.yaml --n-trials 50"
```

### (3) 백그라운드 구동 프로세스
1. `colab run --gpu=a100` 명령이 클라우드 VM을 즉시 프로비저닝합니다.
2. VM 내부에서 [`deploy/colab_bootstrap.sh`](file:///C:/Users/xps/orca/workspaces/tdc-studio/deploy/colab_bootstrap.sh)가 실행되어 `uv`를 통해 의존성을 고속 동기화합니다.
3. 지정된 학습/튜닝 태스크가 수행되며 메트릭이 W&B로 실시간 스트리밍됩니다.
4. 작업이 끝나면 VM이 자동으로 종료(Release)되어 요금이 과다 청구되지 않습니다.

---

## 3단계. 선언적 Optuna 탐색 공간 및 Pruning 설정

`configs/hpo/optuna_default.yaml`에서 파라미터 탐색 범위와 조기 종료(Pruner)를 제어합니다:

```yaml
sampler: "tpe"                # TPESampler (베이지안 최적화)
direction: "minimize"         # regression: minimize, classification: maximize

pruner:
  type: "MedianPruner"        # 중간 성과가 하위 50%인 trial 조기 중단
  n_startup_trials: 5         # 초기 5개 trial은 pruning 없이 완주
  n_warmup_steps: 3           # 각 trial의 초기 3 에포크 동안은 pruning 유예

search_space:
  lr:
    type: "float"
    low: 0.00005
    high: 0.005
    log: true
  hidden_dim:
    type: "categorical"
    choices: [64, 128, 256]
  dropout:
    type: "float"
    low: 0.05
    high: 0.3
```

---

## 4단계. Weights & Biases (W&B) 실험 모니터링

* **설정:** `configs/tracking/wandb_default.yaml`에서 `project` 이름 지정 및 `enabled: true` 설정.
* **추적 항목:**
  * Trial별 Hyperparameter (lr, hidden_dim, dropout 등).
  * 에포크별 검증 메트릭 (MAE, ROC-AUC, Validation Loss).
  * Pruning 여부 (성능 저조 trial 자동 중단 이력).
  * 최종 학습된 최적 모델 가중치 아티팩트(`best_model.pt`).

# configs/ 선언적 설정(Config) 가이드

TDC-Studio는 모든 실험을 `configs/` 폴더 내의 선언적 YAML 파일 조합으로 제어합니다.

---

## 1. 디렉터리 구성

```text
configs/
├── config.yaml                   # 메인 통합 설정 (기본 실행 진입점)
├── data/                         # 데이터셋 및 전처리 설정
│   ├── admet_caco2.yaml          # ADMET 단일 분자 예측
│   └── dta_bindingdb.yaml        # DTA 멀티모달 예측
├── model/                        # 신경망 아키텍처 백본 설정
│   ├── graph_transformer.yaml    # 분자 그래프 트랜스포머
│   └── chemberta.yaml            # 시퀀스 트랜스포머
├── hpo/                          # Optuna 하이퍼파라미터 탐색 공간 설정
│   └── optuna_default.yaml
└── tracking/                     # W&B 로깅 및 모니터링 설정
    └── wandb_default.yaml
```

---

## 2. 주요 설정 필드 명세

### (1) 데이터 설정 (`data/`)
```yaml
type: "admet_loader"             # Registry 등록 키 (admet_loader 또는 dta_loader)
dataset_name: "caco2_wang"       # TDC 데이터셋 고유 명칭
split_type: "scaffold"           # 분할 방식: scaffold, random, cold_drug, cold_target
batch_size: 32                   # 배치 크기
max_epochs: 10                   # 에포크 수
params:
  task_type: "regression"        # regression, binary_classification
  metric_name: "mae"             # 평가 지표 (mae, rmse, roc-auc 등)
```

### (2) 모델 설정 (`model/`)
```yaml
type: "graph_transformer"        # Registry 등록 키 (graph_transformer, graph_transformer_dta 등)
in_dim: 14                       # 원자 피처 입력 차원 (기본 14)
hidden_dim: 128                  # 은닉층 차원
num_layers: 3                    # 그래프/인코더 레이어 수
dropout: 0.1                     # 드롭아웃 비율
```

### (3) HPO 설정 (`hpo/`)
```yaml
sampler: "tpe"                   # tpe, random
direction: "minimize"            # minimize, maximize
pruner:
  type: "MedianPruner"           # 조기 종료 전략
  n_startup_trials: 5            # 초기 보장 완주 trial 수
  n_warmup_steps: 2              # 각 trial별 조기종료 유예 에포크 수
search_space:
  lr:
    type: "float"
    low: 0.00005
    high: 0.005
    log: true
  hidden_dim:
    type: "categorical"
    choices: [64, 128, 256]
```

### (4) 실험 추적 설정 (`tracking/`)
```yaml
project: "tdc-studio-admet"      # W&B 프로젝트명
entity: null                     # W&B 조직/사용자명
enabled: true                    # W&B 로깅 활성화 여부
```

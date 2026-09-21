# 02. 알고리즘 및 모델 백본 선택 가이드 (Algorithm Selection)

TDC(Therapeutics Data Commons)의 생물학적/화학적 데이터는 태스크 성격에 따라 최적의 신경망 아키텍처가 달라집니다. 이 문서는 태스크별 알고리즘 선택 기준과 원리를 설명합니다.

---

## 1. 알고리즘 의사결정 트리 (Decision Tree)

```
                            [TDC 데이터 태스크]
                                     │
         ┌───────────────────────────┴───────────────────────────┐
         ▼                                                       ▼
[단일 분자 물성/활성 예측 (Single-pred)]                 [약물-표적 상호작용 예측 (Multi-pred)]
 (ADME: Caco-2, HIA, Lipo 등)                            (DTA, DTI: BindingDB, DAVIS, KIBA)
 (Tox: hERG, Ames, Tox21 등)                                     │
         │                                                       ▼
         ├─► [선택 1] 분자 그래프 모델 (권장)            [선택 1] 멀티모달 그래프+시퀀스 모델 (권장)
         │   `graph_transformer`                         `graph_transformer_dta`
         │   - 원자의 2D 결합 위상 보존                  - 저분자: 2D Graph (GCN/Transformer)
         │   - 국소 작용기 및 물리화학적 특징            - 표적 단백질: 1D Amino Sequence
         │                                               - Cross-Attention / Joint Pooling
         └─► [선택 2] 시퀀스 트랜스포머                          │
             `sequence_transformer`                      [선택 2] 듀얼 시퀀스 트랜스포머
             - SMILES 문자열 토큰화                      - ChemBERTa (SMILES) + ESM2 (Protein)
             - 대규모 화합물 사전학습 전이 효과
```

---

## 2. 모델 백본별 상세 특징 및 내부 구조

### (1) `graph_transformer` (분자 그래프 트랜스포머 / GNN)
* **표현형:** RDKit을 통해 SMILES를 분자 그래프(원자 = 노드, 화학 결합 = 엣지)로 변환 (`SmilesToGraphTransform`).
* **노드 피처 (14차원):** 원자 번호 One-hot(10종) + 결합 차수 + 형식 전하 + 방향족성 여부 + 수소 개수.
* **아키텍처:**
  * Node Embedding Layer $\rightarrow$ Multi-layer Graph Convolutions $\rightarrow$ Global Mean Readout Pooling $\rightarrow$ MLP Prediction Head.
* **추천 태스크:**
  * ADME 회귀 문제 (`caco2_wang`, `lipophilicity_astrazeneca`, `solubility_aqsoldb`).
  * 입체 배향 및 결합 구조가 물리화학적 성질에 직접적인 영향을 미치는 데이터셋.

### (2) `sequence_transformer` (시퀀스 기반 트랜스포머)
* **표현형:** SMILES 문자열 또는 단백질 FASTA 아미노산 서열을 토큰 인덱스로 변환 (`SequenceTokenizer`).
* **아키텍처:**
  * Token Embedding $\rightarrow$ Multi-Head Self-Attention ($N$ layers) $\rightarrow$ Masked Average Pooling $\rightarrow$ MLP Prediction Head.
* **추천 태스크:**
  * 독성 유무 및 세포 투과성 분류 문제 (`bbb_martins`, `herg`, `ames`).
  * 자연어 처리 방식의 대규모 분자 사전학습 가중치(ChemBERTa, ESM2) 전이학습 시 유리.

### (3) `graph_transformer_dta` (멀티모달 약물-표적 결합력 모델)
* **표현형:** 저분자 화합물(2D Graph) + 표적 단백질(1D Amino Acid Sequence).
* **아키텍처:**
  * Drug Tower: GCN 기반 분자 그래프 임베딩.
  * Target Tower: Bidirectional GRU / Transformer 기반 서열 임베딩.
  * Interaction Tower: 결합 표현형(Joint Representation) 결합 및 친화도(Kd, Ki, IC50) 예측.
* **추천 태스크:**
  * 약물-표적 친화도(DTA) 및 상호작용(DTI) (`bindingdb_kd`, `davis`, `kiba`).

---

## 3. TDC 데이터셋별 권장 매핑 매트릭스

| 카테고리 | 데이터셋 이름 | 태스크 타입 | 기본 평가 지표 | 추천 모델 백본 | 추천 설정 파일 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ADME** | `caco2_wang` | Regression | MAE | `graph_transformer` | `configs/data/admet_caco2.yaml` |
| **ADME** | `lipophilicity_astrazeneca` | Regression | MAE / RMSE | `graph_transformer` | `configs/data/admet_caco2.yaml` |
| **ADME** | `bbb_martins` | Binary Classification | ROC-AUC | `sequence_transformer` | `configs/model/chemberta.yaml` |
| **Tox** | `herg` | Binary Classification | ROC-AUC | `graph_transformer` | `configs/model/graph_transformer.yaml` |
| **Tox** | `ames` | Binary Classification | ROC-AUC | `sequence_transformer` | `configs/model/chemberta.yaml` |
| **DTI / DTA** | `bindingdb_kd` | Regression | MAE / Pearson | `graph_transformer_dta` | `configs/data/dta_bindingdb.yaml` |
| **DTI / DTA** | `davis` | Regression | MSE / Spearman | `graph_transformer_dta` | `configs/data/dta_bindingdb.yaml` |

---

## 4. 손실 함수(Loss Strategy) 자동 매핑 원리

TDC-Studio의 모든 모델은 [`BaseTherapeuticsModel`](file:///C:/Users/xps/orca/workspaces/tdc-studio/tdc_studio/models/base.py)을 상속받아 태스크 타입에 따라 손실 함수를 동적으로 설정합니다:

```python
# Task Type에 따라 Criterion 자동 분기
if task_type == "regression":
    criterion = nn.MSELoss()
elif task_type == "binary_classification":
    criterion = nn.BCEWithLogitsLoss()
elif task_type == "multiclass_classification":
    criterion = nn.CrossEntropyLoss()
```
따라서 사용자는 데이터셋의 `task_type`만 지정하면 손실 함수 코드를 별도로 작성할 필요가 없습니다.

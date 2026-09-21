# 04. 컴포넌트 확장 가이드 (Extending Components)

TDC-Studio는 **느슨한 결합(Loose Coupling)**과 **레지스트리 패턴(Registry Pattern)**을 채택하여, 기존 코드를 한 줄도 수정하지 않고 새로운 모델, 데이터셋, 전처리기를 추가할 수 있습니다.

---

## 1. 신규 모델 아키텍처 추가 (Add New Model)

### 단계 1: 모델 클래스 작성 및 데코레이터 등록
`tdc_studio/models/` 아래에 새로운 파이썬 파일을 생성하고 [`@MODELS.register`](file:///C:/Users/xps/orca/workspaces/tdc-studio/tdc_studio/core/registry.py) 데코레이터를 붙입니다:

```python
# tdc_studio/models/graph/my_gin.py
import torch
import torch.nn as nn
from torch_geometric.nn import GINConv, global_add_pool
from tdc_studio.core.registry import MODELS
from tdc_studio.models.base import BaseTherapeuticsModel

@MODELS.register("my_gin_model")
class MyGINModel(BaseTherapeuticsModel):
    def __init__(self, config: dict):
        super().__init__(config)
        hidden_dim = config.get("hidden_dim", 64)
        nn1 = nn.Sequential(nn.Linear(config.get("in_dim", 14), hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.conv = GINConv(nn1)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, batch: dict) -> torch.Tensor:
        graph = batch["drug_graph"]
        h = self.conv(graph.x, graph.edge_index)
        hg = global_add_pool(h, graph.batch)
        return self.head(hg)
```

### 단계 2: 설정 파일(YAML) 생성
새로운 모델을 사용할 때는 YAML에 등록한 이름(`my_gin_model`)을 지정합니다:

```yaml
# configs/model/my_gin.yaml
type: "my_gin_model"
in_dim: 14
hidden_dim: 128
```

> **결과:** CLI `tdc-studio train --model configs/model/my_gin.yaml` 또는 Optuna HPO에서 즉시 이 모델을 탐색 대상으로 사용할 수 있습니다.

---

## 2. 신규 TDC 데이터셋 로더 추가 (Add New Dataset)

ADMET나 DTA 외에 독성(Tox) 또는 신규 벤치마크 카테고리를 추가할 때:

```python
# tdc_studio/data/tox_pred.py
from tdc_studio.core.registry import DATASETS
from tdc_studio.data.base import BaseTDCDataModule
from tdc_studio.data.transforms import SmilesToGraphTransform

@DATASETS.register("tox_loader")
class ToxDataModule(BaseTDCDataModule):
    def __init__(self, dataset_name: str = "herg", **kwargs):
        super().__init__(
            dataset_name=dataset_name,
            task_type="binary_classification",
            metric_name="roc-auc",
            **kwargs
        )
        self.graph_transform = SmilesToGraphTransform()

    def prepare_data(self) -> None:
        from tdc.single_pred import Tox
        data = Tox(name=self.dataset_name)
        self.splits = data.get_split(method=self.split_type, seed=self.seed)
        self.is_prepared = True

    def setup_loaders(self, batch_size: int = 32, **kwargs):
        # build MolecularDataset and return (train, val, test) Loaders
        ...
```

---

## 3. 커스텀 전처리기 및 피처라이저 추가 (Add Transform)

3D 좌표나 ECFP 지문(Fingerprint) 등 새로운 분자 표현형을 도입할 때:

```python
# tdc_studio/data/transforms.py
from rdkit.Chem import AllChem
from tdc_studio.core.registry import TRANSFORMS

@TRANSFORMS.register("morgan_fingerprint")
class MorganFingerprintTransform:
    def __init__(self, radius: int = 2, n_bits: int = 2048):
        self.radius = radius
        self.n_bits = n_bits

    def __call__(self, smiles: str):
        mol = Chem.MolFromSmiles(smiles)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, nBits=self.n_bits)
        return torch.tensor(list(fp), dtype=torch.float32)
```

---

## 4. 자동 탐색 메커니즘 (Auto-Discovery)

`TDC-Studio`의 모든 CLI 명령어는 실행 시작 시 `auto_import_modules("tdc_studio")`를 호출합니다.  
따라서 `tdc_studio` 패키지 내부 어디에 새로운 클래스를 추가하더라도 별도의 import 명시 없이 레지스트리에 자동 등록됩니다.

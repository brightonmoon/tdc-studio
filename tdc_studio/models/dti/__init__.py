"""DTI model components: ProteinCNNEncoder, BilinearAttentionFusion, GraphDTAModel."""

from tdc_studio.models.dti.dta_model import GraphDTAModel
from tdc_studio.models.dti.fusion import BilinearAttentionFusion
from tdc_studio.models.dti.protein_encoder import ProteinCNNEncoder

__all__ = ["ProteinCNNEncoder", "BilinearAttentionFusion", "GraphDTAModel"]

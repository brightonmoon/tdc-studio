"""DTI model components: ProteinCNNEncoder, BilinearAttentionFusion, GraphDTAModel."""

from tdc_studio.models.dti.protein_encoder import ProteinCNNEncoder
from tdc_studio.models.dti.fusion import BilinearAttentionFusion
from tdc_studio.models.dti.dta_model import GraphDTAModel

__all__ = ["ProteinCNNEncoder", "BilinearAttentionFusion", "GraphDTAModel"]

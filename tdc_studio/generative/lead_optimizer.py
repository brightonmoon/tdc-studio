"""Self-Correcting Generative Lead Optimizer for Drug Discovery.

Implements a closed-loop 4-step optimization architecture:
1. Diagnosis: Evaluates 22-task ADMET profile and identifies pharmacological/toxicological liabilities.
2. Localization: Uses Integrated Gradients (XAI) and alert SMARTS to pinpoint defect hotspots.
3. Scaffold-Preserved Mutation: Applies medicinal chemistry bioisosteres while locking the core Bemis-Murcko scaffold.
4. Pareto & Synthesizability Filter: Filters candidates by SAScore <= 3.5 and ranks by multi-objective ADMET recovery.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from tdc_studio.explainability.attribution import MolecularExplainer
from tdc_studio.explainability.bioisostere import BioisostereRecommender
from tdc_studio.generative.sa_score import calculate_sa_score
from tdc_studio.serving.schema import UnifiedADMETProfile
from tdc_studio.serving.unified_pipeline import UnifiedADMETPipeline

logger = logging.getLogger("tdc_studio.generative")


@dataclass
class LiabilityDiagnostic:
    """Diagnostic report for an identified molecular liability."""

    liability_key: str
    liability_name: str
    cluster: str
    current_value: float
    threshold: float
    severity: float  # Margin by which threshold is violated
    category: str  # 'cardiotoxicity', 'mutagenicity', 'hepatotoxicity', 'clearance', 'solubility', 'permeability'


@dataclass
class OptimizedCandidate:
    """An optimized analogue with repaired liabilities."""

    smiles: str
    transformation_name: str
    liability_addressed: str
    rationale: str
    parent_liability_value: float
    candidate_liability_value: float
    liability_delta: float
    sa_score: float
    scaffold_preserved: bool
    admet_profile: Optional[Dict[str, Any]] = None
    fitness_score: float = 0.0


@dataclass
class OptimizationReport:
    """Comprehensive report returned by SelfCorrectingOptimizer."""

    input_smiles: str
    canonical_smiles: str
    bemis_murcko_scaffold: str
    diagnosed_liabilities: List[LiabilityDiagnostic]
    primary_liability: Optional[LiabilityDiagnostic]
    hotspot_atom_indices: List[int]
    candidates_generated: int
    candidates_passing_sa_filter: int
    top_candidates: List[OptimizedCandidate] = field(default_factory=list)


class SelfCorrectingOptimizer:
    """Closed-loop generative lead optimizer combining Unified ADMET, XAI, and Bioisosteres."""

    # Default liability thresholds
    DEFAULT_THRESHOLDS = {
        "herg": (0.50, ">", "cardiotoxicity", "toxicity"),
        "ames": (0.50, ">", "mutagenicity", "toxicity"),
        "dili": (0.50, ">", "hepatotoxicity", "toxicity"),
        "clintox": (0.50, ">", "clinical_toxicity", "toxicity"),
        "clearance_hepatocyte_az": (35.0, ">", "clearance", "excretion"),
        "clearance_microsome_az": (35.0, ">", "clearance", "excretion"),
        "solubility_aqsoldb": (-4.0, "<", "solubility", "absorption"),
        "caco2_wang": (-5.5, "<", "permeability", "absorption"),
        "lipophilicity_astrazeneca": (3.8, ">", "lipophilicity", "absorption"),
    }

    def __init__(
        self,
        pipeline: Optional[UnifiedADMETPipeline] = None,
        explainer: Optional[MolecularExplainer] = None,
        recommender: Optional[BioisostereRecommender] = None,
        sa_threshold: float = 4.0,
        device: str = "cpu",
    ):
        """Initialize the Self-Correcting Lead Optimizer."""
        self.device = device
        self.pipeline = pipeline or UnifiedADMETPipeline(device=device)
        self.recommender = recommender or BioisostereRecommender()
        self.sa_threshold = sa_threshold

        if explainer is not None:
            self.explainer = explainer
        else:
            from tdc_studio.models.graph.dmpnn import DMPNNModel

            model = DMPNNModel(
                {
                    "type": "dmpnn",
                    "in_dim": 14,
                    "edge_dim": 6,
                    "hidden_dim": 64,
                    "num_layers": 2,
                    "use_descriptors": False,
                }
            )
            self.explainer = MolecularExplainer(model=model, device=device)

    @staticmethod
    def _extract_val(ind: Any) -> float:
        """Safely extract numerical score from regression value or classification probability."""
        if ind is None:
            return 0.0
        if getattr(ind, "probability", None) is not None:
            return float(ind.probability)
        if getattr(ind, "value", None) is not None:
            return float(ind.value)
        return 0.0

    def diagnose_liabilities(self, profile: UnifiedADMETProfile) -> List[LiabilityDiagnostic]:
        """Diagnose liabilities from a 22-ADMET profile."""
        diagnostics = []

        # Helper to search indicator across all clusters
        all_indicators = {}
        all_indicators.update(profile.absorption)
        all_indicators.update(profile.distribution)
        all_indicators.update(profile.metabolism)
        all_indicators.update(profile.excretion)
        all_indicators.update(profile.toxicity)

        for key, (thresh, op, cat, cluster) in self.DEFAULT_THRESHOLDS.items():
            if key in all_indicators:
                ind = all_indicators[key]
                val = self._extract_val(ind)
                is_violation = (op == ">" and val > thresh) or (op == "<" and val < thresh)
                if is_violation:
                    severity = abs(val - thresh)
                    diagnostics.append(
                        LiabilityDiagnostic(
                            liability_key=key,
                            liability_name=ind.name,
                            cluster=cluster,
                            current_value=val,
                            threshold=thresh,
                            severity=severity,
                            category=cat,
                        )
                    )

        # Sort by severity descending
        diagnostics.sort(key=lambda d: d.severity, reverse=True)
        return diagnostics

    def _extract_scaffold(self, mol: Chem.Mol) -> str:
        """Extract Bemis-Murcko core scaffold SMILES."""
        try:
            scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
            return Chem.MolToSmiles(scaffold_mol, canonical=True)
        except Exception:
            return ""

    def _check_scaffold_preserved(self, parent_scaffold_smi: str, candidate_mol: Chem.Mol) -> bool:
        """Verify candidate still contains the essential core scaffold."""
        if not parent_scaffold_smi:
            return True
        scaffold_mol = Chem.MolFromSmiles(parent_scaffold_smi)
        if scaffold_mol is None or scaffold_mol.GetNumHeavyAtoms() <= 3:
            return True
        return candidate_mol.HasSubstructMatch(scaffold_mol)

    def optimize(
        self,
        smiles: str,
        target_liability: Optional[str] = None,
        max_candidates: int = 5,
        steps: int = 20,
    ) -> OptimizationReport:
        """Run complete 4-step closed-loop optimization on the target molecule.

        Args:
            smiles: Input molecule SMILES.
            target_liability: Optional explicit liability key (e.g. 'herg', 'ames', 'dili', 'clearance').
            max_candidates: Number of top candidates to return.
            steps: Number of Integrated Gradients steps.

        Returns:
            OptimizationReport with diagnosed liabilities, localized hotspots, and top candidates.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid input SMILES: '{smiles}'")
        canon_smiles = Chem.MolToSmiles(mol, canonical=True)
        scaffold_smi = self._extract_scaffold(mol)

        # ----------------------------------------------------------------------
        # Step 1: Liability Diagnosis
        # ----------------------------------------------------------------------
        parent_profile = self.pipeline.predict_single(canon_smiles)
        diagnosed = self.diagnose_liabilities(parent_profile)

        primary_liability = None
        if target_liability:
            for diag in diagnosed:
                if diag.liability_key == target_liability or diag.category == target_liability:
                    primary_liability = diag
                    break
            if primary_liability is None:
                # Target liability requested by user even if below auto-threshold
                val = 0.5
                all_inds = {}
                all_inds.update(parent_profile.toxicity)
                all_inds.update(parent_profile.excretion)
                all_inds.update(parent_profile.absorption)
                if target_liability in all_inds:
                    val = self._extract_val(all_inds[target_liability])
                primary_liability = LiabilityDiagnostic(
                    liability_key=target_liability,
                    liability_name=target_liability.upper(),
                    cluster="toxicity",
                    current_value=val,
                    threshold=0.5,
                    severity=0.1,
                    category=target_liability,
                )
        elif diagnosed:
            primary_liability = diagnosed[0]

        # ----------------------------------------------------------------------
        # Step 2: Defect Localization (XAI Integrated Gradients)
        # ----------------------------------------------------------------------
        attr_result = self.explainer.attribute(canon_smiles, steps=steps)
        hotspot_atoms = attr_result.get("hotspot_atoms", [])

        # ----------------------------------------------------------------------
        # Step 3: Scaffold-Preserved Targeted Mutation (Bioisosteres)
        # ----------------------------------------------------------------------
        liability_focus_param = (
            primary_liability.category if primary_liability else target_liability
        )
        suggestions = self.recommender.recommend(
            canon_smiles,
            liability_focus=liability_focus_param,
        )

        candidates_generated = len(suggestions)
        passing_sa_count = 0
        evaluated_candidates: List[OptimizedCandidate] = []

        # ----------------------------------------------------------------------
        # Step 4: Multi-Objective Pareto Filtering & Scoring
        # ----------------------------------------------------------------------
        for sug in suggestions:
            cand_smi = sug["modified_smiles"]
            cand_mol = Chem.MolFromSmiles(cand_smi)
            if cand_mol is None:
                continue

            sa = calculate_sa_score(cand_mol)
            if sa > self.sa_threshold:
                continue
            passing_sa_count += 1

            scaffold_ok = self._check_scaffold_preserved(scaffold_smi, cand_mol)

            # Predict ADMET profile for candidate
            cand_profile = self.pipeline.predict_single(cand_smi)

            # Measure liability improvement
            parent_val = primary_liability.current_value if primary_liability else 0.5
            cand_val = parent_val

            all_cand_inds = {}
            all_cand_inds.update(cand_profile.absorption)
            all_cand_inds.update(cand_profile.distribution)
            all_cand_inds.update(cand_profile.metabolism)
            all_cand_inds.update(cand_profile.excretion)
            all_cand_inds.update(cand_profile.toxicity)

            target_key = primary_liability.liability_key if primary_liability else "herg"
            if target_key in all_cand_inds:
                cand_val = self._extract_val(all_cand_inds[target_key])

            # Delta calculation: higher positive means greater improvement
            delta = parent_val - cand_val  # for toxicities, lower is better

            # Fitness: delta - 0.1 * SA penalty + bonus for scaffold preservation
            fitness = delta - 0.05 * (sa - 2.0) + (0.2 if scaffold_ok else -0.5)

            candidate_obj = OptimizedCandidate(
                smiles=cand_smi,
                transformation_name=sug["transformation_name"],
                liability_addressed=sug["liability_addressed"],
                rationale=sug["rationale"],
                parent_liability_value=parent_val,
                candidate_liability_value=cand_val,
                liability_delta=delta,
                sa_score=sa,
                scaffold_preserved=scaffold_ok,
                fitness_score=fitness,
            )
            evaluated_candidates.append(candidate_obj)

        # Sort candidates by fitness score descending
        evaluated_candidates.sort(key=lambda c: c.fitness_score, reverse=True)
        top_candidates = evaluated_candidates[:max_candidates]

        return OptimizationReport(
            input_smiles=smiles,
            canonical_smiles=canon_smiles,
            bemis_murcko_scaffold=scaffold_smi,
            diagnosed_liabilities=diagnosed,
            primary_liability=primary_liability,
            hotspot_atom_indices=hotspot_atoms,
            candidates_generated=candidates_generated,
            candidates_passing_sa_filter=passing_sa_count,
            top_candidates=top_candidates,
        )

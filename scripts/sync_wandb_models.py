"""W&B Model Registry & Artifacts Synchronization Utility for TDC-Studio.

Synchronizes trained SOTA checkpoints and metadata from Weights & Biases:
- Clearance MTL (Cluster 4: Microsome CL, Hepatocyte CL, Half-life)
- CYP450 Two-Stage (Cluster 3: 5 Inhibitors + 3 Substrates)
- PPBR / VDss Tri-Hybrid Stackers (Cluster 2)
- Safety & Toxicity MTL (Cluster 5: hERG, DILI, AMES)
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("sync_wandb_models")


def sync_artifacts(
    project_path: str = "tdc-studio/tdc-learning",
    output_dir: str = "models/export",
    download: bool = True,
) -> Dict[str, Any]:
    """Sync model artifacts and benchmark metrics from W&B project."""
    import wandb

    api = wandb.Api()
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    manifest_path = out_path / "export_manifest.json"
    manifest: Dict[str, Any] = {}
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception:
            manifest = {}

    runs = api.runs(project_path)
    logger.info("Found %d runs in project '%s'.", len(runs), project_path)

    cluster_models: Dict[str, Dict[str, Any]] = {}

    for run in runs:
        run_name = run.name or ""
        run_id = run.id
        summary = run.summary._json_dict

        # Identify run cluster
        cluster = None
        if "caco2" in run_name.lower() or "absorption" in run_name.lower():
            cluster = "cluster_1_absorption"
        elif "clearance" in run_name.lower():
            cluster = "cluster_4_clearance"
        elif "cyp450" in run_name.lower():
            cluster = "cluster_3_cyp450"
        elif (
            "vdss" in run_name.lower()
            or "distribution" in run_name.lower()
            or "ppbr" in run_name.lower()
        ):
            cluster = "cluster_2_distribution"
        elif "toxicity" in run_name.lower() or "herg" in run_name.lower():
            cluster = "cluster_5_safety"

        if not cluster:
            continue

        artifacts = list(run.logged_artifacts())
        logger.info(
            "Run '%s' (%s) -> Cluster: %s, Artifacts: %d", run_name, run_id, cluster, len(artifacts)
        )

        if cluster not in cluster_models or run.created_at > cluster_models[cluster].get(
            "created_at", ""
        ):
            cluster_models[cluster] = {
                "run_id": run_id,
                "run_name": run_name,
                "created_at": run.created_at,
                "summary": summary,
                "artifacts": [a.name for a in artifacts],
            }

            if download and artifacts:
                latest_art = artifacts[-1]
                target_art_dir = out_path / cluster
                target_art_dir.mkdir(parents=True, exist_ok=True)
                try:
                    latest_art.download(root=str(target_art_dir))
                    logger.info(
                        "Downloaded artifact '%s' to '%s'.", latest_art.name, target_art_dir
                    )
                except Exception as e:
                    logger.warning("Could not download artifact '%s': %s", latest_art.name, e)

    manifest["synced_clusters"] = cluster_models
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Updated export manifest saved to '%s'.", manifest_path)
    return manifest


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sync W&B model artifacts for TDC-Studio.")
    parser.add_argument("--project", default="tdc-studio/tdc-learning", help="W&B project path")
    parser.add_argument("--output-dir", default="models/export", help="Output export directory")
    parser.add_argument("--no-download", action="store_true", help="Only sync metadata manifest")
    args = parser.parse_args()

    sync_artifacts(
        project_path=args.project,
        output_dir=args.output_dir,
        download=not args.no_download,
    )

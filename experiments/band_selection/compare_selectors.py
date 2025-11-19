# experiments/band_selection/compare_selectors.py

import os
import json
import numpy as np
import pandas as pd

from typing import Dict, List


# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------

def load_bands(path: str) -> List[int]:
    return np.load(path).astype(int).tolist()


def load_metrics(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)["mean_metrics"]


def jaccard(set1: set, set2: set) -> float:
    return len(set1 & set2) / len(set1 | set2)


def hamming(set1: set, set2: set, total_bands: int) -> float:
    """Normalised Hamming distance between two band-selection masks."""
    mask1 = np.zeros(total_bands, dtype=int)
    mask2 = np.zeros(total_bands, dtype=int)

    for b in set1:
        mask1[b] = 1
    for b in set2:
        mask2[b] = 1

    return np.mean(mask1 != mask2)


# ----------------------------------------------------------------------
# Main Comparison
# ----------------------------------------------------------------------

def compare_selectors(
    dataset_name: str,
    total_bands: int,
    results_root: str = "results/",
    out_dir: str = "results/selector_comparison/",
):
    """
    Compare all band-selection methods across both models:
    
        1. Hyper3DNet-Lite:
            - lite_ibra_gss
            - lite_ssep
            - lite_srpa

        2. Conv3D Full:
            - 3dcnn_ibra_gss
            - 3dcnn_ssep
            - 3dcnn_srpa

    Loads their band lists & CV performance metrics, computes:
        - intersections
        - Jaccard similarity
        - Hamming distance
        - OA, F1, Kappa comparison table
    """

    os.makedirs(out_dir, exist_ok=True)

    # ==================================================================
    # 1. Locate files for all 6 selectors
    # ==================================================================

    selector_defs = {
        # HYPER3DNET-LITE
        "lite_ibra_gss": {
            "dir": "hyper3dnetlite",
            "pattern": f"{dataset_name}_lite_ibra_gss_",
        },
        "lite_ssep": {
            "dir": "hyper3dnetlite",
            "pattern": f"{dataset_name}_lite_ssep_",
        },
        "lite_srpa": {
            "dir": "hyper3dnetlite",
            "pattern": f"{dataset_name}_lite_srpa_",
        },

        # 3D CNN
        "3dcnn_ibra_gss": {
            "dir": "conv3d_full",
            "pattern": f"{dataset_name}_3dcnn_ibra_gss_",
        },
        "3dcnn_ssep": {
            "dir": "conv3d_full",
            "pattern": f"{dataset_name}_3dcnn_ssep_",
        },
        "3dcnn_srpa": {
            "dir": "conv3d_full",
            "pattern": f"{dataset_name}_3dcnn_srpa_",
        },
    }

    band_paths = {}
    metric_paths = {}

    for name, spec in selector_defs.items():
        folder = os.path.join(results_root, spec["dir"])
        pattern = spec["pattern"]

        band_paths[name] = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.startswith(pattern) and f.endswith("bands.npy")
        ][0]

        metric_paths[name] = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.startswith(pattern) and f.endswith("cv.json")
        ][0]

    # ==================================================================
    # 2. Load band lists & metrics
    # ==================================================================
    band_sets = {name: set(load_bands(path)) for name, path in band_paths.items()}
    mean_metrics = {name: load_metrics(path) for name, path in metric_paths.items()}

    # Save raw band dictionary
    with open(os.path.join(out_dir, "bands.json"), "w") as f:
        json.dump({k: sorted(list(v)) for k, v in band_sets.items()}, f, indent=4)

    # ==================================================================
    # 3. Pairwise Overlaps / Jaccard / Hamming
    # ==================================================================
    selectors = list(band_sets.keys())
    overlap_matrix = {}

    for i in range(len(selectors)):
        for j in range(i + 1, len(selectors)):
            A, B = selectors[i], selectors[j]
            sA, sB = band_sets[A], band_sets[B]

            overlap_matrix[f"{A}_vs_{B}"] = {
                "intersection": sorted(list(sA & sB)),
                "num_intersection": len(sA & sB),
                "jaccard": jaccard(sA, sB),
                "hamming": hamming(sA, sB, total_bands),
            }

    # Save overlap JSON
    with open(os.path.join(out_dir, "band_overlap_matrix.json"), "w") as f:
        json.dump(overlap_matrix, f, indent=4)

    # Create overlap table (CSV)
    rows = []
    for k, v in overlap_matrix.items():
        rows.append([k, v["num_intersection"], v["jaccard"], v["hamming"]])

    df_overlap = pd.DataFrame(rows, columns=["pair", "intersection", "jaccard", "hamming"])
    df_overlap.to_csv(os.path.join(out_dir, "band_overlap_matrix.csv"), index=False)

    # ==================================================================
    # 4. Metrics table — OA/F1/Kappa for all selectors
    # ==================================================================
    metrics_df = pd.DataFrame(mean_metrics).T
    metrics_df.to_json(os.path.join(out_dir, "metrics_table.json"), indent=4)
    metrics_df.to_csv(os.path.join(out_dir, "metrics_table.csv"))

    # ==================================================================
    # 5. Summary TXT
    # ==================================================================
    summary_path = os.path.join(out_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"=== BAND SELECTION COMPARISON for {dataset_name} ===\n\n")

        # Print all band lists
        for name, bset in band_sets.items():
            f.write(f"{name}: {sorted(list(bset))}\n")

        f.write("\n--- Pairwise Overlaps ---\n")
        for pair, info in overlap_matrix.items():
            f.write(f"{pair}:\n")
            f.write(f"   intersection: {info['intersection']}\n")
            f.write(f"   count:        {info['num_intersection']}\n")
            f.write(f"   jaccard:      {info['jaccard']:.4f}\n")
            f.write(f"   hamming:      {info['hamming']:.4f}\n\n")

        f.write("\n--- Mean CV Metrics (OA / F1 / Kappa) ---\n")
        f.write(metrics_df.to_string())
        f.write("\n")

    print(f"\nComparison complete!")
    print(f"Saved band sets:           {out_dir}/bands.json")
    print(f"Saved overlap matrix:      {out_dir}/band_overlap_matrix.json")
    print(f"Saved overlap CSV:         {out_dir}/band_overlap_matrix.csv")
    print(f"Saved metrics table:       {out_dir}/metrics_table.csv")
    print(f"Saved summary:             {out_dir}/summary.txt")

    return {
        "bands": band_sets,
        "overlap": overlap_matrix,
        "metrics": mean_metrics,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--total_bands", type=int, required=True,
                        help="Total number of spectral bands in raw cube")
    args = parser.parse_args()

    compare_selectors(
        dataset_name=args.dataset,
        total_bands=args.total_bands,
    )

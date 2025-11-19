# experiments/ssl/compare_ssl_vs_supervised.py

import os
import json
from typing import Optional


def load_json(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        print(f"[WARN] File not found: {path}")
        return None
    with open(path, "r") as f:
        return json.load(f)


def compare_ssl_vs_supervised(
    dataset_name: str,
    num_bands: int = 20,
    results_dir: str = "results/ssl/",
    use_ibra_gss_baseline: bool = False,
):
    """
    Compare:
      - Supervised baseline (full-bands or IBRA+GSS)
      - SSL IBRA+GSS
      - SSL Progressive (best OA over unlabeled fractions)
    """

    if use_ibra_gss_baseline:
        sup_tag = f"{dataset_name}_supervised_lite_ibra_gss_{num_bands}bands.json"
    else:
        sup_tag = f"{dataset_name}_supervised_lite_fullbands.json"

    ssl_tag = f"{dataset_name}_ssl_lite_ibra_gss_{num_bands}bands.json"
    prog_tag = f"{dataset_name}_ssl_progressive_ibra_gss_{num_bands}bands.json"

    sup_path = os.path.join(results_dir, sup_tag)
    ssl_path = os.path.join(results_dir, ssl_tag)
    prog_path = os.path.join(results_dir, prog_tag)

    sup_res = load_json(sup_path)
    ssl_res = load_json(ssl_path)
    prog_res = load_json(prog_path)

    print("\n=== Comparison: Supervised vs SSL ===")
    print(f"Dataset: {dataset_name}")
    print(f"Num bands (IBRA+GSS): {num_bands}")
    print(f"Use IBRA+GSS as supervised baseline: {use_ibra_gss_baseline}")

    sup_OA = sup_res["metrics"]["OA"] if sup_res is not None else None
    ssl_OA = ssl_res["final_OA"] if ssl_res is not None else None

    prog_best_frac = None
    prog_best_OA = None
    if prog_res is not None:
        for frac_str, v in prog_res["results_by_frac"].items():
            oa = v["OA"]
            if (prog_best_OA is None) or (oa > prog_best_OA):
                prog_best_OA = oa
                prog_best_frac = float(frac_str)

    print("\nOA summary:")
    print("  Supervised baseline:", f"{sup_OA:.4f}" if sup_OA is not None else "N/A")
    print("  SSL IBRA+GSS:        ", f"{ssl_OA:.4f}" if ssl_OA is not None else "N/A")
    if prog_best_OA is not None:
        print(f"  SSL Progressive (best): {prog_best_OA:.4f} at unlabeled_frac={prog_best_frac:.2f}")
    else:
        print("  SSL Progressive (best): N/A")

    return {
        "supervised_OA": sup_OA,
        "ssl_OA": ssl_OA,
        "ssl_progressive_best_OA": prog_best_OA,
        "ssl_progressive_best_frac": prog_best_frac,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--num_bands", type=int, default=20)
    parser.add_argument("--results_dir", type=str, default="results/ssl/")
    parser.add_argument("--use_ibra_gss_baseline", action="store_true")

    args = parser.parse_args()

    compare_ssl_vs_supervised(
        dataset_name=args.dataset,
        num_bands=args.num_bands,
        results_dir=args.results_dir,
        use_ibra_gss_baseline=args.use_ibra_gss_baseline,
    )

import os
import json
import numpy as np
import tensorflow as tf

from hyperspectral_insight.data.loaders import load_dataset
from hyperspectral_insight.data.patches import extract_patches
from hyperspectral_insight.data.normalization import minmax_normalize

from hyperspectral_insight.models.hyper3dnet_lite import build_hyper3dnet_lite
from hyperspectral_insight.evaluation.cross_validation import kfold_cross_validation


def run_lite_fullbands_cv(
    dataset_name: str,
    patch_size: int = 25,
    n_splits: int = 10,
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 1e-4,
    save_dir: str = None,
):

    tf.config.optimizer.set_jit(False)

    cube, gt = load_dataset(dataset_name)
    cube_norm = minmax_normalize(cube)
    X, y = extract_patches(cube_norm, gt, patch_size)

    def model_fn(input_shape, n_classes):
        return build_hyper3dnet_lite(input_shape, n_classes, lr=lr)

    tried_batches = []
    bs = batch_size
    results = None

    def try_train(bs):
        try:
            print(f"Trying batch_size = {bs}")
            return kfold_cross_validation(
                model_fn=model_fn,
                X=X,
                y=y,
                n_splits=n_splits,
                epochs=epochs,
                batch_size=bs,
                shuffle=True,
                random_state=0,
                verbose=1,
            )
        except RuntimeError as e:
            msg = str(e).lower()
            if any(k in msg for k in ["out of memory", "oom", "cudnn", "xla"]):
                print(f"Batch size {bs} failed.")
                return None
            raise e

    while bs >= 8:
        tried_batches.append(bs)
        results = try_train(bs)
        if results is not None:
            break
        bs //= 2

    if results is None:
        raise RuntimeError(f"All attempted batch sizes failed: {tried_batches}")

    final_batch_size = bs
    print(f"Final batch size used: {final_batch_size}")

    if save_dir is None:
        save_dir = f"results/hyper3dnetlite/{dataset_name}/"

    os.makedirs(save_dir, exist_ok=True)

    results["final_batch_size"] = final_batch_size
    results["attempted_batch_sizes"] = tried_batches

    json_path = os.path.join(save_dir, f"{dataset_name}_lite_fullbands_cv.json")
    hist_path = os.path.join(save_dir, f"{dataset_name}_h3dnetlite_fullbands_histories.npy")

    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)

    np.save(hist_path, results["histories"], allow_pickle=True)

    print(f"Saved metrics to: {json_path}")
    print(f"Saved histories to: {hist_path}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--patch", type=int, default=25)
    parser.add_argument("--splits", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1e-4)

    args = parser.parse_args()

    run_lite_fullbands_cv(
        dataset_name=args.dataset,
        patch_size=args.patch,
        n_splits=args.splits,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.learning_rate,
    )

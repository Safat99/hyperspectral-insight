from experiments.baseline.run_shallow_cnn import run_shallow_baseline

DATASETS = [
    "indian_pines",
    "salinas",
    "pavia_university",
    "pavia_centre",
    "ksc"
]

def run_all():
    for ds in DATASETS:
        print(f"\n>>> Running dataset: {ds}")
        run_shallow_baseline(
            dataset_name=ds,
            patch_size=25,
            epochs=50,
            batch_size=32,
            lr=0.001,
            save_dir="results/baseline/"
        )
        
if __name__ == "__main__":
    run_all()
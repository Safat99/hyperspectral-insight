from run_shallow_cnn import run_shallow_baseline

DATASETS = [
    "indian_pines",
    "salinas",
    "pavia",
    "ksc"
]

def run_all():
    for ds in DATASETS:
        run_shallow_baseline(ds, patch_size=25)
        
if __name__ == "__main__":
    run_all()
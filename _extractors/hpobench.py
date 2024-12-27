import os
import pickle


# Dependency on hpolib-extractor.


data_file_names = [
    "australian.pkl",
    "blood_transfusion.pkl",
    "car.pkl",
    "credit_g.pkl",
    "kc1.pkl",
    "phoneme.pkl",
    "segment.pkl",
    "vehicle.pkl",
]
n_seeds = 5
dir_path = os.path.join(os.environ["HOME"], "hpo_benchmarks", "hpobench")
for fn in data_file_names:
    data_path = os.path.join(dir_path, fn)
    dataset = pickle.load(open(data_path, "rb"))
    modified_dataset = {}
    print(fn)
    for i, (config_id, results) in enumerate(dataset.items()):
        print(i, config_id)
        modified_dataset[config_id] = {
            "train_time": [float(v[243]) for v in dataset[config_id]["runtime"]],
            "val_acc": [float(v[243]) for v in dataset[config_id]["bal_acc"]],
            "val_precision": [float(v[243]) for v in dataset[config_id]["precision"]],
            "val_f1": [float(v[243]) for v in dataset[config_id]["f1"]],
        }

    with open(f"hpo_benchmarks/datasets/hpobench/{fn}", mode="wb") as f:
        pickle.dump(modified_dataset, f)

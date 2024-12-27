import os
import pickle


# Dependency on hpolib-extractor.


data_file_names = [
    "naval_propulsion.pkl", "parkinsons_telemonitoring.pkl", "protein_structure.pkl", "slice_localization.pkl"
]
n_seeds = 4
dir_path = os.path.join(os.environ["HOME"], "hpo_benchmarks", "hpolib")
for fn in data_file_names:
    data_path = os.path.join(dir_path, fn)
    dataset = pickle.load(open(data_path, "rb"))
    modified_dataset = {}
    print(fn)
    for i, (config_id, results) in enumerate(dataset.items()):
        print(i, config_id)
        modified_dataset[config_id] = {
            "train_time": [float(v[100]) for v in dataset[config_id]["valid_mse"]],
            "val_loss": [float(v) for v in dataset[config_id]["runtime"]],
            "model_size": [float(dataset[config_id]["n_params"])]*n_seeds,
        }
    with open(fn, mode="wb") as f:
        pickle.dump(modified_dataset, f)

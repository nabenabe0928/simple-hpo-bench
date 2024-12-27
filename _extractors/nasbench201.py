from __future__ import annotations

# NATS-tss-v1_0-3ffb9-simple.tar
# https://github.com/D-X-Y/NATS-Bench/blob/main/nats_bench/api_utils.py#L845-L880
import bz2
import pickle


def extract_result(results: dict[str, Any]) -> dict[str, float]:
    max_epoch = 200
    # valid_epoch_key = f"x-valid@{max_epoch-1}"
    valid_epoch_key = f"x-valid@{max_epoch-1}"
    row = {
        "train_time": results["train_times"][max_epoch - 1],
        "train_loss": min(results["train_losses"][e] for e in range(max_epoch)),
        "train_acc": max(results["train_acc1es"][e] for e in range(max_epoch)),
        "val_time": results["eval_times"][valid_epoch_key],
        "val_loss": results["eval_losses"][valid_epoch_key],
        "val_acc": results["eval_acc1es"][valid_epoch_key],
        "latency": results["latency"][0],
    }
    return {k: float(v) for k, v in row.items()}


def convert_arch_str_to_config_id(arch_str: str) -> str:
    choices = ["none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3"]
    operators = [cmp.split("|")[-1] for cmp in arch_str.split("~")[:-1]]
    config_id = "".join([str(choices.index(op)) for op in operators])
    return config_id


final_results = {"imagenet": {}, "cifar10": {}, "cifar100": {}}
dataset_names = {"imagenet": "ImageNet16-120", "cifar10": "cifar10-valid", "cifar100": "cifar100"}
for i in range(5 ** 6):
    data = pickle.load(bz2.open(f"NATS-tss-v1_0-3ffb9-simple/{i:0>6}.pickle.pbz2", mode="rb"))["200"]
    config_id = convert_arch_str_to_config_id(data["arch_str"])
    all_results = data["all_results"]
    print(i, config_id)
    for dataset_name, key in dataset_names.items():
        final_results[dataset_name][config_id] = []
        for seed in [777, 888, 999]:
            results = all_results.get((key, seed), None)
            if results is None:
                continue

            final_results[dataset_name][config_id].append(extract_result(results))

for dataset_name, results in final_results.items():
    with open(f"{dataset_name}.pkl", mode="wb") as f:
        pickle.dump(results, f)

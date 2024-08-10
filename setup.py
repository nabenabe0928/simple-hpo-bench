import os
import setuptools


requirements = []
with open("requirements.txt", "r") as f:
    for line in f:
        requirements.append(line.strip())


dataset_dir = "hpo_benchmarks/datasets/"
dataset_dirs = [os.path.join(dataset_dir, name) for name in os.listdir(dataset_dir)]
dataset_file_names = []
for target in dataset_dirs:
    dataset_file_names.extend(os.listdir(target))

setuptools.setup(
    name="simple-hpo-bench",
    version="0.0.1",
    author="nabenabe0928",
    author_email="shuhei.watanabe.utokyo@gmail.com",
    url="https://github.com/nabenabe0928/simple-hpo-bench",
    packages=["hpo_benchmarks/"] + dataset_dirs,
    package_data={"": dataset_file_names},
    python_requires=">=3.7",
    platforms=["Linux", "Darwin"],
    install_requires=requirements,
    include_package_data=True,
)

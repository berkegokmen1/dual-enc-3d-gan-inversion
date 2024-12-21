from datasets import load_dataset

dataset = load_dataset("onethousand/LPFF", cache_dir="/home/bilecen/3d_inversion/data")

print(dataset)

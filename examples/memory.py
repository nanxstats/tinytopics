import tinytopics as tt

tt.set_random_seed(42)

n, m, k = 100_000, 100_000, 20
X, true_L, true_F = tt.generate_synthetic_data(n, m, k, avg_doc_length=256 * 256)

size_gb = X.nbytes / (1024**3)
print(f"Memory size of X: {size_gb:.2f} GB")

model, losses = tt.fit_model(X, k=k, num_epochs=200)

tt.plot_loss(losses, output_file="loss.png")

import numpy as np

import tinytopics as tt

tt.set_random_seed(42)

n, m, k = 100_000, 100_000, 20
X, true_L, true_F = tt.generate_synthetic_data(n, m, k, avg_doc_length=256 * 256)

size_gb = X.nbytes / (1024**3)
print(f"Memory size of X: {size_gb:.2f} GB")

data_path = "X.npy"
np.save(data_path, X.cpu().numpy())

del X, true_L, true_F

dataset = tt.NumpyDiskDataset(data_path)
model, losses = tt.fit_model(dataset, k=k, num_epochs=100)

tt.plot_loss(losses, output_file="loss.png")

import numpy as np
from tqdm.auto import tqdm

import tinytopics as tt

tt.set_random_seed(42)

n, m, k = 100_000, 100_000, 20
X, true_L, true_F = tt.generate_synthetic_data(n, m, k, avg_doc_length=256 * 256)

init_path = "X.npy"
np.save(init_path, X.cpu().numpy())

size_gb = X.nbytes / (1024**3)
print(f"Memory size of X: {size_gb:.2f} GB")

del X, true_L, true_F

n_large = 500_000
large_path = "X_large.npy"

shape = (n_large, m)
large_size_gb = (shape[0] * shape[1] * 4) / (1024**3)  # 4 bytes per float32
print(f"Expected size: {large_size_gb:.2f} GB")

large_array = np.lib.format.open_memmap(
    large_path,
    mode="w+",
    dtype=np.float32,
    shape=shape,
)

chunk_size = 10_000
n_chunks = n_large // chunk_size

source_data = np.load(init_path, mmap_mode="r")

for i in tqdm(range(n_chunks), desc="Generating chunks"):
    start_idx = i * chunk_size
    end_idx = start_idx + chunk_size
    indices = np.random.randint(0, n, size=chunk_size)
    large_array[start_idx:end_idx] = source_data[indices]

large_array.flush()

dataset = tt.NumpyDiskDataset(large_path)
model, losses = tt.fit_model(dataset, k=k, num_epochs=20)

tt.plot_loss(losses, output_file="loss.png")

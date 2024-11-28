import time

import torch
import pandas as pd
import matplotlib.pyplot as plt
import tinytopics as tt

tt.set_random_seed(42)

n_values = [1000, 5000]  # Number of documents
m_values = [1000, 5000, 10000, 20000]  # Vocabulary size
k_values = [10, 50, 100]  # Number of topics
avg_doc_length = 256 * 256

benchmark_results = pd.DataFrame()


def benchmark(X, k, device):
    start_time = time.time()
    model, losses = tt.fit_model(X, k, device=device)
    elapsed_time = time.time() - start_time

    return elapsed_time


for n in n_values:
    for m in m_values:
        for k in k_values:
            print(f"Benchmarking for n={n}, m={m}, k={k}...")

            X, true_L, true_F = tt.generate_synthetic_data(
                n, m, k, avg_doc_length=avg_doc_length
            )

            # Benchmark on CPU
            cpu_time = benchmark(X, k, torch.device("cpu"))
            cpu_result = pd.DataFrame(
                [{"n": n, "m": m, "k": k, "device": "CPU", "time": cpu_time}]
            )

            if not cpu_result.isna().all().any():
                benchmark_results = pd.concat(
                    [benchmark_results, cpu_result], ignore_index=True
                )

            # Benchmark on GPU if available
            if torch.cuda.is_available():
                gpu_time = benchmark(X, k, torch.device("cuda"))
                gpu_result = pd.DataFrame(
                    [{"n": n, "m": m, "k": k, "device": "GPU", "time": gpu_time}]
                )

                if not gpu_result.isna().all().any():
                    benchmark_results = pd.concat(
                        [benchmark_results, gpu_result], ignore_index=True
                    )

benchmark_results.to_csv("benchmark-results.csv", index=False)

unique_series = len(n_values) * (2 if torch.cuda.is_available() else 1)
colormap = tt.scale_color_tinytopics(unique_series)
colors_list = [colormap(i) for i in range(unique_series)]

for k in k_values:
    plt.figure(figsize=(7, 4.3), dpi=300)

    color_idx = 0
    for n in n_values:
        subset = benchmark_results[
            (benchmark_results["n"] == n) & (benchmark_results["k"] == k)
        ]

        # Plot CPU results with a specific color
        plt.plot(
            subset[subset["device"] == "CPU"]["m"],
            subset[subset["device"] == "CPU"]["time"],
            label=f"CPU (n={n})",
            linestyle="--",
            marker="o",
            color=colors_list[color_idx],
        )
        color_idx += 1

        # Plot GPU results if available
        if torch.cuda.is_available():
            plt.plot(
                subset[subset["device"] == "GPU"]["m"],
                subset[subset["device"] == "GPU"]["time"],
                label=f"GPU (n={n})",
                linestyle="-",
                marker="x",
                color=colors_list[color_idx],
            )
            color_idx += 1

    plt.xlabel("Vocabulary size (m)")
    plt.ylabel("Training time (seconds)")
    plt.title(f"Training time vs. vocabulary size (k={k})")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"training-time-k-{k}.png", dpi=300)
    plt.close()

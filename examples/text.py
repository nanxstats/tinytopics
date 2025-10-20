from safetensors.numpy import load_file as load_safetensors_numpy
from safetensors.torch import load_file as load_safetensors_torch

import tinytopics as tt


def read_safetensors_numpy(file_path):
    tensors = load_safetensors_numpy(file_path)
    first_key = next(iter(tensors))
    return tensors[first_key]


def read_safetensors_torch(file_path):
    tensors = load_safetensors_torch(file_path)
    first_key = next(iter(tensors))
    return tensors[first_key]


X = read_safetensors_torch("counts.safetensors")

with open("terms.txt") as file:
    terms = [line.strip() for line in file]

tt.set_random_seed(42)

k = 10
model, losses = tt.fit_model(X, k)
tt.plot_loss(losses, output_file="loss.png")

L_tt = model.get_normalized_L().numpy()
F_tt = model.get_normalized_F().numpy()

L_ft = read_safetensors_numpy("L_fastTopics.safetensors")
F_ft = read_safetensors_numpy("F_fastTopics.safetensors")

aligned_indices = tt.align_topics(F_ft, F_tt)
F_aligned_tt = F_tt[aligned_indices]
L_aligned_tt = L_tt[:, aligned_indices]

sorted_indices_ft = tt.sort_documents(L_ft)
L_sorted_ft = L_ft[sorted_indices_ft]
sorted_indices_tt = tt.sort_documents(L_aligned_tt)
L_sorted_tt = L_aligned_tt[sorted_indices_tt]

tt.plot_structure(
    L_sorted_ft,
    title="fastTopics document-topic distributions (sorted)",
    output_file="L-fastTopics.png",
)

tt.plot_structure(
    L_sorted_tt,
    title="tinytopics document-topic distributions (sorted and aligned)",
    output_file="L-tinytopics.png",
)

tt.plot_top_terms(
    F_ft,
    n_top_terms=15,
    term_names=terms,
    title="fastTopics top terms per topic",
    output_file="F-top-terms-fastTopics.png",
)

tt.plot_top_terms(
    F_aligned_tt,
    n_top_terms=15,
    term_names=terms,
    title="tinytopics top terms per topic (aligned)",
    output_file="F-top-terms-tinytopics.png",
)

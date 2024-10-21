from tinytopics.fit import fit_model
from tinytopics.plot import plot_loss, plot_structure, plot_top_terms
from tinytopics.utils import (
    set_random_seed,
    generate_synthetic_data,
    align_topics,
    sort_documents,
)


set_random_seed(42)


n, m, k = 5000, 1000, 10
X, true_L, true_F = generate_synthetic_data(n, m, k, avg_doc_length=256 * 256)


model, losses = fit_model(X, k, learning_rate=0.01)

plot_loss(losses, output_file="loss.png")


learned_L = model.get_normalized_L().numpy()
learned_F = model.get_normalized_F().numpy()


aligned_indices = align_topics(true_F, learned_F)
learned_F_aligned = learned_F[aligned_indices]
learned_L_aligned = learned_L[:, aligned_indices]


sorted_indices = sort_documents(true_L)
true_L_sorted = true_L[sorted_indices]
learned_L_sorted = learned_L_aligned[sorted_indices]


plot_structure(
    true_L_sorted,
    title="True Document-Topic Distributions (Sorted)",
    output_file="L-true.png",
)


plot_structure(
    learned_L_sorted,
    title="Learned Document-Topic Distributions (Sorted and Aligned)",
    output_file="L-learned.png",
)


plot_top_terms(
    true_F,
    n_top_terms=15,
    title="Top Terms per Topic - True F Matrix",
    output_file="F-top-terms-true.png",
)


plot_top_terms(
    learned_F_aligned,
    n_top_terms=15,
    title="Top Terms per Topic - Learned F Matrix (Aligned)",
    output_file="F-top-terms-learned.png",
)

#!/usr/bin/env python
# coding: utf-8

# <!-- `.md` and `.py` files are generated from the `.qmd` file. Please edit that file. -->

# ---
# title: "Get started"
# format: gfm
# eval: false
# ---
#
# !!! tip
#
#     To run the code from this article as a Python script:
#
#     ```bash
#     python3 examples/get-started.py
#     ```
#
# ## Import stuff

# In[ ]:


from tinytopics.fit import fit_model
from tinytopics.plot import plot_loss, plot_structure, plot_top_terms
from tinytopics.utils import (
    set_random_seed,
    generate_synthetic_data,
    align_topics,
    sort_documents,
)


# Set seed for reproducibility

# In[ ]:


set_random_seed(42)


# Generate synthetic data

# In[ ]:


n, m, k = 5000, 1000, 10
X, true_L, true_F = generate_synthetic_data(n, m, k, avg_doc_length=256 * 256)


# ## Training
#
# Train the model

# In[ ]:


model, losses = fit_model(X, k)


# Plot loss curve

# In[ ]:


plot_loss(losses, output_file="loss.png")


# ## Post-process results
#
# Derive matrices

# In[ ]:


learned_L = model.get_normalized_L().numpy()
learned_F = model.get_normalized_F().numpy()


# Align topics

# In[ ]:


aligned_indices = align_topics(true_F, learned_F)
learned_F_aligned = learned_F[aligned_indices]
learned_L_aligned = learned_L[:, aligned_indices]


# Sort documents

# In[ ]:


sorted_indices = sort_documents(true_L)
true_L_sorted = true_L[sorted_indices]
learned_L_sorted = learned_L_aligned[sorted_indices]


# ## Visualize results
#
# STRUCTURE plot

# In[ ]:


plot_structure(
    true_L_sorted,
    title="True Document-Topic Distributions (Sorted)",
    output_file="L_true.png",
)
plot_structure(
    learned_L_sorted,
    title="Learned Document-Topic Distributions (Sorted and Aligned)",
    output_file="L_learned_aligned.png",
)


# Top terms plot

# In[ ]:


plot_top_terms(
    true_F,
    n_top_terms=15,
    title="Top Terms per Topic - True F Matrix",
    output_file="F_top_terms_true.png",
)
plot_top_terms(
    learned_F_aligned,
    n_top_terms=15,
    title="Top Terms per Topic - Learned F Matrix (Aligned)",
    output_file="F_top_terms_learned_aligned.png",
)
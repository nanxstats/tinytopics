curl::curl_download(
  "https://github.com/stephenslab/fastTopics-experiments/raw/refs/heads/main/data/nips.RData",
  destfile = "nips.rda"
)

load("nips.rda")
counts_dense <- as.matrix(counts)
counts_tensor <- torch::torch_tensor(counts_dense, dtype = torch::torch_float())
safetensors::safe_save_file(list(counts = counts_tensor), "counts.safetensors")

writeLines(colnames(counts), con = "terms.txt")

set.seed(42)
fit <- fastTopics::fit_topic_model(counts, k = 10)

L_tensor <- torch::torch_tensor(as.matrix(fit$L), dtype = torch::torch_float())
safetensors::safe_save_file(list(L = L_tensor), "L_fastTopics.safetensors")

F_tensor <- torch::torch_tensor(t(as.matrix(fit$F)), dtype = torch::torch_float())
safetensors::safe_save_file(list(F = F_tensor), "F_fastTopics.safetensors")

curl::curl_download(
  "https://github.com/stephenslab/fastTopics-experiments/raw/refs/heads/main/data/nips.RData",
  destfile = "nips.rda"
)

load("nips.rda")
counts_dense <- as.matrix(counts)
saveRDS(counts_dense, file = "counts.rds")

writeLines(colnames(counts), con = "terms.txt")

set.seed(42)
fit <- fastTopics::fit_topic_model(counts, k = 10)

saveRDS(fit$L, file = "L_fastTopics.rds")
saveRDS(t(fit$F), file = "F_fastTopics.rds")

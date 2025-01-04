import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class NeuralPoissonNMF(nn.Module):
    def __init__(
        self,
        n: int,
        m: int,
        k: int,
        device: torch.device | None = None,
        eps: float = 1e-10,
    ):
        """
        Poisson NMF model with nonnegative factors (via softplus),
        but *no* row normalization in the forward pass.
        The normalization is done *post hoc*, only when calling
        get_normalized_L() or get_normalized_F().

        Args:
            n: Number of documents.
            m: Number of terms (vocabulary size).
            k: Number of topics.
            device: Device to run the model on. Defaults to CPU.
            eps: Small constant for numerical stability.
        """
        super(NeuralPoissonNMF, self).__init__()

        self.device: torch.device = device or torch.device("cpu")
        self.eps = eps

        # Raw embeddings for documents
        self.L_raw: nn.Embedding = nn.Embedding(n, k).to(self.device)
        # Initialize L with near-zero values
        nn.init.uniform_(self.L_raw.weight, a=-0.1, b=0.1)

        # Define F as a parameter and initialize with near-zero values
        self.F_raw: nn.Parameter = nn.Parameter(torch.empty(k, m, device=self.device))
        nn.init.uniform_(self.F_raw, a=-0.1, b=0.1)

    def forward(self, doc_indices: Tensor) -> Tensor:
        """
        Forward pass:
        1. Look up document-topic embeddings -> L_raw(doc_indices).
        2. softplus -> ensures nonnegativity but no normalization.
        3. Same for topic-term embeddings -> F_raw.
        4. Multiply to get reconstruction.

        Args:
            doc_indices: Indices of documents in the batch.

        Returns:
            Reconstructed document-term matrix for the batch.
        """
        # Get the L vectors for the batch
        L_batch_raw: Tensor = self.L_raw(doc_indices)
        L_pos: Tensor = F.softplus(L_batch_raw)

        # Get the F vectors for the batch
        F_pos: Tensor = F.softplus(self.F_raw)

        # Return the matrix product to approximate X_batch
        return torch.matmul(L_pos, F_pos)

    def get_normalized_L(self) -> Tensor:
        """
        Get the learned, *post hoc* normalized document-topic distribution matrix (L).

        Returns:
            Posthoc normalized L matrix on the CPU.
        """
        with torch.no_grad():
            L_pos = F.softplus(self.L_raw.weight)
            row_sums = L_pos.sum(dim=1, keepdim=True) + self.eps
            return (L_pos / row_sums).cpu()

    def get_normalized_F(self) -> Tensor:
        """
        Get the learned, *post hoc* normalized topic-term distribution matrix (F).

        Returns:
            Posthoc normalized F matrix on the CPU.
        """
        with torch.no_grad():
            F_pos = F.softplus(self.F_raw)
            row_sums = F_pos.sum(dim=1, keepdim=True) + self.eps
            return (F_pos / row_sums).cpu()

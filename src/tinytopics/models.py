import torch
import torch.nn as nn


class NeuralPoissonNMF(nn.Module):
    def __init__(self, n, m, k, device=None):
        """
        Neural Poisson NMF model with sum-to-one constraints on document-topic and topic-term distributions.

        Args:
            n (int): Number of documents.
            m (int): Number of terms (vocabulary size).
            k (int): Number of topics.
            device (torch.device, optional): Device to run the model on. Defaults to CPU.
        """
        super(NeuralPoissonNMF, self).__init__()

        self.device = device or torch.device("cpu")

        # Use embedding for L to handle batches efficiently
        self.L = nn.Embedding(n, k).to(self.device)

        # Initialize L with small positive values
        nn.init.uniform_(self.L.weight, a=0.0, b=0.1)

        # Define F as a parameter and initialize with small positive values
        self.F = nn.Parameter(torch.empty(k, m, device=self.device))
        nn.init.uniform_(self.F, a=0.0, b=0.1)

    def forward(self, doc_indices):
        """
        Forward pass of the neural Poisson NMF model.

        Args:
            doc_indices (torch.Tensor): Indices of documents in the batch.

        Returns:
            (torch.Tensor): Reconstructed document-term matrix for the batch.
        """
        # Get the L vectors for the batch
        L_batch = self.L(doc_indices)

        # Sum-to-one constraints across topics for each document
        L_normalized = torch.softmax(L_batch, dim=1)
        # Sum-to-one constraints across terms for each topic
        F_normalized = torch.softmax(self.F, dim=1)

        # Return the matrix product to approximate X_batch
        return torch.matmul(L_normalized, F_normalized)

    def get_normalized_L(self):
        """
        Get the learned, normalized document-topic distribution matrix (L).

        Returns:
            (torch.Tensor): Normalized L matrix on the CPU.
        """
        with torch.no_grad():
            return torch.softmax(self.L.weight, dim=1).cpu()

    def get_normalized_F(self):
        """
        Get the learned, normalized topic-term distribution matrix (F).

        Returns:
            (torch.Tensor): Normalized F matrix on the CPU.
        """
        with torch.no_grad():
            return torch.softmax(self.F, dim=1).cpu()

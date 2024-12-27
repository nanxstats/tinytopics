import torch

from tinytopics.loss import poisson_nmf_loss


def test_poisson_nmf_loss():
    """Test the Poisson NMF loss function."""
    X = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    X_reconstructed = torch.tensor([[1.1, 1.9], [2.9, 4.1]])

    loss = poisson_nmf_loss(X, X_reconstructed)

    # Test with perfect reconstruction
    perfect_loss = poisson_nmf_loss(X, X)

    # Perfect reconstruction should have lower loss
    assert perfect_loss < loss

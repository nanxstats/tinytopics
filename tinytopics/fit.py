import torch
from tqdm import tqdm
from .models import NeuralPoissonNMF


def fit_model(X, k, num_epochs=200, batch_size=64, learning_rate=0.001, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = X.to(device)
    n, m = X.shape

    model = NeuralPoissonNMF(n, m, k, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=5, factor=0.5
    )

    num_batches = n // batch_size
    losses = []

    with tqdm(total=num_epochs, desc="Training Progress") as pbar:
        for epoch in range(num_epochs):
            permutation = torch.randperm(n, device=device)
            epoch_loss = 0.0

            for i in range(num_batches):
                indices = permutation[i * batch_size : (i + 1) * batch_size]
                batch_X = X[indices, :]

                optimizer.zero_grad()
                X_reconstructed = model(indices)
                loss = poisson_nmf_loss(batch_X, X_reconstructed)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            epoch_loss /= num_batches
            losses.append(epoch_loss)
            scheduler.step(epoch_loss)
            pbar.set_postfix({"Loss": f"{epoch_loss:.4f}"})
            pbar.update(1)

    return model, losses


def poisson_nmf_loss(X, X_reconstructed):
    """
    Poisson NMF negative log-likelihood loss function.

    Args:
        X (torch.Tensor): Original data matrix.
        X_reconstructed (torch.Tensor): Reconstructed data matrix from the model.
    """
    return (X_reconstructed - X * torch.log(X_reconstructed + 1e-10)).sum()

import torch
from tqdm import tqdm
from .models import NeuralPoissonNMF


def fit_model(X, k, learning_rate=0.001, num_epochs=200, batch_size=64, device=None):
    """
    Fit topic model via sum-to-one constrained neural Poisson NMF using batch gradient descent.

    Args:
        X (torch.Tensor): Document-term matrix.
        k (int): Number of topics.
        learning_rate (float, optional): Learning rate for Adam optimizer. Default is 0.001.
        num_epochs (int, optional): Number of training epochs. Default is 200.
        batch_size (int, optional): Number of documents per batch. Default is 64.
        device (torch.device, optional): Device to run the training on. Defaults to CUDA if available, otherwise CPU.

    Returns:
        (NeuralPoissonNMF): Trained model.
        (list): List of training losses for each epoch.
    """
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
    Compute the Poisson NMF loss function (negative log-likelihood).

    Args:
        X (torch.Tensor): Original document-term matrix.
        X_reconstructed (torch.Tensor): Reconstructed matrix from the model.
    """
    epsilon = 1e-10
    return (
        X_reconstructed - X * torch.log(torch.clamp(X_reconstructed, min=epsilon))
    ).sum()

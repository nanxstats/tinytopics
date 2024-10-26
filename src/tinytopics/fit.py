import torch
from tqdm import tqdm
from .models import NeuralPoissonNMF


def fit_model(
    X,
    k,
    num_epochs=200,
    batch_size=16,
    base_lr=0.01,
    max_lr=0.05,
    T_0=20,
    T_mult=1,
    weight_decay=1e-5,
    device=None,
):
    """
    Fit topic model using sum-to-one constrained neural Poisson NMF,
    optimized with AdamW and a cosine annealing with warm restarts scheduler.

    Args:
        X (torch.Tensor): Document-term matrix.
        k (int): Number of topics.
        num_epochs (int, optional): Number of training epochs. Default is 200.
        batch_size (int, optional): Number of documents per batch. Default is 16.
        base_lr (float, optional): Minimum learning rate after annealing. Default is 0.01.
        max_lr (float, optional): Starting maximum learning rate. Default is 0.05.
        T_0 (int, optional): Number of epochs until the first restart. Default is 20.
        T_mult (int, optional): Factor by which the restart interval increases after each restart. Default is 1.
        weight_decay (float, optional): Weight decay for the AdamW optimizer. Default is 1e-5.
        device (torch.device, optional): Device to run the training on. Defaults to CUDA if available, otherwise CPU.

    Returns:
        (NeuralPoissonNMF): Trained model.
        (list): List of training losses for each epoch.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = X.to(device)
    n, m = X.shape

    model = NeuralPoissonNMF(n, m, k, device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=max_lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=T_0, T_mult=T_mult, eta_min=base_lr
    )

    losses = []

    with tqdm(total=num_epochs, desc="Training Progress") as pbar:
        for epoch in range(num_epochs):
            permutation = torch.randperm(n, device=device)
            epoch_loss = 0.0
            num_batches = n // batch_size

            for i in range(num_batches):
                indices = permutation[i * batch_size : (i + 1) * batch_size]
                batch_X = X[indices, :]

                optimizer.zero_grad()
                X_reconstructed = model(indices)
                loss = poisson_nmf_loss(batch_X, X_reconstructed)
                loss.backward()

                optimizer.step()
                # Update per batch for cosine annealing with restarts
                scheduler.step(epoch + i / num_batches)

                epoch_loss += loss.item()

            epoch_loss /= num_batches
            losses.append(epoch_loss)
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

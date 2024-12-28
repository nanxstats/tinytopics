from typing import Tuple
from collections.abc import Sequence

import torch
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, Dataset
from accelerate import Accelerator
from tqdm.auto import tqdm

from .data import IndexTrackingDataset
from .models import NeuralPoissonNMF
from .loss import poisson_nmf_loss


def fit_model_distributed(
    X: Tensor | Dataset,
    k: int,
    num_epochs: int = 200,
    batch_size: int = 16,
    base_lr: float = 0.01,
    max_lr: float = 0.05,
    T_0: int = 20,
    T_mult: int = 1,
    weight_decay: float = 1e-5,
    save_path: str | None = "model.pt",
) -> Tuple[NeuralPoissonNMF, Sequence[float]]:
    """
    Fit topic model using sum-to-one constrained neural Poisson NMF with
    distributed training. Supports multi-GPU, multiple node setups via
    Hugging Face Accelerate.

    Args:
        X: Input data, can be:

            - `torch.Tensor`: In-memory document-term matrix.
            - `Dataset`: Custom dataset implementation.
              For example, see `NumpyDiskDataset`.

        k: Number of topics.
        num_epochs: Number of training epochs.
        batch_size: Batch size.
        base_lr: Minimum learning rate after annealing.
        max_lr: Starting maximum learning rate.
        T_0: Cosine annealing param (epochs until first restart).
        T_mult: Cosine annealing param (factor for each restart).
        weight_decay: Weight decay for AdamW optimizer.
        save_path: File path to save the trained model.
            If None, the model will not be saved to disk.

    Returns:
        Tuple containing:

            - Trained NeuralPoissonNMF model.
            - List of training losses per epoch.
    """
    # Initialize Accelerator
    accelerator = Accelerator()

    # Handle different input types
    if isinstance(X, Dataset):
        base_dataset = X
        n = len(X)
        m = X.num_terms if hasattr(X, "num_terms") else X[0].shape[0]
    else:  # torch.Tensor
        # Do NOT X.to(device) manually here as Accelerate will do it later
        n, m = X.shape
        base_dataset = X  # Pass tensor directly

    # Wrap dataset to track indices
    # Use standard DataLoader as Accelerate will handle multi-GPU distribution
    dataset = IndexTrackingDataset(base_dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Let Accelerate handle .to(device)
    model = NeuralPoissonNMF(n=n, m=m, k=k, device=None)
    optimizer = AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=T_0, T_mult=T_mult, eta_min=base_lr
    )

    # Prepare with accelerator. This will:
    # - Move model, optimizer, and dataloader onto the right devices.
    # - Handle gradient synchronization, etc.
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )

    losses: Sequence[float] = []
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        # Show only one tqdm progress bar on the local main process
        loop = tqdm(
            dataloader,
            desc=f"Epoch {epoch+1}/{num_epochs}",
            disable=not accelerator.is_local_main_process,
        )

        for batch_X, batch_indices in loop:
            num_batches += 1

            # batch_X, batch_indices are already on the right device
            optimizer.zero_grad()

            # Forward pass
            X_reconstructed = model(batch_indices)
            loss = poisson_nmf_loss(batch_X, X_reconstructed)

            # Backprop
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step(epoch + (num_batches / len(dataloader)))

            epoch_loss += loss.item()

            # Update progress bar for the main process
            loop.set_postfix({"loss": f"{loss.item():.4f}"})

        epoch_loss /= num_batches
        losses.append(epoch_loss)

        # Only print on the main process
        if accelerator.is_main_process:
            print(f"Epoch {epoch+1}, Loss = {epoch_loss:.4f}")

    # Final model state is already synchronized across all processes.
    # Accelerate says all GPUs should have the same weights.

    # Unwrap and save the model only from rank 0 when save_path is specified
    if accelerator.is_main_process and save_path is not None:
        torch.save(accelerator.unwrap_model(model).state_dict(), save_path)

    return accelerator.unwrap_model(model), losses

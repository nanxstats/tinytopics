import subprocess
from pathlib import Path

import pytest
import torch

from tinytopics.utils import set_random_seed, generate_synthetic_data

# Test data dimensions
N_DOCS = 100
N_TERMS = 100
N_TOPICS = 5


@pytest.fixture
def sample_data():
    """Fixture providing sample document-term matrix for testing."""
    set_random_seed(42)
    return generate_synthetic_data(n=N_DOCS, m=N_TERMS, k=N_TOPICS)


def run_distributed_training(args):
    """Helper to run distributed training via accelerate launch."""
    cmd = ["accelerate", "launch"]
    script_path = Path(__file__).parent / "train_distributed.py"
    cmd.extend([str(script_path)] + args)

    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
    )
    stdout, stderr = process.communicate()

    assert process.returncode == 0, f"Training failed with error: {stderr}"
    return stdout


def test_fit_model_distributed_basic(sample_data, tmp_path):
    """Test basic distributed model fitting functionality."""
    X, _, _ = sample_data
    num_epochs = 2
    save_path = tmp_path / "model.pt"

    # Save test data
    data_path = tmp_path / "data.pt"
    torch.save(X, data_path)

    args = [
        "--data_path",
        str(data_path),
        "--num_topics",
        str(N_TOPICS),
        "--num_epochs",
        str(num_epochs),
        "--batch_size",
        "8",
        "--save_path",
        str(save_path),
    ]

    stdout = run_distributed_training(args)

    # Check model was saved
    assert save_path.exists()

    # Load and verify the losses
    losses = torch.load(tmp_path / "losses.pt", weights_only=True)
    assert len(losses) == num_epochs
    assert losses[-1] < losses[0]  # Loss decreased


def test_fit_model_distributed_multi_gpu(tmp_path):
    """Test model fitting with multiple GPUs if available."""
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("Test requires at least 2 GPUs")

    set_random_seed(42)
    X, _, _ = generate_synthetic_data(n=N_DOCS, m=N_TERMS, k=N_TOPICS)

    # Save test data
    data_path = tmp_path / "data.pt"
    torch.save(X, data_path)

    args = [
        "--data_path",
        str(data_path),
        "--num_topics",
        "3",
        "--num_epochs",
        "2",
        "--multi_gpu",
    ]

    stdout = run_distributed_training(args)
    assert "Training completed successfully" in stdout


def test_fit_model_distributed_batch_size_handling(sample_data, tmp_path):
    """Test model fitting with different batch sizes."""
    X, _, _ = sample_data
    data_path = tmp_path / "data.pt"
    torch.save(X, data_path)

    # Test with different batch sizes
    for batch_size in [len(X), 4]:
        args = [
            "--data_path",
            str(data_path),
            "--num_topics",
            str(N_TOPICS),
            "--num_epochs",
            "2",
            "--batch_size",
            str(batch_size),
        ]
        stdout = run_distributed_training(args)
        assert "Training completed successfully" in stdout


def test_fit_model_distributed_reproducibility(sample_data, tmp_path):
    """Test that training is reproducible with same seed but different with different seeds."""
    X, _, _ = sample_data
    data_path = tmp_path / "data.pt"
    torch.save(X, data_path)

    save_path_1 = tmp_path / "model_1.pt"
    args = [
        "--data_path",
        str(data_path),
        "--num_topics",
        str(N_TOPICS),
        "--num_epochs",
        "2",
        "--save_path",
        str(save_path_1),
        "--seed",
        "42",
    ]
    run_distributed_training(args)

    save_path_2 = tmp_path / "model_2.pt"
    args = [
        "--data_path",
        str(data_path),
        "--num_topics",
        str(N_TOPICS),
        "--num_epochs",
        "2",
        "--save_path",
        str(save_path_2),
        "--seed",
        "42",
    ]
    run_distributed_training(args)

    save_path_3 = tmp_path / "model_3.pt"
    args = [
        "--data_path",
        str(data_path),
        "--num_topics",
        str(N_TOPICS),
        "--num_epochs",
        "2",
        "--save_path",
        str(save_path_3),
        "--seed",
        "43",
    ]
    run_distributed_training(args)

    # Load losses from all runs
    losses_1 = torch.load(tmp_path / "losses_1.pt", weights_only=True)
    losses_2 = torch.load(tmp_path / "losses_2.pt", weights_only=True)
    losses_3 = torch.load(tmp_path / "losses_3.pt", weights_only=True)

    # Same seed should give identical results
    assert torch.allclose(torch.tensor(losses_1), torch.tensor(losses_2))

    # Different seeds should give different results
    assert not torch.allclose(torch.tensor(losses_1), torch.tensor(losses_3))

import pytest
import torch

from tinytopics.fit import fit_model
from tinytopics.utils import set_random_seed, generate_synthetic_data

# Test data dimensions
N_DOCS = 50
N_TERMS = 100
N_TOPICS = 5


@pytest.fixture
def sample_data():
    """Fixture providing sample document-term matrix for testing."""
    set_random_seed(42)
    return generate_synthetic_data(n=N_DOCS, m=N_TERMS, k=N_TOPICS)


def test_fit_model_basic(sample_data):
    """Test basic model fitting functionality."""
    X, _, _ = sample_data
    num_epochs = 2

    model, losses = fit_model(X=X, k=N_TOPICS, num_epochs=num_epochs, batch_size=8)

    # Check if model was trained
    assert len(losses) == num_epochs
    assert all(isinstance(loss, float) for loss in losses)

    # Check if losses decrease - last loss should be less than first loss
    assert losses[-1] < losses[0]

    # Check model parameters
    assert isinstance(model.get_normalized_L(), torch.Tensor)
    assert isinstance(model.get_normalized_F(), torch.Tensor)


def test_fit_model_device_handling():
    """Test model fitting with different devices."""
    set_random_seed(42)
    data = generate_synthetic_data(n=N_DOCS, m=N_TERMS, k=N_TOPICS)
    X, _, _ = data
    k = 3
    num_epochs = 2

    # Test CPU
    model_cpu, _ = fit_model(
        X=X, k=k, num_epochs=num_epochs, device=torch.device("cpu")
    )
    assert model_cpu.device == torch.device("cpu")

    # Test CUDA if available
    if torch.cuda.is_available():
        model_cuda, _ = fit_model(
            X=X, k=k, num_epochs=num_epochs, device=torch.device("cuda")
        )
        assert model_cuda.device == torch.device("cuda")


def test_fit_model_batch_size_handling(sample_data):
    """Test model fitting with different batch sizes."""
    X, _, _ = sample_data
    num_epochs = 2

    # Test with batch size equal to dataset size
    model1, losses1 = fit_model(
        X=X, k=N_TOPICS, num_epochs=num_epochs, batch_size=len(X)
    )

    # Test with small batch size
    model2, losses2 = fit_model(X=X, k=N_TOPICS, num_epochs=num_epochs, batch_size=4)

    # Both models should produce valid results
    assert len(losses1) == num_epochs
    assert len(losses2) == num_epochs

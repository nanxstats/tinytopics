import pytest
import torch

from tinytopics.models import NeuralPoissonNMF

N_DOCS = 100
N_TERMS = 200
N_TOPICS = 5


@pytest.fixture
def sample_model():
    """Fixture providing a small NeuralPoissonNMF model for testing."""
    return NeuralPoissonNMF(n=N_DOCS, m=N_TERMS, k=N_TOPICS)


def test_model_initialization():
    """Test proper initialization of model parameters."""
    model = NeuralPoissonNMF(n=N_DOCS, m=N_TERMS, k=N_TOPICS)

    # Check dimensions
    assert model.L.weight.shape == (N_DOCS, N_TOPICS)
    assert model.F.shape == (N_TOPICS, N_TERMS)

    # Check if parameters are on correct device
    assert model.L.weight.device == torch.device("cpu")
    assert model.F.device == torch.device("cpu")

    # Check if values are initialized within expected range
    assert torch.all((0.0 <= model.L.weight) & (model.L.weight <= 0.1))
    assert torch.all((0.0 <= model.F) & (model.F <= 0.1))


def test_forward_pass(sample_model):
    """Test the forward pass of the model."""
    batch_size = 4
    doc_indices = torch.tensor([0, 1, 2, 3])

    output = sample_model(doc_indices)

    # Check output dimensions
    assert output.shape == (batch_size, N_TERMS)  # (batch_size, num_terms)

    # Check if output sums approximately to 1 (due to softmax)
    assert torch.allclose(output.sum(dim=1), torch.ones(batch_size), atol=1e-6)

    # Check if all values are positive (probability distribution)
    assert torch.all(output >= 0)


def test_get_normalized_matrices(sample_model):
    """Test retrieval of normalized L and F matrices."""
    L_norm = sample_model.get_normalized_L()
    F_norm = sample_model.get_normalized_F()

    # Check dimensions
    assert L_norm.shape == (N_DOCS, N_TOPICS)  # (n_docs, n_topics)
    assert F_norm.shape == (N_TOPICS, N_TERMS)  # (n_topics, n_terms)

    # Check if matrices sum to 1 along correct dimensions
    assert torch.allclose(L_norm.sum(dim=1), torch.ones(N_DOCS), atol=1e-6)
    assert torch.allclose(F_norm.sum(dim=1), torch.ones(N_TOPICS), atol=1e-6)

    # Check if all values are positive and <= 1
    assert torch.all((0 <= L_norm) & (L_norm <= 1))
    assert torch.all((0 <= F_norm) & (F_norm <= 1))


def test_device_placement():
    """Test if model correctly handles device placement."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda:0")
    model = NeuralPoissonNMF(n=N_DOCS, m=N_TERMS, k=N_TOPICS, device=device)

    assert model.L.weight.device.type == device.type
    assert model.F.device.type == device.type

    # Test forward pass on GPU
    doc_indices = torch.tensor([0, 1, 2], device=device)
    output = model(doc_indices)
    assert output.device == device

import pytest
import torch

from src.model.cnn_model import SimpleCNN


def test_model_batch_size_one():
    model = SimpleCNN(num_classes=17)
    x = torch.randn(1, 3, 128, 128)
    y = model(x)
    assert y.shape == (1, 17)


def test_model_non_square_input():
    model = SimpleCNN(num_classes=17)
    x = torch.randn(2, 3, 128, 96)
    # model uses pooling that should handle non-square,
    # final flatten dims will differ and break fc layer
    with pytest.raises(RuntimeError):
        _ = model(x)


def test_model_wrong_channels():
    model = SimpleCNN(num_classes=17)
    x = torch.randn(2, 1, 128, 128)
    with pytest.raises(RuntimeError):
        _ = model(x)

import pytest  # noqa: F401  # 暂时未使用，但将来可能需要
import torch

from src.model.cnn_model import SimpleCNN


class TestCNNModel:
    """测试CNN模型"""

    def test_model_creation(self):
        """测试模型创建"""
        model = SimpleCNN(num_classes=17)
        assert model is not None

    def test_model_forward_pass(self):
        """测试模型前向传播"""
        model = SimpleCNN(num_classes=17)
        dummy_input = torch.randn(2, 3, 128, 128)
        output = model(dummy_input)
        assert output.shape == (2, 17)
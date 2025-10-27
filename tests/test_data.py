import pytest  # noqa: F401  # 暂时未使用，但将来可能需要

from src.data.data_loader import FlowerDataLoader


class TestDataLoader:
    """测试数据加载器"""

    def test_config_loading(self):
        """测试配置加载"""
        from src.utils.config import load_config

        config = load_config()
        assert "DATA_DIR" in config
        assert "BATCH_SIZE" in config

    def test_data_loader_initialization(self):
        """测试数据加载器初始化"""
        data_loader = FlowerDataLoader()
        assert data_loader is not None

import os
import tempfile
import shutil
import pytest
from src.data.data_loader import FlowerDataLoader


def test_prepare_data_with_empty_dataset_dir(monkeypatch):
    tmp = tempfile.mkdtemp()
    try:
        # monkeypatch DATA_DIR/TRAIN_DIR/TEST_DIR to temp locations
        monkeypatch.setenv("DATA_DIR", os.path.join(tmp, "dataset"))
        monkeypatch.setenv("TRAIN_DIR", os.path.join(tmp, "train"))
        monkeypatch.setenv("TEST_DIR", os.path.join(tmp, "test"))
        os.makedirs(os.environ["DATA_DIR"], exist_ok=True)

        dl = FlowerDataLoader()
        train_dir, test_dir = dl.prepare_data()
        # empty dataset -> should still create train/test dirs and not crash
        assert os.path.isdir(train_dir)
        assert os.path.isdir(test_dir)
    finally:
        shutil.rmtree(tmp)


def test_get_data_loaders_on_empty_dirs(monkeypatch):
    tmp = tempfile.mkdtemp()
    try:
        monkeypatch.setenv("TRAIN_DIR", os.path.join(tmp, "train"))
        monkeypatch.setenv("TEST_DIR", os.path.join(tmp, "test"))
        os.makedirs(os.environ["TRAIN_DIR"], exist_ok=True)
        os.makedirs(os.environ["TEST_DIR"], exist_ok=True)

        dl = FlowerDataLoader()
        # Empty folders will create empty datasets; accessing DataLoader len should be 0-sized datasets
        train_loader, test_loader, mapping = dl.get_data_loaders()
        assert hasattr(train_loader, "batch_size")
        assert isinstance(mapping, dict)
    finally:
        shutil.rmtree(tmp)

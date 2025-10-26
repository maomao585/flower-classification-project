import os
import shutil

import mlflow  # noqa: F401  # 暂时未使用，但将来可能需要
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class FlowerDataLoader:
    """
    简化的花卉图像数据加载器
    """

    def __init__(self):
        from src.utils.config import load_config

        self.config = load_config()

    def prepare_data(self):
        """
        准备训练和测试数据
        """
        dataset_dir = self.config["DATA_DIR"]
        train_dir = self.config["TRAIN_DIR"]
        test_dir = self.config["TEST_DIR"]

        # 创建目录
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        # 获取所有类别
        categories = [
            d
            for d in os.listdir(dataset_dir)
            if os.path.isdir(os.path.join(dataset_dir, d))
        ]

        print(f"发现 {len(categories)} 个花卉类别")

        # 分割数据
        for category in categories:
            category_path = os.path.join(dataset_dir, category)
            all_images = [
                f
                for f in os.listdir(category_path)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]

            if not all_images:
                continue

            train_images, test_images = train_test_split(
                all_images, test_size=0.2, random_state=42
            )

            # 创建类别目录
            train_category_dir = os.path.join(train_dir, category)
            test_category_dir = os.path.join(test_dir, category)
            os.makedirs(train_category_dir, exist_ok=True)
            os.makedirs(test_category_dir, exist_ok=True)

            # 复制文件
            for img in train_images:
                shutil.copy(
                    os.path.join(category_path, img),
                    os.path.join(train_category_dir, img),
                )

            for img in test_images:
                shutil.copy(
                    os.path.join(category_path, img),
                    os.path.join(test_category_dir, img),
                )

        print(f"数据准备完成: 训练集 -> {train_dir}, 测试集 -> {test_dir}")
        return train_dir, test_dir

    def get_data_loaders(self):
        """
        获取数据加载器
        """
        # 数据预处理
        transform = transforms.Compose(
            [
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # 加载数据集
        train_data = datasets.ImageFolder(
            root=self.config["TRAIN_DIR"], transform=transform
        )
        test_data = datasets.ImageFolder(
            root=self.config["TEST_DIR"], transform=transform
        )

        # 创建数据加载器
        train_loader = DataLoader(
            train_data, batch_size=self.config["BATCH_SIZE"], shuffle=True
        )
        test_loader = DataLoader(
            test_data, batch_size=self.config["BATCH_SIZE"], shuffle=False
        )

        print(f"训练集大小: {len(train_data)}, 测试集大小: {len(test_data)}")

        return train_loader, test_loader, train_data.class_to_idx
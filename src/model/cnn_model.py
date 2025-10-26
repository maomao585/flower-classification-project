import mlflow  # noqa: F401  # 暂时未使用，但将来可能需要
import torch  # noqa: F401  # 暂时未使用，但将来可能需要
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    简化的CNN模型用于花卉图像分类
    """

    def __init__(self, num_classes=17):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def create_model(num_classes=17, device="cpu"):
    """
    创建并初始化模型
    """
    model = SimpleCNN(num_classes=num_classes)
    model = model.to(device)

    # 记录模型参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"模型参数总数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")

    return model

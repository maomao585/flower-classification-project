import os

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report


def train_model(model, train_loader, test_loader, class_names):
    """
    训练模型并记录实验
    """
    from src.utils.config import load_config

    config = load_config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 设置MLflow实验
    mlflow.set_experiment(config["MLFLOW_EXPERIMENT_NAME"])

    with mlflow.start_run():
        # provenance tags
        try:
            import subprocess
            git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
            mlflow.set_tag("git_sha", git_sha)
        except Exception:
            mlflow.set_tag("git_sha", "unknown")
        mlflow.set_tag("dataset_version", os.getenv("DATA_VERSION", "unknown"))
        # 记录超参数
        mlflow.log_params(
            {
                "learning_rate": config["LEARNING_RATE"],
                "epochs": config["EPOCHS"],
                "batch_size": config["BATCH_SIZE"],
                "num_classes": config["NUM_CLASSES"],
            }
        )

        # 初始化模型、损失函数和优化器
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config["LEARNING_RATE"])

        train_losses = []
        train_accuracies = []

        print("开始训练...")
        for epoch in range(config["EPOCHS"]):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

            epoch_loss = running_loss / len(train_loader)
            epoch_acc = correct / total

            train_losses.append(epoch_loss)
            train_accuracies.append(epoch_acc)

            # 记录每个epoch的指标
            mlflow.log_metric("train_loss", epoch_loss, step=epoch)
            mlflow.log_metric("train_accuracy", epoch_acc, step=epoch)

            print(
                f"Epoch {epoch+1}/{config['EPOCHS']}, "
                f"Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}"
            )

            # 每5个epoch进行一次验证
            if (epoch + 1) % 5 == 0:
                test_accuracy = evaluate_model(model, test_loader, device)
                mlflow.log_metric("test_accuracy", test_accuracy, step=epoch)
                print(f"验证准确率: {test_accuracy:.4f}")

        # 最终评估
        final_test_accuracy, class_report = evaluate_model_detailed(
            model, test_loader, class_names, device
        )
        mlflow.log_metric("final_test_accuracy", final_test_accuracy)
        mlflow.log_text(class_report, "classification_report.txt")

        # 保存模型
        os.makedirs(config["MODEL_SAVE_PATH"], exist_ok=True)
        model_save_path = os.path.join(config["MODEL_SAVE_PATH"], "flower_model.pth")
        torch.save(model.state_dict(), model_save_path)
        mlflow.log_artifact(model_save_path)

        # 使用MLflow记录PyTorch模型
        mlflow.pytorch.log_model(model, "pytorch_model")

        print(f"最终测试准确率: {final_test_accuracy:.4f}")
        print("分类报告:")
        print(class_report)

        return model, train_losses, train_accuracies


def evaluate_model(model, test_loader, device):
    """
    评估模型准确率
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    return accuracy


def evaluate_model_detailed(model, test_loader, class_names, device):
    """
    详细评估模型性能
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    class_report = classification_report(
        all_labels, all_preds, target_names=class_names
    )

    return accuracy, class_report

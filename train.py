#!/usr/bin/env python3
"""
花卉图像分类训练脚本
"""

import argparse

import mlflow

from src.data.data_loader import FlowerDataLoader
from src.model.cnn_model import create_model
from src.model.train import train_model
from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description='花卉图像分类训练')
    parser.add_argument('--data-dir', type=str, help='数据目录路径')
    parser.add_argument('--epochs', type=int, help='训练轮数')
    parser.add_argument('--batch-size', type=int, help='批次大小')
    parser.add_argument('--learning-rate', type=float, help='学习率')
    
    args = parser.parse_args()
    config = load_config()
    
    # 覆盖配置（如果提供了命令行参数）
    if args.data_dir:
        config['DATA_DIR'] = args.data_dir
    if args.epochs:
        config['EPOCHS'] = args.epochs
    if args.batch_size:
        config['BATCH_SIZE'] = args.batch_size
    if args.learning_rate:
        config['LEARNING_RATE'] = args.learning_rate
    
    # 设置MLflow
    mlflow.set_tracking_uri(config['MLFLOW_TRACKING_URI'])
    
    print("=== 花卉图像分类训练开始 ===")
    print(f"数据目录: {config['DATA_DIR']}")
    print(f"训练轮数: {config['EPOCHS']}")
    print(f"批次大小: {config['BATCH_SIZE']}")
    print(f"学习率: {config['LEARNING_RATE']}")
    
    # 初始化数据加载器
    data_loader = FlowerDataLoader()
    
    # 准备数据
    print("\n1. 准备数据...")
    train_dir, test_dir = data_loader.prepare_data()
    
    # 加载数据
    print("\n2. 加载数据...")
    train_loader, test_loader, class_mapping = data_loader.get_data_loaders()
    class_names = list(class_mapping.keys())
    print(f"类别: {class_names}")
    
    # 创建模型
    print("\n3. 创建模型...")
    model = create_model(num_classes=config['NUM_CLASSES'])
    
    # 训练模型
    print("\n4. 训练模型...")
    model, train_losses, train_accuracies = train_model(
        model, train_loader, test_loader, class_names
    )
    
    print("\n=== 训练完成 ===")
    print(f"最终训练准确率: {train_accuracies[-1]:.4f}")

if __name__ == "__main__":
    main()
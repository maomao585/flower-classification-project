#!/usr/bin/env python3
"""
花卉分类模型训练主程序
"""

import os
from src.data.data_loader import FlowerDataLoader
from src.model.train import ModelTrainer
from src.utils.config import config

def main():
    """主函数"""
    print("=== 花卉分类模型训练 ===")
    
    # 准备数据
    data_loader = FlowerDataLoader()
    if not data_loader.prepare_data():
        print("数据准备失败，请检查数据路径")
        return
    
    # 获取数据加载器
    train_loader, val_loader, test_loader = data_loader.get_data_loaders()
    print(f"训练样本数: {len(train_loader.dataset)}")
    print(f"验证样本数: {len(val_loader.dataset)}")
    
    # 创建并训练模型
    trainer = ModelTrainer()
    trainer.train(train_loader, val_loader)
    
    print("训练完成！")

if __name__ == "__main__":
    main()
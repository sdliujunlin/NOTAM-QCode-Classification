"""
GPU 环境检测工具

用于检测当前运行环境中是否有可用的 CUDA GPU，
并输出 GPU 型号、显存等基本信息。
"""

import torch


def check_gpu():
    """检查并输出 GPU 环境信息"""
    print(f"CUDA 可用: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU 数量: {torch.cuda.device_count()}")
        print(f"当前 GPU: {torch.cuda.current_device()}")
        print(f"GPU 名称: {torch.cuda.get_device_name(0)}")

        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3

        print(f"\n总显存: {total_memory:.2f} GB")
        print(f"已分配显存: {allocated:.2f} GB")
        print(f"缓存显存: {reserved:.2f} GB")
    else:
        print("警告: 未检测到可用 GPU，BERT 训练将使用 CPU（速度较慢）")


if __name__ == "__main__":
    check_gpu()
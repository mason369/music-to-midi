#!/bin/bash
# YourMT3 代码库安装脚本
# 用于安装 YourMT3 MoE 转写所需的代码依赖（基于 Hugging Face Spaces 仓库）

set -e

echo "=========================================="
echo "YourMT3 代码库安装"
echo "=========================================="

if ! command -v git >/dev/null 2>&1; then
  echo "错误: 未找到 git，请先安装 Git。"
  exit 1
fi

if command -v git-lfs >/dev/null 2>&1; then
  git lfs install
else
  echo "警告: 未检测到 Git LFS，请先安装 Git LFS 后再重试。"
  echo "      继续执行可能导致模型文件未完整拉取。"
fi

if [ -d "YourMT3" ]; then
  echo "YourMT3 目录已存在，跳过克隆。"
else
  echo "克隆 YourMT3 仓库..."
  git clone https://huggingface.co/spaces/mimbres/YourMT3
fi

echo ""
echo "安装依赖..."
(
  cd "YourMT3"
  pip install -r requirements.txt
)

echo ""
echo "可选: GuitarSet 预处理需要 sox (Linux): sudo apt-get install sox"
echo "=========================================="
echo "✔ YourMT3 代码库准备完成"
echo "=========================================="

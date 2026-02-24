#!/bin/bash
# YourMT3 代码库安装脚本
# 用法: bash install_yourmt3_code.sh [--pip /path/to/pip]

set -e

# ───── 参数解析 ─────
PIP_BIN="pip"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --pip) PIP_BIN="$2"; shift 2 ;;
        *) shift ;;
    esac
done

echo "=========================================="
echo "YourMT3 代码库安装"
echo "=========================================="
echo "使用 pip: $PIP_BIN"

if ! command -v git >/dev/null 2>&1; then
  echo "错误: 未找到 git，请先安装 Git。"
  exit 1
fi


if [ -d "YourMT3" ]; then
  echo "YourMT3 目录已存在，跳过克隆。"
else
  echo "克隆 YourMT3 代码库（仅代码，不含模型权重）..."
  GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --progress \
      https://huggingface.co/spaces/mimbres/YourMT3 YourMT3
fi

echo ""
echo "安装 YourMT3 Python 依赖..."
(
  cd "YourMT3"
  "$PIP_BIN" install -r requirements.txt
)

echo ""
echo "=========================================="
echo "✔ YourMT3 代码库准备完成"
echo "=========================================="

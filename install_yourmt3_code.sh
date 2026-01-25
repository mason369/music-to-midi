#!/bin/bash
# YourMT3 代码库安装脚本
# 用于安装 YourMT3 MoE 转写所需的代码依赖

set -e

echo "=========================================="
echo "YourMT3 代码库安装"
echo "=========================================="

# 检查是否已安装
if python3 -c "import amt" 2>/dev/null; then
    echo "✓ YourMT3 已安装"
    exit 0
fi

echo ""
echo "安装 YourMT3 代码库..."
echo ""

# 克隆仓库到临时目录
TEMP_DIR=$(mktemp -d)
echo "克隆 YourMT3 仓库..."
git clone https://github.com/mimbres/YourMT3.git "$TEMP_DIR/YourMT3"

# 安装
cd "$TEMP_DIR/YourMT3"
echo ""
echo "安装依赖..."
pip install -e .

# 清理
cd -
rm -rf "$TEMP_DIR"

echo ""
echo "=========================================="
echo "✅ YourMT3 代码库安装完成！"
echo "=========================================="

# 验证
python3 -c "import amt; print('✓ 验证成功: YourMT3 可用')"

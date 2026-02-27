#!/bin/bash

# 设置环境名称
ENV_NAME="lamgen"

echo "🚀 开始创建 Conda 环境: $ENV_NAME ..."

# 1. 创建环境 (Python 3.10)
conda create -n $ENV_NAME python=3.10 -y

# 2. 初始化 shell 以便在脚本中使用 conda activate
# 注意：直接在脚本里用 'source activate' 有时会失效，建议使用 conda 内部命令
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

echo "📦 正在安装 PyTorch (CUDA 11.8)..."

# 3. 安装 PyTorch 相关组件
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu118

echo "📦 正在安装 Transformers 和 Pandas..."

# 4. 安装其他依赖
pip install transformers==4.24.0 pandas

echo "✅ 环境 '$ENV_NAME' 配置完成！"
echo "💡 请运行以下命令进入环境: conda activate $ENV_NAME"

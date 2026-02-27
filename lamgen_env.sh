#!/bin/bash

# 1. 创建环境 (Python 3.10)
echo "🚀 正在创建 Conda 环境: lamgen ..."
conda create -n lamgen python=3.10 -y

# 2. 激活环境
eval "$(conda shell.bash hook)"
conda activate lamgen

# 3. 安装 PyTorch (指定官方 CUDA 11.8 源，必须带 +cu118 后缀)
echo "📦 正在安装 PyTorch 2.4.0 (CUDA 11.8)..."
pip install torch==2.4.0+cu118 torchvision==0.19.0+cu118 torchaudio==2.4.0+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118

# 4. 安装其他依赖
echo "📦 正在安装 Transformers 和 Pandas..."
pip install transformers==4.24.0 pandas

echo "✅ 环境 'lamgen' 配置完成！"
echo "💡 请运行以下命令进入环境: conda activate lamgen"

#!/bin/bash

echo "启动AI Eye Remote Control PyQt6版本..."

# 检查虚拟环境
if [ -d "../mediapipe_env" ]; then
    echo "激活虚拟环境..."
    source ../mediapipe_env/bin/activate
else
    echo "未找到虚拟环境，使用系统Python环境"
fi

# 启动应用（使用PyQt6版本）
echo "启动应用..."
python3 main_widget.py

echo "应用已退出"
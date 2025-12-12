#!/bin/bash

echo "Activate the eye controller..."

# 检查虚拟环境
if [ -d "../mediapipe_env" ]; then
    echo "Activate virtual environment..."
    source ../mediapipe_env/bin/activate
else
    echo "Virtual environment not found, using system Python environment"
fi

echo "启动应用..."
python3 main_widget.py

echo "应用已退出"
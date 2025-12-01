#!/bin/bash

echo "启动简化版AI眼睛遥控器..."

# 检查虚拟环境是否存在
if [ -d "../mediapipe_env" ]; then
    source ../mediapipe_env/bin/activate
    echo "已激活虚拟环境"
else
    echo "未找到虚拟环境，使用系统Python环境"
fi

# 设置环境变量
export DISPLAY=:0
export XAUTHORITY=$HOME/.Xauthority

# 减少 TensorFlow 日志输出
export TF_CPP_MIN_LOG_LEVEL=2


# 运行主程序
python3 main_simple.py

echo "程序已退出"
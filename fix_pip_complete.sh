#!/bin/bash
echo "开始修复 pip..."

# 1. 清理旧文件
echo "清理旧文件..."
sudo apt remove --purge python3-pip -y 2>/dev/null
sudo rm -f /usr/bin/pip /usr/bin/pip3
sudo rm -rf /usr/lib/python3.10/site-packages/pip* 2>/dev/null

# 2. 确保目录存在
echo "创建必要目录..."
sudo mkdir -p /usr/lib/python3.10/site-packages
sudo chmod 755 /usr/lib/python3.10/site-packages

# 3. 安装依赖
echo "安装依赖..."
sudo apt update
sudo apt install python3-distutils python3-setuptools python3-wheel -y

# 4. 安装 pip
echo "安装 pip..."
cd /tmp
wget -q https://bootstrap.pypa.io/get-pip.py
sudo python3 get-pip.py

# 5. 创建执行脚本
echo "创建 pip 脚本..."
sudo bash -c 'cat > /usr/bin/pip3 << "END"
#!/usr/bin/python3
import sys
sys.path.insert(0, "/usr/lib/python3.10/site-packages")
sys.path.insert(0, "/usr/local/lib/python3.10/dist-packages")
try:
    from pip._internal.cli.main import main
except ImportError:
    from pip import main
sys.exit(main())
END'
sudo chmod +x /usr/bin/pip3
sudo ln -sf /usr/bin/pip3 /usr/bin/pip

# 6. 验证
echo "验证安装..."
if python3 -c "import pip; print('成功: pip', pip.__version__)" 2>/dev/null; then
    echo "✓ pip 修复成功!"
    pip3 --version
else
    echo "✗ pip 修复失败，尝试备用方案..."
    # 备用方案
    sudo apt install --reinstall python3-pip -y
fi

echo "修复完成!"

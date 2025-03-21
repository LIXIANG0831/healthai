#!/bin/bash

# 获取当前脚本的目录（确保路径正确）
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# 进入脚本所在目录（保证路径正确）
cd "$DIR" || exit

# 使用 nohup 启动后台任务，并将日志输出到当前目录的 [nohup.log] 文件
nohup python3 train.py > nohup.log 2>&1 &

# 显示提示信息
echo "脚本已在后台运行！日志路径：$DIR/nohup.log"
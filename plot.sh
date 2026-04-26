#!/bin/bash
# 超参数范围定义：64 -> 192，步长16
for units in $(seq 64 16 192); do
    echo "===== 运行参数：$units ====="
    python run_train_dg_cls.py --prompt_plot $units
    echo "--------------------------"
done
#!/bin/bash

# 设置参数范围
train_batch_sizes=($(seq 512 128 768))  # 16到32，步长2
learning_rates=($(seq 8e-6 1e-6 1e-5))  # 6e-5到1e-4，步长1e-5

# 基础命令
base_cmd="python run_train_dg_cls.py"

# 遍历所有参数组合
for bs in "${train_batch_sizes[@]}"; do
    for lr in "${learning_rates[@]}"; do
        # 构造完整命令
        cmd="$base_cmd --prompt_dim $bs --learning_rate $lr"
        
        # 打印当前运行的命令（可选）
        echo "Running: $cmd"
        
        # 执行命令
        eval $cmd
        
        # 可选：每次运行后暂停（防止资源冲突）
        sleep 2
    done
done

echo "All experiments completed!"
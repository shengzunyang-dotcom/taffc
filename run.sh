#!/bin/bash
# 自动化训练脚本 - 遍历学习率并重复10次
# 创建时间：$(date +%F)

# 配置参数
TRAIN_SCRIPT="run_train_dg_cls.py"  # 训练程序
SEEDS=(3404 3407 3408 3403 3402 3405 3406 3409 3410 3417)
# LEARNING_RATES=(6e-5 7e-5 8e-5 1e-4 2e-4)

echo "===== 开始自动化训练 ====="
echo "配置: $REPEATS 次重复, 学习率范围 $MIN_LR 到 $MAX_LR"


# 内层循环：遍历学习率
for lr in "${SEEDS[@]}"; do
    printf "运行中: seed=%s\n" "$lr"
    # 执行训练程序
    python $TRAIN_SCRIPT --seed "$lr"
    
    # 检查执行状态
    if [ $? -ne 0 ]; then
        echo "错误: 训练失败! 学习率=$lr" >&2
        exit 1
    fi
done


echo -e "\n===== 所有任务完成 ====="
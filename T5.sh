

for ((rep=1; rep<=10; rep++)); do
        
        # 执行训练程序
        python T5test.py
        
        # 检查执行状态
        if [ $? -ne 0 ]; then
            exit 1
        fi
    done
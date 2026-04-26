file_obj = open('task_lamda_mosei.sh', 'w')
cmd1 = '''
monitor_gpu_processes() {
    while true; do
        # 使用nvidia-smi命令获取指定GPU上正在运行的进程数量
        gpu_processes=$(ps -ef | grep run_knowledge_mosi_mosei | wc -l)

        if [ "$gpu_processes" -gt 1 ]; then
            echo "GPU $gpu_index 上有 $gpu_processes 个程序在运行，将睡眠3分钟..."
            sleep 60  # 睡眠3分钟
        else
            echo "GPU $gpu_index 上没有程序在运行，退出循环。"
            break
        fi
    done
}
'''
file_obj.writelines(cmd1 + '\n')

count = 0
for i in range(1, 11):
    for j in range(0, 8):
        cmd = f'export CUDA_VISIBLE_DEVICES={j%4}; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-0.{i}-a100-{count} --domain_type 2 --lamda 0.{i} > log/0.log 2>&1 &'
        file_obj.writelines(cmd +'\n')
        count += 1
        if (count) % 2 == 0:
            file_obj.writelines('sleep 30\n')
            file_obj.writelines('monitor_gpu_processes\n')
file_obj.close()
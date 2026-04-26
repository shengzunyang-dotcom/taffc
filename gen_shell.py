
file_obj = open('task_mosi.sh', 'w')
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

for i in range(0, 1000):
    cmd = f'export CUDA_VISIBLE_DEVICES={ (i+2) %  2 + 2}; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-{i} --domain_type 1 --unimodal text> log/{i}.log 2>&1 &'
    file_obj.writelines(cmd +'\n')
    if (i+1) % 4 ==0:
        file_obj.writelines('sleep 30\n')
        file_obj.writelines('monitor_gpu_processes\n')
file_obj.close()
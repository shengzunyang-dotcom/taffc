# export CUDA_VISIBLE_DEVICES=1; python run_knowledge.py --wandb_name text-m-multimodal-freeze-1-1 --domain_type 
monitor_gpu_processes() {
    while true; do
        # 使用nvidia-smi命令获取指定GPU上正在运行的进程数量
        gpu_processes=$(ps -ef | grep run_knowledge_unimodal | wc -l)

        if [ "$gpu_processes" -gt 1 ]; then
            echo "GPU $gpu_index 上有 $gpu_processes 个程序在运行，将睡眠3分钟..."
            sleep 180  # 睡眠3分钟
        else
            echo "GPU $gpu_index 上没有程序在运行，退出循环。"
            break
        fi
    done
}


export CUDA_VISIBLE_DEVICES=0; nohup python run_knowledge_unimodal.py --wandb_name text-1 --domain_type 1 > log/1.log 2>&1 &
export CUDA_VISIBLE_DEVICES=1; nohup python run_knowledge_unimodal.py --wandb_name text-2 --domain_type 1 > log/1.log 2>&1 & 
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name text-3 --domain_type 1 > log/1.log 2>&1 & 
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name text-4 --domain_type 1 > log/1.log 2>&1 & 

sleep 120
monitor_gpu_processes



export CUDA_VISIBLE_DEVICES=0; nohup python run_knowledge_unimodal.py --wandb_name text-5 --domain_type 1 > log/1.log 2>&1 &
export CUDA_VISIBLE_DEVICES=1; nohup python run_knowledge_unimodal.py --wandb_name text-6 --domain_type 1 > log/1.log 2>&1 & 
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name text-7 --domain_type 1 > log/1.log 2>&1 & 
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name text-8 --domain_type 1 > log/1.log 2>&1 & 

sleep 120
monitor_gpu_processes

export CUDA_VISIBLE_DEVICES=0; nohup python run_knowledge_unimodal.py --wandb_name text-9 --domain_type 1 > log/1.log 2>&1 &
export CUDA_VISIBLE_DEVICES=1; nohup python run_knowledge_unimodal.py --wandb_name text-10 --domain_type 1 > log/1.log 2>&1 & 
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name text-11 --domain_type 1 > log/1.log 2>&1 & 
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name text-12 --domain_type 1 > log/1.log 2>&1 & 

sleep 120
monitor_gpu_processes

export CUDA_VISIBLE_DEVICES=0; nohup python run_knowledge_unimodal.py --wandb_name text-13 --domain_type 1 > log/1.log 2>&1 &
export CUDA_VISIBLE_DEVICES=1; nohup python run_knowledge_unimodal.py --wandb_name text-14 --domain_type 1 > log/1.log 2>&1 & 
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name text-15 --domain_type 1 > log/1.log 2>&1 & 
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name text-16 --domain_type 1 > log/1.log 2>&1 & 

sleep 120
monitor_gpu_processes

export CUDA_VISIBLE_DEVICES=0; nohup python run_knowledge_unimodal.py --wandb_name text-17 --domain_type 1 > log/1.log 2>&1 &
export CUDA_VISIBLE_DEVICES=1; nohup python run_knowledge_unimodal.py --wandb_name text-18 --domain_type 1 > log/1.log 2>&1 & 
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name text-19 --domain_type 1 > log/1.log 2>&1 & 
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name text-20 --domain_type 1 > log/1.log 2>&1 & 

sleep 120
monitor_gpu_processes

export CUDA_VISIBLE_DEVICES=0; nohup python run_knowledge_unimodal.py --wandb_name text-21 --domain_type 1 > log/1.log 2>&1 &
export CUDA_VISIBLE_DEVICES=1; nohup python run_knowledge_unimodal.py --wandb_name text-22 --domain_type 1 > log/1.log 2>&1 & 
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name text-23 --domain_type 1 > log/1.log 2>&1 & 
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name text-24 --domain_type 1 > log/1.log 2>&1 & 

sleep 120
monitor_gpu_processes

export CUDA_VISIBLE_DEVICES=0; nohup python run_knowledge_unimodal.py --wandb_name text-25 --domain_type 1 > log/1.log 2>&1 &
export CUDA_VISIBLE_DEVICES=1; nohup python run_knowledge_unimodal.py --wandb_name text-26 --domain_type 1 > log/1.log 2>&1 & 
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name text-27 --domain_type 1 > log/1.log 2>&1 & 
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name text-28 --domain_type 1 > log/1.log 2>&1 & 

sleep 120
monitor_gpu_processes

export CUDA_VISIBLE_DEVICES=0; nohup python run_knowledge_unimodal.py --wandb_name text-29 --domain_type 1 > log/1.log 2>&1 &
export CUDA_VISIBLE_DEVICES=1; nohup python run_knowledge_unimodal.py --wandb_name text-30 --domain_type 1 > log/1.log 2>&1 & 
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name text-31 --domain_type 1 > log/1.log 2>&1 & 
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name text-32 --domain_type 1 > log/1.log 2>&1 & 

sleep 120
monitor_gpu_processes

export CUDA_VISIBLE_DEVICES=0; nohup python run_knowledge_unimodal.py --wandb_name text-33 --domain_type 1 > log/1.log 2>&1 &
export CUDA_VISIBLE_DEVICES=1; nohup python run_knowledge_unimodal.py --wandb_name text-34 --domain_type 1 > log/1.log 2>&1 & 
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name text-35 --domain_type 1 > log/1.log 2>&1 & 
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name text-36 --domain_type 1 > log/1.log 2>&1 & 

sleep 120
monitor_gpu_processes

export CUDA_VISIBLE_DEVICES=0; nohup python run_knowledge_unimodal.py --wandb_name text-33 --domain_type 1 > log/1.log 2>&1 &
export CUDA_VISIBLE_DEVICES=1; nohup python run_knowledge_unimodal.py --wandb_name text-34 --domain_type 1 > log/1.log 2>&1 & 
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name text-35 --domain_type 1 > log/1.log 2>&1 & 
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name text-36 --domain_type 1 > log/1.log 2>&1 & 

sleep 120
monitor_gpu_processes

export CUDA_VISIBLE_DEVICES=0; nohup python run_knowledge_unimodal.py --wandb_name text-33 --domain_type 1 > log/1.log 2>&1 &
export CUDA_VISIBLE_DEVICES=1; nohup python run_knowledge_unimodal.py --wandb_name text-34 --domain_type 1 > log/1.log 2>&1 & 
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name text-35 --domain_type 1 > log/1.log 2>&1 & 
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name text-36 --domain_type 1 > log/1.log 2>&1 & 

sleep 120
monitor_gpu_processes

export CUDA_VISIBLE_DEVICES=0; nohup python run_knowledge_unimodal.py --wandb_name text-33 --domain_type 1 > log/1.log 2>&1 &
export CUDA_VISIBLE_DEVICES=1; nohup python run_knowledge_unimodal.py --wandb_name text-34 --domain_type 1 > log/1.log 2>&1 & 
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name text-35 --domain_type 1 > log/1.log 2>&1 & 
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name text-36 --domain_type 1 > log/1.log 2>&1 & 

sleep 120
monitor_gpu_processes

export CUDA_VISIBLE_DEVICES=0; nohup python run_knowledge_unimodal.py --wandb_name text-33 --domain_type 1 > log/1.log 2>&1 &
export CUDA_VISIBLE_DEVICES=1; nohup python run_knowledge_unimodal.py --wandb_name text-34 --domain_type 1 > log/1.log 2>&1 & 
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name text-35 --domain_type 1 > log/1.log 2>&1 & 
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name text-36 --domain_type 1 > log/1.log 2>&1 & 

sleep 120
monitor_gpu_processes

export CUDA_VISIBLE_DEVICES=0; nohup python run_knowledge_unimodal.py --wandb_name text-33 --domain_type 1 > log/1.log 2>&1 &
export CUDA_VISIBLE_DEVICES=1; nohup python run_knowledge_unimodal.py --wandb_name text-34 --domain_type 1 > log/1.log 2>&1 & 
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name text-35 --domain_type 1 > log/1.log 2>&1 & 
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name text-36 --domain_type 1 > log/1.log 2>&1 & 

sleep 120
monitor_gpu_processes

export CUDA_VISIBLE_DEVICES=0; nohup python run_knowledge_unimodal.py --wandb_name text-33 --domain_type 1 > log/1.log 2>&1 &
export CUDA_VISIBLE_DEVICES=1; nohup python run_knowledge_unimodal.py --wandb_name text-34 --domain_type 1 > log/1.log 2>&1 & 
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name text-35 --domain_type 1 > log/1.log 2>&1 & 
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name text-36 --domain_type 1 > log/1.log 2>&1 & 
sleep 120
monitor_gpu_processes

export CUDA_VISIBLE_DEVICES=0; nohup python run_knowledge_unimodal.py --wandb_name text-33 --domain_type 1 > log/1.log 2>&1 &
export CUDA_VISIBLE_DEVICES=1; nohup python run_knowledge_unimodal.py --wandb_name text-34 --domain_type 1 > log/1.log 2>&1 & 
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name text-35 --domain_type 1 > log/1.log 2>&1 & 
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name text-36 --domain_type 1 > log/1.log 2>&1 & 

sleep 120
monitor_gpu_processes

export CUDA_VISIBLE_DEVICES=0; nohup python run_knowledge_unimodal.py --wandb_name text-33 --domain_type 1 > log/1.log 2>&1 &
export CUDA_VISIBLE_DEVICES=1; nohup python run_knowledge_unimodal.py --wandb_name text-34 --domain_type 1 > log/1.log 2>&1 & 
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name text-35 --domain_type 1 > log/1.log 2>&1 & 
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name text-36 --domain_type 1 > log/1.log 2>&1 & 
sleep 120
monitor_gpu_processes

export CUDA_VISIBLE_DEVICES=0; nohup python run_knowledge_unimodal.py --wandb_name text-33 --domain_type 1 > log/1.log 2>&1 &
export CUDA_VISIBLE_DEVICES=1; nohup python run_knowledge_unimodal.py --wandb_name text-34 --domain_type 1 > log/1.log 2>&1 & 
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name text-35 --domain_type 1 > log/1.log 2>&1 & 
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name text-36 --domain_type 1 > log/1.log 2>&1 & 
sleep 120
monitor_gpu_processes

export CUDA_VISIBLE_DEVICES=0; nohup python run_knowledge_unimodal.py --wandb_name text-33 --domain_type 1 > log/1.log 2>&1 &
export CUDA_VISIBLE_DEVICES=1; nohup python run_knowledge_unimodal.py --wandb_name text-34 --domain_type 1 > log/1.log 2>&1 & 
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name text-35 --domain_type 1 > log/1.log 2>&1 & 
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name text-36 --domain_type 1 > log/1.log 2>&1 & 
sleep 120
monitor_gpu_processes

export CUDA_VISIBLE_DEVICES=0; nohup python run_knowledge_unimodal.py --wandb_name text-33 --domain_type 1 > log/1.log 2>&1 &
export CUDA_VISIBLE_DEVICES=1; nohup python run_knowledge_unimodal.py --wandb_name text-34 --domain_type 1 > log/1.log 2>&1 & 
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name text-35 --domain_type 1 > log/1.log 2>&1 & 
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name text-36 --domain_type 1 > log/1.log 2>&1 & 
sleep 120
monitor_gpu_processes

export CUDA_VISIBLE_DEVICES=0; nohup python run_knowledge_unimodal.py --wandb_name text-33 --domain_type 1 > log/1.log 2>&1 &
export CUDA_VISIBLE_DEVICES=1; nohup python run_knowledge_unimodal.py --wandb_name text-34 --domain_type 1 > log/1.log 2>&1 & 
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name text-35 --domain_type 1 > log/1.log 2>&1 & 
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name text-36 --domain_type 1 > log/1.log 2>&1 & 
sleep 120
monitor_gpu_processes

export CUDA_VISIBLE_DEVICES=0; nohup python run_knowledge_unimodal.py --wandb_name text-33 --domain_type 1 > log/1.log 2>&1 &
export CUDA_VISIBLE_DEVICES=1; nohup python run_knowledge_unimodal.py --wandb_name text-34 --domain_type 1 > log/1.log 2>&1 & 
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name text-35 --domain_type 1 > log/1.log 2>&1 & 
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name text-36 --domain_type 1 > log/1.log 2>&1 & 
sleep 120
monitor_gpu_processes

export CUDA_VISIBLE_DEVICES=0; nohup python run_knowledge_unimodal.py --wandb_name text-33 --domain_type 1 > log/1.log 2>&1 &
export CUDA_VISIBLE_DEVICES=1; nohup python run_knowledge_unimodal.py --wandb_name text-34 --domain_type 1 > log/1.log 2>&1 & 
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name text-35 --domain_type 1 > log/1.log 2>&1 & 
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name text-36 --domain_type 1 > log/1.log 2>&1 & 
sleep 120
monitor_gpu_processes

export CUDA_VISIBLE_DEVICES=0; nohup python run_knowledge_unimodal.py --wandb_name text-33 --domain_type 1 > log/1.log 2>&1 &
export CUDA_VISIBLE_DEVICES=1; nohup python run_knowledge_unimodal.py --wandb_name text-34 --domain_type 1 > log/1.log 2>&1 & 
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name text-35 --domain_type 1 > log/1.log 2>&1 & 
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name text-36 --domain_type 1 > log/1.log 2>&1 & 
sleep 120
monitor_gpu_processes

export CUDA_VISIBLE_DEVICES=0; nohup python run_knowledge_unimodal.py --wandb_name text-33 --domain_type 1 > log/1.log 2>&1 &
export CUDA_VISIBLE_DEVICES=1; nohup python run_knowledge_unimodal.py --wandb_name text-34 --domain_type 1 > log/1.log 2>&1 & 
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name text-35 --domain_type 1 > log/1.log 2>&1 & 
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name text-36 --domain_type 1 > log/1.log 2>&1 & 
sleep 120
monitor_gpu_processes

export CUDA_VISIBLE_DEVICES=0; nohup python run_knowledge_unimodal.py --wandb_name text-33 --domain_type 1 > log/1.log 2>&1 &
export CUDA_VISIBLE_DEVICES=1; nohup python run_knowledge_unimodal.py --wandb_name text-34 --domain_type 1 > log/1.log 2>&1 & 
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name text-35 --domain_type 1 > log/1.log 2>&1 & 
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name text-36 --domain_type 1 > log/1.log 2>&1 & 
sleep 120
monitor_gpu_processes

export CUDA_VISIBLE_DEVICES=0; nohup python run_knowledge_unimodal.py --wandb_name text-33 --domain_type 1 > log/1.log 2>&1 &
export CUDA_VISIBLE_DEVICES=1; nohup python run_knowledge_unimodal.py --wandb_name text-34 --domain_type 1 > log/1.log 2>&1 & 
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name text-35 --domain_type 1 > log/1.log 2>&1 & 
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name text-36 --domain_type 1 > log/1.log 2>&1 & 
sleep 120
monitor_gpu_processes

export CUDA_VISIBLE_DEVICES=0; nohup python run_knowledge_unimodal.py --wandb_name text-33 --domain_type 1 > log/1.log 2>&1 &
export CUDA_VISIBLE_DEVICES=1; nohup python run_knowledge_unimodal.py --wandb_name text-34 --domain_type 1 > log/1.log 2>&1 & 
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name text-35 --domain_type 1 > log/1.log 2>&1 & 
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name text-36 --domain_type 1 > log/1.log 2>&1 & 
sleep 120
monitor_gpu_processes

export CUDA_VISIBLE_DEVICES=0; nohup python run_knowledge_unimodal.py --wandb_name text-33 --domain_type 1 > log/1.log 2>&1 &
export CUDA_VISIBLE_DEVICES=1; nohup python run_knowledge_unimodal.py --wandb_name text-34 --domain_type 1 > log/1.log 2>&1 & 
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name text-35 --domain_type 1 > log/1.log 2>&1 & 
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name text-36 --domain_type 1 > log/1.log 2>&1 & 
sleep 120
monitor_gpu_processes

export CUDA_VISIBLE_DEVICES=0; nohup python run_knowledge_unimodal.py --wandb_name text-33 --domain_type 1 > log/1.log 2>&1 &
export CUDA_VISIBLE_DEVICES=1; nohup python run_knowledge_unimodal.py --wandb_name text-34 --domain_type 1 > log/1.log 2>&1 & 
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name text-35 --domain_type 1 > log/1.log 2>&1 & 
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name text-36 --domain_type 1 > log/1.log 2>&1 & 
sleep 120
monitor_gpu_processes

export CUDA_VISIBLE_DEVICES=0; nohup python run_knowledge_unimodal.py --wandb_name text-33 --domain_type 1 > log/1.log 2>&1 &
export CUDA_VISIBLE_DEVICES=1; nohup python run_knowledge_unimodal.py --wandb_name text-34 --domain_type 1 > log/1.log 2>&1 & 
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name text-35 --domain_type 1 > log/1.log 2>&1 & 
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name text-36 --domain_type 1 > log/1.log 2>&1 & 
sleep 120
monitor_gpu_processes

export CUDA_VISIBLE_DEVICES=0; nohup python run_knowledge_unimodal.py --wandb_name text-33 --domain_type 1 > log/1.log 2>&1 &
export CUDA_VISIBLE_DEVICES=1; nohup python run_knowledge_unimodal.py --wandb_name text-34 --domain_type 1 > log/1.log 2>&1 & 
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name text-35 --domain_type 1 > log/1.log 2>&1 & 
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name text-36 --domain_type 1 > log/1.log 2>&1 & 
sleep 120
monitor_gpu_processes

export CUDA_VISIBLE_DEVICES=0; nohup python run_knowledge_unimodal.py --wandb_name text-33 --domain_type 1 > log/1.log 2>&1 &
export CUDA_VISIBLE_DEVICES=1; nohup python run_knowledge_unimodal.py --wandb_name text-34 --domain_type 1 > log/1.log 2>&1 & 
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name text-35 --domain_type 1 > log/1.log 2>&1 & 
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name text-36 --domain_type 1 > log/1.log 2>&1 & 
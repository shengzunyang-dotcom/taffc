
monitor_gpu_processes() {
    while true; do
        # 使用nvidia-smi命令获取指定GPU上正在运行的进程数量
        gpu_processes=$(ps -ef | grep run_knowledge_unimodal | wc -l)

        if [ "$gpu_processes" -gt 1 ]; then
            echo "GPU $gpu_index 上有 $gpu_processes 个程序在运行，将睡眠3分钟..."
            sleep 60  # 睡眠3分钟
        else
            echo "GPU $gpu_index 上没有程序在运行，退出循环。"
            break
        fi
    done
}

export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-0 --domain_type 1 --unimodal text> log/0.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-1 --domain_type 1 --unimodal text> log/1.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-2 --domain_type 1 --unimodal text> log/2.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-3 --domain_type 1 --unimodal text> log/3.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-4 --domain_type 1 --unimodal text> log/4.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-5 --domain_type 1 --unimodal text> log/5.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-6 --domain_type 1 --unimodal text> log/6.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-7 --domain_type 1 --unimodal text> log/7.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-8 --domain_type 1 --unimodal text> log/8.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-9 --domain_type 1 --unimodal text> log/9.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-10 --domain_type 1 --unimodal text> log/10.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-11 --domain_type 1 --unimodal text> log/11.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-12 --domain_type 1 --unimodal text> log/12.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-13 --domain_type 1 --unimodal text> log/13.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-14 --domain_type 1 --unimodal text> log/14.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-15 --domain_type 1 --unimodal text> log/15.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-16 --domain_type 1 --unimodal text> log/16.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-17 --domain_type 1 --unimodal text> log/17.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-18 --domain_type 1 --unimodal text> log/18.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-19 --domain_type 1 --unimodal text> log/19.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-20 --domain_type 1 --unimodal text> log/20.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-21 --domain_type 1 --unimodal text> log/21.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-22 --domain_type 1 --unimodal text> log/22.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-23 --domain_type 1 --unimodal text> log/23.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-24 --domain_type 1 --unimodal text> log/24.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-25 --domain_type 1 --unimodal text> log/25.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-26 --domain_type 1 --unimodal text> log/26.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-27 --domain_type 1 --unimodal text> log/27.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-28 --domain_type 1 --unimodal text> log/28.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-29 --domain_type 1 --unimodal text> log/29.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-30 --domain_type 1 --unimodal text> log/30.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-31 --domain_type 1 --unimodal text> log/31.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-32 --domain_type 1 --unimodal text> log/32.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-33 --domain_type 1 --unimodal text> log/33.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-34 --domain_type 1 --unimodal text> log/34.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-35 --domain_type 1 --unimodal text> log/35.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-36 --domain_type 1 --unimodal text> log/36.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-37 --domain_type 1 --unimodal text> log/37.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-38 --domain_type 1 --unimodal text> log/38.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-39 --domain_type 1 --unimodal text> log/39.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-40 --domain_type 1 --unimodal text> log/40.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-41 --domain_type 1 --unimodal text> log/41.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-42 --domain_type 1 --unimodal text> log/42.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-43 --domain_type 1 --unimodal text> log/43.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-44 --domain_type 1 --unimodal text> log/44.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-45 --domain_type 1 --unimodal text> log/45.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-46 --domain_type 1 --unimodal text> log/46.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-47 --domain_type 1 --unimodal text> log/47.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-48 --domain_type 1 --unimodal text> log/48.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-49 --domain_type 1 --unimodal text> log/49.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-50 --domain_type 1 --unimodal text> log/50.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-51 --domain_type 1 --unimodal text> log/51.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-52 --domain_type 1 --unimodal text> log/52.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-53 --domain_type 1 --unimodal text> log/53.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-54 --domain_type 1 --unimodal text> log/54.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-55 --domain_type 1 --unimodal text> log/55.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-56 --domain_type 1 --unimodal text> log/56.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-57 --domain_type 1 --unimodal text> log/57.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-58 --domain_type 1 --unimodal text> log/58.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-59 --domain_type 1 --unimodal text> log/59.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-60 --domain_type 1 --unimodal text> log/60.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-61 --domain_type 1 --unimodal text> log/61.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-62 --domain_type 1 --unimodal text> log/62.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-63 --domain_type 1 --unimodal text> log/63.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-64 --domain_type 1 --unimodal text> log/64.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-65 --domain_type 1 --unimodal text> log/65.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-66 --domain_type 1 --unimodal text> log/66.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-67 --domain_type 1 --unimodal text> log/67.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-68 --domain_type 1 --unimodal text> log/68.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-69 --domain_type 1 --unimodal text> log/69.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-70 --domain_type 1 --unimodal text> log/70.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-71 --domain_type 1 --unimodal text> log/71.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-72 --domain_type 1 --unimodal text> log/72.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-73 --domain_type 1 --unimodal text> log/73.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-74 --domain_type 1 --unimodal text> log/74.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-75 --domain_type 1 --unimodal text> log/75.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-76 --domain_type 1 --unimodal text> log/76.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-77 --domain_type 1 --unimodal text> log/77.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-78 --domain_type 1 --unimodal text> log/78.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-79 --domain_type 1 --unimodal text> log/79.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-80 --domain_type 1 --unimodal text> log/80.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-81 --domain_type 1 --unimodal text> log/81.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-82 --domain_type 1 --unimodal text> log/82.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-83 --domain_type 1 --unimodal text> log/83.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-84 --domain_type 1 --unimodal text> log/84.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-85 --domain_type 1 --unimodal text> log/85.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-86 --domain_type 1 --unimodal text> log/86.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-87 --domain_type 1 --unimodal text> log/87.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-88 --domain_type 1 --unimodal text> log/88.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-89 --domain_type 1 --unimodal text> log/89.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-90 --domain_type 1 --unimodal text> log/90.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-91 --domain_type 1 --unimodal text> log/91.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-92 --domain_type 1 --unimodal text> log/92.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-93 --domain_type 1 --unimodal text> log/93.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-94 --domain_type 1 --unimodal text> log/94.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-95 --domain_type 1 --unimodal text> log/95.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-96 --domain_type 1 --unimodal text> log/96.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-97 --domain_type 1 --unimodal text> log/97.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-98 --domain_type 1 --unimodal text> log/98.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-99 --domain_type 1 --unimodal text> log/99.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-100 --domain_type 1 --unimodal text> log/100.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-101 --domain_type 1 --unimodal text> log/101.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-102 --domain_type 1 --unimodal text> log/102.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-103 --domain_type 1 --unimodal text> log/103.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-104 --domain_type 1 --unimodal text> log/104.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-105 --domain_type 1 --unimodal text> log/105.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-106 --domain_type 1 --unimodal text> log/106.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-107 --domain_type 1 --unimodal text> log/107.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-108 --domain_type 1 --unimodal text> log/108.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-109 --domain_type 1 --unimodal text> log/109.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-110 --domain_type 1 --unimodal text> log/110.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-111 --domain_type 1 --unimodal text> log/111.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-112 --domain_type 1 --unimodal text> log/112.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-113 --domain_type 1 --unimodal text> log/113.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-114 --domain_type 1 --unimodal text> log/114.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-115 --domain_type 1 --unimodal text> log/115.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-116 --domain_type 1 --unimodal text> log/116.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-117 --domain_type 1 --unimodal text> log/117.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-118 --domain_type 1 --unimodal text> log/118.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-119 --domain_type 1 --unimodal text> log/119.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-120 --domain_type 1 --unimodal text> log/120.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-121 --domain_type 1 --unimodal text> log/121.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-122 --domain_type 1 --unimodal text> log/122.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-123 --domain_type 1 --unimodal text> log/123.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-124 --domain_type 1 --unimodal text> log/124.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-125 --domain_type 1 --unimodal text> log/125.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-126 --domain_type 1 --unimodal text> log/126.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-127 --domain_type 1 --unimodal text> log/127.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-128 --domain_type 1 --unimodal text> log/128.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-129 --domain_type 1 --unimodal text> log/129.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-130 --domain_type 1 --unimodal text> log/130.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-131 --domain_type 1 --unimodal text> log/131.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-132 --domain_type 1 --unimodal text> log/132.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-133 --domain_type 1 --unimodal text> log/133.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-134 --domain_type 1 --unimodal text> log/134.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-135 --domain_type 1 --unimodal text> log/135.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-136 --domain_type 1 --unimodal text> log/136.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-137 --domain_type 1 --unimodal text> log/137.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-138 --domain_type 1 --unimodal text> log/138.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-139 --domain_type 1 --unimodal text> log/139.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-140 --domain_type 1 --unimodal text> log/140.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-141 --domain_type 1 --unimodal text> log/141.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-142 --domain_type 1 --unimodal text> log/142.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-143 --domain_type 1 --unimodal text> log/143.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-144 --domain_type 1 --unimodal text> log/144.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-145 --domain_type 1 --unimodal text> log/145.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-146 --domain_type 1 --unimodal text> log/146.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-147 --domain_type 1 --unimodal text> log/147.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-148 --domain_type 1 --unimodal text> log/148.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-149 --domain_type 1 --unimodal text> log/149.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-150 --domain_type 1 --unimodal text> log/150.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-151 --domain_type 1 --unimodal text> log/151.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-152 --domain_type 1 --unimodal text> log/152.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-153 --domain_type 1 --unimodal text> log/153.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-154 --domain_type 1 --unimodal text> log/154.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-155 --domain_type 1 --unimodal text> log/155.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-156 --domain_type 1 --unimodal text> log/156.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-157 --domain_type 1 --unimodal text> log/157.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-158 --domain_type 1 --unimodal text> log/158.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-159 --domain_type 1 --unimodal text> log/159.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-160 --domain_type 1 --unimodal text> log/160.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-161 --domain_type 1 --unimodal text> log/161.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-162 --domain_type 1 --unimodal text> log/162.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-163 --domain_type 1 --unimodal text> log/163.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-164 --domain_type 1 --unimodal text> log/164.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-165 --domain_type 1 --unimodal text> log/165.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-166 --domain_type 1 --unimodal text> log/166.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-167 --domain_type 1 --unimodal text> log/167.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-168 --domain_type 1 --unimodal text> log/168.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-169 --domain_type 1 --unimodal text> log/169.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-170 --domain_type 1 --unimodal text> log/170.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-171 --domain_type 1 --unimodal text> log/171.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-172 --domain_type 1 --unimodal text> log/172.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-173 --domain_type 1 --unimodal text> log/173.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-174 --domain_type 1 --unimodal text> log/174.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-175 --domain_type 1 --unimodal text> log/175.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-176 --domain_type 1 --unimodal text> log/176.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-177 --domain_type 1 --unimodal text> log/177.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-178 --domain_type 1 --unimodal text> log/178.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-179 --domain_type 1 --unimodal text> log/179.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-180 --domain_type 1 --unimodal text> log/180.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-181 --domain_type 1 --unimodal text> log/181.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-182 --domain_type 1 --unimodal text> log/182.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-183 --domain_type 1 --unimodal text> log/183.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-184 --domain_type 1 --unimodal text> log/184.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-185 --domain_type 1 --unimodal text> log/185.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-186 --domain_type 1 --unimodal text> log/186.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-187 --domain_type 1 --unimodal text> log/187.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-188 --domain_type 1 --unimodal text> log/188.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-189 --domain_type 1 --unimodal text> log/189.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-190 --domain_type 1 --unimodal text> log/190.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-191 --domain_type 1 --unimodal text> log/191.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-192 --domain_type 1 --unimodal text> log/192.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-193 --domain_type 1 --unimodal text> log/193.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-194 --domain_type 1 --unimodal text> log/194.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-195 --domain_type 1 --unimodal text> log/195.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-196 --domain_type 1 --unimodal text> log/196.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-197 --domain_type 1 --unimodal text> log/197.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-198 --domain_type 1 --unimodal text> log/198.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-199 --domain_type 1 --unimodal text> log/199.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-200 --domain_type 1 --unimodal text> log/200.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-201 --domain_type 1 --unimodal text> log/201.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-202 --domain_type 1 --unimodal text> log/202.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-203 --domain_type 1 --unimodal text> log/203.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-204 --domain_type 1 --unimodal text> log/204.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-205 --domain_type 1 --unimodal text> log/205.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-206 --domain_type 1 --unimodal text> log/206.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-207 --domain_type 1 --unimodal text> log/207.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-208 --domain_type 1 --unimodal text> log/208.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-209 --domain_type 1 --unimodal text> log/209.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-210 --domain_type 1 --unimodal text> log/210.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-211 --domain_type 1 --unimodal text> log/211.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-212 --domain_type 1 --unimodal text> log/212.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-213 --domain_type 1 --unimodal text> log/213.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-214 --domain_type 1 --unimodal text> log/214.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-215 --domain_type 1 --unimodal text> log/215.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-216 --domain_type 1 --unimodal text> log/216.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-217 --domain_type 1 --unimodal text> log/217.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-218 --domain_type 1 --unimodal text> log/218.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-219 --domain_type 1 --unimodal text> log/219.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-220 --domain_type 1 --unimodal text> log/220.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-221 --domain_type 1 --unimodal text> log/221.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-222 --domain_type 1 --unimodal text> log/222.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-223 --domain_type 1 --unimodal text> log/223.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-224 --domain_type 1 --unimodal text> log/224.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-225 --domain_type 1 --unimodal text> log/225.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-226 --domain_type 1 --unimodal text> log/226.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-227 --domain_type 1 --unimodal text> log/227.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-228 --domain_type 1 --unimodal text> log/228.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-229 --domain_type 1 --unimodal text> log/229.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-230 --domain_type 1 --unimodal text> log/230.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-231 --domain_type 1 --unimodal text> log/231.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-232 --domain_type 1 --unimodal text> log/232.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-233 --domain_type 1 --unimodal text> log/233.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-234 --domain_type 1 --unimodal text> log/234.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-235 --domain_type 1 --unimodal text> log/235.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-236 --domain_type 1 --unimodal text> log/236.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-237 --domain_type 1 --unimodal text> log/237.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-238 --domain_type 1 --unimodal text> log/238.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-239 --domain_type 1 --unimodal text> log/239.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-240 --domain_type 1 --unimodal text> log/240.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-241 --domain_type 1 --unimodal text> log/241.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-242 --domain_type 1 --unimodal text> log/242.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-243 --domain_type 1 --unimodal text> log/243.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-244 --domain_type 1 --unimodal text> log/244.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-245 --domain_type 1 --unimodal text> log/245.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-246 --domain_type 1 --unimodal text> log/246.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-247 --domain_type 1 --unimodal text> log/247.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-248 --domain_type 1 --unimodal text> log/248.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-249 --domain_type 1 --unimodal text> log/249.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-250 --domain_type 1 --unimodal text> log/250.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-251 --domain_type 1 --unimodal text> log/251.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-252 --domain_type 1 --unimodal text> log/252.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-253 --domain_type 1 --unimodal text> log/253.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-254 --domain_type 1 --unimodal text> log/254.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-255 --domain_type 1 --unimodal text> log/255.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-256 --domain_type 1 --unimodal text> log/256.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-257 --domain_type 1 --unimodal text> log/257.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-258 --domain_type 1 --unimodal text> log/258.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-259 --domain_type 1 --unimodal text> log/259.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-260 --domain_type 1 --unimodal text> log/260.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-261 --domain_type 1 --unimodal text> log/261.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-262 --domain_type 1 --unimodal text> log/262.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-263 --domain_type 1 --unimodal text> log/263.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-264 --domain_type 1 --unimodal text> log/264.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-265 --domain_type 1 --unimodal text> log/265.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-266 --domain_type 1 --unimodal text> log/266.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-267 --domain_type 1 --unimodal text> log/267.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-268 --domain_type 1 --unimodal text> log/268.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-269 --domain_type 1 --unimodal text> log/269.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-270 --domain_type 1 --unimodal text> log/270.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-271 --domain_type 1 --unimodal text> log/271.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-272 --domain_type 1 --unimodal text> log/272.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-273 --domain_type 1 --unimodal text> log/273.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-274 --domain_type 1 --unimodal text> log/274.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-275 --domain_type 1 --unimodal text> log/275.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-276 --domain_type 1 --unimodal text> log/276.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-277 --domain_type 1 --unimodal text> log/277.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-278 --domain_type 1 --unimodal text> log/278.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-279 --domain_type 1 --unimodal text> log/279.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-280 --domain_type 1 --unimodal text> log/280.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-281 --domain_type 1 --unimodal text> log/281.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-282 --domain_type 1 --unimodal text> log/282.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-283 --domain_type 1 --unimodal text> log/283.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-284 --domain_type 1 --unimodal text> log/284.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-285 --domain_type 1 --unimodal text> log/285.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-286 --domain_type 1 --unimodal text> log/286.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-287 --domain_type 1 --unimodal text> log/287.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-288 --domain_type 1 --unimodal text> log/288.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-289 --domain_type 1 --unimodal text> log/289.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-290 --domain_type 1 --unimodal text> log/290.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-291 --domain_type 1 --unimodal text> log/291.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-292 --domain_type 1 --unimodal text> log/292.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-293 --domain_type 1 --unimodal text> log/293.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-294 --domain_type 1 --unimodal text> log/294.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-295 --domain_type 1 --unimodal text> log/295.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-296 --domain_type 1 --unimodal text> log/296.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-297 --domain_type 1 --unimodal text> log/297.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-298 --domain_type 1 --unimodal text> log/298.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-299 --domain_type 1 --unimodal text> log/299.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-300 --domain_type 1 --unimodal text> log/300.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-301 --domain_type 1 --unimodal text> log/301.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-302 --domain_type 1 --unimodal text> log/302.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-303 --domain_type 1 --unimodal text> log/303.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-304 --domain_type 1 --unimodal text> log/304.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-305 --domain_type 1 --unimodal text> log/305.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-306 --domain_type 1 --unimodal text> log/306.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-307 --domain_type 1 --unimodal text> log/307.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-308 --domain_type 1 --unimodal text> log/308.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-309 --domain_type 1 --unimodal text> log/309.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-310 --domain_type 1 --unimodal text> log/310.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-311 --domain_type 1 --unimodal text> log/311.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-312 --domain_type 1 --unimodal text> log/312.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-313 --domain_type 1 --unimodal text> log/313.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-314 --domain_type 1 --unimodal text> log/314.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-315 --domain_type 1 --unimodal text> log/315.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-316 --domain_type 1 --unimodal text> log/316.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-317 --domain_type 1 --unimodal text> log/317.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-318 --domain_type 1 --unimodal text> log/318.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-319 --domain_type 1 --unimodal text> log/319.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-320 --domain_type 1 --unimodal text> log/320.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-321 --domain_type 1 --unimodal text> log/321.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-322 --domain_type 1 --unimodal text> log/322.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-323 --domain_type 1 --unimodal text> log/323.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-324 --domain_type 1 --unimodal text> log/324.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-325 --domain_type 1 --unimodal text> log/325.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-326 --domain_type 1 --unimodal text> log/326.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-327 --domain_type 1 --unimodal text> log/327.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-328 --domain_type 1 --unimodal text> log/328.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-329 --domain_type 1 --unimodal text> log/329.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-330 --domain_type 1 --unimodal text> log/330.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-331 --domain_type 1 --unimodal text> log/331.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-332 --domain_type 1 --unimodal text> log/332.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-333 --domain_type 1 --unimodal text> log/333.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-334 --domain_type 1 --unimodal text> log/334.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-335 --domain_type 1 --unimodal text> log/335.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-336 --domain_type 1 --unimodal text> log/336.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-337 --domain_type 1 --unimodal text> log/337.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-338 --domain_type 1 --unimodal text> log/338.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-339 --domain_type 1 --unimodal text> log/339.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-340 --domain_type 1 --unimodal text> log/340.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-341 --domain_type 1 --unimodal text> log/341.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-342 --domain_type 1 --unimodal text> log/342.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-343 --domain_type 1 --unimodal text> log/343.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-344 --domain_type 1 --unimodal text> log/344.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-345 --domain_type 1 --unimodal text> log/345.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-346 --domain_type 1 --unimodal text> log/346.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-347 --domain_type 1 --unimodal text> log/347.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-348 --domain_type 1 --unimodal text> log/348.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-349 --domain_type 1 --unimodal text> log/349.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-350 --domain_type 1 --unimodal text> log/350.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-351 --domain_type 1 --unimodal text> log/351.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-352 --domain_type 1 --unimodal text> log/352.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-353 --domain_type 1 --unimodal text> log/353.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-354 --domain_type 1 --unimodal text> log/354.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-355 --domain_type 1 --unimodal text> log/355.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-356 --domain_type 1 --unimodal text> log/356.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-357 --domain_type 1 --unimodal text> log/357.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-358 --domain_type 1 --unimodal text> log/358.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-359 --domain_type 1 --unimodal text> log/359.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-360 --domain_type 1 --unimodal text> log/360.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-361 --domain_type 1 --unimodal text> log/361.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-362 --domain_type 1 --unimodal text> log/362.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-363 --domain_type 1 --unimodal text> log/363.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-364 --domain_type 1 --unimodal text> log/364.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-365 --domain_type 1 --unimodal text> log/365.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-366 --domain_type 1 --unimodal text> log/366.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-367 --domain_type 1 --unimodal text> log/367.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-368 --domain_type 1 --unimodal text> log/368.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-369 --domain_type 1 --unimodal text> log/369.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-370 --domain_type 1 --unimodal text> log/370.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-371 --domain_type 1 --unimodal text> log/371.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-372 --domain_type 1 --unimodal text> log/372.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-373 --domain_type 1 --unimodal text> log/373.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-374 --domain_type 1 --unimodal text> log/374.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-375 --domain_type 1 --unimodal text> log/375.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-376 --domain_type 1 --unimodal text> log/376.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-377 --domain_type 1 --unimodal text> log/377.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-378 --domain_type 1 --unimodal text> log/378.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-379 --domain_type 1 --unimodal text> log/379.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-380 --domain_type 1 --unimodal text> log/380.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-381 --domain_type 1 --unimodal text> log/381.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-382 --domain_type 1 --unimodal text> log/382.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-383 --domain_type 1 --unimodal text> log/383.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-384 --domain_type 1 --unimodal text> log/384.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-385 --domain_type 1 --unimodal text> log/385.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-386 --domain_type 1 --unimodal text> log/386.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-387 --domain_type 1 --unimodal text> log/387.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-388 --domain_type 1 --unimodal text> log/388.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-389 --domain_type 1 --unimodal text> log/389.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-390 --domain_type 1 --unimodal text> log/390.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-391 --domain_type 1 --unimodal text> log/391.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-392 --domain_type 1 --unimodal text> log/392.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-393 --domain_type 1 --unimodal text> log/393.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-394 --domain_type 1 --unimodal text> log/394.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-395 --domain_type 1 --unimodal text> log/395.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-396 --domain_type 1 --unimodal text> log/396.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-397 --domain_type 1 --unimodal text> log/397.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-398 --domain_type 1 --unimodal text> log/398.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-399 --domain_type 1 --unimodal text> log/399.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-400 --domain_type 1 --unimodal text> log/400.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-401 --domain_type 1 --unimodal text> log/401.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-402 --domain_type 1 --unimodal text> log/402.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-403 --domain_type 1 --unimodal text> log/403.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-404 --domain_type 1 --unimodal text> log/404.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-405 --domain_type 1 --unimodal text> log/405.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-406 --domain_type 1 --unimodal text> log/406.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-407 --domain_type 1 --unimodal text> log/407.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-408 --domain_type 1 --unimodal text> log/408.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-409 --domain_type 1 --unimodal text> log/409.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-410 --domain_type 1 --unimodal text> log/410.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-411 --domain_type 1 --unimodal text> log/411.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-412 --domain_type 1 --unimodal text> log/412.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-413 --domain_type 1 --unimodal text> log/413.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-414 --domain_type 1 --unimodal text> log/414.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-415 --domain_type 1 --unimodal text> log/415.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-416 --domain_type 1 --unimodal text> log/416.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-417 --domain_type 1 --unimodal text> log/417.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-418 --domain_type 1 --unimodal text> log/418.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-419 --domain_type 1 --unimodal text> log/419.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-420 --domain_type 1 --unimodal text> log/420.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-421 --domain_type 1 --unimodal text> log/421.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-422 --domain_type 1 --unimodal text> log/422.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-423 --domain_type 1 --unimodal text> log/423.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-424 --domain_type 1 --unimodal text> log/424.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-425 --domain_type 1 --unimodal text> log/425.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-426 --domain_type 1 --unimodal text> log/426.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-427 --domain_type 1 --unimodal text> log/427.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-428 --domain_type 1 --unimodal text> log/428.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-429 --domain_type 1 --unimodal text> log/429.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-430 --domain_type 1 --unimodal text> log/430.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-431 --domain_type 1 --unimodal text> log/431.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-432 --domain_type 1 --unimodal text> log/432.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-433 --domain_type 1 --unimodal text> log/433.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-434 --domain_type 1 --unimodal text> log/434.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-435 --domain_type 1 --unimodal text> log/435.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-436 --domain_type 1 --unimodal text> log/436.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-437 --domain_type 1 --unimodal text> log/437.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-438 --domain_type 1 --unimodal text> log/438.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-439 --domain_type 1 --unimodal text> log/439.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-440 --domain_type 1 --unimodal text> log/440.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-441 --domain_type 1 --unimodal text> log/441.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-442 --domain_type 1 --unimodal text> log/442.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-443 --domain_type 1 --unimodal text> log/443.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-444 --domain_type 1 --unimodal text> log/444.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-445 --domain_type 1 --unimodal text> log/445.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-446 --domain_type 1 --unimodal text> log/446.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-447 --domain_type 1 --unimodal text> log/447.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-448 --domain_type 1 --unimodal text> log/448.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-449 --domain_type 1 --unimodal text> log/449.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-450 --domain_type 1 --unimodal text> log/450.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-451 --domain_type 1 --unimodal text> log/451.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-452 --domain_type 1 --unimodal text> log/452.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-453 --domain_type 1 --unimodal text> log/453.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-454 --domain_type 1 --unimodal text> log/454.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-455 --domain_type 1 --unimodal text> log/455.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-456 --domain_type 1 --unimodal text> log/456.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-457 --domain_type 1 --unimodal text> log/457.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-458 --domain_type 1 --unimodal text> log/458.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-459 --domain_type 1 --unimodal text> log/459.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-460 --domain_type 1 --unimodal text> log/460.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-461 --domain_type 1 --unimodal text> log/461.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-462 --domain_type 1 --unimodal text> log/462.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-463 --domain_type 1 --unimodal text> log/463.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-464 --domain_type 1 --unimodal text> log/464.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-465 --domain_type 1 --unimodal text> log/465.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-466 --domain_type 1 --unimodal text> log/466.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-467 --domain_type 1 --unimodal text> log/467.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-468 --domain_type 1 --unimodal text> log/468.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-469 --domain_type 1 --unimodal text> log/469.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-470 --domain_type 1 --unimodal text> log/470.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-471 --domain_type 1 --unimodal text> log/471.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-472 --domain_type 1 --unimodal text> log/472.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-473 --domain_type 1 --unimodal text> log/473.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-474 --domain_type 1 --unimodal text> log/474.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-475 --domain_type 1 --unimodal text> log/475.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-476 --domain_type 1 --unimodal text> log/476.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-477 --domain_type 1 --unimodal text> log/477.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-478 --domain_type 1 --unimodal text> log/478.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-479 --domain_type 1 --unimodal text> log/479.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-480 --domain_type 1 --unimodal text> log/480.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-481 --domain_type 1 --unimodal text> log/481.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-482 --domain_type 1 --unimodal text> log/482.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-483 --domain_type 1 --unimodal text> log/483.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-484 --domain_type 1 --unimodal text> log/484.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-485 --domain_type 1 --unimodal text> log/485.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-486 --domain_type 1 --unimodal text> log/486.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-487 --domain_type 1 --unimodal text> log/487.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-488 --domain_type 1 --unimodal text> log/488.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-489 --domain_type 1 --unimodal text> log/489.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-490 --domain_type 1 --unimodal text> log/490.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-491 --domain_type 1 --unimodal text> log/491.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-492 --domain_type 1 --unimodal text> log/492.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-493 --domain_type 1 --unimodal text> log/493.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-494 --domain_type 1 --unimodal text> log/494.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-495 --domain_type 1 --unimodal text> log/495.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-496 --domain_type 1 --unimodal text> log/496.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-497 --domain_type 1 --unimodal text> log/497.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-498 --domain_type 1 --unimodal text> log/498.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-499 --domain_type 1 --unimodal text> log/499.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-500 --domain_type 1 --unimodal text> log/500.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-501 --domain_type 1 --unimodal text> log/501.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-502 --domain_type 1 --unimodal text> log/502.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-503 --domain_type 1 --unimodal text> log/503.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-504 --domain_type 1 --unimodal text> log/504.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-505 --domain_type 1 --unimodal text> log/505.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-506 --domain_type 1 --unimodal text> log/506.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-507 --domain_type 1 --unimodal text> log/507.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-508 --domain_type 1 --unimodal text> log/508.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-509 --domain_type 1 --unimodal text> log/509.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-510 --domain_type 1 --unimodal text> log/510.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-511 --domain_type 1 --unimodal text> log/511.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-512 --domain_type 1 --unimodal text> log/512.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-513 --domain_type 1 --unimodal text> log/513.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-514 --domain_type 1 --unimodal text> log/514.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-515 --domain_type 1 --unimodal text> log/515.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-516 --domain_type 1 --unimodal text> log/516.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-517 --domain_type 1 --unimodal text> log/517.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-518 --domain_type 1 --unimodal text> log/518.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-519 --domain_type 1 --unimodal text> log/519.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-520 --domain_type 1 --unimodal text> log/520.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-521 --domain_type 1 --unimodal text> log/521.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-522 --domain_type 1 --unimodal text> log/522.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-523 --domain_type 1 --unimodal text> log/523.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-524 --domain_type 1 --unimodal text> log/524.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-525 --domain_type 1 --unimodal text> log/525.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-526 --domain_type 1 --unimodal text> log/526.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-527 --domain_type 1 --unimodal text> log/527.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-528 --domain_type 1 --unimodal text> log/528.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-529 --domain_type 1 --unimodal text> log/529.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-530 --domain_type 1 --unimodal text> log/530.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-531 --domain_type 1 --unimodal text> log/531.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-532 --domain_type 1 --unimodal text> log/532.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-533 --domain_type 1 --unimodal text> log/533.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-534 --domain_type 1 --unimodal text> log/534.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-535 --domain_type 1 --unimodal text> log/535.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-536 --domain_type 1 --unimodal text> log/536.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-537 --domain_type 1 --unimodal text> log/537.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-538 --domain_type 1 --unimodal text> log/538.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-539 --domain_type 1 --unimodal text> log/539.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-540 --domain_type 1 --unimodal text> log/540.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-541 --domain_type 1 --unimodal text> log/541.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-542 --domain_type 1 --unimodal text> log/542.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-543 --domain_type 1 --unimodal text> log/543.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-544 --domain_type 1 --unimodal text> log/544.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-545 --domain_type 1 --unimodal text> log/545.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-546 --domain_type 1 --unimodal text> log/546.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-547 --domain_type 1 --unimodal text> log/547.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-548 --domain_type 1 --unimodal text> log/548.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-549 --domain_type 1 --unimodal text> log/549.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-550 --domain_type 1 --unimodal text> log/550.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-551 --domain_type 1 --unimodal text> log/551.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-552 --domain_type 1 --unimodal text> log/552.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-553 --domain_type 1 --unimodal text> log/553.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-554 --domain_type 1 --unimodal text> log/554.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-555 --domain_type 1 --unimodal text> log/555.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-556 --domain_type 1 --unimodal text> log/556.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-557 --domain_type 1 --unimodal text> log/557.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-558 --domain_type 1 --unimodal text> log/558.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-559 --domain_type 1 --unimodal text> log/559.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-560 --domain_type 1 --unimodal text> log/560.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-561 --domain_type 1 --unimodal text> log/561.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-562 --domain_type 1 --unimodal text> log/562.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-563 --domain_type 1 --unimodal text> log/563.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-564 --domain_type 1 --unimodal text> log/564.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-565 --domain_type 1 --unimodal text> log/565.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-566 --domain_type 1 --unimodal text> log/566.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-567 --domain_type 1 --unimodal text> log/567.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-568 --domain_type 1 --unimodal text> log/568.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-569 --domain_type 1 --unimodal text> log/569.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-570 --domain_type 1 --unimodal text> log/570.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-571 --domain_type 1 --unimodal text> log/571.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-572 --domain_type 1 --unimodal text> log/572.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-573 --domain_type 1 --unimodal text> log/573.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-574 --domain_type 1 --unimodal text> log/574.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-575 --domain_type 1 --unimodal text> log/575.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-576 --domain_type 1 --unimodal text> log/576.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-577 --domain_type 1 --unimodal text> log/577.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-578 --domain_type 1 --unimodal text> log/578.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-579 --domain_type 1 --unimodal text> log/579.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-580 --domain_type 1 --unimodal text> log/580.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-581 --domain_type 1 --unimodal text> log/581.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-582 --domain_type 1 --unimodal text> log/582.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-583 --domain_type 1 --unimodal text> log/583.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-584 --domain_type 1 --unimodal text> log/584.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-585 --domain_type 1 --unimodal text> log/585.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-586 --domain_type 1 --unimodal text> log/586.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-587 --domain_type 1 --unimodal text> log/587.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-588 --domain_type 1 --unimodal text> log/588.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-589 --domain_type 1 --unimodal text> log/589.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-590 --domain_type 1 --unimodal text> log/590.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-591 --domain_type 1 --unimodal text> log/591.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-592 --domain_type 1 --unimodal text> log/592.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-593 --domain_type 1 --unimodal text> log/593.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-594 --domain_type 1 --unimodal text> log/594.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-595 --domain_type 1 --unimodal text> log/595.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-596 --domain_type 1 --unimodal text> log/596.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-597 --domain_type 1 --unimodal text> log/597.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-598 --domain_type 1 --unimodal text> log/598.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-599 --domain_type 1 --unimodal text> log/599.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-600 --domain_type 1 --unimodal text> log/600.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-601 --domain_type 1 --unimodal text> log/601.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-602 --domain_type 1 --unimodal text> log/602.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-603 --domain_type 1 --unimodal text> log/603.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-604 --domain_type 1 --unimodal text> log/604.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-605 --domain_type 1 --unimodal text> log/605.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-606 --domain_type 1 --unimodal text> log/606.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-607 --domain_type 1 --unimodal text> log/607.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-608 --domain_type 1 --unimodal text> log/608.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-609 --domain_type 1 --unimodal text> log/609.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-610 --domain_type 1 --unimodal text> log/610.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-611 --domain_type 1 --unimodal text> log/611.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-612 --domain_type 1 --unimodal text> log/612.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-613 --domain_type 1 --unimodal text> log/613.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-614 --domain_type 1 --unimodal text> log/614.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-615 --domain_type 1 --unimodal text> log/615.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-616 --domain_type 1 --unimodal text> log/616.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-617 --domain_type 1 --unimodal text> log/617.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-618 --domain_type 1 --unimodal text> log/618.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-619 --domain_type 1 --unimodal text> log/619.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-620 --domain_type 1 --unimodal text> log/620.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-621 --domain_type 1 --unimodal text> log/621.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-622 --domain_type 1 --unimodal text> log/622.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-623 --domain_type 1 --unimodal text> log/623.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-624 --domain_type 1 --unimodal text> log/624.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-625 --domain_type 1 --unimodal text> log/625.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-626 --domain_type 1 --unimodal text> log/626.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-627 --domain_type 1 --unimodal text> log/627.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-628 --domain_type 1 --unimodal text> log/628.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-629 --domain_type 1 --unimodal text> log/629.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-630 --domain_type 1 --unimodal text> log/630.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-631 --domain_type 1 --unimodal text> log/631.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-632 --domain_type 1 --unimodal text> log/632.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-633 --domain_type 1 --unimodal text> log/633.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-634 --domain_type 1 --unimodal text> log/634.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-635 --domain_type 1 --unimodal text> log/635.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-636 --domain_type 1 --unimodal text> log/636.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-637 --domain_type 1 --unimodal text> log/637.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-638 --domain_type 1 --unimodal text> log/638.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-639 --domain_type 1 --unimodal text> log/639.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-640 --domain_type 1 --unimodal text> log/640.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-641 --domain_type 1 --unimodal text> log/641.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-642 --domain_type 1 --unimodal text> log/642.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-643 --domain_type 1 --unimodal text> log/643.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-644 --domain_type 1 --unimodal text> log/644.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-645 --domain_type 1 --unimodal text> log/645.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-646 --domain_type 1 --unimodal text> log/646.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-647 --domain_type 1 --unimodal text> log/647.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-648 --domain_type 1 --unimodal text> log/648.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-649 --domain_type 1 --unimodal text> log/649.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-650 --domain_type 1 --unimodal text> log/650.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-651 --domain_type 1 --unimodal text> log/651.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-652 --domain_type 1 --unimodal text> log/652.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-653 --domain_type 1 --unimodal text> log/653.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-654 --domain_type 1 --unimodal text> log/654.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-655 --domain_type 1 --unimodal text> log/655.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-656 --domain_type 1 --unimodal text> log/656.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-657 --domain_type 1 --unimodal text> log/657.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-658 --domain_type 1 --unimodal text> log/658.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-659 --domain_type 1 --unimodal text> log/659.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-660 --domain_type 1 --unimodal text> log/660.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-661 --domain_type 1 --unimodal text> log/661.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-662 --domain_type 1 --unimodal text> log/662.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-663 --domain_type 1 --unimodal text> log/663.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-664 --domain_type 1 --unimodal text> log/664.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-665 --domain_type 1 --unimodal text> log/665.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-666 --domain_type 1 --unimodal text> log/666.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-667 --domain_type 1 --unimodal text> log/667.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-668 --domain_type 1 --unimodal text> log/668.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-669 --domain_type 1 --unimodal text> log/669.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-670 --domain_type 1 --unimodal text> log/670.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-671 --domain_type 1 --unimodal text> log/671.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-672 --domain_type 1 --unimodal text> log/672.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-673 --domain_type 1 --unimodal text> log/673.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-674 --domain_type 1 --unimodal text> log/674.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-675 --domain_type 1 --unimodal text> log/675.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-676 --domain_type 1 --unimodal text> log/676.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-677 --domain_type 1 --unimodal text> log/677.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-678 --domain_type 1 --unimodal text> log/678.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-679 --domain_type 1 --unimodal text> log/679.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-680 --domain_type 1 --unimodal text> log/680.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-681 --domain_type 1 --unimodal text> log/681.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-682 --domain_type 1 --unimodal text> log/682.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-683 --domain_type 1 --unimodal text> log/683.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-684 --domain_type 1 --unimodal text> log/684.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-685 --domain_type 1 --unimodal text> log/685.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-686 --domain_type 1 --unimodal text> log/686.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-687 --domain_type 1 --unimodal text> log/687.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-688 --domain_type 1 --unimodal text> log/688.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-689 --domain_type 1 --unimodal text> log/689.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-690 --domain_type 1 --unimodal text> log/690.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-691 --domain_type 1 --unimodal text> log/691.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-692 --domain_type 1 --unimodal text> log/692.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-693 --domain_type 1 --unimodal text> log/693.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-694 --domain_type 1 --unimodal text> log/694.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-695 --domain_type 1 --unimodal text> log/695.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-696 --domain_type 1 --unimodal text> log/696.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-697 --domain_type 1 --unimodal text> log/697.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-698 --domain_type 1 --unimodal text> log/698.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-699 --domain_type 1 --unimodal text> log/699.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-700 --domain_type 1 --unimodal text> log/700.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-701 --domain_type 1 --unimodal text> log/701.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-702 --domain_type 1 --unimodal text> log/702.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-703 --domain_type 1 --unimodal text> log/703.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-704 --domain_type 1 --unimodal text> log/704.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-705 --domain_type 1 --unimodal text> log/705.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-706 --domain_type 1 --unimodal text> log/706.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-707 --domain_type 1 --unimodal text> log/707.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-708 --domain_type 1 --unimodal text> log/708.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-709 --domain_type 1 --unimodal text> log/709.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-710 --domain_type 1 --unimodal text> log/710.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-711 --domain_type 1 --unimodal text> log/711.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-712 --domain_type 1 --unimodal text> log/712.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-713 --domain_type 1 --unimodal text> log/713.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-714 --domain_type 1 --unimodal text> log/714.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-715 --domain_type 1 --unimodal text> log/715.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-716 --domain_type 1 --unimodal text> log/716.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-717 --domain_type 1 --unimodal text> log/717.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-718 --domain_type 1 --unimodal text> log/718.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-719 --domain_type 1 --unimodal text> log/719.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-720 --domain_type 1 --unimodal text> log/720.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-721 --domain_type 1 --unimodal text> log/721.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-722 --domain_type 1 --unimodal text> log/722.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-723 --domain_type 1 --unimodal text> log/723.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-724 --domain_type 1 --unimodal text> log/724.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-725 --domain_type 1 --unimodal text> log/725.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-726 --domain_type 1 --unimodal text> log/726.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-727 --domain_type 1 --unimodal text> log/727.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-728 --domain_type 1 --unimodal text> log/728.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-729 --domain_type 1 --unimodal text> log/729.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-730 --domain_type 1 --unimodal text> log/730.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-731 --domain_type 1 --unimodal text> log/731.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-732 --domain_type 1 --unimodal text> log/732.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-733 --domain_type 1 --unimodal text> log/733.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-734 --domain_type 1 --unimodal text> log/734.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-735 --domain_type 1 --unimodal text> log/735.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-736 --domain_type 1 --unimodal text> log/736.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-737 --domain_type 1 --unimodal text> log/737.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-738 --domain_type 1 --unimodal text> log/738.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-739 --domain_type 1 --unimodal text> log/739.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-740 --domain_type 1 --unimodal text> log/740.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-741 --domain_type 1 --unimodal text> log/741.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-742 --domain_type 1 --unimodal text> log/742.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-743 --domain_type 1 --unimodal text> log/743.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-744 --domain_type 1 --unimodal text> log/744.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-745 --domain_type 1 --unimodal text> log/745.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-746 --domain_type 1 --unimodal text> log/746.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-747 --domain_type 1 --unimodal text> log/747.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-748 --domain_type 1 --unimodal text> log/748.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-749 --domain_type 1 --unimodal text> log/749.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-750 --domain_type 1 --unimodal text> log/750.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-751 --domain_type 1 --unimodal text> log/751.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-752 --domain_type 1 --unimodal text> log/752.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-753 --domain_type 1 --unimodal text> log/753.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-754 --domain_type 1 --unimodal text> log/754.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-755 --domain_type 1 --unimodal text> log/755.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-756 --domain_type 1 --unimodal text> log/756.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-757 --domain_type 1 --unimodal text> log/757.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-758 --domain_type 1 --unimodal text> log/758.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-759 --domain_type 1 --unimodal text> log/759.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-760 --domain_type 1 --unimodal text> log/760.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-761 --domain_type 1 --unimodal text> log/761.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-762 --domain_type 1 --unimodal text> log/762.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-763 --domain_type 1 --unimodal text> log/763.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-764 --domain_type 1 --unimodal text> log/764.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-765 --domain_type 1 --unimodal text> log/765.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-766 --domain_type 1 --unimodal text> log/766.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-767 --domain_type 1 --unimodal text> log/767.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-768 --domain_type 1 --unimodal text> log/768.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-769 --domain_type 1 --unimodal text> log/769.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-770 --domain_type 1 --unimodal text> log/770.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-771 --domain_type 1 --unimodal text> log/771.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-772 --domain_type 1 --unimodal text> log/772.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-773 --domain_type 1 --unimodal text> log/773.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-774 --domain_type 1 --unimodal text> log/774.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-775 --domain_type 1 --unimodal text> log/775.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-776 --domain_type 1 --unimodal text> log/776.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-777 --domain_type 1 --unimodal text> log/777.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-778 --domain_type 1 --unimodal text> log/778.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-779 --domain_type 1 --unimodal text> log/779.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-780 --domain_type 1 --unimodal text> log/780.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-781 --domain_type 1 --unimodal text> log/781.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-782 --domain_type 1 --unimodal text> log/782.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-783 --domain_type 1 --unimodal text> log/783.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-784 --domain_type 1 --unimodal text> log/784.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-785 --domain_type 1 --unimodal text> log/785.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-786 --domain_type 1 --unimodal text> log/786.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-787 --domain_type 1 --unimodal text> log/787.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-788 --domain_type 1 --unimodal text> log/788.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-789 --domain_type 1 --unimodal text> log/789.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-790 --domain_type 1 --unimodal text> log/790.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-791 --domain_type 1 --unimodal text> log/791.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-792 --domain_type 1 --unimodal text> log/792.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-793 --domain_type 1 --unimodal text> log/793.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-794 --domain_type 1 --unimodal text> log/794.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-795 --domain_type 1 --unimodal text> log/795.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-796 --domain_type 1 --unimodal text> log/796.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-797 --domain_type 1 --unimodal text> log/797.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-798 --domain_type 1 --unimodal text> log/798.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-799 --domain_type 1 --unimodal text> log/799.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-800 --domain_type 1 --unimodal text> log/800.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-801 --domain_type 1 --unimodal text> log/801.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-802 --domain_type 1 --unimodal text> log/802.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-803 --domain_type 1 --unimodal text> log/803.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-804 --domain_type 1 --unimodal text> log/804.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-805 --domain_type 1 --unimodal text> log/805.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-806 --domain_type 1 --unimodal text> log/806.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-807 --domain_type 1 --unimodal text> log/807.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-808 --domain_type 1 --unimodal text> log/808.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-809 --domain_type 1 --unimodal text> log/809.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-810 --domain_type 1 --unimodal text> log/810.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-811 --domain_type 1 --unimodal text> log/811.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-812 --domain_type 1 --unimodal text> log/812.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-813 --domain_type 1 --unimodal text> log/813.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-814 --domain_type 1 --unimodal text> log/814.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-815 --domain_type 1 --unimodal text> log/815.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-816 --domain_type 1 --unimodal text> log/816.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-817 --domain_type 1 --unimodal text> log/817.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-818 --domain_type 1 --unimodal text> log/818.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-819 --domain_type 1 --unimodal text> log/819.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-820 --domain_type 1 --unimodal text> log/820.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-821 --domain_type 1 --unimodal text> log/821.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-822 --domain_type 1 --unimodal text> log/822.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-823 --domain_type 1 --unimodal text> log/823.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-824 --domain_type 1 --unimodal text> log/824.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-825 --domain_type 1 --unimodal text> log/825.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-826 --domain_type 1 --unimodal text> log/826.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-827 --domain_type 1 --unimodal text> log/827.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-828 --domain_type 1 --unimodal text> log/828.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-829 --domain_type 1 --unimodal text> log/829.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-830 --domain_type 1 --unimodal text> log/830.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-831 --domain_type 1 --unimodal text> log/831.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-832 --domain_type 1 --unimodal text> log/832.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-833 --domain_type 1 --unimodal text> log/833.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-834 --domain_type 1 --unimodal text> log/834.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-835 --domain_type 1 --unimodal text> log/835.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-836 --domain_type 1 --unimodal text> log/836.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-837 --domain_type 1 --unimodal text> log/837.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-838 --domain_type 1 --unimodal text> log/838.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-839 --domain_type 1 --unimodal text> log/839.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-840 --domain_type 1 --unimodal text> log/840.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-841 --domain_type 1 --unimodal text> log/841.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-842 --domain_type 1 --unimodal text> log/842.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-843 --domain_type 1 --unimodal text> log/843.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-844 --domain_type 1 --unimodal text> log/844.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-845 --domain_type 1 --unimodal text> log/845.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-846 --domain_type 1 --unimodal text> log/846.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-847 --domain_type 1 --unimodal text> log/847.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-848 --domain_type 1 --unimodal text> log/848.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-849 --domain_type 1 --unimodal text> log/849.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-850 --domain_type 1 --unimodal text> log/850.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-851 --domain_type 1 --unimodal text> log/851.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-852 --domain_type 1 --unimodal text> log/852.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-853 --domain_type 1 --unimodal text> log/853.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-854 --domain_type 1 --unimodal text> log/854.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-855 --domain_type 1 --unimodal text> log/855.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-856 --domain_type 1 --unimodal text> log/856.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-857 --domain_type 1 --unimodal text> log/857.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-858 --domain_type 1 --unimodal text> log/858.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-859 --domain_type 1 --unimodal text> log/859.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-860 --domain_type 1 --unimodal text> log/860.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-861 --domain_type 1 --unimodal text> log/861.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-862 --domain_type 1 --unimodal text> log/862.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-863 --domain_type 1 --unimodal text> log/863.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-864 --domain_type 1 --unimodal text> log/864.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-865 --domain_type 1 --unimodal text> log/865.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-866 --domain_type 1 --unimodal text> log/866.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-867 --domain_type 1 --unimodal text> log/867.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-868 --domain_type 1 --unimodal text> log/868.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-869 --domain_type 1 --unimodal text> log/869.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-870 --domain_type 1 --unimodal text> log/870.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-871 --domain_type 1 --unimodal text> log/871.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-872 --domain_type 1 --unimodal text> log/872.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-873 --domain_type 1 --unimodal text> log/873.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-874 --domain_type 1 --unimodal text> log/874.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-875 --domain_type 1 --unimodal text> log/875.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-876 --domain_type 1 --unimodal text> log/876.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-877 --domain_type 1 --unimodal text> log/877.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-878 --domain_type 1 --unimodal text> log/878.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-879 --domain_type 1 --unimodal text> log/879.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-880 --domain_type 1 --unimodal text> log/880.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-881 --domain_type 1 --unimodal text> log/881.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-882 --domain_type 1 --unimodal text> log/882.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-883 --domain_type 1 --unimodal text> log/883.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-884 --domain_type 1 --unimodal text> log/884.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-885 --domain_type 1 --unimodal text> log/885.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-886 --domain_type 1 --unimodal text> log/886.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-887 --domain_type 1 --unimodal text> log/887.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-888 --domain_type 1 --unimodal text> log/888.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-889 --domain_type 1 --unimodal text> log/889.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-890 --domain_type 1 --unimodal text> log/890.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-891 --domain_type 1 --unimodal text> log/891.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-892 --domain_type 1 --unimodal text> log/892.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-893 --domain_type 1 --unimodal text> log/893.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-894 --domain_type 1 --unimodal text> log/894.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-895 --domain_type 1 --unimodal text> log/895.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-896 --domain_type 1 --unimodal text> log/896.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-897 --domain_type 1 --unimodal text> log/897.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-898 --domain_type 1 --unimodal text> log/898.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-899 --domain_type 1 --unimodal text> log/899.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-900 --domain_type 1 --unimodal text> log/900.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-901 --domain_type 1 --unimodal text> log/901.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-902 --domain_type 1 --unimodal text> log/902.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-903 --domain_type 1 --unimodal text> log/903.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-904 --domain_type 1 --unimodal text> log/904.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-905 --domain_type 1 --unimodal text> log/905.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-906 --domain_type 1 --unimodal text> log/906.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-907 --domain_type 1 --unimodal text> log/907.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-908 --domain_type 1 --unimodal text> log/908.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-909 --domain_type 1 --unimodal text> log/909.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-910 --domain_type 1 --unimodal text> log/910.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-911 --domain_type 1 --unimodal text> log/911.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-912 --domain_type 1 --unimodal text> log/912.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-913 --domain_type 1 --unimodal text> log/913.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-914 --domain_type 1 --unimodal text> log/914.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-915 --domain_type 1 --unimodal text> log/915.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-916 --domain_type 1 --unimodal text> log/916.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-917 --domain_type 1 --unimodal text> log/917.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-918 --domain_type 1 --unimodal text> log/918.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-919 --domain_type 1 --unimodal text> log/919.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-920 --domain_type 1 --unimodal text> log/920.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-921 --domain_type 1 --unimodal text> log/921.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-922 --domain_type 1 --unimodal text> log/922.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-923 --domain_type 1 --unimodal text> log/923.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-924 --domain_type 1 --unimodal text> log/924.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-925 --domain_type 1 --unimodal text> log/925.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-926 --domain_type 1 --unimodal text> log/926.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-927 --domain_type 1 --unimodal text> log/927.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-928 --domain_type 1 --unimodal text> log/928.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-929 --domain_type 1 --unimodal text> log/929.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-930 --domain_type 1 --unimodal text> log/930.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-931 --domain_type 1 --unimodal text> log/931.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-932 --domain_type 1 --unimodal text> log/932.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-933 --domain_type 1 --unimodal text> log/933.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-934 --domain_type 1 --unimodal text> log/934.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-935 --domain_type 1 --unimodal text> log/935.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-936 --domain_type 1 --unimodal text> log/936.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-937 --domain_type 1 --unimodal text> log/937.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-938 --domain_type 1 --unimodal text> log/938.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-939 --domain_type 1 --unimodal text> log/939.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-940 --domain_type 1 --unimodal text> log/940.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-941 --domain_type 1 --unimodal text> log/941.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-942 --domain_type 1 --unimodal text> log/942.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-943 --domain_type 1 --unimodal text> log/943.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-944 --domain_type 1 --unimodal text> log/944.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-945 --domain_type 1 --unimodal text> log/945.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-946 --domain_type 1 --unimodal text> log/946.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-947 --domain_type 1 --unimodal text> log/947.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-948 --domain_type 1 --unimodal text> log/948.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-949 --domain_type 1 --unimodal text> log/949.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-950 --domain_type 1 --unimodal text> log/950.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-951 --domain_type 1 --unimodal text> log/951.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-952 --domain_type 1 --unimodal text> log/952.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-953 --domain_type 1 --unimodal text> log/953.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-954 --domain_type 1 --unimodal text> log/954.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-955 --domain_type 1 --unimodal text> log/955.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-956 --domain_type 1 --unimodal text> log/956.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-957 --domain_type 1 --unimodal text> log/957.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-958 --domain_type 1 --unimodal text> log/958.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-959 --domain_type 1 --unimodal text> log/959.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-960 --domain_type 1 --unimodal text> log/960.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-961 --domain_type 1 --unimodal text> log/961.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-962 --domain_type 1 --unimodal text> log/962.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-963 --domain_type 1 --unimodal text> log/963.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-964 --domain_type 1 --unimodal text> log/964.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-965 --domain_type 1 --unimodal text> log/965.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-966 --domain_type 1 --unimodal text> log/966.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-967 --domain_type 1 --unimodal text> log/967.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-968 --domain_type 1 --unimodal text> log/968.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-969 --domain_type 1 --unimodal text> log/969.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-970 --domain_type 1 --unimodal text> log/970.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-971 --domain_type 1 --unimodal text> log/971.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-972 --domain_type 1 --unimodal text> log/972.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-973 --domain_type 1 --unimodal text> log/973.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-974 --domain_type 1 --unimodal text> log/974.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-975 --domain_type 1 --unimodal text> log/975.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-976 --domain_type 1 --unimodal text> log/976.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-977 --domain_type 1 --unimodal text> log/977.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-978 --domain_type 1 --unimodal text> log/978.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-979 --domain_type 1 --unimodal text> log/979.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-980 --domain_type 1 --unimodal text> log/980.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-981 --domain_type 1 --unimodal text> log/981.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-982 --domain_type 1 --unimodal text> log/982.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-983 --domain_type 1 --unimodal text> log/983.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-984 --domain_type 1 --unimodal text> log/984.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-985 --domain_type 1 --unimodal text> log/985.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-986 --domain_type 1 --unimodal text> log/986.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-987 --domain_type 1 --unimodal text> log/987.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-988 --domain_type 1 --unimodal text> log/988.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-989 --domain_type 1 --unimodal text> log/989.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-990 --domain_type 1 --unimodal text> log/990.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-991 --domain_type 1 --unimodal text> log/991.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-992 --domain_type 1 --unimodal text> log/992.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-993 --domain_type 1 --unimodal text> log/993.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-994 --domain_type 1 --unimodal text> log/994.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-995 --domain_type 1 --unimodal text> log/995.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-996 --domain_type 1 --unimodal text> log/996.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-997 --domain_type 1 --unimodal text> log/997.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-998 --domain_type 1 --unimodal text> log/998.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_unimodal.py --wandb_name mosi-a100-t-999 --domain_type 1 --unimodal text> log/999.log 2>&1 &
sleep 30
monitor_gpu_processes

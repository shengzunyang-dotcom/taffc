
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
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-0 --domain_type 2 > log/0.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-1 --domain_type 2 > log/1.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-2 --domain_type 2 > log/2.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-3 --domain_type 2 > log/3.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-4 --domain_type 2 > log/4.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-5 --domain_type 2 > log/5.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-6 --domain_type 2 > log/6.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-7 --domain_type 2 > log/7.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-8 --domain_type 2 > log/8.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-9 --domain_type 2 > log/9.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-10 --domain_type 2 > log/10.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-11 --domain_type 2 > log/11.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-12 --domain_type 2 > log/12.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-13 --domain_type 2 > log/13.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-14 --domain_type 2 > log/14.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-15 --domain_type 2 > log/15.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-16 --domain_type 2 > log/16.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-17 --domain_type 2 > log/17.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-18 --domain_type 2 > log/18.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-19 --domain_type 2 > log/19.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-20 --domain_type 2 > log/20.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-21 --domain_type 2 > log/21.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-22 --domain_type 2 > log/22.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-23 --domain_type 2 > log/23.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-24 --domain_type 2 > log/24.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-25 --domain_type 2 > log/25.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-26 --domain_type 2 > log/26.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-27 --domain_type 2 > log/27.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-28 --domain_type 2 > log/28.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-29 --domain_type 2 > log/29.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-30 --domain_type 2 > log/30.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-31 --domain_type 2 > log/31.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-32 --domain_type 2 > log/32.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-33 --domain_type 2 > log/33.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-34 --domain_type 2 > log/34.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-35 --domain_type 2 > log/35.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-36 --domain_type 2 > log/36.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-37 --domain_type 2 > log/37.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-38 --domain_type 2 > log/38.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-39 --domain_type 2 > log/39.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-40 --domain_type 2 > log/40.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-41 --domain_type 2 > log/41.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-42 --domain_type 2 > log/42.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-43 --domain_type 2 > log/43.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-44 --domain_type 2 > log/44.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-45 --domain_type 2 > log/45.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-46 --domain_type 2 > log/46.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-47 --domain_type 2 > log/47.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-48 --domain_type 2 > log/48.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-49 --domain_type 2 > log/49.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-50 --domain_type 2 > log/50.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-51 --domain_type 2 > log/51.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-52 --domain_type 2 > log/52.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-53 --domain_type 2 > log/53.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-54 --domain_type 2 > log/54.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-55 --domain_type 2 > log/55.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-56 --domain_type 2 > log/56.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-57 --domain_type 2 > log/57.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-58 --domain_type 2 > log/58.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-59 --domain_type 2 > log/59.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-60 --domain_type 2 > log/60.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-61 --domain_type 2 > log/61.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-62 --domain_type 2 > log/62.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-63 --domain_type 2 > log/63.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-64 --domain_type 2 > log/64.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-65 --domain_type 2 > log/65.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-66 --domain_type 2 > log/66.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-67 --domain_type 2 > log/67.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-68 --domain_type 2 > log/68.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-69 --domain_type 2 > log/69.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-70 --domain_type 2 > log/70.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-71 --domain_type 2 > log/71.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-72 --domain_type 2 > log/72.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-73 --domain_type 2 > log/73.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-74 --domain_type 2 > log/74.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-75 --domain_type 2 > log/75.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-76 --domain_type 2 > log/76.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-77 --domain_type 2 > log/77.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-78 --domain_type 2 > log/78.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-79 --domain_type 2 > log/79.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-80 --domain_type 2 > log/80.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-81 --domain_type 2 > log/81.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-82 --domain_type 2 > log/82.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-83 --domain_type 2 > log/83.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-84 --domain_type 2 > log/84.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-85 --domain_type 2 > log/85.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-86 --domain_type 2 > log/86.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-87 --domain_type 2 > log/87.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-88 --domain_type 2 > log/88.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-89 --domain_type 2 > log/89.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-90 --domain_type 2 > log/90.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-91 --domain_type 2 > log/91.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-92 --domain_type 2 > log/92.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-93 --domain_type 2 > log/93.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-94 --domain_type 2 > log/94.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-95 --domain_type 2 > log/95.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-96 --domain_type 2 > log/96.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-97 --domain_type 2 > log/97.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-98 --domain_type 2 > log/98.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-99 --domain_type 2 > log/99.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-100 --domain_type 2 > log/100.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-101 --domain_type 2 > log/101.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-102 --domain_type 2 > log/102.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-103 --domain_type 2 > log/103.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-104 --domain_type 2 > log/104.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-105 --domain_type 2 > log/105.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-106 --domain_type 2 > log/106.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-107 --domain_type 2 > log/107.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-108 --domain_type 2 > log/108.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-109 --domain_type 2 > log/109.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-110 --domain_type 2 > log/110.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-111 --domain_type 2 > log/111.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-112 --domain_type 2 > log/112.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-113 --domain_type 2 > log/113.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-114 --domain_type 2 > log/114.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-115 --domain_type 2 > log/115.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-116 --domain_type 2 > log/116.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-117 --domain_type 2 > log/117.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-118 --domain_type 2 > log/118.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-119 --domain_type 2 > log/119.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-120 --domain_type 2 > log/120.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-121 --domain_type 2 > log/121.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-122 --domain_type 2 > log/122.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-123 --domain_type 2 > log/123.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-124 --domain_type 2 > log/124.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-125 --domain_type 2 > log/125.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-126 --domain_type 2 > log/126.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-127 --domain_type 2 > log/127.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-128 --domain_type 2 > log/128.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-129 --domain_type 2 > log/129.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-130 --domain_type 2 > log/130.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-131 --domain_type 2 > log/131.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-132 --domain_type 2 > log/132.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-133 --domain_type 2 > log/133.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-134 --domain_type 2 > log/134.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-135 --domain_type 2 > log/135.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-136 --domain_type 2 > log/136.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-137 --domain_type 2 > log/137.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-138 --domain_type 2 > log/138.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-139 --domain_type 2 > log/139.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-140 --domain_type 2 > log/140.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-141 --domain_type 2 > log/141.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-142 --domain_type 2 > log/142.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-143 --domain_type 2 > log/143.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-144 --domain_type 2 > log/144.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-145 --domain_type 2 > log/145.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-146 --domain_type 2 > log/146.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-147 --domain_type 2 > log/147.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-148 --domain_type 2 > log/148.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-149 --domain_type 2 > log/149.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-150 --domain_type 2 > log/150.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-151 --domain_type 2 > log/151.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-152 --domain_type 2 > log/152.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-153 --domain_type 2 > log/153.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-154 --domain_type 2 > log/154.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-155 --domain_type 2 > log/155.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-156 --domain_type 2 > log/156.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-157 --domain_type 2 > log/157.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-158 --domain_type 2 > log/158.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-159 --domain_type 2 > log/159.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-160 --domain_type 2 > log/160.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-161 --domain_type 2 > log/161.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-162 --domain_type 2 > log/162.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-163 --domain_type 2 > log/163.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-164 --domain_type 2 > log/164.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-165 --domain_type 2 > log/165.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-166 --domain_type 2 > log/166.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-167 --domain_type 2 > log/167.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-168 --domain_type 2 > log/168.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-169 --domain_type 2 > log/169.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-170 --domain_type 2 > log/170.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-171 --domain_type 2 > log/171.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-172 --domain_type 2 > log/172.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-173 --domain_type 2 > log/173.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-174 --domain_type 2 > log/174.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-175 --domain_type 2 > log/175.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-176 --domain_type 2 > log/176.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-177 --domain_type 2 > log/177.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-178 --domain_type 2 > log/178.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-179 --domain_type 2 > log/179.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-180 --domain_type 2 > log/180.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-181 --domain_type 2 > log/181.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-182 --domain_type 2 > log/182.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-183 --domain_type 2 > log/183.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-184 --domain_type 2 > log/184.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-185 --domain_type 2 > log/185.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-186 --domain_type 2 > log/186.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-187 --domain_type 2 > log/187.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-188 --domain_type 2 > log/188.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-189 --domain_type 2 > log/189.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-190 --domain_type 2 > log/190.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-191 --domain_type 2 > log/191.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-192 --domain_type 2 > log/192.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-193 --domain_type 2 > log/193.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-194 --domain_type 2 > log/194.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-195 --domain_type 2 > log/195.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-196 --domain_type 2 > log/196.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-197 --domain_type 2 > log/197.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-198 --domain_type 2 > log/198.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-199 --domain_type 2 > log/199.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-200 --domain_type 2 > log/200.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-201 --domain_type 2 > log/201.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-202 --domain_type 2 > log/202.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-203 --domain_type 2 > log/203.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-204 --domain_type 2 > log/204.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-205 --domain_type 2 > log/205.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-206 --domain_type 2 > log/206.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-207 --domain_type 2 > log/207.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-208 --domain_type 2 > log/208.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-209 --domain_type 2 > log/209.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-210 --domain_type 2 > log/210.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-211 --domain_type 2 > log/211.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-212 --domain_type 2 > log/212.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-213 --domain_type 2 > log/213.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-214 --domain_type 2 > log/214.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-215 --domain_type 2 > log/215.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-216 --domain_type 2 > log/216.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-217 --domain_type 2 > log/217.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-218 --domain_type 2 > log/218.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-219 --domain_type 2 > log/219.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-220 --domain_type 2 > log/220.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-221 --domain_type 2 > log/221.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-222 --domain_type 2 > log/222.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-223 --domain_type 2 > log/223.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-224 --domain_type 2 > log/224.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-225 --domain_type 2 > log/225.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-226 --domain_type 2 > log/226.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-227 --domain_type 2 > log/227.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-228 --domain_type 2 > log/228.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-229 --domain_type 2 > log/229.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-230 --domain_type 2 > log/230.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-231 --domain_type 2 > log/231.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-232 --domain_type 2 > log/232.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-233 --domain_type 2 > log/233.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-234 --domain_type 2 > log/234.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-235 --domain_type 2 > log/235.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-236 --domain_type 2 > log/236.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-237 --domain_type 2 > log/237.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-238 --domain_type 2 > log/238.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-239 --domain_type 2 > log/239.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-240 --domain_type 2 > log/240.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-241 --domain_type 2 > log/241.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-242 --domain_type 2 > log/242.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-243 --domain_type 2 > log/243.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-244 --domain_type 2 > log/244.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-245 --domain_type 2 > log/245.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-246 --domain_type 2 > log/246.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-247 --domain_type 2 > log/247.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-248 --domain_type 2 > log/248.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-249 --domain_type 2 > log/249.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-250 --domain_type 2 > log/250.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-251 --domain_type 2 > log/251.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-252 --domain_type 2 > log/252.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-253 --domain_type 2 > log/253.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-254 --domain_type 2 > log/254.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-255 --domain_type 2 > log/255.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-256 --domain_type 2 > log/256.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-257 --domain_type 2 > log/257.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-258 --domain_type 2 > log/258.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-259 --domain_type 2 > log/259.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-260 --domain_type 2 > log/260.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-261 --domain_type 2 > log/261.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-262 --domain_type 2 > log/262.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-263 --domain_type 2 > log/263.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-264 --domain_type 2 > log/264.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-265 --domain_type 2 > log/265.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-266 --domain_type 2 > log/266.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-267 --domain_type 2 > log/267.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-268 --domain_type 2 > log/268.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-269 --domain_type 2 > log/269.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-270 --domain_type 2 > log/270.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-271 --domain_type 2 > log/271.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-272 --domain_type 2 > log/272.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-273 --domain_type 2 > log/273.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-274 --domain_type 2 > log/274.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-275 --domain_type 2 > log/275.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-276 --domain_type 2 > log/276.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-277 --domain_type 2 > log/277.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-278 --domain_type 2 > log/278.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-279 --domain_type 2 > log/279.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-280 --domain_type 2 > log/280.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-281 --domain_type 2 > log/281.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-282 --domain_type 2 > log/282.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-283 --domain_type 2 > log/283.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-284 --domain_type 2 > log/284.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-285 --domain_type 2 > log/285.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-286 --domain_type 2 > log/286.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-287 --domain_type 2 > log/287.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-288 --domain_type 2 > log/288.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-289 --domain_type 2 > log/289.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-290 --domain_type 2 > log/290.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-291 --domain_type 2 > log/291.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-292 --domain_type 2 > log/292.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-293 --domain_type 2 > log/293.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-294 --domain_type 2 > log/294.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-295 --domain_type 2 > log/295.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-296 --domain_type 2 > log/296.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-297 --domain_type 2 > log/297.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-298 --domain_type 2 > log/298.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-299 --domain_type 2 > log/299.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-300 --domain_type 2 > log/300.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-301 --domain_type 2 > log/301.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-302 --domain_type 2 > log/302.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-303 --domain_type 2 > log/303.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-304 --domain_type 2 > log/304.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-305 --domain_type 2 > log/305.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-306 --domain_type 2 > log/306.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-307 --domain_type 2 > log/307.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-308 --domain_type 2 > log/308.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-309 --domain_type 2 > log/309.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-310 --domain_type 2 > log/310.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-311 --domain_type 2 > log/311.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-312 --domain_type 2 > log/312.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-313 --domain_type 2 > log/313.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-314 --domain_type 2 > log/314.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-315 --domain_type 2 > log/315.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-316 --domain_type 2 > log/316.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-317 --domain_type 2 > log/317.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-318 --domain_type 2 > log/318.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-319 --domain_type 2 > log/319.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-320 --domain_type 2 > log/320.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-321 --domain_type 2 > log/321.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-322 --domain_type 2 > log/322.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-323 --domain_type 2 > log/323.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-324 --domain_type 2 > log/324.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-325 --domain_type 2 > log/325.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-326 --domain_type 2 > log/326.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-327 --domain_type 2 > log/327.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-328 --domain_type 2 > log/328.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-329 --domain_type 2 > log/329.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-330 --domain_type 2 > log/330.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-331 --domain_type 2 > log/331.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-332 --domain_type 2 > log/332.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-333 --domain_type 2 > log/333.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-334 --domain_type 2 > log/334.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-335 --domain_type 2 > log/335.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-336 --domain_type 2 > log/336.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-337 --domain_type 2 > log/337.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-338 --domain_type 2 > log/338.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-339 --domain_type 2 > log/339.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-340 --domain_type 2 > log/340.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-341 --domain_type 2 > log/341.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-342 --domain_type 2 > log/342.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-343 --domain_type 2 > log/343.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-344 --domain_type 2 > log/344.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-345 --domain_type 2 > log/345.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-346 --domain_type 2 > log/346.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-347 --domain_type 2 > log/347.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-348 --domain_type 2 > log/348.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-349 --domain_type 2 > log/349.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-350 --domain_type 2 > log/350.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-351 --domain_type 2 > log/351.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-352 --domain_type 2 > log/352.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-353 --domain_type 2 > log/353.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-354 --domain_type 2 > log/354.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-355 --domain_type 2 > log/355.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-356 --domain_type 2 > log/356.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-357 --domain_type 2 > log/357.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-358 --domain_type 2 > log/358.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-359 --domain_type 2 > log/359.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-360 --domain_type 2 > log/360.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-361 --domain_type 2 > log/361.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-362 --domain_type 2 > log/362.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-363 --domain_type 2 > log/363.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-364 --domain_type 2 > log/364.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-365 --domain_type 2 > log/365.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-366 --domain_type 2 > log/366.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-367 --domain_type 2 > log/367.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-368 --domain_type 2 > log/368.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-369 --domain_type 2 > log/369.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-370 --domain_type 2 > log/370.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-371 --domain_type 2 > log/371.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-372 --domain_type 2 > log/372.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-373 --domain_type 2 > log/373.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-374 --domain_type 2 > log/374.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-375 --domain_type 2 > log/375.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-376 --domain_type 2 > log/376.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-377 --domain_type 2 > log/377.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-378 --domain_type 2 > log/378.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-379 --domain_type 2 > log/379.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-380 --domain_type 2 > log/380.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-381 --domain_type 2 > log/381.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-382 --domain_type 2 > log/382.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-383 --domain_type 2 > log/383.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-384 --domain_type 2 > log/384.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-385 --domain_type 2 > log/385.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-386 --domain_type 2 > log/386.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-387 --domain_type 2 > log/387.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-388 --domain_type 2 > log/388.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-389 --domain_type 2 > log/389.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-390 --domain_type 2 > log/390.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-391 --domain_type 2 > log/391.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-392 --domain_type 2 > log/392.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-393 --domain_type 2 > log/393.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-394 --domain_type 2 > log/394.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-395 --domain_type 2 > log/395.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-396 --domain_type 2 > log/396.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-397 --domain_type 2 > log/397.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-398 --domain_type 2 > log/398.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-399 --domain_type 2 > log/399.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-400 --domain_type 2 > log/400.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-401 --domain_type 2 > log/401.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-402 --domain_type 2 > log/402.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-403 --domain_type 2 > log/403.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-404 --domain_type 2 > log/404.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-405 --domain_type 2 > log/405.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-406 --domain_type 2 > log/406.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-407 --domain_type 2 > log/407.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-408 --domain_type 2 > log/408.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-409 --domain_type 2 > log/409.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-410 --domain_type 2 > log/410.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-411 --domain_type 2 > log/411.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-412 --domain_type 2 > log/412.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-413 --domain_type 2 > log/413.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-414 --domain_type 2 > log/414.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-415 --domain_type 2 > log/415.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-416 --domain_type 2 > log/416.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-417 --domain_type 2 > log/417.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-418 --domain_type 2 > log/418.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-419 --domain_type 2 > log/419.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-420 --domain_type 2 > log/420.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-421 --domain_type 2 > log/421.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-422 --domain_type 2 > log/422.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-423 --domain_type 2 > log/423.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-424 --domain_type 2 > log/424.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-425 --domain_type 2 > log/425.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-426 --domain_type 2 > log/426.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-427 --domain_type 2 > log/427.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-428 --domain_type 2 > log/428.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-429 --domain_type 2 > log/429.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-430 --domain_type 2 > log/430.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-431 --domain_type 2 > log/431.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-432 --domain_type 2 > log/432.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-433 --domain_type 2 > log/433.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-434 --domain_type 2 > log/434.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-435 --domain_type 2 > log/435.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-436 --domain_type 2 > log/436.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-437 --domain_type 2 > log/437.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-438 --domain_type 2 > log/438.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-439 --domain_type 2 > log/439.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-440 --domain_type 2 > log/440.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-441 --domain_type 2 > log/441.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-442 --domain_type 2 > log/442.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-443 --domain_type 2 > log/443.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-444 --domain_type 2 > log/444.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-445 --domain_type 2 > log/445.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-446 --domain_type 2 > log/446.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-447 --domain_type 2 > log/447.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-448 --domain_type 2 > log/448.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-449 --domain_type 2 > log/449.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-450 --domain_type 2 > log/450.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-451 --domain_type 2 > log/451.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-452 --domain_type 2 > log/452.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-453 --domain_type 2 > log/453.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-454 --domain_type 2 > log/454.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-455 --domain_type 2 > log/455.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-456 --domain_type 2 > log/456.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-457 --domain_type 2 > log/457.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-458 --domain_type 2 > log/458.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-459 --domain_type 2 > log/459.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-460 --domain_type 2 > log/460.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-461 --domain_type 2 > log/461.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-462 --domain_type 2 > log/462.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-463 --domain_type 2 > log/463.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-464 --domain_type 2 > log/464.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-465 --domain_type 2 > log/465.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-466 --domain_type 2 > log/466.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-467 --domain_type 2 > log/467.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-468 --domain_type 2 > log/468.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-469 --domain_type 2 > log/469.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-470 --domain_type 2 > log/470.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-471 --domain_type 2 > log/471.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-472 --domain_type 2 > log/472.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-473 --domain_type 2 > log/473.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-474 --domain_type 2 > log/474.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-475 --domain_type 2 > log/475.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-476 --domain_type 2 > log/476.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-477 --domain_type 2 > log/477.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-478 --domain_type 2 > log/478.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-479 --domain_type 2 > log/479.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-480 --domain_type 2 > log/480.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-481 --domain_type 2 > log/481.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-482 --domain_type 2 > log/482.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-483 --domain_type 2 > log/483.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-484 --domain_type 2 > log/484.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-485 --domain_type 2 > log/485.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-486 --domain_type 2 > log/486.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-487 --domain_type 2 > log/487.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-488 --domain_type 2 > log/488.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-489 --domain_type 2 > log/489.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-490 --domain_type 2 > log/490.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-491 --domain_type 2 > log/491.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-492 --domain_type 2 > log/492.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-493 --domain_type 2 > log/493.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-494 --domain_type 2 > log/494.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-495 --domain_type 2 > log/495.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-496 --domain_type 2 > log/496.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-497 --domain_type 2 > log/497.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-498 --domain_type 2 > log/498.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-499 --domain_type 2 > log/499.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-500 --domain_type 2 > log/500.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-501 --domain_type 2 > log/501.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-502 --domain_type 2 > log/502.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-503 --domain_type 2 > log/503.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-504 --domain_type 2 > log/504.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-505 --domain_type 2 > log/505.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-506 --domain_type 2 > log/506.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-507 --domain_type 2 > log/507.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-508 --domain_type 2 > log/508.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-509 --domain_type 2 > log/509.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-510 --domain_type 2 > log/510.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-511 --domain_type 2 > log/511.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-512 --domain_type 2 > log/512.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-513 --domain_type 2 > log/513.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-514 --domain_type 2 > log/514.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-515 --domain_type 2 > log/515.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-516 --domain_type 2 > log/516.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-517 --domain_type 2 > log/517.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-518 --domain_type 2 > log/518.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-519 --domain_type 2 > log/519.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-520 --domain_type 2 > log/520.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-521 --domain_type 2 > log/521.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-522 --domain_type 2 > log/522.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-523 --domain_type 2 > log/523.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-524 --domain_type 2 > log/524.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-525 --domain_type 2 > log/525.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-526 --domain_type 2 > log/526.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-527 --domain_type 2 > log/527.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-528 --domain_type 2 > log/528.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-529 --domain_type 2 > log/529.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-530 --domain_type 2 > log/530.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-531 --domain_type 2 > log/531.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-532 --domain_type 2 > log/532.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-533 --domain_type 2 > log/533.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-534 --domain_type 2 > log/534.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-535 --domain_type 2 > log/535.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-536 --domain_type 2 > log/536.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-537 --domain_type 2 > log/537.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-538 --domain_type 2 > log/538.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-539 --domain_type 2 > log/539.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-540 --domain_type 2 > log/540.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-541 --domain_type 2 > log/541.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-542 --domain_type 2 > log/542.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-543 --domain_type 2 > log/543.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-544 --domain_type 2 > log/544.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-545 --domain_type 2 > log/545.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-546 --domain_type 2 > log/546.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-547 --domain_type 2 > log/547.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-548 --domain_type 2 > log/548.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-549 --domain_type 2 > log/549.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-550 --domain_type 2 > log/550.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-551 --domain_type 2 > log/551.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-552 --domain_type 2 > log/552.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-553 --domain_type 2 > log/553.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-554 --domain_type 2 > log/554.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-555 --domain_type 2 > log/555.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-556 --domain_type 2 > log/556.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-557 --domain_type 2 > log/557.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-558 --domain_type 2 > log/558.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-559 --domain_type 2 > log/559.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-560 --domain_type 2 > log/560.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-561 --domain_type 2 > log/561.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-562 --domain_type 2 > log/562.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-563 --domain_type 2 > log/563.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-564 --domain_type 2 > log/564.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-565 --domain_type 2 > log/565.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-566 --domain_type 2 > log/566.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-567 --domain_type 2 > log/567.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-568 --domain_type 2 > log/568.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-569 --domain_type 2 > log/569.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-570 --domain_type 2 > log/570.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-571 --domain_type 2 > log/571.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-572 --domain_type 2 > log/572.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-573 --domain_type 2 > log/573.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-574 --domain_type 2 > log/574.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-575 --domain_type 2 > log/575.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-576 --domain_type 2 > log/576.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-577 --domain_type 2 > log/577.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-578 --domain_type 2 > log/578.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-579 --domain_type 2 > log/579.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-580 --domain_type 2 > log/580.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-581 --domain_type 2 > log/581.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-582 --domain_type 2 > log/582.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-583 --domain_type 2 > log/583.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-584 --domain_type 2 > log/584.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-585 --domain_type 2 > log/585.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-586 --domain_type 2 > log/586.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-587 --domain_type 2 > log/587.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-588 --domain_type 2 > log/588.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-589 --domain_type 2 > log/589.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-590 --domain_type 2 > log/590.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-591 --domain_type 2 > log/591.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-592 --domain_type 2 > log/592.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-593 --domain_type 2 > log/593.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-594 --domain_type 2 > log/594.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-595 --domain_type 2 > log/595.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-596 --domain_type 2 > log/596.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-597 --domain_type 2 > log/597.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-598 --domain_type 2 > log/598.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-599 --domain_type 2 > log/599.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-600 --domain_type 2 > log/600.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-601 --domain_type 2 > log/601.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-602 --domain_type 2 > log/602.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-603 --domain_type 2 > log/603.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-604 --domain_type 2 > log/604.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-605 --domain_type 2 > log/605.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-606 --domain_type 2 > log/606.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-607 --domain_type 2 > log/607.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-608 --domain_type 2 > log/608.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-609 --domain_type 2 > log/609.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-610 --domain_type 2 > log/610.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-611 --domain_type 2 > log/611.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-612 --domain_type 2 > log/612.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-613 --domain_type 2 > log/613.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-614 --domain_type 2 > log/614.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-615 --domain_type 2 > log/615.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-616 --domain_type 2 > log/616.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-617 --domain_type 2 > log/617.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-618 --domain_type 2 > log/618.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-619 --domain_type 2 > log/619.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-620 --domain_type 2 > log/620.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-621 --domain_type 2 > log/621.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-622 --domain_type 2 > log/622.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-623 --domain_type 2 > log/623.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-624 --domain_type 2 > log/624.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-625 --domain_type 2 > log/625.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-626 --domain_type 2 > log/626.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-627 --domain_type 2 > log/627.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-628 --domain_type 2 > log/628.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-629 --domain_type 2 > log/629.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-630 --domain_type 2 > log/630.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-631 --domain_type 2 > log/631.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-632 --domain_type 2 > log/632.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-633 --domain_type 2 > log/633.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-634 --domain_type 2 > log/634.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-635 --domain_type 2 > log/635.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-636 --domain_type 2 > log/636.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-637 --domain_type 2 > log/637.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-638 --domain_type 2 > log/638.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-639 --domain_type 2 > log/639.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-640 --domain_type 2 > log/640.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-641 --domain_type 2 > log/641.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-642 --domain_type 2 > log/642.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-643 --domain_type 2 > log/643.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-644 --domain_type 2 > log/644.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-645 --domain_type 2 > log/645.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-646 --domain_type 2 > log/646.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-647 --domain_type 2 > log/647.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-648 --domain_type 2 > log/648.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-649 --domain_type 2 > log/649.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-650 --domain_type 2 > log/650.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-651 --domain_type 2 > log/651.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-652 --domain_type 2 > log/652.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-653 --domain_type 2 > log/653.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-654 --domain_type 2 > log/654.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-655 --domain_type 2 > log/655.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-656 --domain_type 2 > log/656.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-657 --domain_type 2 > log/657.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-658 --domain_type 2 > log/658.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-659 --domain_type 2 > log/659.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-660 --domain_type 2 > log/660.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-661 --domain_type 2 > log/661.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-662 --domain_type 2 > log/662.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-663 --domain_type 2 > log/663.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-664 --domain_type 2 > log/664.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-665 --domain_type 2 > log/665.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-666 --domain_type 2 > log/666.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-667 --domain_type 2 > log/667.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-668 --domain_type 2 > log/668.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-669 --domain_type 2 > log/669.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-670 --domain_type 2 > log/670.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-671 --domain_type 2 > log/671.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-672 --domain_type 2 > log/672.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-673 --domain_type 2 > log/673.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-674 --domain_type 2 > log/674.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-675 --domain_type 2 > log/675.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-676 --domain_type 2 > log/676.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-677 --domain_type 2 > log/677.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-678 --domain_type 2 > log/678.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-679 --domain_type 2 > log/679.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-680 --domain_type 2 > log/680.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-681 --domain_type 2 > log/681.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-682 --domain_type 2 > log/682.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-683 --domain_type 2 > log/683.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-684 --domain_type 2 > log/684.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-685 --domain_type 2 > log/685.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-686 --domain_type 2 > log/686.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-687 --domain_type 2 > log/687.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-688 --domain_type 2 > log/688.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-689 --domain_type 2 > log/689.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-690 --domain_type 2 > log/690.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-691 --domain_type 2 > log/691.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-692 --domain_type 2 > log/692.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-693 --domain_type 2 > log/693.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-694 --domain_type 2 > log/694.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-695 --domain_type 2 > log/695.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-696 --domain_type 2 > log/696.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-697 --domain_type 2 > log/697.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-698 --domain_type 2 > log/698.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-699 --domain_type 2 > log/699.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-700 --domain_type 2 > log/700.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-701 --domain_type 2 > log/701.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-702 --domain_type 2 > log/702.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-703 --domain_type 2 > log/703.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-704 --domain_type 2 > log/704.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-705 --domain_type 2 > log/705.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-706 --domain_type 2 > log/706.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-707 --domain_type 2 > log/707.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-708 --domain_type 2 > log/708.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-709 --domain_type 2 > log/709.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-710 --domain_type 2 > log/710.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-711 --domain_type 2 > log/711.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-712 --domain_type 2 > log/712.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-713 --domain_type 2 > log/713.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-714 --domain_type 2 > log/714.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-715 --domain_type 2 > log/715.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-716 --domain_type 2 > log/716.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-717 --domain_type 2 > log/717.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-718 --domain_type 2 > log/718.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-719 --domain_type 2 > log/719.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-720 --domain_type 2 > log/720.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-721 --domain_type 2 > log/721.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-722 --domain_type 2 > log/722.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-723 --domain_type 2 > log/723.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-724 --domain_type 2 > log/724.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-725 --domain_type 2 > log/725.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-726 --domain_type 2 > log/726.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-727 --domain_type 2 > log/727.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-728 --domain_type 2 > log/728.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-729 --domain_type 2 > log/729.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-730 --domain_type 2 > log/730.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-731 --domain_type 2 > log/731.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-732 --domain_type 2 > log/732.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-733 --domain_type 2 > log/733.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-734 --domain_type 2 > log/734.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-735 --domain_type 2 > log/735.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-736 --domain_type 2 > log/736.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-737 --domain_type 2 > log/737.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-738 --domain_type 2 > log/738.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-739 --domain_type 2 > log/739.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-740 --domain_type 2 > log/740.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-741 --domain_type 2 > log/741.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-742 --domain_type 2 > log/742.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-743 --domain_type 2 > log/743.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-744 --domain_type 2 > log/744.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-745 --domain_type 2 > log/745.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-746 --domain_type 2 > log/746.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-747 --domain_type 2 > log/747.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-748 --domain_type 2 > log/748.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-749 --domain_type 2 > log/749.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-750 --domain_type 2 > log/750.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-751 --domain_type 2 > log/751.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-752 --domain_type 2 > log/752.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-753 --domain_type 2 > log/753.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-754 --domain_type 2 > log/754.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-755 --domain_type 2 > log/755.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-756 --domain_type 2 > log/756.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-757 --domain_type 2 > log/757.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-758 --domain_type 2 > log/758.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-759 --domain_type 2 > log/759.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-760 --domain_type 2 > log/760.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-761 --domain_type 2 > log/761.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-762 --domain_type 2 > log/762.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-763 --domain_type 2 > log/763.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-764 --domain_type 2 > log/764.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-765 --domain_type 2 > log/765.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-766 --domain_type 2 > log/766.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-767 --domain_type 2 > log/767.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-768 --domain_type 2 > log/768.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-769 --domain_type 2 > log/769.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-770 --domain_type 2 > log/770.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-771 --domain_type 2 > log/771.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-772 --domain_type 2 > log/772.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-773 --domain_type 2 > log/773.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-774 --domain_type 2 > log/774.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-775 --domain_type 2 > log/775.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-776 --domain_type 2 > log/776.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-777 --domain_type 2 > log/777.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-778 --domain_type 2 > log/778.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-779 --domain_type 2 > log/779.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-780 --domain_type 2 > log/780.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-781 --domain_type 2 > log/781.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-782 --domain_type 2 > log/782.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-783 --domain_type 2 > log/783.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-784 --domain_type 2 > log/784.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-785 --domain_type 2 > log/785.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-786 --domain_type 2 > log/786.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-787 --domain_type 2 > log/787.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-788 --domain_type 2 > log/788.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-789 --domain_type 2 > log/789.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-790 --domain_type 2 > log/790.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-791 --domain_type 2 > log/791.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-792 --domain_type 2 > log/792.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-793 --domain_type 2 > log/793.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-794 --domain_type 2 > log/794.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-795 --domain_type 2 > log/795.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-796 --domain_type 2 > log/796.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-797 --domain_type 2 > log/797.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-798 --domain_type 2 > log/798.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-799 --domain_type 2 > log/799.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-800 --domain_type 2 > log/800.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-801 --domain_type 2 > log/801.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-802 --domain_type 2 > log/802.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-803 --domain_type 2 > log/803.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-804 --domain_type 2 > log/804.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-805 --domain_type 2 > log/805.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-806 --domain_type 2 > log/806.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-807 --domain_type 2 > log/807.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-808 --domain_type 2 > log/808.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-809 --domain_type 2 > log/809.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-810 --domain_type 2 > log/810.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-811 --domain_type 2 > log/811.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-812 --domain_type 2 > log/812.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-813 --domain_type 2 > log/813.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-814 --domain_type 2 > log/814.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-815 --domain_type 2 > log/815.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-816 --domain_type 2 > log/816.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-817 --domain_type 2 > log/817.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-818 --domain_type 2 > log/818.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-819 --domain_type 2 > log/819.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-820 --domain_type 2 > log/820.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-821 --domain_type 2 > log/821.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-822 --domain_type 2 > log/822.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-823 --domain_type 2 > log/823.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-824 --domain_type 2 > log/824.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-825 --domain_type 2 > log/825.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-826 --domain_type 2 > log/826.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-827 --domain_type 2 > log/827.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-828 --domain_type 2 > log/828.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-829 --domain_type 2 > log/829.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-830 --domain_type 2 > log/830.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-831 --domain_type 2 > log/831.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-832 --domain_type 2 > log/832.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-833 --domain_type 2 > log/833.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-834 --domain_type 2 > log/834.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-835 --domain_type 2 > log/835.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-836 --domain_type 2 > log/836.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-837 --domain_type 2 > log/837.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-838 --domain_type 2 > log/838.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-839 --domain_type 2 > log/839.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-840 --domain_type 2 > log/840.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-841 --domain_type 2 > log/841.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-842 --domain_type 2 > log/842.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-843 --domain_type 2 > log/843.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-844 --domain_type 2 > log/844.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-845 --domain_type 2 > log/845.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-846 --domain_type 2 > log/846.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-847 --domain_type 2 > log/847.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-848 --domain_type 2 > log/848.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-849 --domain_type 2 > log/849.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-850 --domain_type 2 > log/850.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-851 --domain_type 2 > log/851.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-852 --domain_type 2 > log/852.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-853 --domain_type 2 > log/853.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-854 --domain_type 2 > log/854.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-855 --domain_type 2 > log/855.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-856 --domain_type 2 > log/856.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-857 --domain_type 2 > log/857.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-858 --domain_type 2 > log/858.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-859 --domain_type 2 > log/859.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-860 --domain_type 2 > log/860.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-861 --domain_type 2 > log/861.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-862 --domain_type 2 > log/862.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-863 --domain_type 2 > log/863.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-864 --domain_type 2 > log/864.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-865 --domain_type 2 > log/865.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-866 --domain_type 2 > log/866.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-867 --domain_type 2 > log/867.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-868 --domain_type 2 > log/868.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-869 --domain_type 2 > log/869.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-870 --domain_type 2 > log/870.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-871 --domain_type 2 > log/871.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-872 --domain_type 2 > log/872.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-873 --domain_type 2 > log/873.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-874 --domain_type 2 > log/874.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-875 --domain_type 2 > log/875.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-876 --domain_type 2 > log/876.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-877 --domain_type 2 > log/877.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-878 --domain_type 2 > log/878.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-879 --domain_type 2 > log/879.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-880 --domain_type 2 > log/880.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-881 --domain_type 2 > log/881.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-882 --domain_type 2 > log/882.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-883 --domain_type 2 > log/883.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-884 --domain_type 2 > log/884.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-885 --domain_type 2 > log/885.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-886 --domain_type 2 > log/886.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-887 --domain_type 2 > log/887.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-888 --domain_type 2 > log/888.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-889 --domain_type 2 > log/889.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-890 --domain_type 2 > log/890.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-891 --domain_type 2 > log/891.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-892 --domain_type 2 > log/892.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-893 --domain_type 2 > log/893.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-894 --domain_type 2 > log/894.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-895 --domain_type 2 > log/895.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-896 --domain_type 2 > log/896.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-897 --domain_type 2 > log/897.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-898 --domain_type 2 > log/898.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-899 --domain_type 2 > log/899.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-900 --domain_type 2 > log/900.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-901 --domain_type 2 > log/901.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-902 --domain_type 2 > log/902.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-903 --domain_type 2 > log/903.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-904 --domain_type 2 > log/904.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-905 --domain_type 2 > log/905.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-906 --domain_type 2 > log/906.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-907 --domain_type 2 > log/907.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-908 --domain_type 2 > log/908.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-909 --domain_type 2 > log/909.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-910 --domain_type 2 > log/910.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-911 --domain_type 2 > log/911.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-912 --domain_type 2 > log/912.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-913 --domain_type 2 > log/913.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-914 --domain_type 2 > log/914.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-915 --domain_type 2 > log/915.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-916 --domain_type 2 > log/916.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-917 --domain_type 2 > log/917.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-918 --domain_type 2 > log/918.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-919 --domain_type 2 > log/919.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-920 --domain_type 2 > log/920.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-921 --domain_type 2 > log/921.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-922 --domain_type 2 > log/922.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-923 --domain_type 2 > log/923.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-924 --domain_type 2 > log/924.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-925 --domain_type 2 > log/925.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-926 --domain_type 2 > log/926.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-927 --domain_type 2 > log/927.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-928 --domain_type 2 > log/928.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-929 --domain_type 2 > log/929.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-930 --domain_type 2 > log/930.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-931 --domain_type 2 > log/931.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-932 --domain_type 2 > log/932.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-933 --domain_type 2 > log/933.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-934 --domain_type 2 > log/934.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-935 --domain_type 2 > log/935.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-936 --domain_type 2 > log/936.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-937 --domain_type 2 > log/937.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-938 --domain_type 2 > log/938.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-939 --domain_type 2 > log/939.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-940 --domain_type 2 > log/940.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-941 --domain_type 2 > log/941.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-942 --domain_type 2 > log/942.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-943 --domain_type 2 > log/943.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-944 --domain_type 2 > log/944.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-945 --domain_type 2 > log/945.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-946 --domain_type 2 > log/946.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-947 --domain_type 2 > log/947.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-948 --domain_type 2 > log/948.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-949 --domain_type 2 > log/949.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-950 --domain_type 2 > log/950.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-951 --domain_type 2 > log/951.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-952 --domain_type 2 > log/952.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-953 --domain_type 2 > log/953.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-954 --domain_type 2 > log/954.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-955 --domain_type 2 > log/955.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-956 --domain_type 2 > log/956.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-957 --domain_type 2 > log/957.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-958 --domain_type 2 > log/958.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-959 --domain_type 2 > log/959.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-960 --domain_type 2 > log/960.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-961 --domain_type 2 > log/961.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-962 --domain_type 2 > log/962.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-963 --domain_type 2 > log/963.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-964 --domain_type 2 > log/964.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-965 --domain_type 2 > log/965.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-966 --domain_type 2 > log/966.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-967 --domain_type 2 > log/967.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-968 --domain_type 2 > log/968.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-969 --domain_type 2 > log/969.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-970 --domain_type 2 > log/970.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-971 --domain_type 2 > log/971.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-972 --domain_type 2 > log/972.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-973 --domain_type 2 > log/973.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-974 --domain_type 2 > log/974.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-975 --domain_type 2 > log/975.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-976 --domain_type 2 > log/976.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-977 --domain_type 2 > log/977.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-978 --domain_type 2 > log/978.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-979 --domain_type 2 > log/979.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-980 --domain_type 2 > log/980.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-981 --domain_type 2 > log/981.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-982 --domain_type 2 > log/982.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-983 --domain_type 2 > log/983.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-984 --domain_type 2 > log/984.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-985 --domain_type 2 > log/985.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-986 --domain_type 2 > log/986.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-987 --domain_type 2 > log/987.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-988 --domain_type 2 > log/988.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-989 --domain_type 2 > log/989.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-990 --domain_type 2 > log/990.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-991 --domain_type 2 > log/991.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-992 --domain_type 2 > log/992.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-993 --domain_type 2 > log/993.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-994 --domain_type 2 > log/994.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-995 --domain_type 2 > log/995.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-996 --domain_type 2 > log/996.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-997 --domain_type 2 > log/997.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=2; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-998 --domain_type 2 > log/998.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_knowledge_mosi_mosei.py --wandb_name mosei-A100-999 --domain_type 2 > log/999.log 2>&1 &
sleep 30
monitor_gpu_processes

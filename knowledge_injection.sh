monitor_gpu_processes() {
    while true; do
        # 使用nvidia-smi命令获取指定GPU上正在运行的进程数量
        gpu_processes=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | wc -l)

        if [ "$gpu_processes" -gt 0 ]; then
            echo "GPU $gpu_index 上有 $gpu_processes 个程序在运行，将睡眠3分钟..."
            sleep 180  # 睡眠3分钟
        else
            echo "GPU $gpu_index 上没有程序在运行，退出循环。"
            break
        fi
    done
}
export CUDA_VISIBLE_DEVICES=1; nohup python run_sparse_mm_m.py --wandb_name m-multimodal-freeze-1-1 --domain_type 1  --freeze freeze --unimodal multimodal > log/1.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_sparse_mm_m.py --wandb_name m-multimodal-freeze-1-2 --domain_type 1  --freeze freeze --unimodal multimodal > log/2.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_sparse_mm_m.py --wandb_name m-multimodal-freeze-1-3 --domain_type 1  --freeze freeze --unimodal multimodal > log/3.log 2>&1 &
export CUDA_VISIBLE_DEVICES=0; nohup python run_sparse_mm_m.py --wandb_name m-multimodal-freeze-1-4 --domain_type 1  --freeze freeze --unimodal multimodal > log/4.log 2>&1 &
sleep 10
export CUDA_VISIBLE_DEVICES=1; nohup python run_sparse_mm_m.py --wandb_name m-multimodal-freeze-2-1 --domain_type 2  --freeze freeze --unimodal multimodal > log/5.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_sparse_mm_m.py --wandb_name m-multimodal-freeze-2-2 --domain_type 2  --freeze freeze --unimodal multimodal > log/6.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_sparse_mm_m.py --wandb_name m-multimodal-freeze-2-3 --domain_type 2  --freeze freeze --unimodal multimodal > log/7.log 2>&1 &
export CUDA_VISIBLE_DEVICES=0; nohup python run_sparse_mm_m.py --wandb_name m-multimodal-freeze-2-4 --domain_type 2  --freeze freeze --unimodal multimodal > log/8.log 2>&1 &
sleep 180
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=1; nohup python run_sparse_mm_m.py --wandb_name m-multimodal-freeze-3-1 --domain_type 3  --freeze freeze --unimodal multimodal > log/9.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_sparse_mm_m.py --wandb_name m-multimodal-freeze-3-2 --domain_type 3  --freeze freeze --unimodal multimodal > log/10.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_sparse_mm_m.py --wandb_name m-multimodal-freeze-3-3 --domain_type 3  --freeze freeze --unimodal multimodal > log/11.log 2>&1 &
export CUDA_VISIBLE_DEVICES=0; nohup python run_sparse_mm_m.py --wandb_name m-multimodal-freeze-3-4 --domain_type 3  --freeze freeze --unimodal multimodal > log/12.log 2>&1 &
sleep 10
export CUDA_VISIBLE_DEVICES=1; nohup python run_sparse_mm_m.py --wandb_name m-multimodal-freeze-4-1 --domain_type 4  --freeze freeze --unimodal multimodal > log/13.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_sparse_mm_m.py --wandb_name m-multimodal-freeze-4-2 --domain_type 4  --freeze freeze --unimodal multimodal > log/14.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_sparse_mm_m.py --wandb_name m-multimodal-freeze-4-3 --domain_type 4  --freeze freeze --unimodal multimodal > log/15.log 2>&1 &
export CUDA_VISIBLE_DEVICES=0; nohup python run_sparse_mm_m.py --wandb_name m-multimodal-freeze-4-4 --domain_type 4  --freeze freeze --unimodal multimodal > log/16.log 2>&1 &
sleep 180
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=1; nohup python run_sparse_mm_m.py --wandb_name m-multimodal-freeze-5-1 --domain_type 5  --freeze freeze --unimodal multimodal > log/17.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_sparse_mm_m.py --wandb_name m-multimodal-freeze-5-2 --domain_type 5  --freeze freeze --unimodal multimodal > log/18.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_sparse_mm_m.py --wandb_name m-multimodal-freeze-5-3 --domain_type 5  --freeze freeze --unimodal multimodal > log/19.log 2>&1 &
export CUDA_VISIBLE_DEVICES=0; nohup python run_sparse_mm_m.py --wandb_name m-multimodal-freeze-5-4 --domain_type 5  --freeze freeze --unimodal multimodal > log/20.log 2>&1 &
sleep 10
export CUDA_VISIBLE_DEVICES=1; nohup python run_sparse_mm_m.py --wandb_name m-multimodal-freeze-6-1 --domain_type 6  --freeze freeze --unimodal multimodal > log/21.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_sparse_mm_m.py --wandb_name m-multimodal-freeze-6-2 --domain_type 6  --freeze freeze --unimodal multimodal > log/22.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_sparse_mm_m.py --wandb_name m-multimodal-freeze-6-3 --domain_type 6  --freeze freeze --unimodal multimodal > log/23.log 2>&1 &
export CUDA_VISIBLE_DEVICES=0; nohup python run_sparse_mm_m.py --wandb_name m-multimodal-freeze-6-4 --domain_type 6  --freeze freeze --unimodal multimodal > log/24.log 2>&1 &
sleep 180
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=1; nohup python run_sparse_mm_m.py --wandb_name m-multimodal-wogl-freeze-1-1 --domain_type 1  --freeze freeze --unimodal multimodal-wogl > log/25.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_sparse_mm_m.py --wandb_name m-multimodal-wogl-freeze-1-2 --domain_type 1  --freeze freeze --unimodal multimodal-wogl > log/26.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_sparse_mm_m.py --wandb_name m-multimodal-wogl-freeze-1-3 --domain_type 1  --freeze freeze --unimodal multimodal-wogl > log/27.log 2>&1 &
export CUDA_VISIBLE_DEVICES=0; nohup python run_sparse_mm_m.py --wandb_name m-multimodal-wogl-freeze-1-4 --domain_type 1  --freeze freeze --unimodal multimodal-wogl > log/28.log 2>&1 &
sleep 10
export CUDA_VISIBLE_DEVICES=1; nohup python run_sparse_mm_m.py --wandb_name m-multimodal-wogl-freeze-2-1 --domain_type 2  --freeze freeze --unimodal multimodal-wogl > log/29.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_sparse_mm_m.py --wandb_name m-multimodal-wogl-freeze-2-2 --domain_type 2  --freeze freeze --unimodal multimodal-wogl > log/30.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_sparse_mm_m.py --wandb_name m-multimodal-wogl-freeze-2-3 --domain_type 2  --freeze freeze --unimodal multimodal-wogl > log/31.log 2>&1 &
export CUDA_VISIBLE_DEVICES=0; nohup python run_sparse_mm_m.py --wandb_name m-multimodal-wogl-freeze-2-4 --domain_type 2  --freeze freeze --unimodal multimodal-wogl > log/32.log 2>&1 &
sleep 180
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=1; nohup python run_sparse_mm_m.py --wandb_name m-multimodal-wogl-freeze-3-1 --domain_type 3  --freeze freeze --unimodal multimodal-wogl > log/33.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_sparse_mm_m.py --wandb_name m-multimodal-wogl-freeze-3-2 --domain_type 3  --freeze freeze --unimodal multimodal-wogl > log/34.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_sparse_mm_m.py --wandb_name m-multimodal-wogl-freeze-3-3 --domain_type 3  --freeze freeze --unimodal multimodal-wogl > log/35.log 2>&1 &
export CUDA_VISIBLE_DEVICES=0; nohup python run_sparse_mm_m.py --wandb_name m-multimodal-wogl-freeze-3-4 --domain_type 3  --freeze freeze --unimodal multimodal-wogl > log/36.log 2>&1 &
sleep 10
export CUDA_VISIBLE_DEVICES=1; nohup python run_sparse_mm_m.py --wandb_name m-multimodal-wogl-freeze-4-1 --domain_type 4  --freeze freeze --unimodal multimodal-wogl > log/37.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_sparse_mm_m.py --wandb_name m-multimodal-wogl-freeze-4-2 --domain_type 4  --freeze freeze --unimodal multimodal-wogl > log/38.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_sparse_mm_m.py --wandb_name m-multimodal-wogl-freeze-4-3 --domain_type 4  --freeze freeze --unimodal multimodal-wogl > log/39.log 2>&1 &
export CUDA_VISIBLE_DEVICES=0; nohup python run_sparse_mm_m.py --wandb_name m-multimodal-wogl-freeze-4-4 --domain_type 4  --freeze freeze --unimodal multimodal-wogl > log/40.log 2>&1 &
sleep 180
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=1; nohup python run_sparse_mm_m.py --wandb_name m-multimodal-wogl-freeze-5-1 --domain_type 5  --freeze freeze --unimodal multimodal-wogl > log/41.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_sparse_mm_m.py --wandb_name m-multimodal-wogl-freeze-5-2 --domain_type 5  --freeze freeze --unimodal multimodal-wogl > log/42.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_sparse_mm_m.py --wandb_name m-multimodal-wogl-freeze-5-3 --domain_type 5  --freeze freeze --unimodal multimodal-wogl > log/43.log 2>&1 &
export CUDA_VISIBLE_DEVICES=0; nohup python run_sparse_mm_m.py --wandb_name m-multimodal-wogl-freeze-5-4 --domain_type 5  --freeze freeze --unimodal multimodal-wogl > log/44.log 2>&1 &
sleep 10
export CUDA_VISIBLE_DEVICES=1; nohup python run_sparse_mm_m.py --wandb_name m-multimodal-wogl-freeze-6-1 --domain_type 6  --freeze freeze --unimodal multimodal-wogl > log/45.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_sparse_mm_m.py --wandb_name m-multimodal-wogl-freeze-6-2 --domain_type 6  --freeze freeze --unimodal multimodal-wogl > log/46.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_sparse_mm_m.py --wandb_name m-multimodal-wogl-freeze-6-3 --domain_type 6  --freeze freeze --unimodal multimodal-wogl > log/47.log 2>&1 &
export CUDA_VISIBLE_DEVICES=0; nohup python run_sparse_mm_m.py --wandb_name m-multimodal-wogl-freeze-6-4 --domain_type 6  --freeze freeze --unimodal multimodal-wogl > log/48.log 2>&1 &
sleep 180
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=1; nohup python run_sparse_mm_m.py --wandb_name m-text-freeze-1-1 --domain_type 1  --freeze freeze --unimodal text > log/49.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_sparse_mm_m.py --wandb_name m-text-freeze-1-2 --domain_type 1  --freeze freeze --unimodal text > log/50.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_sparse_mm_m.py --wandb_name m-text-freeze-1-3 --domain_type 1  --freeze freeze --unimodal text > log/51.log 2>&1 &
export CUDA_VISIBLE_DEVICES=0; nohup python run_sparse_mm_m.py --wandb_name m-text-freeze-1-4 --domain_type 1  --freeze freeze --unimodal text > log/52.log 2>&1 &
sleep 10
export CUDA_VISIBLE_DEVICES=1; nohup python run_sparse_mm_m.py --wandb_name m-text-freeze-2-1 --domain_type 2  --freeze freeze --unimodal text > log/53.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_sparse_mm_m.py --wandb_name m-text-freeze-2-2 --domain_type 2  --freeze freeze --unimodal text > log/54.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_sparse_mm_m.py --wandb_name m-text-freeze-2-3 --domain_type 2  --freeze freeze --unimodal text > log/55.log 2>&1 &
export CUDA_VISIBLE_DEVICES=0; nohup python run_sparse_mm_m.py --wandb_name m-text-freeze-2-4 --domain_type 2  --freeze freeze --unimodal text > log/56.log 2>&1 &
sleep 180
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=1; nohup python run_sparse_mm_m.py --wandb_name m-text-freeze-3-1 --domain_type 3  --freeze freeze --unimodal text > log/57.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_sparse_mm_m.py --wandb_name m-text-freeze-3-2 --domain_type 3  --freeze freeze --unimodal text > log/58.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_sparse_mm_m.py --wandb_name m-text-freeze-3-3 --domain_type 3  --freeze freeze --unimodal text > log/59.log 2>&1 &
export CUDA_VISIBLE_DEVICES=0; nohup python run_sparse_mm_m.py --wandb_name m-text-freeze-3-4 --domain_type 3  --freeze freeze --unimodal text > log/60.log 2>&1 &
sleep 10
export CUDA_VISIBLE_DEVICES=1; nohup python run_sparse_mm_m.py --wandb_name m-text-freeze-4-1 --domain_type 4  --freeze freeze --unimodal text > log/61.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_sparse_mm_m.py --wandb_name m-text-freeze-4-2 --domain_type 4  --freeze freeze --unimodal text > log/62.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_sparse_mm_m.py --wandb_name m-text-freeze-4-3 --domain_type 4  --freeze freeze --unimodal text > log/63.log 2>&1 &
export CUDA_VISIBLE_DEVICES=0; nohup python run_sparse_mm_m.py --wandb_name m-text-freeze-4-4 --domain_type 4  --freeze freeze --unimodal text > log/64.log 2>&1 &
sleep 180
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=1; nohup python run_sparse_mm_m.py --wandb_name m-text-freeze-5-1 --domain_type 5  --freeze freeze --unimodal text > log/65.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_sparse_mm_m.py --wandb_name m-text-freeze-5-2 --domain_type 5  --freeze freeze --unimodal text > log/66.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_sparse_mm_m.py --wandb_name m-text-freeze-5-3 --domain_type 5  --freeze freeze --unimodal text > log/67.log 2>&1 &
export CUDA_VISIBLE_DEVICES=0; nohup python run_sparse_mm_m.py --wandb_name m-text-freeze-5-4 --domain_type 5  --freeze freeze --unimodal text > log/68.log 2>&1 &
sleep 10
export CUDA_VISIBLE_DEVICES=1; nohup python run_sparse_mm_m.py --wandb_name m-text-freeze-6-1 --domain_type 6  --freeze freeze --unimodal text > log/69.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_sparse_mm_m.py --wandb_name m-text-freeze-6-2 --domain_type 6  --freeze freeze --unimodal text > log/70.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_sparse_mm_m.py --wandb_name m-text-freeze-6-3 --domain_type 6  --freeze freeze --unimodal text > log/71.log 2>&1 &
export CUDA_VISIBLE_DEVICES=0; nohup python run_sparse_mm_m.py --wandb_name m-text-freeze-6-4 --domain_type 6  --freeze freeze --unimodal text > log/72.log 2>&1 &
sleep 180
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=1; nohup python run_sparse_mm_m.py --wandb_name m-visual-freeze-1-1 --domain_type 1  --freeze freeze --unimodal visual > log/73.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_sparse_mm_m.py --wandb_name m-visual-freeze-1-2 --domain_type 1  --freeze freeze --unimodal visual > log/74.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_sparse_mm_m.py --wandb_name m-visual-freeze-1-3 --domain_type 1  --freeze freeze --unimodal visual > log/75.log 2>&1 &
export CUDA_VISIBLE_DEVICES=0; nohup python run_sparse_mm_m.py --wandb_name m-visual-freeze-1-4 --domain_type 1  --freeze freeze --unimodal visual > log/76.log 2>&1 &
sleep 10
export CUDA_VISIBLE_DEVICES=1; nohup python run_sparse_mm_m.py --wandb_name m-visual-freeze-2-1 --domain_type 2  --freeze freeze --unimodal visual > log/77.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_sparse_mm_m.py --wandb_name m-visual-freeze-2-2 --domain_type 2  --freeze freeze --unimodal visual > log/78.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_sparse_mm_m.py --wandb_name m-visual-freeze-2-3 --domain_type 2  --freeze freeze --unimodal visual > log/79.log 2>&1 &
export CUDA_VISIBLE_DEVICES=0; nohup python run_sparse_mm_m.py --wandb_name m-visual-freeze-2-4 --domain_type 2  --freeze freeze --unimodal visual > log/80.log 2>&1 &
sleep 180
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=1; nohup python run_sparse_mm_m.py --wandb_name m-visual-freeze-3-1 --domain_type 3  --freeze freeze --unimodal visual > log/81.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_sparse_mm_m.py --wandb_name m-visual-freeze-3-2 --domain_type 3  --freeze freeze --unimodal visual > log/82.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_sparse_mm_m.py --wandb_name m-visual-freeze-3-3 --domain_type 3  --freeze freeze --unimodal visual > log/83.log 2>&1 &
export CUDA_VISIBLE_DEVICES=0; nohup python run_sparse_mm_m.py --wandb_name m-visual-freeze-3-4 --domain_type 3  --freeze freeze --unimodal visual > log/84.log 2>&1 &
sleep 10
export CUDA_VISIBLE_DEVICES=1; nohup python run_sparse_mm_m.py --wandb_name m-visual-freeze-4-1 --domain_type 4  --freeze freeze --unimodal visual > log/85.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_sparse_mm_m.py --wandb_name m-visual-freeze-4-2 --domain_type 4  --freeze freeze --unimodal visual > log/86.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_sparse_mm_m.py --wandb_name m-visual-freeze-4-3 --domain_type 4  --freeze freeze --unimodal visual > log/87.log 2>&1 &
export CUDA_VISIBLE_DEVICES=0; nohup python run_sparse_mm_m.py --wandb_name m-visual-freeze-4-4 --domain_type 4  --freeze freeze --unimodal visual > log/88.log 2>&1 &
sleep 180
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=1; nohup python run_sparse_mm_m.py --wandb_name m-visual-freeze-5-1 --domain_type 5  --freeze freeze --unimodal visual > log/89.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_sparse_mm_m.py --wandb_name m-visual-freeze-5-2 --domain_type 5  --freeze freeze --unimodal visual > log/90.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_sparse_mm_m.py --wandb_name m-visual-freeze-5-3 --domain_type 5  --freeze freeze --unimodal visual > log/91.log 2>&1 &
export CUDA_VISIBLE_DEVICES=0; nohup python run_sparse_mm_m.py --wandb_name m-visual-freeze-5-4 --domain_type 5  --freeze freeze --unimodal visual > log/92.log 2>&1 &
sleep 10
export CUDA_VISIBLE_DEVICES=1; nohup python run_sparse_mm_m.py --wandb_name m-visual-freeze-6-1 --domain_type 6  --freeze freeze --unimodal visual > log/93.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_sparse_mm_m.py --wandb_name m-visual-freeze-6-2 --domain_type 6  --freeze freeze --unimodal visual > log/94.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_sparse_mm_m.py --wandb_name m-visual-freeze-6-3 --domain_type 6  --freeze freeze --unimodal visual > log/95.log 2>&1 &
export CUDA_VISIBLE_DEVICES=0; nohup python run_sparse_mm_m.py --wandb_name m-visual-freeze-6-4 --domain_type 6  --freeze freeze --unimodal visual > log/96.log 2>&1 &
sleep 180
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=1; nohup python run_sparse_mm_m.py --wandb_name m-visual-gl-freeze-1-1 --domain_type 1  --freeze freeze --unimodal visual-gl > log/97.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_sparse_mm_m.py --wandb_name m-visual-gl-freeze-1-2 --domain_type 1  --freeze freeze --unimodal visual-gl > log/98.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_sparse_mm_m.py --wandb_name m-visual-gl-freeze-1-3 --domain_type 1  --freeze freeze --unimodal visual-gl > log/99.log 2>&1 &
export CUDA_VISIBLE_DEVICES=0; nohup python run_sparse_mm_m.py --wandb_name m-visual-gl-freeze-1-4 --domain_type 1  --freeze freeze --unimodal visual-gl > log/100.log 2>&1 &
sleep 10
export CUDA_VISIBLE_DEVICES=1; nohup python run_sparse_mm_m.py --wandb_name m-visual-gl-freeze-2-1 --domain_type 2  --freeze freeze --unimodal visual-gl > log/101.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_sparse_mm_m.py --wandb_name m-visual-gl-freeze-2-2 --domain_type 2  --freeze freeze --unimodal visual-gl > log/102.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_sparse_mm_m.py --wandb_name m-visual-gl-freeze-2-3 --domain_type 2  --freeze freeze --unimodal visual-gl > log/103.log 2>&1 &
export CUDA_VISIBLE_DEVICES=0; nohup python run_sparse_mm_m.py --wandb_name m-visual-gl-freeze-2-4 --domain_type 2  --freeze freeze --unimodal visual-gl > log/104.log 2>&1 &
sleep 180
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=1; nohup python run_sparse_mm_m.py --wandb_name m-visual-gl-freeze-3-1 --domain_type 3  --freeze freeze --unimodal visual-gl > log/105.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_sparse_mm_m.py --wandb_name m-visual-gl-freeze-3-2 --domain_type 3  --freeze freeze --unimodal visual-gl > log/106.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_sparse_mm_m.py --wandb_name m-visual-gl-freeze-3-3 --domain_type 3  --freeze freeze --unimodal visual-gl > log/107.log 2>&1 &
export CUDA_VISIBLE_DEVICES=0; nohup python run_sparse_mm_m.py --wandb_name m-visual-gl-freeze-3-4 --domain_type 3  --freeze freeze --unimodal visual-gl > log/108.log 2>&1 &
sleep 10
export CUDA_VISIBLE_DEVICES=1; nohup python run_sparse_mm_m.py --wandb_name m-visual-gl-freeze-4-1 --domain_type 4  --freeze freeze --unimodal visual-gl > log/109.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_sparse_mm_m.py --wandb_name m-visual-gl-freeze-4-2 --domain_type 4  --freeze freeze --unimodal visual-gl > log/110.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_sparse_mm_m.py --wandb_name m-visual-gl-freeze-4-3 --domain_type 4  --freeze freeze --unimodal visual-gl > log/111.log 2>&1 &
export CUDA_VISIBLE_DEVICES=0; nohup python run_sparse_mm_m.py --wandb_name m-visual-gl-freeze-4-4 --domain_type 4  --freeze freeze --unimodal visual-gl > log/112.log 2>&1 &
sleep 180
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=1; nohup python run_sparse_mm_m.py --wandb_name m-visual-gl-freeze-5-1 --domain_type 5  --freeze freeze --unimodal visual-gl > log/113.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_sparse_mm_m.py --wandb_name m-visual-gl-freeze-5-2 --domain_type 5  --freeze freeze --unimodal visual-gl > log/114.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_sparse_mm_m.py --wandb_name m-visual-gl-freeze-5-3 --domain_type 5  --freeze freeze --unimodal visual-gl > log/115.log 2>&1 &
export CUDA_VISIBLE_DEVICES=0; nohup python run_sparse_mm_m.py --wandb_name m-visual-gl-freeze-5-4 --domain_type 5  --freeze freeze --unimodal visual-gl > log/116.log 2>&1 &
sleep 10
export CUDA_VISIBLE_DEVICES=1; nohup python run_sparse_mm_m.py --wandb_name m-visual-gl-freeze-6-1 --domain_type 6  --freeze freeze --unimodal visual-gl > log/117.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_sparse_mm_m.py --wandb_name m-visual-gl-freeze-6-2 --domain_type 6  --freeze freeze --unimodal visual-gl > log/118.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_sparse_mm_m.py --wandb_name m-visual-gl-freeze-6-3 --domain_type 6  --freeze freeze --unimodal visual-gl > log/119.log 2>&1 &
export CUDA_VISIBLE_DEVICES=0; nohup python run_sparse_mm_m.py --wandb_name m-visual-gl-freeze-6-4 --domain_type 6  --freeze freeze --unimodal visual-gl > log/120.log 2>&1 &
sleep 180
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=1; nohup python run_sparse_mm_m.py --wandb_name m-multimodal-nofreeze-1-1 --domain_type 1  --freeze nofreeze --unimodal multimodal > log/121.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_sparse_mm_m.py --wandb_name m-multimodal-nofreeze-1-2 --domain_type 1  --freeze nofreeze --unimodal multimodal > log/122.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_sparse_mm_m.py --wandb_name m-multimodal-nofreeze-1-3 --domain_type 1  --freeze nofreeze --unimodal multimodal > log/123.log 2>&1 &
export CUDA_VISIBLE_DEVICES=0; nohup python run_sparse_mm_m.py --wandb_name m-multimodal-nofreeze-1-4 --domain_type 1  --freeze nofreeze --unimodal multimodal > log/124.log 2>&1 &
sleep 10
export CUDA_VISIBLE_DEVICES=1; nohup python run_sparse_mm_m.py --wandb_name m-multimodal-nofreeze-2-1 --domain_type 2  --freeze nofreeze --unimodal multimodal > log/125.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_sparse_mm_m.py --wandb_name m-multimodal-nofreeze-2-2 --domain_type 2  --freeze nofreeze --unimodal multimodal > log/126.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_sparse_mm_m.py --wandb_name m-multimodal-nofreeze-2-3 --domain_type 2  --freeze nofreeze --unimodal multimodal > log/127.log 2>&1 &
export CUDA_VISIBLE_DEVICES=0; nohup python run_sparse_mm_m.py --wandb_name m-multimodal-nofreeze-2-4 --domain_type 2  --freeze nofreeze --unimodal multimodal > log/128.log 2>&1 &
sleep 180
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=1; nohup python run_sparse_mm_m.py --wandb_name m-multimodal-nofreeze-3-1 --domain_type 3  --freeze nofreeze --unimodal multimodal > log/129.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_sparse_mm_m.py --wandb_name m-multimodal-nofreeze-3-2 --domain_type 3  --freeze nofreeze --unimodal multimodal > log/130.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_sparse_mm_m.py --wandb_name m-multimodal-nofreeze-3-3 --domain_type 3  --freeze nofreeze --unimodal multimodal > log/131.log 2>&1 &
export CUDA_VISIBLE_DEVICES=0; nohup python run_sparse_mm_m.py --wandb_name m-multimodal-nofreeze-3-4 --domain_type 3  --freeze nofreeze --unimodal multimodal > log/132.log 2>&1 &
sleep 10
export CUDA_VISIBLE_DEVICES=1; nohup python run_sparse_mm_m.py --wandb_name m-multimodal-nofreeze-4-1 --domain_type 4  --freeze nofreeze --unimodal multimodal > log/133.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_sparse_mm_m.py --wandb_name m-multimodal-nofreeze-4-2 --domain_type 4  --freeze nofreeze --unimodal multimodal > log/134.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_sparse_mm_m.py --wandb_name m-multimodal-nofreeze-4-3 --domain_type 4  --freeze nofreeze --unimodal multimodal > log/135.log 2>&1 &
export CUDA_VISIBLE_DEVICES=0; nohup python run_sparse_mm_m.py --wandb_name m-multimodal-nofreeze-4-4 --domain_type 4  --freeze nofreeze --unimodal multimodal > log/136.log 2>&1 &
sleep 180
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=1; nohup python run_sparse_mm_m.py --wandb_name m-multimodal-nofreeze-5-1 --domain_type 5  --freeze nofreeze --unimodal multimodal > log/137.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_sparse_mm_m.py --wandb_name m-multimodal-nofreeze-5-2 --domain_type 5  --freeze nofreeze --unimodal multimodal > log/138.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_sparse_mm_m.py --wandb_name m-multimodal-nofreeze-5-3 --domain_type 5  --freeze nofreeze --unimodal multimodal > log/139.log 2>&1 &
export CUDA_VISIBLE_DEVICES=0; nohup python run_sparse_mm_m.py --wandb_name m-multimodal-nofreeze-5-4 --domain_type 5  --freeze nofreeze --unimodal multimodal > log/140.log 2>&1 &
sleep 10
export CUDA_VISIBLE_DEVICES=1; nohup python run_sparse_mm_m.py --wandb_name m-multimodal-nofreeze-6-1 --domain_type 6  --freeze nofreeze --unimodal multimodal > log/141.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_sparse_mm_m.py --wandb_name m-multimodal-nofreeze-6-2 --domain_type 6  --freeze nofreeze --unimodal multimodal > log/142.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_sparse_mm_m.py --wandb_name m-multimodal-nofreeze-6-3 --domain_type 6  --freeze nofreeze --unimodal multimodal > log/143.log 2>&1 &
export CUDA_VISIBLE_DEVICES=0; nohup python run_sparse_mm_m.py --wandb_name m-multimodal-nofreeze-6-4 --domain_type 6  --freeze nofreeze --unimodal multimodal > log/144.log 2>&1 &
sleep 180
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=1; nohup python run_sparse_mm_m.py --wandb_name m-multimodal-wogl-nofreeze-1-1 --domain_type 1  --freeze nofreeze --unimodal multimodal-wogl > log/145.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_sparse_mm_m.py --wandb_name m-multimodal-wogl-nofreeze-1-2 --domain_type 1  --freeze nofreeze --unimodal multimodal-wogl > log/146.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_sparse_mm_m.py --wandb_name m-multimodal-wogl-nofreeze-1-3 --domain_type 1  --freeze nofreeze --unimodal multimodal-wogl > log/147.log 2>&1 &
export CUDA_VISIBLE_DEVICES=0; nohup python run_sparse_mm_m.py --wandb_name m-multimodal-wogl-nofreeze-1-4 --domain_type 1  --freeze nofreeze --unimodal multimodal-wogl > log/148.log 2>&1 &
sleep 10
export CUDA_VISIBLE_DEVICES=1; nohup python run_sparse_mm_m.py --wandb_name m-multimodal-wogl-nofreeze-2-1 --domain_type 2  --freeze nofreeze --unimodal multimodal-wogl > log/149.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_sparse_mm_m.py --wandb_name m-multimodal-wogl-nofreeze-2-2 --domain_type 2  --freeze nofreeze --unimodal multimodal-wogl > log/150.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_sparse_mm_m.py --wandb_name m-multimodal-wogl-nofreeze-2-3 --domain_type 2  --freeze nofreeze --unimodal multimodal-wogl > log/151.log 2>&1 &
export CUDA_VISIBLE_DEVICES=0; nohup python run_sparse_mm_m.py --wandb_name m-multimodal-wogl-nofreeze-2-4 --domain_type 2  --freeze nofreeze --unimodal multimodal-wogl > log/152.log 2>&1 &
sleep 180
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=1; nohup python run_sparse_mm_m.py --wandb_name m-multimodal-wogl-nofreeze-3-1 --domain_type 3  --freeze nofreeze --unimodal multimodal-wogl > log/153.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_sparse_mm_m.py --wandb_name m-multimodal-wogl-nofreeze-3-2 --domain_type 3  --freeze nofreeze --unimodal multimodal-wogl > log/154.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_sparse_mm_m.py --wandb_name m-multimodal-wogl-nofreeze-3-3 --domain_type 3  --freeze nofreeze --unimodal multimodal-wogl > log/155.log 2>&1 &
export CUDA_VISIBLE_DEVICES=0; nohup python run_sparse_mm_m.py --wandb_name m-multimodal-wogl-nofreeze-3-4 --domain_type 3  --freeze nofreeze --unimodal multimodal-wogl > log/156.log 2>&1 &
sleep 10
export CUDA_VISIBLE_DEVICES=1; nohup python run_sparse_mm_m.py --wandb_name m-multimodal-wogl-nofreeze-4-1 --domain_type 4  --freeze nofreeze --unimodal multimodal-wogl > log/157.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_sparse_mm_m.py --wandb_name m-multimodal-wogl-nofreeze-4-2 --domain_type 4  --freeze nofreeze --unimodal multimodal-wogl > log/158.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_sparse_mm_m.py --wandb_name m-multimodal-wogl-nofreeze-4-3 --domain_type 4  --freeze nofreeze --unimodal multimodal-wogl > log/159.log 2>&1 &
export CUDA_VISIBLE_DEVICES=0; nohup python run_sparse_mm_m.py --wandb_name m-multimodal-wogl-nofreeze-4-4 --domain_type 4  --freeze nofreeze --unimodal multimodal-wogl > log/160.log 2>&1 &
sleep 180
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=1; nohup python run_sparse_mm_m.py --wandb_name m-multimodal-wogl-nofreeze-5-1 --domain_type 5  --freeze nofreeze --unimodal multimodal-wogl > log/161.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_sparse_mm_m.py --wandb_name m-multimodal-wogl-nofreeze-5-2 --domain_type 5  --freeze nofreeze --unimodal multimodal-wogl > log/162.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_sparse_mm_m.py --wandb_name m-multimodal-wogl-nofreeze-5-3 --domain_type 5  --freeze nofreeze --unimodal multimodal-wogl > log/163.log 2>&1 &
export CUDA_VISIBLE_DEVICES=0; nohup python run_sparse_mm_m.py --wandb_name m-multimodal-wogl-nofreeze-5-4 --domain_type 5  --freeze nofreeze --unimodal multimodal-wogl > log/164.log 2>&1 &
sleep 10
export CUDA_VISIBLE_DEVICES=1; nohup python run_sparse_mm_m.py --wandb_name m-multimodal-wogl-nofreeze-6-1 --domain_type 6  --freeze nofreeze --unimodal multimodal-wogl > log/165.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_sparse_mm_m.py --wandb_name m-multimodal-wogl-nofreeze-6-2 --domain_type 6  --freeze nofreeze --unimodal multimodal-wogl > log/166.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_sparse_mm_m.py --wandb_name m-multimodal-wogl-nofreeze-6-3 --domain_type 6  --freeze nofreeze --unimodal multimodal-wogl > log/167.log 2>&1 &
export CUDA_VISIBLE_DEVICES=0; nohup python run_sparse_mm_m.py --wandb_name m-multimodal-wogl-nofreeze-6-4 --domain_type 6  --freeze nofreeze --unimodal multimodal-wogl > log/168.log 2>&1 &
sleep 180
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=1; nohup python run_sparse_mm_m.py --wandb_name m-text-nofreeze-1-1 --domain_type 1  --freeze nofreeze --unimodal text > log/169.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_sparse_mm_m.py --wandb_name m-text-nofreeze-1-2 --domain_type 1  --freeze nofreeze --unimodal text > log/170.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_sparse_mm_m.py --wandb_name m-text-nofreeze-1-3 --domain_type 1  --freeze nofreeze --unimodal text > log/171.log 2>&1 &
export CUDA_VISIBLE_DEVICES=0; nohup python run_sparse_mm_m.py --wandb_name m-text-nofreeze-1-4 --domain_type 1  --freeze nofreeze --unimodal text > log/172.log 2>&1 &
sleep 10
export CUDA_VISIBLE_DEVICES=1; nohup python run_sparse_mm_m.py --wandb_name m-text-nofreeze-2-1 --domain_type 2  --freeze nofreeze --unimodal text > log/173.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_sparse_mm_m.py --wandb_name m-text-nofreeze-2-2 --domain_type 2  --freeze nofreeze --unimodal text > log/174.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_sparse_mm_m.py --wandb_name m-text-nofreeze-2-3 --domain_type 2  --freeze nofreeze --unimodal text > log/175.log 2>&1 &
export CUDA_VISIBLE_DEVICES=0; nohup python run_sparse_mm_m.py --wandb_name m-text-nofreeze-2-4 --domain_type 2  --freeze nofreeze --unimodal text > log/176.log 2>&1 &
sleep 180
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=1; nohup python run_sparse_mm_m.py --wandb_name m-text-nofreeze-3-1 --domain_type 3  --freeze nofreeze --unimodal text > log/177.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_sparse_mm_m.py --wandb_name m-text-nofreeze-3-2 --domain_type 3  --freeze nofreeze --unimodal text > log/178.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_sparse_mm_m.py --wandb_name m-text-nofreeze-3-3 --domain_type 3  --freeze nofreeze --unimodal text > log/179.log 2>&1 &
export CUDA_VISIBLE_DEVICES=0; nohup python run_sparse_mm_m.py --wandb_name m-text-nofreeze-3-4 --domain_type 3  --freeze nofreeze --unimodal text > log/180.log 2>&1 &
sleep 10
export CUDA_VISIBLE_DEVICES=1; nohup python run_sparse_mm_m.py --wandb_name m-text-nofreeze-4-1 --domain_type 4  --freeze nofreeze --unimodal text > log/181.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_sparse_mm_m.py --wandb_name m-text-nofreeze-4-2 --domain_type 4  --freeze nofreeze --unimodal text > log/182.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_sparse_mm_m.py --wandb_name m-text-nofreeze-4-3 --domain_type 4  --freeze nofreeze --unimodal text > log/183.log 2>&1 &
export CUDA_VISIBLE_DEVICES=0; nohup python run_sparse_mm_m.py --wandb_name m-text-nofreeze-4-4 --domain_type 4  --freeze nofreeze --unimodal text > log/184.log 2>&1 &
sleep 180
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=1; nohup python run_sparse_mm_m.py --wandb_name m-text-nofreeze-5-1 --domain_type 5  --freeze nofreeze --unimodal text > log/185.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_sparse_mm_m.py --wandb_name m-text-nofreeze-5-2 --domain_type 5  --freeze nofreeze --unimodal text > log/186.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_sparse_mm_m.py --wandb_name m-text-nofreeze-5-3 --domain_type 5  --freeze nofreeze --unimodal text > log/187.log 2>&1 &
export CUDA_VISIBLE_DEVICES=0; nohup python run_sparse_mm_m.py --wandb_name m-text-nofreeze-5-4 --domain_type 5  --freeze nofreeze --unimodal text > log/188.log 2>&1 &
sleep 10
export CUDA_VISIBLE_DEVICES=1; nohup python run_sparse_mm_m.py --wandb_name m-text-nofreeze-6-1 --domain_type 6  --freeze nofreeze --unimodal text > log/189.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_sparse_mm_m.py --wandb_name m-text-nofreeze-6-2 --domain_type 6  --freeze nofreeze --unimodal text > log/190.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_sparse_mm_m.py --wandb_name m-text-nofreeze-6-3 --domain_type 6  --freeze nofreeze --unimodal text > log/191.log 2>&1 &
export CUDA_VISIBLE_DEVICES=0; nohup python run_sparse_mm_m.py --wandb_name m-text-nofreeze-6-4 --domain_type 6  --freeze nofreeze --unimodal text > log/192.log 2>&1 &
sleep 180
monitor_gpu_processes
single-source
export CUDA_VISIBLE_DEVICES=1; nohup python run_sparse_mm_s.py --wandb_name s-multimodal-freeze-1-1 --domain_type 1  --freeze freeze --unimodal multimodal > log/193.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_sparse_mm_s.py --wandb_name s-multimodal-freeze-1-2 --domain_type 1  --freeze freeze --unimodal multimodal > log/194.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_sparse_mm_s.py --wandb_name s-multimodal-freeze-1-3 --domain_type 1  --freeze freeze --unimodal multimodal > log/195.log 2>&1 &
export CUDA_VISIBLE_DEVICES=0; nohup python run_sparse_mm_s.py --wandb_name s-multimodal-freeze-1-4 --domain_type 1  --freeze freeze --unimodal multimodal > log/196.log 2>&1 &
sleep 10
export CUDA_VISIBLE_DEVICES=1; nohup python run_sparse_mm_s.py --wandb_name s-multimodal-freeze-2-1 --domain_type 2  --freeze freeze --unimodal multimodal > log/197.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_sparse_mm_s.py --wandb_name s-multimodal-freeze-2-2 --domain_type 2  --freeze freeze --unimodal multimodal > log/198.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_sparse_mm_s.py --wandb_name s-multimodal-freeze-2-3 --domain_type 2  --freeze freeze --unimodal multimodal > log/199.log 2>&1 &
export CUDA_VISIBLE_DEVICES=0; nohup python run_sparse_mm_s.py --wandb_name s-multimodal-freeze-2-4 --domain_type 2  --freeze freeze --unimodal multimodal > log/200.log 2>&1 &
sleep 180
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=1; nohup python run_sparse_mm_s.py --wandb_name s-multimodal-freeze-3-1 --domain_type 3  --freeze freeze --unimodal multimodal > log/201.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_sparse_mm_s.py --wandb_name s-multimodal-freeze-3-2 --domain_type 3  --freeze freeze --unimodal multimodal > log/202.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_sparse_mm_s.py --wandb_name s-multimodal-freeze-3-3 --domain_type 3  --freeze freeze --unimodal multimodal > log/203.log 2>&1 &
export CUDA_VISIBLE_DEVICES=0; nohup python run_sparse_mm_s.py --wandb_name s-multimodal-freeze-3-4 --domain_type 3  --freeze freeze --unimodal multimodal > log/204.log 2>&1 &
sleep 10
export CUDA_VISIBLE_DEVICES=1; nohup python run_sparse_mm_s.py --wandb_name s-multimodal-wogl-freeze-1-1 --domain_type 1  --freeze freeze --unimodal multimodal-wogl > log/205.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_sparse_mm_s.py --wandb_name s-multimodal-wogl-freeze-1-2 --domain_type 1  --freeze freeze --unimodal multimodal-wogl > log/206.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_sparse_mm_s.py --wandb_name s-multimodal-wogl-freeze-1-3 --domain_type 1  --freeze freeze --unimodal multimodal-wogl > log/207.log 2>&1 &
export CUDA_VISIBLE_DEVICES=0; nohup python run_sparse_mm_s.py --wandb_name s-multimodal-wogl-freeze-1-4 --domain_type 1  --freeze freeze --unimodal multimodal-wogl > log/208.log 2>&1 &
sleep 180
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=1; nohup python run_sparse_mm_s.py --wandb_name s-multimodal-wogl-freeze-2-1 --domain_type 2  --freeze freeze --unimodal multimodal-wogl > log/209.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_sparse_mm_s.py --wandb_name s-multimodal-wogl-freeze-2-2 --domain_type 2  --freeze freeze --unimodal multimodal-wogl > log/210.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_sparse_mm_s.py --wandb_name s-multimodal-wogl-freeze-2-3 --domain_type 2  --freeze freeze --unimodal multimodal-wogl > log/211.log 2>&1 &
export CUDA_VISIBLE_DEVICES=0; nohup python run_sparse_mm_s.py --wandb_name s-multimodal-wogl-freeze-2-4 --domain_type 2  --freeze freeze --unimodal multimodal-wogl > log/212.log 2>&1 &
sleep 10
export CUDA_VISIBLE_DEVICES=1; nohup python run_sparse_mm_s.py --wandb_name s-multimodal-wogl-freeze-3-1 --domain_type 3  --freeze freeze --unimodal multimodal-wogl > log/213.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_sparse_mm_s.py --wandb_name s-multimodal-wogl-freeze-3-2 --domain_type 3  --freeze freeze --unimodal multimodal-wogl > log/214.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_sparse_mm_s.py --wandb_name s-multimodal-wogl-freeze-3-3 --domain_type 3  --freeze freeze --unimodal multimodal-wogl > log/215.log 2>&1 &
export CUDA_VISIBLE_DEVICES=0; nohup python run_sparse_mm_s.py --wandb_name s-multimodal-wogl-freeze-3-4 --domain_type 3  --freeze freeze --unimodal multimodal-wogl > log/216.log 2>&1 &
sleep 180
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=1; nohup python run_sparse_mm_s.py --wandb_name s-text-freeze-1-1 --domain_type 1  --freeze freeze --unimodal text > log/217.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_sparse_mm_s.py --wandb_name s-text-freeze-1-2 --domain_type 1  --freeze freeze --unimodal text > log/218.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_sparse_mm_s.py --wandb_name s-text-freeze-1-3 --domain_type 1  --freeze freeze --unimodal text > log/219.log 2>&1 &
export CUDA_VISIBLE_DEVICES=0; nohup python run_sparse_mm_s.py --wandb_name s-text-freeze-1-4 --domain_type 1  --freeze freeze --unimodal text > log/220.log 2>&1 &
sleep 10
export CUDA_VISIBLE_DEVICES=1; nohup python run_sparse_mm_s.py --wandb_name s-text-freeze-2-1 --domain_type 2  --freeze freeze --unimodal text > log/221.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_sparse_mm_s.py --wandb_name s-text-freeze-2-2 --domain_type 2  --freeze freeze --unimodal text > log/222.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_sparse_mm_s.py --wandb_name s-text-freeze-2-3 --domain_type 2  --freeze freeze --unimodal text > log/223.log 2>&1 &
export CUDA_VISIBLE_DEVICES=0; nohup python run_sparse_mm_s.py --wandb_name s-text-freeze-2-4 --domain_type 2  --freeze freeze --unimodal text > log/224.log 2>&1 &
sleep 180
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=1; nohup python run_sparse_mm_s.py --wandb_name s-text-freeze-3-1 --domain_type 3  --freeze freeze --unimodal text > log/225.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_sparse_mm_s.py --wandb_name s-text-freeze-3-2 --domain_type 3  --freeze freeze --unimodal text > log/226.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_sparse_mm_s.py --wandb_name s-text-freeze-3-3 --domain_type 3  --freeze freeze --unimodal text > log/227.log 2>&1 &
export CUDA_VISIBLE_DEVICES=0; nohup python run_sparse_mm_s.py --wandb_name s-text-freeze-3-4 --domain_type 3  --freeze freeze --unimodal text > log/228.log 2>&1 &
sleep 10
export CUDA_VISIBLE_DEVICES=1; nohup python run_sparse_mm_s.py --wandb_name s-visual-freeze-1-1 --domain_type 1  --freeze freeze --unimodal visual > log/229.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_sparse_mm_s.py --wandb_name s-visual-freeze-1-2 --domain_type 1  --freeze freeze --unimodal visual > log/230.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_sparse_mm_s.py --wandb_name s-visual-freeze-1-3 --domain_type 1  --freeze freeze --unimodal visual > log/231.log 2>&1 &
export CUDA_VISIBLE_DEVICES=0; nohup python run_sparse_mm_s.py --wandb_name s-visual-freeze-1-4 --domain_type 1  --freeze freeze --unimodal visual > log/232.log 2>&1 &
sleep 180
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=1; nohup python run_sparse_mm_s.py --wandb_name s-visual-freeze-2-1 --domain_type 2  --freeze freeze --unimodal visual > log/233.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_sparse_mm_s.py --wandb_name s-visual-freeze-2-2 --domain_type 2  --freeze freeze --unimodal visual > log/234.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_sparse_mm_s.py --wandb_name s-visual-freeze-2-3 --domain_type 2  --freeze freeze --unimodal visual > log/235.log 2>&1 &
export CUDA_VISIBLE_DEVICES=0; nohup python run_sparse_mm_s.py --wandb_name s-visual-freeze-2-4 --domain_type 2  --freeze freeze --unimodal visual > log/236.log 2>&1 &
sleep 10
export CUDA_VISIBLE_DEVICES=1; nohup python run_sparse_mm_s.py --wandb_name s-visual-freeze-3-1 --domain_type 3  --freeze freeze --unimodal visual > log/237.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_sparse_mm_s.py --wandb_name s-visual-freeze-3-2 --domain_type 3  --freeze freeze --unimodal visual > log/238.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_sparse_mm_s.py --wandb_name s-visual-freeze-3-3 --domain_type 3  --freeze freeze --unimodal visual > log/239.log 2>&1 &
export CUDA_VISIBLE_DEVICES=0; nohup python run_sparse_mm_s.py --wandb_name s-visual-freeze-3-4 --domain_type 3  --freeze freeze --unimodal visual > log/240.log 2>&1 &
sleep 180
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=1; nohup python run_sparse_mm_s.py --wandb_name s-visual-gl-freeze-1-1 --domain_type 1  --freeze freeze --unimodal visual-gl > log/241.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_sparse_mm_s.py --wandb_name s-visual-gl-freeze-1-2 --domain_type 1  --freeze freeze --unimodal visual-gl > log/242.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_sparse_mm_s.py --wandb_name s-visual-gl-freeze-1-3 --domain_type 1  --freeze freeze --unimodal visual-gl > log/243.log 2>&1 &
export CUDA_VISIBLE_DEVICES=0; nohup python run_sparse_mm_s.py --wandb_name s-visual-gl-freeze-1-4 --domain_type 1  --freeze freeze --unimodal visual-gl > log/244.log 2>&1 &
sleep 10
export CUDA_VISIBLE_DEVICES=1; nohup python run_sparse_mm_s.py --wandb_name s-visual-gl-freeze-2-1 --domain_type 2  --freeze freeze --unimodal visual-gl > log/245.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_sparse_mm_s.py --wandb_name s-visual-gl-freeze-2-2 --domain_type 2  --freeze freeze --unimodal visual-gl > log/246.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_sparse_mm_s.py --wandb_name s-visual-gl-freeze-2-3 --domain_type 2  --freeze freeze --unimodal visual-gl > log/247.log 2>&1 &
export CUDA_VISIBLE_DEVICES=0; nohup python run_sparse_mm_s.py --wandb_name s-visual-gl-freeze-2-4 --domain_type 2  --freeze freeze --unimodal visual-gl > log/248.log 2>&1 &
sleep 180
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=1; nohup python run_sparse_mm_s.py --wandb_name s-visual-gl-freeze-3-1 --domain_type 3  --freeze freeze --unimodal visual-gl > log/249.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_sparse_mm_s.py --wandb_name s-visual-gl-freeze-3-2 --domain_type 3  --freeze freeze --unimodal visual-gl > log/250.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_sparse_mm_s.py --wandb_name s-visual-gl-freeze-3-3 --domain_type 3  --freeze freeze --unimodal visual-gl > log/251.log 2>&1 &
export CUDA_VISIBLE_DEVICES=0; nohup python run_sparse_mm_s.py --wandb_name s-visual-gl-freeze-3-4 --domain_type 3  --freeze freeze --unimodal visual-gl > log/252.log 2>&1 &
sleep 10
export CUDA_VISIBLE_DEVICES=1; nohup python run_sparse_mm_s.py --wandb_name s-multimodal-nofreeze-1-1 --domain_type 1  --freeze nofreeze --unimodal multimodal > log/253.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_sparse_mm_s.py --wandb_name s-multimodal-nofreeze-1-2 --domain_type 1  --freeze nofreeze --unimodal multimodal > log/254.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_sparse_mm_s.py --wandb_name s-multimodal-nofreeze-1-3 --domain_type 1  --freeze nofreeze --unimodal multimodal > log/255.log 2>&1 &
export CUDA_VISIBLE_DEVICES=0; nohup python run_sparse_mm_s.py --wandb_name s-multimodal-nofreeze-1-4 --domain_type 1  --freeze nofreeze --unimodal multimodal > log/256.log 2>&1 &
sleep 180
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=1; nohup python run_sparse_mm_s.py --wandb_name s-multimodal-nofreeze-2-1 --domain_type 2  --freeze nofreeze --unimodal multimodal > log/257.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_sparse_mm_s.py --wandb_name s-multimodal-nofreeze-2-2 --domain_type 2  --freeze nofreeze --unimodal multimodal > log/258.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_sparse_mm_s.py --wandb_name s-multimodal-nofreeze-2-3 --domain_type 2  --freeze nofreeze --unimodal multimodal > log/259.log 2>&1 &
export CUDA_VISIBLE_DEVICES=0; nohup python run_sparse_mm_s.py --wandb_name s-multimodal-nofreeze-2-4 --domain_type 2  --freeze nofreeze --unimodal multimodal > log/260.log 2>&1 &
sleep 10
export CUDA_VISIBLE_DEVICES=1; nohup python run_sparse_mm_s.py --wandb_name s-multimodal-nofreeze-3-1 --domain_type 3  --freeze nofreeze --unimodal multimodal > log/261.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_sparse_mm_s.py --wandb_name s-multimodal-nofreeze-3-2 --domain_type 3  --freeze nofreeze --unimodal multimodal > log/262.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_sparse_mm_s.py --wandb_name s-multimodal-nofreeze-3-3 --domain_type 3  --freeze nofreeze --unimodal multimodal > log/263.log 2>&1 &
export CUDA_VISIBLE_DEVICES=0; nohup python run_sparse_mm_s.py --wandb_name s-multimodal-nofreeze-3-4 --domain_type 3  --freeze nofreeze --unimodal multimodal > log/264.log 2>&1 &
sleep 180
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=1; nohup python run_sparse_mm_s.py --wandb_name s-multimodal-wogl-nofreeze-1-1 --domain_type 1  --freeze nofreeze --unimodal multimodal-wogl > log/265.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_sparse_mm_s.py --wandb_name s-multimodal-wogl-nofreeze-1-2 --domain_type 1  --freeze nofreeze --unimodal multimodal-wogl > log/266.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_sparse_mm_s.py --wandb_name s-multimodal-wogl-nofreeze-1-3 --domain_type 1  --freeze nofreeze --unimodal multimodal-wogl > log/267.log 2>&1 &
export CUDA_VISIBLE_DEVICES=0; nohup python run_sparse_mm_s.py --wandb_name s-multimodal-wogl-nofreeze-1-4 --domain_type 1  --freeze nofreeze --unimodal multimodal-wogl > log/268.log 2>&1 &
sleep 10
export CUDA_VISIBLE_DEVICES=1; nohup python run_sparse_mm_s.py --wandb_name s-multimodal-wogl-nofreeze-2-1 --domain_type 2  --freeze nofreeze --unimodal multimodal-wogl > log/269.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_sparse_mm_s.py --wandb_name s-multimodal-wogl-nofreeze-2-2 --domain_type 2  --freeze nofreeze --unimodal multimodal-wogl > log/270.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_sparse_mm_s.py --wandb_name s-multimodal-wogl-nofreeze-2-3 --domain_type 2  --freeze nofreeze --unimodal multimodal-wogl > log/271.log 2>&1 &
export CUDA_VISIBLE_DEVICES=0; nohup python run_sparse_mm_s.py --wandb_name s-multimodal-wogl-nofreeze-2-4 --domain_type 2  --freeze nofreeze --unimodal multimodal-wogl > log/272.log 2>&1 &
sleep 180
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=1; nohup python run_sparse_mm_s.py --wandb_name s-multimodal-wogl-nofreeze-3-1 --domain_type 3  --freeze nofreeze --unimodal multimodal-wogl > log/273.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_sparse_mm_s.py --wandb_name s-multimodal-wogl-nofreeze-3-2 --domain_type 3  --freeze nofreeze --unimodal multimodal-wogl > log/274.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_sparse_mm_s.py --wandb_name s-multimodal-wogl-nofreeze-3-3 --domain_type 3  --freeze nofreeze --unimodal multimodal-wogl > log/275.log 2>&1 &
export CUDA_VISIBLE_DEVICES=0; nohup python run_sparse_mm_s.py --wandb_name s-multimodal-wogl-nofreeze-3-4 --domain_type 3  --freeze nofreeze --unimodal multimodal-wogl > log/276.log 2>&1 &
sleep 10
export CUDA_VISIBLE_DEVICES=1; nohup python run_sparse_mm_s.py --wandb_name s-text-nofreeze-1-1 --domain_type 1  --freeze nofreeze --unimodal text > log/277.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_sparse_mm_s.py --wandb_name s-text-nofreeze-1-2 --domain_type 1  --freeze nofreeze --unimodal text > log/278.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_sparse_mm_s.py --wandb_name s-text-nofreeze-1-3 --domain_type 1  --freeze nofreeze --unimodal text > log/279.log 2>&1 &
export CUDA_VISIBLE_DEVICES=0; nohup python run_sparse_mm_s.py --wandb_name s-text-nofreeze-1-4 --domain_type 1  --freeze nofreeze --unimodal text > log/280.log 2>&1 &
sleep 180
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=1; nohup python run_sparse_mm_s.py --wandb_name s-text-nofreeze-2-1 --domain_type 2  --freeze nofreeze --unimodal text > log/281.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_sparse_mm_s.py --wandb_name s-text-nofreeze-2-2 --domain_type 2  --freeze nofreeze --unimodal text > log/282.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_sparse_mm_s.py --wandb_name s-text-nofreeze-2-3 --domain_type 2  --freeze nofreeze --unimodal text > log/283.log 2>&1 &
export CUDA_VISIBLE_DEVICES=0; nohup python run_sparse_mm_s.py --wandb_name s-text-nofreeze-2-4 --domain_type 2  --freeze nofreeze --unimodal text > log/284.log 2>&1 &
sleep 10
export CUDA_VISIBLE_DEVICES=1; nohup python run_sparse_mm_s.py --wandb_name s-text-nofreeze-3-1 --domain_type 3  --freeze nofreeze --unimodal text > log/285.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run_sparse_mm_s.py --wandb_name s-text-nofreeze-3-2 --domain_type 3  --freeze nofreeze --unimodal text > log/286.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run_sparse_mm_s.py --wandb_name s-text-nofreeze-3-3 --domain_type 3  --freeze nofreeze --unimodal text > log/287.log 2>&1 &
export CUDA_VISIBLE_DEVICES=0; nohup python run_sparse_mm_s.py --wandb_name s-text-nofreeze-3-4 --domain_type 3  --freeze nofreeze --unimodal text > log/288.log 2>&1 &
sleep 180
monitor_gpu_processes

# compare_costs.py
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
import os, time, json, csv, math, statistics, itertools
from dataclasses import dataclass, asdict
from typing import Callable, Dict, Any, Tuple, List, Optional
import torch
import argparse
from RAGmodel import *
from MULTmodel import *
# ============== 你只需要改下面三段（构建模型） =================
def build_ours(device="cuda", dtype=torch.float16, use_retrieval=True):
    """
    TODO: 返回你的模型实例（.eval()），支持 use_retrieval 开关。
    例如：
        model = OurModel(use_retrieval=use_retrieval, ...)
        model.load_state_dict(torch.load('ours.pth', map_location='cpu'))
        model = model.to(device=device, dtype=dtype).eval()
        return model
    """
    parser = argparse.ArgumentParser()
    # parser.add_argument("--cuda_no", type=str, default=os.environ["CUDA_VISIBLE_DEVICES"])
    parser.add_argument("--dataset", type=str, choices=["mosi", "mosei"], default=DATASETS)
    parser.add_argument("--max_seq_length", type=int, default=50)
    parser.add_argument("--train_batch_size", type=int, default=8)#BATCH_SIZE
    parser.add_argument("--dev_batch_size", type=int, default=128)
    parser.add_argument("--test_batch_size", type=int, default=128)
    parser.add_argument("--n_epochs", type=int, default=EPOCHS)
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--gradient_accumulation_step", type=int, default=1)
    parser.add_argument("--warmup_proportion", type=float, default=0.1)
    parser.add_argument("--seed", type=seed, default="random")
    parser.add_argument("--best_acc", type=float, default=0.1)
    parser.add_argument("--wandb_name", type=str, default='none')
    parser.add_argument("--domain_type", type=int, default=1)
    parser.add_argument("--freeze", type=str, default='freeze')
    parser.add_argument("--unimodal", type=str, default='text')
    parser.add_argument("--layer", type=int, default=1)

    parser.add_argument("--warm_up", type=int, default=5)

    parser.add_argument("--test", type=int, default=0)


    parser.add_argument("--t_dim", type=int, default=4096)
    parser.add_argument("--v_dim", type=int, default=512)
    parser.add_argument("--a_dim", type=int, default=1024)

    parser.add_argument("--dg_label_dim", type=int, default=2)
    parser.add_argument("--ds_label_dim", type=int, default=1)
    parser.add_argument("--dsbert",type=str,default="T5")

    parser.add_argument("--prompt_dim",type=int,default=128)
    parser.add_argument("--prompt_len",type=int,default=100)
    parser.add_argument("--t_len",type=int,default=100)
    parser.add_argument("--v_len",type=int,default=100)
    parser.add_argument("--a_len",type=int,default=100)
    parser.add_argument("--dim_head",type=int,default=96)
    parser.add_argument("--dropout",type=float,default=0.1)#0.2
    parser.add_argument("--depth",type=int,default=1)
    parser.add_argument("--scale_dim",type=int,default=4)
    parser.add_argument("--loss_rate",type=int,default=1)
    parser.add_argument("--classifier_dropout",type=float,default=0.0)
    parser.add_argument("--layers",type=int,default=6)
    parser.add_argument("--attn_dropout",type=float,default=0.1)
    parser.add_argument("--relu_dropout",type=float,default=0.1)
    parser.add_argument("--res_dropout",type=float,default=0.1)
    parser.add_argument("--embed_dropout",type=float,default=0.25)
    parser.add_argument("--attn_mask",action='store_false',
                        help='use attention mask for Transformer (default: true)')
    parser.add_argument("--prompt_plot",type=int,default=128)
    args = parser.parse_args()
    if use_retrieval is True:
        model = RAGModel(args).to(device,dtype=dtype)
    else:
        model = NORAGModel(args).to(device,dtype=dtype)
    return model
    raise NotImplementedError("fill build_ours()")

def build_mult(device="cuda", dtype=torch.float16):
    """
    TODO: 返回 MULT 模型实例（.eval()），保持相同的输入接口
    """
    parser = argparse.ArgumentParser(description='MOSEI Sentiment Analysis')
    parser.add_argument('-f', default='', type=str)

    # Fixed
    parser.add_argument('--model', type=str, default='MulT',
                        help='name of the model to use (Transformer, etc.)')

    # Tasks
    parser.add_argument('--vonly', action='store_true',
                        help='use the crossmodal fusion into v (default: False)')
    parser.add_argument('--aonly', action='store_true',
                        help='use the crossmodal fusion into a (default: False)')
    parser.add_argument('--lonly', action='store_true',
                        help='use the crossmodal fusion into l (default: False)')
    parser.add_argument('--aligned', action='store_true',
                        help='consider aligned experiment or not (default: False)')
    parser.add_argument('--dataset', type=str, default='mosei_senti',
                        help='dataset to use (default: mosei_senti)')
    parser.add_argument('--data_path', type=str, default='data',
                        help='path for storing the dataset')

    # Dropouts
    parser.add_argument('--attn_dropout', type=float, default=0.1,
                        help='attention dropout')
    parser.add_argument('--attn_dropout_a', type=float, default=0.0,
                        help='attention dropout (for audio)')
    parser.add_argument('--attn_dropout_v', type=float, default=0.0,
                        help='attention dropout (for visual)')
    parser.add_argument('--relu_dropout', type=float, default=0.1,
                        help='relu dropout')
    parser.add_argument('--embed_dropout', type=float, default=0.25,
                        help='embedding dropout')
    parser.add_argument('--res_dropout', type=float, default=0.1,
                        help='residual block dropout')
    parser.add_argument('--out_dropout', type=float, default=0.0,
                        help='output layer dropout')

    # Architecture
    parser.add_argument('--nlevels', type=int, default=5,
                        help='number of layers in the network (default: 5)')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='number of heads for the transformer network (default: 5)')
    parser.add_argument('--attn_mask', action='store_false',
                        help='use attention mask for Transformer (default: true)')

    # Tuning
    parser.add_argument('--batch_size', type=int, default=24, metavar='N',
                        help='batch size (default: 24)')
    parser.add_argument('--clip', type=float, default=0.8,
                        help='gradient clip value (default: 0.8)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='initial learning rate (default: 1e-3)')
    parser.add_argument('--optim', type=str, default='Adam',
                        help='optimizer to use (default: Adam)')
    parser.add_argument('--num_epochs', type=int, default=40,
                        help='number of epochs (default: 40)')
    parser.add_argument('--when', type=int, default=20,
                        help='when to decay learning rate (default: 20)')
    parser.add_argument('--batch_chunk', type=int, default=1,
                        help='number of chunks per batch (default: 1)')

    # Logistics
    parser.add_argument('--log_interval', type=int, default=30,
                        help='frequency of result logging (default: 30)')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    # parser.add_argument('--no_cuda', action='store_true',
    #                     help='do not use cuda')
    parser.add_argument('--name', type=str, default='mult',
                        help='name of the trial (default: "mult")')
    parser.add_argument('--output_dim', type=int, default=1,
                        help='name of the trial (default: "mult")')
    args = parser.parse_args()
    valid_partial_mode = args.lonly + args.vonly + args.aonly
    if valid_partial_mode == 0:
        args.lonly = args.vonly = args.aonly = True
    elif valid_partial_mode != 1:
        raise ValueError("You can only choose one of {l/v/a}only.")
    model = MULTModel(args).to(device,dtype=dtype)
    return model
    raise NotImplementedError("fill build_mult()")

# def build_dtn(device="cuda", dtype=torch.float16):
#     """
#     TODO: 返回 DTN 模型实例（.eval()），保持相同的输入接口
#     """
#     raise NotImplementedError("fill build_dtn()")
# ===========================================================

# ------------- 输入构造（按你的任务调整） --------------------
def make_inputs(
    batch_size=8,
    L = 50,
    d_l=768,        # 隐维
    d_a=1024,
    d_v=512,
    device="cuda",
    dtype=torch.float16
):
    """
    按需返回 tuple 或 dict，与三种模型 forward 对齐。
    下面是一个常见多模态示例：text、video，以及可选的 retrieved_docs。
    若你的模型只需要 text，删掉 video 即可。
    """
    text = torch.randn(batch_size, L, d_l, device=device, dtype=dtype).to(device)
    audio = torch.randn(batch_size, L, d_a, device=device, dtype=dtype).to(device)
    video = torch.randn(batch_size, L, d_v, device=device, dtype=dtype).to(device)
    # 检索增强：形状 [B, R, L_r, d]

    # 三个模型最好统一 forward 签名；如果不一致，可用 wrapper 适配
    feed = {"text": text, "audio":audio , "visual": video}
    return feed
# -----------------------------------------------------------

# -------------------- 公共工具函数 ---------------------------
def try_flops(model, sample_inputs):
    # 优先 thop，其次 fvcore；都不可用则返回 None
    # 注意：FLOPs 定义在不同库不同，文中需说明计算口径
    # try:
        from thop import profile
        # thop 只接受 tuple/list 作为输入
        if isinstance(sample_inputs, dict):
            # 常见 forward(text=..., video=..., retrieved=...)
            args = (sample_inputs.get("text"), sample_inputs.get("audio"), sample_inputs.get("visual"))
            args = tuple(x for x in args if x is not None)
        elif isinstance(sample_inputs, (tuple, list)):
            args = sample_inputs
        else:
            args = (sample_inputs,)
        
        macs, params = profile(model, inputs=args, verbose=False)
        return 2 * macs  # FLOPs ≈ 2*MACs（常见口径）
    # except Exception:
    #     try:
    #         from fvcore.nn import FlopCountAnalysis
    #         if isinstance(sample_inputs, dict):
    #             # fvcore 支持 kwargs
    #             flops = FlopCountAnalysis(model, sample_inputs).total()
    #         else:
    #             flops = FlopCountAnalysis(model, sample_inputs).total()
    #         return flops
    #     except Exception:
    #         return None

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def model_size_mb(model):
    bytes_ = sum(p.numel() * p.element_size() for p in model.parameters())
    return bytes_ / (1024**2)

@torch.inference_mode()
def bench_infer(model, make_inputs_fn, repeats=30, warmup=10, amp=True, dtype=torch.float16):
    device = next(model.parameters()).device
    # 预热
    for _ in range(warmup):
        feed = make_inputs_fn()
        if amp and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=dtype):
                _ = model(**feed)
        else:
            _ = model(**feed)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    lat = []
    bs = None
    for _ in range(repeats):
        feed = make_inputs_fn()
        # 尝试推断 batch size
        any_t = next((v for v in feed.values() if torch.is_tensor(v)), None)
        if any_t is not None:
            bs = any_t.shape[0]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        if amp and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=dtype):
                _ = model(**feed)
        else:
            _ = model(**feed)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        lat.append(t1 - t0)

    lat_ms = [x * 1000 for x in lat]
    stats = {
        "latency_ms_mean": statistics.mean(lat_ms),
        "latency_ms_std": statistics.pstdev(lat_ms),
        "latency_ms_p50": statistics.median(lat_ms),
        "latency_ms_p95": sorted(lat_ms)[int(0.95 * len(lat_ms)) - 1],
        "throughput_samples_s": (bs * repeats) / sum(lat) if bs else None,
        "peak_mem_MB": (torch.cuda.max_memory_allocated(device) / (1024**2)) if torch.cuda.is_available() else None,
        "repeats": repeats
    }
    return stats

# -------------------- 试验入口（可改 sweep） -------------------
def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    torch.set_float32_matmul_precision("high")  # 对 Ampere+ 有帮助

    # 统一输入设置（按论文对齐）
    B = 8
    # L_list = [50,100]       # 文本长度
    L_t = 100
    d_l = 4096
    d_a = 1024
    d_v = 512

    rows = []
    # 准备三个模型（Ours 无/有检索 两种实例，MULT/DTN 无检索）
    # 如果 MULT/DTN 也支持检索，可同样加一组
    try:
        ours_noR = build_ours(device=device, dtype=dtype, use_retrieval=False)
        ours_R   = build_ours(device=device, dtype=dtype, use_retrieval=True)
        mult     = build_mult(device=device, dtype=dtype)
        # dtn      = build_dtn(device=device, dtype=dtype)
    except NotImplementedError as e:
        print("请先实现 build_ours/build_mult/build_dtn：", e)
        return

    # for L_t in L_list:
    cfg = dict(B=B, L=L_t, d_l=d_l,d_a = d_a,d_v = d_v, device=device, dtype=dtype)

    def make_inputs_fn():
                return make_inputs(batch_size=B, L=L_t, d_l=d_l,d_a = d_a,d_v = d_v, device=device, dtype=dtype)

    # 选择模型：R=0 -> ours_noR；R>0 -> ours_R
    model_ours = ours_R 

    # 样例一次输入用于 FLOPs
    sample = make_inputs_fn()
    # 统计函数
    def collect(name, model):
        total, trainable = count_params(model)
        flops = try_flops(model, sample)  # 可能为 None
        infer_stats = bench_infer(model, make_inputs_fn, repeats=30, warmup=10, amp=True, dtype=dtype)
        row = {
            "model": name,
            "B": B, "L": L_t, "d_l": d_l,"d_a": d_a,"d_v": d_v,
            "params_M": round(total / 1e6, 3),
            "trainable_M": round(trainable / 1e6, 3),
            "model_size_MB": round(model_size_mb(model), 2),
            "FLOPs_G": (round(flops / 1e9, 3) if flops is not None else None),
            **infer_stats
        }
        print(row)
        rows.append(row)

    collect("Ours", model_ours)
    collect("Ours w\o RAG", ours_noR)
    collect("MULT", mult)
    # collect("DTN", dtn)

    # 输出 CSV
    out = "/data/yangshengzun/knowledge-injection/compare_costs/compare_costs_result.csv"
    fieldnames = list(rows[0].keys())
    with open(out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(rows)
    print(f"=> Saved {out}")

if __name__ == "__main__":
    main()

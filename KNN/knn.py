#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
knn_baseline_with_dataloader.py

复用你现有的 get_dataset/get_dataloader 加载流程：
- 冻结本地 T5/FLAN-T5 Encoder 抽文本嵌入（mean pooling）
- 音频/视频是二维序列 [L, D]，做时间池化（mean/max/meanstd）
- 评测两种 KNN 回归基线：
    1) KNN-Concat：三模态池化后拼接向量做 KNN 回归（CV 选 k/metric/weight/scaler）
    2) KNN-Modal-Retrieval：各模态各自 KNN 回归，再用 dev 集选融合权重 α,β,γ（α+β+γ=1）

依赖：transformers, torch, numpy, scikit-learn
"""
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
import os
import argparse
import numpy as np
import torch
from typing import Dict, Optional, Tuple
from torch.utils.data import Dataset, DataLoader, BatchSampler
# ====== 这里按你的项目路径修改导入 ======
# 你可以：1) 改成你的模块名；或 2) 直接把 get_dataset/get_dataloader 粘到本文件中
# =====================================
from data.dataset import *
from transformers import AutoTokenizer, T5EncoderModel
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
from utils.utils import *
from global_configs import *
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
import csv

def multiclass_acc(preds, truths):
    """
    这里假设 multiclass_acc 已经在你项目里定义过
    如果没有，可以写一个简单版本：
    """
    preds = np.round(preds)
    truths = np.round(truths)
    return accuracy_score(truths, preds)

def compute_accurracy(preds, y_test, use_zero=False):
    preds = np.array(preds)
    y_test = np.array(y_test)
    test_preds_a7 = np.clip(preds, a_min=-3., a_max=3.)
    test_truth_a7 = np.clip(y_test, a_min=-3., a_max=3.)
    acc7 = multiclass_acc(test_preds_a7, test_truth_a7)

    # 去掉 y=0 样本
    non_zeros = np.array([i for i, e in enumerate(y_test) if e != 0 or use_zero])
    preds = preds[non_zeros]
    y_test = y_test[non_zeros]

    mae = np.mean(np.absolute(preds - y_test))
    corr = np.corrcoef(preds, y_test)[0][1]

    preds_bin = preds >= 0
    y_test_bin = y_test >= 0

    # f_score = f1_score(y_test_bin, preds_bin, average="weighted")
    f_score_w = f1_score(y_test_bin, preds_bin, average="weighted")
    f_score_b = f1_score(y_test_bin,preds_bin, average="binary")
    f_score_mi = f1_score(y_test_bin, preds_bin, average="micro")
    f_score_ma = f1_score(y_test_bin, preds_bin, average="macro")
    recall = recall_score(y_test_bin, preds_bin, average="weighted")
    pre = precision_score(y_test_bin, preds_bin, average="weighted")
    acc = accuracy_score(y_test_bin, preds_bin)

    return acc, f_score_w, f_score_b,f_score_mi,f_score_ma,recall, pre, mae, corr, acc7

def get_dataset():
    tokenizer = AutoTokenizer.from_pretrained(T5_PRETRAIN_PATH)
    # tokenizer = BertTokenizer.from_pretrained(PRETRAIN_PATH)
    # tokenizer = CLIPTokenizer.from_pretrained(PRETRAIN_PATH)
    mosi_path = 'merge/mosi_vgg_hubert.pkl'
    mosei_path = 'merge/mosei_vgg_hubert.pkl'
    meld_path = 'merge/meld_vgg_hubert.pkl'

    mosi_data_path = os.path.join(PATH, mosi_path)
    mosei_data_path = os.path.join(PATH, mosei_path)
    meld_data_path = os.path.join(PATH, meld_path)

    # with open(mosi_data_path, "rb") as handle:
    #     mosi_data = pickle.load(handle)

    with open(mosei_data_path, "rb") as handle:
        mosei_data = pickle.load(handle)

    # with open(meld_data_path, "rb") as handle:
    #     meld_data = pickle.load(handle)



    # train_dataset = mosi_data['train']
    # #print(train_dataset[0][0][2].shape)
    # dev_dataset = mosi_data['dev']
    # test_dataset = mosi_data['test']

    train_dataset = mosei_data['train']
    #print(train_dataset[0][0][2].shape)
    dev_dataset = mosei_data['dev']#除10余8
    test_dataset = mosei_data['test']#除10余1

    print(len(train_dataset)+len(dev_dataset)+len(test_dataset))

    # train_dataset = mosi_data['train'] + mosei_data['train'] + meld_data['train']
    # dev_dataset = mosi_data['dev'] + mosei_data['dev'] + meld_data['dev']
    # test_dataset = mosi_data['test'] + mosei_data['test'] + meld_data['test']


    train_data = MultimodalDataset(train_dataset, tokenizer)
    dev_data = MultimodalDataset(dev_dataset, tokenizer)
    test_data = MultimodalDataset(test_dataset, tokenizer)


    return train_data, dev_data, test_data




def get_dataloader(
        train_data, 
        dev_data, 
        test_data, 
        ):
    common = dict(collate_fn=padding_collate_fn, num_workers=4, pin_memory=True, persistent_workers=True)
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=8, **common)
    dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=8, **common)
    test_dataloader = DataLoader(test_data, shuffle=False, batch_size=8, **common)
    # dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=1, collate_fn=padding_collate_fn)
    # test_dataloader = DataLoader(test_data, shuffle=False, batch_size=1, collate_fn=padding_collate_fn)
    num_train_optimization_steps = 0
    return train_dataloader, dev_dataloader, test_dataloader
# ---------- 根据你 batch 的键名做一次统一映射（按需改动） ----------
# 下面是假设：文本 input_ids/attention_mask；音频 'audio' 可选 'audio_mask'；视频 'video' 可选 'video_mask'；目标 'label'
KEYS = dict(
    text_ids="input_ids",
    text_mask="attention_masks",
    audio="audio",
    audio_mask="audio_mask",     # 没有就置 None
    video="video",
    video_mask="video_mask",     # 没有就置 None
    target="label"               # 或 "y" / "target"
)
# ------------------------------------------------------------------

def select_concat_on_dev(Xt_tr, Xa_tr, Xv_tr, y_tr, Xt_dev, Xa_dev, Xv_dev, y_dev):
    """
    在 dev 上轻量选参，避免 GridSearchCV 带来的大规模内存复制。
    """
    # 拼接（注意：np.concatenate 会分配一份新的矩阵，dtype 尽量用 float32，减少内存）
    X_tr_concat = np.concatenate(
        [Xt_tr.astype(np.float32, copy=False),
         Xa_tr.astype(np.float32, copy=False),
         Xv_tr.astype(np.float32, copy=False)], axis=1)
    X_dev_concat = np.concatenate(
        [Xt_dev.astype(np.float32, copy=False),
         Xa_dev.astype(np.float32, copy=False),
         Xv_dev.astype(np.float32, copy=False)], axis=1)

    best = None
    # 精简的网格，已经足够稳定；需要更快可以把 k 列表再缩短
    for metric in ("cosine", "euclidean"):
        for k in (5, 10, 20):
            for weighted in (True, False):
                y_dev_pred = knn_regress_predict(
                    X_tr_concat, y_tr, X_dev_concat,
                    n_neighbors=k, metric=metric, weighted=weighted
                )
                # 用你的函数计算指标；这里用 MAE 选参（更稳）
                acc, _,_,_,_, _, _, _, _, _ = compute_accurracy(y_dev_pred, y_dev)
                if (best is None) or (acc < best[-1]):
                    best = (metric, k, weighted, acc)
    metric, k, weighted, acc = best
    return {"metric": metric, "n_neighbors": k, "weighted": weighted}

def mean_pool_tokens(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()  # [B,L,1]
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1e-6)
    return summed / denom  # [B,H]

def _t5_embed_device(encoder: T5EncoderModel):
    try:
        return encoder.encoder.embed_tokens.weight.device
    except Exception:
        try:
            return next(encoder.parameters()).device
        except Exception:
            return torch.device("cpu")
        
@torch.no_grad()
def encode_text_batch(encoder: T5EncoderModel, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    # out = encoder(input_ids=input_ids, attention_mask=attention_mask)
    # pooled = mean_pool_tokens(out.last_hidden_state, attention_mask)  # [B,H]
    # return pooled
    dev = _t5_embed_device(encoder)
    input_ids = input_ids.to(dev, non_blocking=True)
    attention_mask = attention_mask.to(dev, non_blocking=True)

    use_amp = (dev.type == "cuda")
    # 用 bfloat16/float16 仅影响内部计算精度，不会改变输入设备
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
        out = encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = mean_pool_tokens(out.last_hidden_state, attention_mask)  # [B,H]
    return pooled


def pool_sequence_numpy(x: np.ndarray, mask: Optional[np.ndarray], mode: str = "meanstd") -> np.ndarray:
    """
    x: [B, L, D]  序列特征
    mask: [B, L]  布尔型 (True=padding, False=有效) 或 数值型 (1=有效, 0=padding)
    mode: "mean", "max", "meanstd"
    返回: 池化后的向量
    """
    if mask is None:
        mask = np.ones(x.shape[:2], dtype=np.float32)
    else:
        # 转成 numpy
        mask = mask.astype(np.float32)
        # 如果是 bool，True=padding → 转成 0；False=有效 → 转成 1
        if mask.dtype == np.bool_:
            mask = (~mask).astype(np.float32)

    mask = mask[..., None]                        # [B, L, 1]
    lengths = np.clip(mask.sum(axis=1), 1e-6, None)  # [B, 1]

    mean = (x * mask).sum(axis=1) / lengths       # [B, D]

    if mode == "mean":
        return mean
    elif mode == "max":
        neg_inf = -1e9
        x_f = np.where(mask > 0, x, neg_inf)
        return x_f.max(axis=1)
    elif mode == "meanstd":
        ex2 = (np.square(x) * mask).sum(axis=1) / lengths
        var = np.clip(ex2 - np.square(mean), 0.0, None)
        std = np.sqrt(var + 1e-6)
        return np.concatenate([mean, std], axis=1)
    else:
        raise ValueError(f"Unknown pooling mode: {mode}")



def harvest_embeddings(
    dataloader,
    t5_dir: str,
    device: str = "cuda",
    prefer_8bit: bool = False,
    prefer_4bit: bool = False,
    audio_pool: str = "meanstd",
    video_pool: str = "meanstd",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    遍历 dataloader，抽取三模态向量与标签：
      - 文本：冻结 T5 Encoder，token mean pooling
      - 音频/视频：[L,D] 做时间池化 -> 向量
      - 返回：text_feats [N,Dt], audio_feats [N,Da'], video_feats [N,Dv'], y [N]
    """
    # 加载本地 T5 Encoder（冻结）
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    device_map = "auto" if torch.cuda.is_available() else None
    load_kwargs = dict(device_map=device_map, torch_dtype=dtype)
    if prefer_8bit:
        load_kwargs.update(dict(load_in_8bit=True))
    elif prefer_4bit:
        load_kwargs.update(dict(load_in_4bit=True))

    tokenizer = AutoTokenizer.from_pretrained(t5_dir, use_fast=True, local_files_only=True)
    try:
        encoder = T5EncoderModel.from_pretrained(t5_dir, local_files_only=True, **load_kwargs)
    except Exception as e:
        print(f"[Warn] 8/4-bit load failed: {e}\nFalling back to regular dtype.")
        encoder = T5EncoderModel.from_pretrained(t5_dir, local_files_only=True,
                                                 device_map=device_map, torch_dtype=dtype)
    if device_map is None:
        encoder = encoder.to(device)
    encoder.eval()

    txt_list, aud_list, vid_list, y_list = [], [], [], []

    for batch in dataloader:
        # 取出 batch 中各字段（键名按需修改）
        sentence, input_ids, attn_mask, text_mask, video_seq, visual_len, video_mask, audio_seq, audio_len, audio_mask, source_label, label_id, segment = batch
        # input_ids = batch[KEYS["text_ids"]]
        # attn_mask = batch[KEYS["text_mask"]]
        # 如果数据集里文本不是 tokenized，而是原始 string，请在你的 MultimodalDataset 里做 tokenize
        # 这里假设已经是张量 [B,L]

        # if device_map is None:
        #     input_ids = input_ids.to(device)
        #     attn_mask = attn_mask.to(device)

        # with torch.autocast(device_type="cuda", dtype=dtype, enabled=(device=="cuda")):
        #     txt_emb = encode_text_batch(encoder, input_ids, attn_mask)  # [B,H]
        # txt_np = txt_emb.float().cpu().numpy()  # [B,H]

        txt_emb = encode_text_batch(encoder, input_ids, attn_mask)  # [B,H]
        txt_np = txt_emb.float().cpu().numpy()

        # audio_seq = batch[KEYS["audio"]].cpu().numpy()       # [B, L_a, D_a]
        # video_seq = batch[KEYS["video"]].cpu().numpy()       # [B, L_v, D_v]
        # audio_mask = batch.get(KEYS["audio_mask"], None)
        # video_mask = batch.get(KEYS["video_mask"], None)
        if audio_mask is not None: audio_mask = audio_mask.cpu().numpy()
        if video_mask is not None: video_mask = video_mask.cpu().numpy()

        aud_vec = pool_sequence_numpy(audio_seq, audio_mask, mode=audio_pool)
        vid_vec = pool_sequence_numpy(video_seq, video_mask, mode=video_pool)

        y = label_id.cpu().numpy().reshape(-1)  # [B]

        txt_list.append(txt_np)
        aud_list.append(aud_vec)
        vid_list.append(vid_vec)
        y_list.append(y)

    X_text = np.concatenate(txt_list, axis=0)
    X_audio = np.concatenate(aud_list, axis=0)
    X_video = np.concatenate(vid_list, axis=0)
    y = np.concatenate(y_list, axis=0).astype(float)
    return X_text, X_audio, X_video, y


def knn_regress_predict(X_tr, y_tr, X_te, n_neighbors=10, metric="cosine", weighted=True):
    norm = Normalizer('l2') if metric == "cosine" else StandardScaler()
    X_tr_n = norm.fit_transform(X_tr)
    X_te_n = norm.transform(X_te)
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric).fit(X_tr_n)
    dists, idxs = nn.kneighbors(X_te_n, return_distance=True)
    eps = 1e-6
    if weighted:
        w = 1.0 / (dists + eps)
        w = w / (w.sum(axis=1, keepdims=True) + eps)
        return (y_tr[idxs] * w).sum(axis=1)
    else:
        return y_tr[idxs].mean(axis=1)


def pick_best_modal(X_tr, y_tr, k_list=(5,10,20,50), metrics=("cosine","euclidean"), n_splits=5):
    best = (None, None, 1e18)
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    for k in k_list:
        for m in metrics:
            fold = []
            for tr_idx, va_idx in cv.split(X_tr):
                y_hat = knn_regress_predict(X_tr[tr_idx], y_tr[tr_idx], X_tr[va_idx],
                                            n_neighbors=k, metric=m, weighted=True)
                fold.append(mean_absolute_error(y_tr[va_idx], y_hat))
            mae = float(np.mean(fold))
            if mae < best[2]:
                best = (k, m, mae)
    return best[0], best[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t5_dir", help="本地 T5/FLAN-T5 Encoder 目录（含 config.json 等）",default=T5_PRETRAIN_PATH)
    ap.add_argument("--prefer_8bit", action="store_true")
    ap.add_argument("--prefer_4bit", action="store_true")
    ap.add_argument("--audio_pool", choices=["mean","max","meanstd"], default="meanstd")
    ap.add_argument("--video_pool", choices=["mean","max","meanstd"], default="meanstd")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_csv", type=str, default="/data/yangshengzun/knowledge-injection/KNN/result.csv")
    args = ap.parse_args()

    # ====== 1) 用你原来的方式拿到 dataloader ======
    train_data, dev_data, test_data = get_dataset()              # 走你的 tokenizer/T5_PRETRAIN_PATH
    train_loader, dev_loader, test_loader = get_dataloader(train_data, dev_data, test_data)

    # ====== 2) 抽取三模态向量 ======
    print("[Train] 抽取三模态向量 ...")
    Xt_tr, Xa_tr, Xv_tr, y_tr = harvest_embeddings(
        train_loader, args.t5_dir, args.device, args.prefer_8bit, args.prefer_4bit,
        audio_pool=args.audio_pool, video_pool=args.video_pool
    )
    print("[Dev] 抽取三模态向量 ...")
    Xt_dev, Xa_dev, Xv_dev, y_dev = harvest_embeddings(
        dev_loader, args.t5_dir, args.device, args.prefer_8bit, args.prefer_4bit,
        audio_pool=args.audio_pool, video_pool=args.video_pool
    )
    print("[Test] 抽取三模态向量 ...")
    Xt_te,  Xa_te,  Xv_te,  y_te  = harvest_embeddings(
        test_loader, args.t5_dir, args.device, args.prefer_8bit, args.prefer_4bit,
        audio_pool=args.audio_pool, video_pool=args.video_pool
    )

    # ====== 3) 基线一：KNN-Concat（在 train 上CV 选超参，dev 不参与CV；最终在 test 上评估） ======
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    # X_tr_concat = np.concatenate([Xt_tr, Xa_tr, Xv_tr], axis=1)
    # X_te_concat = np.concatenate([Xt_te, Xa_te, Xv_te], axis=1)

    # pipe_concat = Pipeline([
    #     ("scaler", Normalizer('l2')),  # 文本/语义嵌入通常配 L2+cosine
    #     ("knn", KNeighborsRegressor())
    # ])
    # param_grid = {
    #     "scaler": [Normalizer('l2'), StandardScaler()],
    #     "knn__n_neighbors": [1, 3, 5, 10, 20, 50],
    #     "knn__weights": ["uniform", "distance"],
    #     "knn__metric": ["cosine", "euclidean"],
    # }
    # cv = KFold(n_splits=5, shuffle=True, random_state=0)
    # g = GridSearchCV(pipe_concat, param_grid, scoring="neg_mean_absolute_error", cv=cv, n_jobs=-1, verbose=1)
    # g.fit(X_tr_concat, y_tr)
    # y_pred_c = g.predict(X_te_concat)
    best_cfg = select_concat_on_dev(Xt_tr, Xa_tr, Xv_tr, y_tr, Xt_dev, Xa_dev, Xv_dev, y_dev)
    print("[Concat] best on dev:", best_cfg)

    # 用最佳配置在 test 上评一次
    X_tr_concat = np.concatenate(
        [Xt_tr.astype(np.float32, copy=False),
        Xa_tr.astype(np.float32, copy=False),
        Xv_tr.astype(np.float32, copy=False)], axis=1)
    X_te_concat = np.concatenate(
        [Xt_te.astype(np.float32, copy=False),
        Xa_te.astype(np.float32, copy=False),
        Xv_te.astype(np.float32, copy=False)], axis=1)

    y_pred_c = knn_regress_predict(
        X_tr_concat, y_tr, X_te_concat,
        n_neighbors=best_cfg["n_neighbors"],
        metric=best_cfg["metric"],
        weighted=best_cfg["weighted"]
    )

    acc_c, f_score_w_c, f_score_b_c,f_score_mi_c,f_score_ma_c,recall_C, pre_c, mae_c, corr_c, acc7_c = compute_accurracy(y_pred_c, y_te)

    print("=== KNN-Concat ===")
    print("Best params:", str(best_cfg))
    print(f"Test: Acc={acc_c:.4f}  F1_w={f_score_w_c:.4f}  F1_b={f_score_b_c:.4f}  F1_mi={f_score_mi_c:.4f}  F1_ma={f_score_ma_c:.4f}  Recall={recall_C:.4f}  Precision={pre_c:.4f}  "
        f"MAE={mae_c:.4f}  Corr={corr_c:.4f}  Acc7={acc7_c:.4f}")

    # # ====== 4) 基线二：KNN-Modal-Retrieval（各模态各自 KNN；在 dev 上选融合权重） ======
    # bt_k, bt_m = pick_best_modal(Xt_tr, y_tr)
    # ba_k, ba_m = pick_best_modal(Xa_tr, y_tr)
    # bv_k, bv_m = pick_best_modal(Xv_tr, y_tr)
    # print(f"[Per-modal best] text(k={bt_k},{bt_m}) audio(k={ba_k},{ba_m}) video(k={bv_k},{bv_m})")

    # # 用 train 拟合 -> 在 dev 上出三路预测
    # y_t_dev = knn_regress_predict(Xt_tr, y_tr, Xt_dev, bt_k, bt_m, True)
    # y_a_dev = knn_regress_predict(Xa_tr, y_tr, Xa_dev, ba_k, ba_m, True)
    # y_v_dev = knn_regress_predict(Xv_tr, y_tr, Xv_dev, bv_k, bv_m, True)

    # # 在 dev 上搜索融合权重（α+β+γ=1）
    # grid = [0.0, 0.25, 0.5, 0.75, 1.0]
    # best_mae, best_w = 1e18, None
    # for a in grid:
    #     for b in grid:
    #         c = 1.0 - a - b
    #         if c < 0: continue
    #         y_hat_dev = a*y_t_dev + b*y_a_dev + c*y_v_dev
    #         mae = mean_absolute_error(y_dev, y_hat_dev)
    #         if mae < best_mae:
    #             best_mae, best_w = mae, (a, b, c)
    # print(f"[Dev] best fusion weights (alpha,beta,gamma) = {best_w}")

    # # 固定最佳权重到 test
    # y_t_te = knn_regress_predict(Xt_tr, y_tr, Xt_te, bt_k, bt_m, True)
    # y_a_te = knn_regress_predict(Xa_tr, y_tr, Xa_te, ba_k, ba_m, True)
    # y_v_te = knn_regress_predict(Xv_tr, y_tr, Xv_te, bv_k, bv_m, True)
    # y_pred_m = best_w[0]*y_t_te + best_w[1]*y_a_te + best_w[2]*y_v_te

    # y_pred_m = best_w[0]*y_t_te + best_w[1]*y_a_te + best_w[2]*y_v_te

    # acc_m, f_score_w_c, f_score_b_c,f_score_mi_c,f_score_ma_c,recall_C, pre_c, mae_c, corr_c, acc7_c = compute_accurracy(y_pred_m, y_te)

    # print("=== KNN-Modal-Retrieval ===")
    # print(f"Fusion weights (alpha,beta,gamma) = {best_w}")
    # print(f"Test: Acc={acc_m:.4f}  F1={f1_m:.4f}  Recall={rec_m:.4f}  Precision={pre_m:.4f}  "
    #     f"MAE={mae_m:.4f}  Corr={corr_m:.4f}  Acc7={acc7_m:.4f}")
    
    if args.out_csv:
        rows = [
            {"method":"KNN-Concat",
            "Acc":acc_c, "F1_w":f_score_w_c, "F1_b":f_score_b_c,"F1_mi":f_score_mi_c,"F1_ma":f_score_ma_c,"Recall":recall_C, "Precision":pre_c,
            "MAE":mae_c, "Corr":corr_c, "Acc7":acc7_c,
            "best_params":str(best_cfg)},
            # {"method":"KNN-Modal-Retrieval",
            # "Acc":acc_m, "F1":f1_m, "Recall":rec_m, "Precision":pre_m,
            # "MAE":mae_m, "Corr":corr_m, "Acc7":acc7_m,
            # "best_params":f"weights={best_w}; per-modal(text={bt_k},{bt_m}; audio={ba_k},{ba_m}; video={bv_k},{bv_m})"}
        ]
        header = list(rows[0].keys())
        write_header = (not os.path.exists(args.out_csv)) or (os.path.getsize(args.out_csv) == 0)
        with open(args.out_csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header)
            if write_header: w.writeheader()          # 显式表头
            for r in rows: w.writerow(r)
        print(f"=> 指标已写入 {args.out_csv}")


if __name__ == "__main__":
    main()

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, BatchSampler
from sklearn.model_selection import train_test_split
from tqdm import tqdm, trange
from transformers.models.clip.tokenization_clip import CLIPTokenizer
import argparse
# from utils.utils import *
import pickle
# from data.dataset import *
from torch.nn import CrossEntropyLoss, L1Loss, MSELoss
from sklearn.metrics import accuracy_score, f1_score, recall_score,precision_score
from transformers import AutoTokenizer,T5Tokenizer
from sklearn.metrics import accuracy_score
from contextlib import redirect_stdout
from transformers import AutoModel,AutoConfig,T5ForConditionalGeneration
from torch.utils.data import ConcatDataset
import json
import torch
import numpy as np
import os
import random
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import argparse
import math
import re
import csv
from typing import Optional
from transformers import set_seed

T5_PRETRAIN_PATH = "/data/zhaoxianbing/CMU/pretrain/flan-t5-xxl"
#/data/zhaoxianbing/CMU/pretrain/flan-t5-xl
PATH  = '/data/zhaoxianbing/cmu-features'
# /data/zhaoxianbing/CMU/pretrain/flan-t5-xxl
# /data/zhaoxianbing/CMU/pretrain/t5-v1_1-xxl
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='mosi')
parser.add_argument("--path", type=str, default=T5_PRETRAIN_PATH)

args = parser.parse_args()

def padding_collate_fn(data_batch):
    sentence_list = [] 
    input_ids_list = []
    attention_mask_list = []

    source_label_list = []
    label_id_list = []
    segment_list = []
    
    for item in data_batch:
        sentence, input_ids, attention_mask, source_label, label_id, segment = item

        sentence_list.append(sentence)
        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        
        source_label_list.append(source_label)
        label_id_list.append(label_id)

        segment_list.append(segment)

    sentence = sentence_list
    input_ids = torch.stack(input_ids_list)
    attention_mask = torch.stack(attention_mask_list)

    source_label = torch.stack(source_label_list)
    label_id = torch.stack(label_id_list)
    segment = segment_list

    return (sentence, input_ids, attention_mask, source_label, label_id, segment)

_float_re = re.compile(r"([-+]?\d{1,2}(?:\.\d+)?|\.\d+)")

def parse_numeric_from_text(s: str, default: Optional[float] = float('nan')) -> float:
    """
    从生成文本中提取第一个浮点数（或整数），并 clamp 到 [-3.0, 3.0]。
    返回 float；解析失败则返回 default（默认为 NaN）。
    """
    if s is None:
        return default
    s = str(s).strip()
    # 先找浮点/整数（允许像 '.5', '2.0', '-1.5'）
    m = _float_re.search(s)
    if m:
        try:
            val = float(m.group(0))
            # clamp
            val = max(-3.0, min(3.0, val))
            return val
        except Exception:
            pass
    # 如果没找到浮点，尝试提取数字字符（包含负号）
    digits = re.findall(r"[-+]?\d+", s)
    if digits:
        try:
            val = float(digits[0])
            val = max(-3.0, min(3.0, val))
            return val
        except Exception:
            pass
    return default

def compute_accurracy(preds: list, y_test: list, use_zero: bool = False):
    """
    保留原接口与多个 F1 指标，同时增加 mae, corr, acc7, acc5。

    Returns:
        acc, f_w, f_b, f_mi, f_ma, recall, prec, mae, corr, acc7, acc5
    """
    preds = np.array(preds, dtype=float)
    y_test = np.array(y_test, dtype=float)

    # 过滤 y==0（除非 use_zero=True）
    nz_idx = np.array([i for i, e in enumerate(y_test) if (e != 0 or use_zero)])
    preds_nz = preds[nz_idx]
    y_nz = y_test[nz_idx]

    # 二值化评估（非负为正类）
    if preds_nz.size == 0:
        # 没有样本时返回 nan / 0 合理占位
        acc = f_w = f_b = f_mi = f_ma = recall = prec = float('nan')
        mae = corr = acc7 = acc5 = float('nan')
        return acc, f_w, f_b, f_mi, f_ma, recall, prec, mae, corr, acc7, acc5

    preds_bin = preds_nz >= 0
    y_bin = y_nz >= 0

    acc = float(accuracy_score(y_bin, preds_bin))
    f_w = float(f1_score(y_bin, preds_bin, average='weighted'))
    # 对 binary average 要保证正负两类存在，否则 sklearn 会报错 — 用 try/except 保障
    try:
        f_b = float(f1_score(y_bin, preds_bin, average='binary'))
    except Exception:
        f_b = float('nan')
    f_mi = float(f1_score(y_bin, preds_bin, average='micro'))
    try:
        f_ma = float(f1_score(y_bin, preds_bin, average='macro'))
    except Exception:
        f_ma = float('nan')
    try:
        recall = float(recall_score(y_bin, preds_bin, average='weighted'))
    except Exception:
        recall = float('nan')
    try:
        prec = float(precision_score(y_bin, preds_bin, average='weighted'))
    except Exception:
        prec = float('nan')

    # 回归指标：MAE, Corr（在去掉中性后计算）
    mae = float(np.mean(np.abs(preds_nz - y_nz)))
    try:
        # 若所有值相同，corr 会产生 nan 或除零异常，捕获并返回 nan
        corr_mat = np.corrcoef(preds_nz, y_nz)
        corr = float(corr_mat[0, 1])
    except Exception:
        corr = float('nan')

    # multiclass accuracy on clipped & rounded bins
    # acc7: clip to [-3,3], round -> integer classes
    p7 = np.rint(np.clip(preds_nz, -3.0, 3.0)).astype(int)
    t7 = np.rint(np.clip(y_nz, -3.0, 3.0)).astype(int)
    try:
        acc7 = float(accuracy_score(t7, p7))
    except Exception:
        acc7 = float('nan')

    # acc5: clip to [-2,2]
    p5 = np.rint(np.clip(preds_nz, -2.0, 2.0)).astype(int)
    t5 = np.rint(np.clip(y_nz, -2.0, 2.0)).astype(int)
    try:
        acc5 = float(accuracy_score(t5, p5))
    except Exception:
        acc5 = float('nan')

    return acc, f_w, f_b, f_mi, f_ma, recall, prec, mae, corr, acc7, acc5

class MultimodalDataset(Dataset):
    def __init__(self, data, tokenizer=None):
        self.sentences = []
        self.input_ids = []
        self.attention_masks = []
        self.senti_labels = []
        self.segments = []
        self.conv_labels = []

        for content in data:
            (text, visual, audio), senti_label, video_file = content

            if isinstance(senti_label, np.ndarray):
                conv_label = senti_label[0][0]
                senti_label = senti_label[0][0]
            else:
                conv_label = senti_label

            # taskname = "Multimodal Aspect-Based Sentiment Analysis task."
            # taskdefinition = "Given the text data, assign a sentiment label from ['negative', 'neutral', 'positive']."
            # outputformat= "Return label only without any other text."
            # question= "what is the sentiment about the text?"
            # option1 = "(a) neutral (b) negative (c) positive"
            # option2 = "neutral or negative or positive"
            # # prompt = f"Please perform{taskname} {taskdefinition} {outputformat} Sentence: {text} Label:"
            # # prompt = f"User: Please perform{taskname} {taskdefinition} {outputformat} Sentence: {text} Question: {question} :<answer>"
            # # prompt = f"Please perform{taskname} {taskdefinition} {outputformat} Sentence: {text} Question: {question} Answer:"
            # # prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\
            # # ### Instruction: Please perform{taskname} {taskdefinition} {outputformat} Sentence: {text}  \
            # # ### Instruction:  {question} Options: {option1} ### Response: "
            # # prompt = f"Please perform{taskname} {taskdefinition} {outputformat} Sentence: {text} {question}"
            # # prompt = f"Please perform{taskname} {taskdefinition} {outputformat} Sentence: {text} Question: {question} Options: {option1} Answer:"
            # # prompt = f"The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user‘s questions.\
            # # Human: Please perform{taskname} {taskdefinition} {outputformat} Human: {text} Human: {question} AI:"
            # # prompt = f"Please perform{taskname} {taskdefinition} {outputformat} Sentence: {text} Question: {question} Options: {option2} Answer:"
            # # prompt = f"Please perform{taskname} {taskdefinition} {outputformat} Sentence: {text} "
            # # prompt = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\
            # ### Instruction:  Please perform{taskname} {taskdefinition} {outputformat} ### Input: {text}  ### Input: {question} ### Response:"
            # prompt = f"The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. \
            # Human: Please perform{taskname} {taskdefinition} {outputformat} Human: {text} Human: {question} AI:"

            # system_content = {
            #     "role": "system",
            #     "content": [
            #         {"type": "text", "text": (
            #             "You are a helpful assistant. "
            #             "Given the multimodal content (an image and a text), predict the sentiment intensity "
            #             "as a single numeric value in the range [-3.0, 3.0]. "
            #             "Only reply with the numeric value (optionally with a sign and at most one decimal), "
            #             "and nothing else. Example valid responses: '2.0', '-1.5', '0', '3'."
            #         )}
            #     ]
            # }

            # user_content = {
            #     "role": "user",
            #     "content": [
            #         {"type": "image", "url": pil_image},
            #         {"type": "text", "text": text},
            #         # 下面这句是你指定的 prompt，保持最后的 “response: The sentiment is”
            #         {"type": "text", "text": "Please predict the sentiment intensity of the above multimodal content in the range [-3.0, 3.0]. response: The sentiment is"}
            #     ]
            # }
            prompt = f"Given the multimodal content (an image and a text), predict the sentiment intensity as a single numeric value in the range [-3.0, 3.0].Only reply with the numeric value \
                (optionally with a sign and at most one decimal),and nothing else. Example valid responses: '2.0', '-1.5', '0', '3'.text:{text}"
            # prompt = f"情感分类任务:将文本分类为积极或消极.待分类文本:{text}"
            tokenized = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)
            self.sentences.append(text)
            self.input_ids.append(tokenized['input_ids'].squeeze(0))
            self.attention_masks.append(tokenized['attention_mask'].squeeze(0))

            senti_label = np.array([[senti_label]])
            senti_label = torch.tensor(senti_label.astype(np.float32)).cpu().detach()
            self.senti_labels.append(senti_label)

            conv_label = np.array([[conv_label]])
            conv_label = torch.tensor(conv_label.astype(np.float32)).cpu().detach()
            self.conv_labels.append(conv_label)
            self.segments.append(video_file)

    def __len__(self):
        return len(self.senti_labels)
    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        input_ids = self.input_ids[idx]
        attention_mask = self.attention_masks[idx]
        senti_label = self.senti_labels[idx]
        conv_label = self.conv_labels[idx]

        segment = self.segments[idx]
        return sentence, input_ids, attention_mask, senti_label, conv_label, segment


# def compute_accurracy(preds, y_test,use_zero=False):
#     preds = np.array(preds)
#     y_test = np.array(y_test)
#     # test_preds_a7 = np.clip(preds, a_min=-3., a_max=3.)
#     # test_truth_a7 = np.clip(y_test, a_min=-3., a_max=3.)
#     # test_preds_a5 = np.clip(preds, a_min=-2., a_max=2.)
#     # test_truth_a5 = np.clip(y_test, a_min=-2., a_max=2.)
#     # acc7 = multiclass_acc(test_preds_a7, test_truth_a7)
#     # acc5 = multiclass_acc(test_preds_a5, test_truth_a5)
#     non_zeros = np.array([i for i, e in enumerate(y_test) if e != 0 ])
#     preds = preds[non_zeros]
#     y_test = y_test[non_zeros]
#     # mae = np.mean(np.absolute(preds - y_test))
#     # corr = np.corrcoef(preds, y_test)[0][1]
#     preds = preds >= 0
#     y_test = y_test >= 0
#     f_score = f1_score(y_test, preds, average="weighted")
#     recall = recall_score(y_test,preds,average="weighted")
#     pre = precision_score(y_test,preds,average="weighted")
#     acc = accuracy_score(y_test, preds)
#     #return acc, mae, corr, f_score, acc5, acc7
#     return acc, f_score, recall, pre 


def get_dataset(tokenizer,args):
    mosi_path = 'merge/mosi_vgg_hubert.pkl'
    mosei_path = 'merge/mosei_vgg_hubert.pkl'
    meld_path = 'merge/meld_vgg_hubert.pkl'

    mosi_data_path = os.path.join(PATH, mosi_path)
    mosei_data_path = os.path.join(PATH, mosei_path)
    meld_data_path = os.path.join(PATH, meld_path)

    if args.dataset == 'mosi':
        with open(mosi_data_path, "rb") as handle:
            mosi_data = pickle.load(handle)

        train_dataset = mosi_data['train']
        dev_dataset = mosi_data['dev']
        test_dataset = mosi_data['test']
    elif args.dataset == 'mosei':
        with open(mosei_data_path, "rb") as handle:
            mosei_data = pickle.load(handle)

        train_dataset = mosei_data['train']
        dev_dataset = mosei_data['dev']#除10余8
        test_dataset = mosei_data['test']#除10余1

    print(len(train_dataset)+len(dev_dataset)+len(test_dataset))

    # train_data = MultimodalDataset(train_dataset, tokenizer)
    # dev_data = MultimodalDataset(dev_dataset, tokenizer)
    test_data = MultimodalDataset(test_dataset, tokenizer)
    # full_data = ConcatDataset([train_data, dev_data, test_data])


    return test_data




def get_dataloader(
        test_data, 
        ):
    # train_dataloader = DataLoader(train_data, shuffle=True, batch_size=1, collate_fn=padding_collate_fn)
    # dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=1, collate_fn=padding_collate_fn)
    test_dataloader = DataLoader(test_data, shuffle=False, batch_size=1, collate_fn=padding_collate_fn)
    # full_dataloader = DataLoader(full_data, shuffle=False, batch_size=1, collate_fn=padding_collate_fn)
    return test_dataloader

# ,max_new_tokens=2


def test(model, dev_dataloader,tokenizer,args):
    # model.eval()  
    y_test = []
    preds = []
    results = []
    # with torch.no_grad():
    for step, batch in enumerate(tqdm(dev_dataloader, desc="Iteration")):
        sentence, input_ids, attention_mask, source_label, label_id, segment = batch
        input_ids, attention_mask, label_id = input_ids.to(model.device), attention_mask.to(model.device), label_id.to(model.device)
        outputs= model.generate(input_ids=input_ids ,attention_mask=attention_mask)
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        val = parse_numeric_from_text(result, default=float('nan'))
        y_test_val = label_id.view(-1).cpu().detach().tolist()
        if math.isnan(val):
                # choose fallback policy, e.g., 0.0
            val = 0.0
        preds.append(val)
        y_test += y_test_val
        temp = {'pred':preds,'y':y_test}
        dataset_name = args.dataset
        model_name = os.path.basename(os.path.normpath(args.path))
        with open(f'/data/yangshengzun/LLM/LLMresults/{model_name}_{dataset_name}my_list.pkl', 'wb') as file:
            pickle.dump(temp, file)
    # 计算指标
    acc, f_w, f_b, f_mi, f_ma, recall, prec, mae, corr, acc7, acc5 = compute_accurracy(preds, y_test)
    return acc, f_w, f_b, f_mi, f_ma, recall, prec, mae, corr, acc7, acc5
        


tokenizer = T5Tokenizer.from_pretrained(T5_PRETRAIN_PATH)
test_data = get_dataset(tokenizer,args)
test_dataloader= get_dataloader(test_data)
model = T5ForConditionalGeneration.from_pretrained(T5_PRETRAIN_PATH, device_map="auto")
seeds = [3407,3408,3409,3410,3411]
all_results = []
for seed in seeds:
        print(f"\n=== Running seed={seed} ===")
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        acc, f_w, f_b, f_mi, f_ma, recall, prec, mae, corr, acc7, acc5 = test(model,test_dataloader,tokenizer,args)
        print(f"Seed={seed} Results: Acc={acc:.4f}, F1_w={f_w:.4f}, F1_b={f_b:.4f}, "
          f"F1_mi={f_mi:.4f}, F1_ma={f_ma:.4f}, Recall={recall:.4f}, Precision={prec:.4f}, "
          f"MAE={mae:.4f}, Corr={(corr if not np.isnan(corr) else 'nan')}, Acc7={acc7:.4f}, Acc5={acc5:.4f}", flush=True)
        result = {
            "seed": int(seed),
            "acc": float(acc),
            "f1_weighted": float(f_w),
            "f1_binary": float(f_b) if not np.isnan(f_b) else "",
            "f1_micro": float(f_mi),
            "f1_macro": float(f_ma) if not np.isnan(f_ma) else "",
            "recall": float(recall) if not np.isnan(recall) else "",
            "precision": float(prec) if not np.isnan(prec) else "",
            "mae": float(mae) if not np.isnan(mae) else "",
            "corr": float(corr) if not np.isnan(corr) else "",
            "acc7": float(acc7),
            "acc5": float(acc5),
        }

        all_results.append(result)


    # 保存 CSV（每个 seed 的结果 + mean/std）
fieldnames = list(all_results[0].keys())
dataset_name = args.dataset
model_name = os.path.basename(os.path.normpath(args.path))
out_csv = f"/data/yangshengzun/LLM/LLMresults/{model_name}_{dataset_name}_multi_seed.csv"
with open(out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(all_results)
    # mean/std
    metrics = [k for k in fieldnames if k != "seed"]
    mean_row = {"seed":"mean"}
    std_row = {"seed":"std"}
    for m in metrics:
        vals = [r[m] for r in all_results]
        mean_row[m] = float(np.mean(vals))
        std_row[m] = float(np.std(vals))
    w.writerow(mean_row)
    w.writerow(std_row)
print(f"=> Results saved to {out_csv}")

# if __name__ == "__main__":
#     main()
# acc, f_score, recall, pre = test(model,test_dataloader,tokenizer,args)
# print(acc, f_score, recall, pre)

# Tokenize 与生成
# input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
# outputs = model.generate(input_ids, max_new_tokens=2)  # 限制生成长度
# result = tokenizer.decode(outputs[0], skip_special_tokens=True)

# 解析输出
# sentiment = 1 if result.strip() == "1" else 0
# print(f"文本: {input_text} → 情感: {sentiment}")
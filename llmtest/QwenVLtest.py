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
from transformers import AutoTokenizer,T5Tokenizer,LlamaTokenizer
from sklearn.metrics import accuracy_score
from contextlib import redirect_stdout
from transformers import AutoModel,AutoConfig,T5ForConditionalGeneration,LlamaForCausalLM,AutoModelForCausalLM
from torch.utils.data import ConcatDataset
import json
import torch
import numpy as np
import os
import random
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import argparse
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
try:
    # 若你使用 HF 的 multimodal processor（推荐 Qwen-VL 等），可以 import
    from transformers import AutoProcessor
except Exception:
    AutoProcessor = None
import os
import random


T5_PRETRAIN_PATH = "/data/zhaoxianbing/CMU/pretrain/flan-t5-xxl"
LLAMA2_7B_PATH = "/data1/lvyiqing/.model/LLM-Research/llama-2-7b"
LLAMA2_13B_PATH = "/data1/lvyiqing/.model/LLM-Research/llama-2-13b"
PATH  = '/data/zhaoxianbing/cmu-features'
# /data/zhaoxianbing/CMU/pretrain/flan-t5-xxl
# /data/zhaoxianbing/CMU/pretrain/t5-v1_1-xxl
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='mosi')
parser.add_argument("--path", type=str, default='/data1/lvyiqing/.model/LLM-Research/llama-2-7b')

args = parser.parse_args()
PROMPT_TEMPLATES = ["Please perform{taskname} {taskdefinition} {outputformat} Sentence: {text} Label:",
"User: Please perform{taskname} {taskdefinition} {outputformat} Sentence: {text} Question: {question} :<answer>",
"Please perform{taskname} {taskdefinition} {outputformat} Sentence: {text} Question: {question} Answer:",
"Below is an instruction that describes a task. Write a response that appropriately completes the request.\
### Instruction: Please perform{taskname} {taskdefinition} {outputformat} Sentence: {text}  \
### Instruction:  {question} Options: {option1} ### Response: ",
"Please perform{taskname} {taskdefinition} {outputformat} Sentence: {text} {question}",
"Please perform{taskname} {taskdefinition} {outputformat} Sentence: {text} Question: {question} Options: {option1} Answer:",
"The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\
Human: Please perform{taskname} {taskdefinition} {outputformat} Human: {text} Human: {question} AI:",
"Please perform{taskname} {taskdefinition} {outputformat} Sentence: {text} Question: {question} Options: {option2} Answer:",
"Please perform{taskname} {taskdefinition} {outputformat} Sentence: {text} ",
"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\
### Instruction:  Please perform{taskname} {taskdefinition} {outputformat} ### Input: {text}  ### Input: {question} ### Response:",
"The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. \
Human: Please perform{taskname} {taskdefinition} {outputformat} Human: {text} Human: {question} AI:"
]
def get_random_image_path(folder_path):
    # 获取目录下所有文件
    files = os.listdir(folder_path)
    # 只保留图片类型
    image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    
    if not image_files:
        raise FileNotFoundError(f"目录 {folder_path} 下没有找到图片文件")
    
    # 随机选择一张图片
    chosen_file = random.choice(image_files)
    return os.path.join(folder_path, chosen_file)


def make_collate_fn(tokenizer, image_size=384, return_pil_for_qwen=True):
    """
    return_pil_for_qwen=True 时，返回的 batch tuple 最后一个元素是 PIL.Image 列表（len=B）
    这样 test_qwen 会把每张 PIL 直接传给 tokenizer.from_list_format。
    若 return_pil_for_qwen=False，则保持原来返回 images tensor 的逻辑（兼容其他模型）。
    """
    transform = T.Compose([
        T.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    def padding_collate_fn(data_batch):
        sentence_list = []
        source_label_list = []
        label_id_list = []
        segment_list = []
        pil_imgs = []
        tensor_imgs = []
        mosi_path = "/data/zhaoxianbing/multimodal-dataset/cmu/CMU-RAW/MOSI/Raw/Video/Frames"
        mosei_path = "/data/zhaoxianbing/multimodal-dataset/cmu/CMU-RAW/MOSEI/mosei-frames"
        for item in data_batch:
            sentence, source_label, label_id, segment = item
            if args.dataset == 'mosi':
                segment = os.path.join(mosi_path, segment)
            elif args.dataset == 'mosei':
                segment = os.path.join(mosei_path, segment)
            segment = get_random_image_path(segment)
            sentence_list.append(sentence)
            source_label_list.append(source_label)
            label_id_list.append(label_id)
            segment_list.append(segment)

            # 读取图像（segment 既可以是文件路径，也可以是 URL，但这里我们假设是本地路径）
            try:
                pil = Image.open(segment).convert("RGB")
            except Exception as e:
                print(f"[WARN] cannot open image {segment}: {e}", flush=True)
                pil = Image.new("RGB", (image_size, image_size), "black")
            pil_imgs.append(pil)
            # 如果需要 tensor 版本，也可以同时生成
            tensor_imgs.append(transform(pil))

        # tokenized = tokenizer(sentence_list, return_tensors="pt", truncation=True, padding=True, max_length=512)
        # input_ids = tokenized["input_ids"]
        # attention_mask = tokenized["attention_mask"]
        source_label = torch.stack(source_label_list)
        label_id = torch.stack(label_id_list)

        if return_pil_for_qwen:
            # 返回 PIL 列表供 Qwen tokenizer 使用
            return (sentence_list, source_label, label_id, segment_list, pil_imgs)
        else:
            images_tensor = torch.stack(tensor_imgs, dim=0)  # [B, C, H, W]
            return (sentence_list, source_label, label_id, segment_list, images_tensor)

    return padding_collate_fn



def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_results(all_results, out_csv):
    # 写每个种子的结果
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)

        # 计算均值和标准差
        metrics = [k for k in all_results[0].keys() if k != "seed"]
        stats_mean = {"seed": "mean"}
        stats_std = {"seed": "std"}
        for m in metrics:
            vals = [r[m] for r in all_results]
            stats_mean[m] = float(np.mean(vals))
            stats_std[m] = float(np.std(vals))
        writer.writerow(stats_mean)
        writer.writerow(stats_std)

    print(f"=> 结果和均值/标准差已写入 {out_csv}")



class MultimodalDataset(Dataset):
    def __init__(self, data, tokenizer=None):
        self.sentences = []
        # self.input_ids = []
        # self.attention_masks = []
        self.senti_labels = []
        self.segments = []
        self.conv_labels = []

        for content in data:
            (text, visual, audio), senti_label, video_file = content
            # print(video_file)
            if isinstance(senti_label, np.ndarray):
                conv_label = senti_label[0][0]
                senti_label = senti_label[0][0]
            else:
                conv_label = senti_label

            taskname = "Multimodal Aspect-Based Sentiment Analysis task."
            taskdefinition = "Given the text-image pair, assign a sentiment label from ['negative', 'neutral', 'positive']."
            outputformat= "Return label only without any other text."
            question= "what is the sentiment about the text-image pair?"
            option1 = "(a) neutral (b) negative (c) positive"
            option2 = "neutral or negative or positive"
            # prompt = f"Please perform{taskname} {taskdefinition} {outputformat} Sentence: {text} Label:"
            # prompt = f"User: Please perform{taskname} {taskdefinition} {outputformat} Sentence: {text} Question: {question} :<answer>"
            # prompt = f"Please perform{taskname} {taskdefinition} {outputformat} Sentence: {text} Question: {question} Answer:"
            prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\
            ### Instruction: Please perform{taskname} {taskdefinition} {outputformat} Sentence: {text}  \
            ### Instruction:  {question} Options: {option1} ### Response: "
            # prompt = f"Please perform{taskname} {taskdefinition} {outputformat} Sentence: {text} {question}"
            # prompt = f"Please perform{taskname} {taskdefinition} {outputformat} Sentence: {text} Question: {question} Options: {option1} Answer:"
            # prompt = f"The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user‘s questions.\
            #         Human: Please perform{taskname} {taskdefinition} {outputformat} Human: {text} Human: {question} AI:"
            # prompt = f"Please perform{taskname} {taskdefinition} {outputformat} Sentence: {text} Question: {question} Options: {option2} Answer:"
            # prompt = f"Please perform{taskname} {taskdefinition} {outputformat} Sentence: {text} "
            # prompt = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\
            # ## Instruction:  Please perform{taskname} {taskdefinition} {outputformat} ### Input: {text}  ### Input: {question} ### Response:"
            # prompt = f"The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. \
            #     Human: Please perform{taskname} {taskdefinition} {outputformat} Human: {text} Human: {question} AI:"
            self.sentences.append(prompt)
            # self.input_ids.append(tokenized['input_ids'].squeeze(0))
            # self.attention_masks.append(tokenized['attention_mask'].squeeze(0))

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
        # input_ids = self.input_ids[idx]
        # attention_mask = self.attention_masks[idx]
        senti_label = self.senti_labels[idx]
        conv_label = self.conv_labels[idx]

        segment = self.segments[idx]
        # return sentence, input_ids, attention_mask, senti_label, conv_label, segment
        return sentence, senti_label, conv_label, segment


def compute_accurracy(preds, y_test,use_zero=False):
    preds = np.array(preds)
    y_test = np.array(y_test)
    # test_preds_a7 = np.clip(preds, a_min=-3., a_max=3.)
    # test_truth_a7 = np.clip(y_test, a_min=-3., a_max=3.)
    # test_preds_a5 = np.clip(preds, a_min=-2., a_max=2.)
    # test_truth_a5 = np.clip(y_test, a_min=-2., a_max=2.)
    # acc7 = multiclass_acc(test_preds_a7, test_truth_a7)
    # acc5 = multiclass_acc(test_preds_a5, test_truth_a5)
    non_zeros = np.array([i for i, e in enumerate(y_test) if e != 0 ])
    preds = preds[non_zeros]
    y_test = y_test[non_zeros]
    # mae = np.mean(np.absolute(preds - y_test))
    # corr = np.corrcoef(preds, y_test)[0][1]
    preds = preds >= 0
    y_test = y_test >= 0
    f_score_w = f1_score(y_test, preds, average="weighted")
    f_score_b = f1_score(y_test, preds, average="binary")
    f_score_mi = f1_score(y_test, preds, average="micro")
    f_score_ma = f1_score(y_test, preds, average="macro")
    recall = recall_score(y_test,preds,average="weighted")
    pre = precision_score(y_test,preds,average="weighted")
    acc = accuracy_score(y_test, preds)
    #return acc, mae, corr, f_score, acc5, acc7
    return acc, f_score_w,f_score_b,f_score_mi,f_score_ma, recall, pre 


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

    train_data = MultimodalDataset(train_dataset, tokenizer)
    dev_data = MultimodalDataset(dev_dataset, tokenizer)
    test_data = MultimodalDataset(test_dataset, tokenizer)
    full_data = ConcatDataset([train_data, dev_data, test_data])


    return train_data, dev_data, test_data,full_data




def get_dataloader(train_data, dev_data, test_data, full_data, tokenizer, image_size=384):
    collate = make_collate_fn(tokenizer, image_size=image_size)
    common = dict(collate_fn=collate, num_workers=4, pin_memory=True, persistent_workers=True)
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=16, **common)
    dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=16, **common)
    test_dataloader = DataLoader(test_data, shuffle=False, batch_size=16, **common)
    full_dataloader = DataLoader(full_data, shuffle=False, batch_size=16, **common)
    return train_dataloader, dev_dataloader, test_dataloader, full_dataloader

# ,max_new_tokens=2

def test_qwen(model, dev_dataloader, tokenizer, args):
    """
    对每个样本使用 Qwen 的 chat API：query = tokenizer.from_list_format([{'image': pil}, {'text': sentence}])
    model.chat(tokenizer, query=query, history=None) -> (response, history)
    返回同原 compute_accurracy 兼容的指标。
    """
    y_test = []
    preds = []

    model.eval()
    # 保证 tokenizer 有 pad_token
    if tokenizer.pad_token is None and hasattr(tokenizer, "eos_token"):
        tokenizer.pad_token = tokenizer.eos_token

    for step, batch in enumerate(tqdm(dev_dataloader, desc="Iteration")):
        # batch: (sentence_list, input_ids, attention_mask, source_label, label_id, segment_list, pil_imgs)
        sentence_list, source_label, label_id, segment_list, pil_imgs = batch

        B = len(sentence_list)
        # 对于 Qwen 的 chat，我们逐样本调用（确保 query 为单样本格式）
        for i in range(B):
            pil = pil_imgs[i]
            sent = sentence_list[i]
            # 构造 query（和官方示例一致）
            # from_list_format 接受 [{'image': <url_or_PIL>}, {'text': '...'}]
            query = tokenizer.from_list_format([
                {'image': segment_list[i]},
                {'text': sent}
            ])

            # model.chat 返回 (response, history)
            
            response, _history = model.chat(tokenizer, query=query, history=None)
            # response 可能是字符串或结构（依据 Qwen 定义）；示例中 response 是字符串结果
            if isinstance(response, (list, tuple)):
                # 有些版本返回 (str,) 或类似结构，取第一个
                resp_text = response[0] if len(response) > 0 else ""
            else:
                resp_text = str(response)

            txt = resp_text.lower()
            # 你的情感判定逻辑：包含 "positive"/"neutral" 或其他 -> -1
            if "positive" in txt or "积极" in txt:
                sentiment = 1
            elif "neutral" in txt or "中立" in txt or "neutral" in txt:
                sentiment = 0
            else:
                sentiment = -1
            # print(txt)
            # print(sentiment)
            preds.append(sentiment)

        # collect ground-truths for this batch
        y_vals = label_id.view(-1).cpu().detach().tolist()
        y_test.extend(y_vals)

    # 统计指标（使用你之前的 compute_accurracy）
    acc, f_score_w, f_score_b, f_score_mi, f_score_ma, recall, pre = compute_accurracy(preds, y_test)
    # 保存中间结果以便检查
    temp = {'pred': preds, 'y': y_test}
    dataset_name = args.dataset
    model_name = os.path.basename(os.path.normpath(args.path))
    with open(f'/data/yangshengzun/LLM/LLMresults/{model_name}_{dataset_name}_qwen_my_list.pkl', 'wb') as file:
        pickle.dump(temp, file)

    return acc, f_score_w, f_score_b, f_score_mi, f_score_ma, recall, pre

# tokenizer = LlamaTokenizer.from_pretrained(LLAMA2_7B_PATH)
# train_data, dev_data, test_data,full_data = get_dataset(tokenizer,args)
# train_dataloader, dev_dataloader, test_dataloader,full_dataloader = get_dataloader(train_data, dev_data, test_data,full_data)
# model = LlamaForCausalLM.from_pretrained(LLAMA2_7B_PATH, device_map="auto")
# acc, f_score_w,f_score_b,f_score_mi,f_score_ma, recall, pre = test(model,test_dataloader,tokenizer,args)
# print(acc, f_score_w,f_score_b,f_score_mi,f_score_ma, recall, pre)

# Tokenize 与生成
# input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
# outputs = model.generate(input_ids, max_new_tokens=2)  # 限制生成长度
# result = tokenizer.decode(outputs[0], skip_special_tokens=True)

# 解析输出
# sentiment = 1 if result.strip() == "1" else 0
# print(f"文本: {input_text} → 情感: {sentiment}")
def run_once(seed, args):
    print(f"\n===== Running with seed={seed} =====")
    set_seed(seed)

    # 1) 加载 Qwen tokenizer（trust_remote_code=True）
    tokenizer = AutoTokenizer.from_pretrained(args.path, trust_remote_code=True)

    # 2) 准备 dataset/dataloader（collate 返回 PIL 列表）
    train_data, dev_data, test_data, full_data = get_dataset(tokenizer, args)
    collate = make_collate_fn(tokenizer, image_size=384, return_pil_for_qwen=True)
    # 使用单进程或少 worker 以便 debug；上线上可以调大 num_workers
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=16, collate_fn=collate, num_workers=0)
    dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=16, collate_fn=collate, num_workers=0)
    test_dataloader = DataLoader(test_data, shuffle=False, batch_size=16, collate_fn=collate, num_workers=0)

    # 3) 加载 Qwen 模型（device_map="auto" 或 "cuda"）
    # 你在示例中用 device_map="cuda"
    model = AutoModelForCausalLM.from_pretrained(args.path, device_map="cuda", trust_remote_code=True).eval()

    # 4) 运行测试（Qwen 专用）
    acc, f_score_w, f_score_b, f_score_mi, f_score_ma, recall, pre = test_qwen(model, test_dataloader, tokenizer, args)

    print(f"Seed={seed} Results:",
          f"Acc={acc:.4f}, F1_w={f_score_w:.4f}, F1_b={f_score_b:.4f}, "
          f"F1_mi={f_score_mi:.4f}, F1_ma={f_score_ma:.4f}, Recall={recall:.4f}, Precision={pre:.4f}")

    result = {
        "seed": seed,
        "acc": acc,
        "f1_weighted": f_score_w,
        "f1_binary": f_score_b,
        "f1_micro": f_score_mi,
        "f1_macro": f_score_ma,
        "recall": recall,
        "precision": pre,
    }
    return result


SEEDS = [3407, 3408, 3409, 3410, 3411]
if __name__ == "__main__":
    all_results = []
    for seed in SEEDS:
        res = run_once(seed, args)
        all_results.append(res)

    # 保存成 CSV
    import csv
    dataset_name = args.dataset
    model_name = os.path.basename(os.path.normpath(args.path))
    out_csv = f"/data/yangshengzun/LLM/LLMresults/{model_name}_{dataset_name}_multi_seed.csv"
    # with open(out_csv, "w", newline="") as f:
    #     writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
    #     writer.writeheader()
    #     writer.writerows(all_results)
    # print(f"\n所有种子结果已保存到 {out_csv}")
    save_results(all_results, out_csv)



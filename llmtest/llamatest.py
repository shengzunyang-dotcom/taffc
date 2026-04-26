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
from transformers import AutoModel,AutoConfig,T5ForConditionalGeneration,LlamaForCausalLM
from torch.utils.data import ConcatDataset
import json
import torch
import numpy as np
import os
import random
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import argparse


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

def make_collate_fn(tokenizer):
    def padding_collate_fn(data_batch):
        sentence_list = [] 
        source_label_list = []
        label_id_list = []
        segment_list = []
        
        for item in data_batch:
            sentence, source_label, label_id, segment = item
            sentence_list.append(sentence)
            source_label_list.append(source_label)
            label_id_list.append(label_id)
            segment_list.append(segment)

        tokenized = tokenizer(sentence_list, return_tensors="pt", truncation=True, padding=True, max_length=512)
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]

        source_label = torch.stack(source_label_list)
        label_id = torch.stack(label_id_list)
        return (sentence_list, input_ids, attention_mask, source_label, label_id, segment_list)
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
    def __init__(self, data, tokenizer,prompt_template):
        self.sentences = []
        # self.input_ids = []
        # self.attention_masks = []
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


            prompt = prompt_template.format(
                    taskname = "Multimodal Aspect-Based Sentiment Analysis task.",
                    taskdefinition = "Given the text, assign a sentiment label from ['negative', 'neutral', 'positive'].",
                    outputformat= "Return label only without any other text.",
                    question= "what is the sentiment about the text?",
                    option1 = "(a) neutral (b) negative (c) positive",
                    option2 = "neutral or negative or positive",
                    text = text
                )

            # taskname = "Multimodal Aspect-Based Sentiment Analysis task."
            # taskdefinition = "Given the text, assign a sentiment label from ['negative', 'neutral', 'positive']."
            # outputformat= "Return label only without any other text."
            # question= "what is the sentiment about the text?"
            # option1 = "(a) neutral (b) negative (c) positive"
            # option2 = "neutral or negative or positive"
            # prompt = f"Please perform{taskname} {taskdefinition} {outputformat} Sentence: {text} Label:"
            # prompt = f"User: Please perform{taskname} {taskdefinition} {outputformat} Sentence: {text} Question: {question} :<answer>"
            # prompt = f"Please perform{taskname} {taskdefinition} {outputformat} Sentence: {text} Question: {question} Answer:"
            # prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\
            # ### Instruction: Please perform{taskname} {taskdefinition} {outputformat} Sentence: {text}  \
            # ### Instruction:  {question} Options: {option1} ### Response: "
            # prompt = f"Please perform{taskname} {taskdefinition} {outputformat} Sentence: {text} {question}"
            # prompt = f"Please perform{taskname} {taskdefinition} {outputformat} Sentence: {text} Question: {question} Options: {option1} Answer:"
            # prompt = f"The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user‘s questions.\
                    # Human: Please perform{taskname} {taskdefinition} {outputformat} Human: {text} Human: {question} AI:"
            # prompt = f"Please perform{taskname} {taskdefinition} {outputformat} Sentence: {text} Question: {question} Options: {option2} Answer:"
                    # prompt = f"Please perform{taskname} {taskdefinition} {outputformat} Sentence: {text} "
                    # prompt = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\
                    ### Instruction:  Please perform{taskname} {taskdefinition} {outputformat} ### Input: {text}  ### Input: {question} ### Response:"
            # prompt = f"The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. \
                    # Human: Please perform{taskname} {taskdefinition} {outputformat} Human: {text} Human: {question} AI:"
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


def get_dataset(tokenizer,args,prompt_template):
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

    train_data = MultimodalDataset(train_dataset, tokenizer,prompt_template)
    dev_data = MultimodalDataset(dev_dataset, tokenizer,prompt_template)
    test_data = MultimodalDataset(test_dataset, tokenizer,prompt_template)
    full_data = ConcatDataset([train_data, dev_data, test_data])


    return train_data, dev_data, test_data,full_data




def get_dataloader(
        train_data, 
        dev_data, 
        test_data, 
        full_data,
        tokenizer,
        ):
    collate = make_collate_fn(tokenizer)
    common = dict(collate_fn=collate, num_workers=4, pin_memory=True, persistent_workers=True)
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=16, **common)
    dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=16, **common)
    test_dataloader = DataLoader(test_data, shuffle=False, batch_size=16,**common)
    full_dataloader = DataLoader(full_data, shuffle=False, batch_size=16, **common)
    return train_dataloader, dev_dataloader, test_dataloader,full_dataloader

# ,max_new_tokens=2

def test(model, dev_dataloader,tokenizer,args):
    # model.eval()  
    y_test = []
    preds = []
    results = []
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    # with torch.no_grad():
    for step, batch in enumerate(tqdm(dev_dataloader, desc="Iteration")):
        sentence, input_ids, attention_mask, source_label, label_id, segment = batch
        input_ids, attention_mask, label_id = input_ids.to(model.device), attention_mask.to(model.device), label_id.to(model.device)
        outputs = model.generate(input_ids=input_ids ,attention_mask=attention_mask,max_new_tokens=10)
        # result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # result = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        new_texts = []
        for i in range(outputs.size(0)):
            # 在原始输入里找最后一个非 pad 的索引（兼容 left/right padding）
            non_pad_pos = (input_ids[i] != pad_id).nonzero(as_tuple=True)[0]
            if non_pad_pos.numel() == 0:
                start = 0
            else:
                start = int(non_pad_pos[-1].item()) + 1  # 新 token 从这里开始

            # 注意：outputs[i] 的前缀和 input_ids[i] 对齐（包含 pad）
            new_tokens = outputs[i, start:]
            # 也可以：若你想严谨一点，截断到第一个 eos
            # if (new_tokens == eos_id).any():
            #     first_eos = (new_tokens == eos_id).nonzero(as_tuple=True)[0][0].item()
            #     new_tokens = new_tokens[:first_eos]

            txt = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            new_texts.append(txt)
                # print(result)
        # results.extend(result)
        for txt in new_texts:
            t = txt.lower()
            
            if "positive" in t:
                sentiment = 1
            elif "neutral" in t:
                sentiment = 0
            else:
                sentiment = -1
            # print(t)
            # print(sentiment)
            preds.append(sentiment)
        y_test_val = label_id.view(-1).cpu().detach().tolist()
        # preds_val = outputs.cpu().detach().tolist()
        # preds += preds_val
        # preds.append(sentiment)
        y_test += y_test_val
    acc, f_score_w,f_score_b,f_score_mi,f_score_ma, recall, pre = compute_accurracy(preds,y_test)
    temp = {'pred':preds,'y':y_test}
    dataset_name = args.dataset
    model_name = os.path.basename(os.path.normpath(args.path))
    with open(f'/data/yangshengzun/LLM/LLMresults/{model_name}_{dataset_name}my_list.pkl', 'wb') as file:
        pickle.dump(temp, file)
    # print(results)
    return acc, f_score_w,f_score_b,f_score_mi,f_score_ma, recall, pre 


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
def run_once(seed, args,prompt_template):
    print(f"\n===== Running with seed={seed} =====")
    set_seed(seed)

    # 每次重新加载 tokenizer/dataset/model，避免受前一次缓存干扰
    tokenizer = LlamaTokenizer.from_pretrained(args.path)
    train_data, dev_data, test_data, full_data = get_dataset(tokenizer, args,prompt_template)
    train_dataloader, dev_dataloader, test_dataloader, full_dataloader = get_dataloader(
        train_data, dev_data, test_data, full_data,tokenizer
    )
    model = LlamaForCausalLM.from_pretrained(
        args.path,
        device_map="auto"
    )

    acc, f_score_w,f_score_b,f_score_mi,f_score_ma, recall, pre = test(model, test_dataloader, tokenizer, args)

    print(f"Seed={seed} Results:",
          f"Acc={acc:.4f}, F1_w={f_score_w:.4f}, F1_b={f_score_b:.4f}, "
          f"F1_mi={f_score_mi:.4f}, F1_ma={f_score_ma:.4f}, Recall={recall:.4f}, Precision={pre:.4f}")

    # 保存结果
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
    for prompt_idx, prompt_template in enumerate(PROMPT_TEMPLATES):
        print(f"\n=== Testing Prompt {prompt_idx+1} ===")
        all_results = []
        for seed in SEEDS:
            res = run_once(seed, args,prompt_template)
            all_results.append(res)
        mean_acc = np.mean([r["acc"] for r in all_results])
        print(f"Prompt {prompt_idx+1} average acc: {mean_acc:.4f}")

        if mean_acc > best_acc:
            best_acc = mean_acc
            best_prompt = prompt_template
            best_results = all_results
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
    save_results(best_results, out_csv)
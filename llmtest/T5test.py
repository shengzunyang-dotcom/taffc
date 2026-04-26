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

T5_PRETRAIN_PATH = "/data/zhaoxianbing/CMU/pretrain/flan-t5-xxl"
PATH  = '/data/zhaoxianbing/cmu-features'
# /data/zhaoxianbing/CMU/pretrain/flan-t5-xxl
# /data/zhaoxianbing/CMU/pretrain/t5-v1_1-xxl
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='mosi')
parser.add_argument("--path", type=str, default='/data/zhaoxianbing/CMU/pretrain/flan-t5-xxl')

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

            taskname = "Multimodal Aspect-Based Sentiment Analysis task."
            taskdefinition = "Given the text data, assign a sentiment label from ['negative', 'neutral', 'positive']."
            outputformat= "Return label only without any other text."
            question= "what is the sentiment about the text?"
            option1 = "(a) neutral (b) negative (c) positive"
            option2 = "neutral or negative or positive"
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
            prompt = f"The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. \
            Human: Please perform{taskname} {taskdefinition} {outputformat} Human: {text} Human: {question} AI:"
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
    f_score = f1_score(y_test, preds, average="weighted")
    recall = recall_score(y_test,preds,average="weighted")
    pre = precision_score(y_test,preds,average="weighted")
    acc = accuracy_score(y_test, preds)
    #return acc, mae, corr, f_score, acc5, acc7
    return acc, f_score, recall, pre 


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




def get_dataloader(
        train_data, 
        dev_data, 
        test_data, 
        full_data,
        ):
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=1, collate_fn=padding_collate_fn)
    dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=1, collate_fn=padding_collate_fn)
    test_dataloader = DataLoader(test_data, shuffle=False, batch_size=1, collate_fn=padding_collate_fn)
    full_dataloader = DataLoader(full_data, shuffle=False, batch_size=1, collate_fn=padding_collate_fn)
    return train_dataloader, dev_dataloader, test_dataloader,full_dataloader

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
        results.append(result)
        if "positive" in result:
            sentiment = 1
        elif "neutral" in result:
            sentiment = 0
        else:  # 隐含"negative"或其他情况
            sentiment = -1
        y_test_val = label_id.view(-1).cpu().detach().tolist()
        # preds_val = outputs.cpu().detach().tolist()
        # preds += preds_val
        preds.append(sentiment)
        y_test += y_test_val
    acc, f_score, recall, pre = compute_accurracy(preds,y_test)
    temp = {'pred':preds,'y':y_test}
    dataset_name = args.dataset
    model_name = os.path.basename(os.path.normpath(args.path))
    with open(f'/data/yangshengzun/LLM/LLMresults/{model_name}_{dataset_name}my_list.pkl', 'wb') as file:
        pickle.dump(temp, file)
    print(results)
    return acc, f_score, recall, pre 


tokenizer = T5Tokenizer.from_pretrained(T5_PRETRAIN_PATH)
train_data, dev_data, test_data,full_data = get_dataset(tokenizer,args)
train_dataloader, dev_dataloader, test_dataloader,full_dataloader = get_dataloader(train_data, dev_data, test_data,full_data)
model = T5ForConditionalGeneration.from_pretrained(T5_PRETRAIN_PATH, device_map="auto")
acc, f_score, recall, pre = test(model,test_dataloader,tokenizer,args)
print(acc, f_score, recall, pre)

# Tokenize 与生成
# input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
# outputs = model.generate(input_ids, max_new_tokens=2)  # 限制生成长度
# result = tokenizer.decode(outputs[0], skip_special_tokens=True)

# 解析输出
# sentiment = 1 if result.strip() == "1" else 0
# print(f"文本: {input_text} → 情感: {sentiment}")
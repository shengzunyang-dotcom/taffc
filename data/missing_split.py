'''
Author: zhaoxianbing
Date: 2023-06-02 18:56:52
LastEditTime: 2023-06-02 20:49:04
LastEditors: Please set LastEditors
Description: file description
FilePath: /BERT_multimodal_transformer/process_data_clip.py
'''
from lib2to3.pgen2 import token
import pickle
from types import new_class
from transformers import BertTokenizer, BertModel
import os
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from PIL import Image
import requests
from transformers import AutoProcessor, CLIPVisionModelWithProjection

DEVICE = torch.device("cuda:0")

path = '/home/zhaoxianbing/CMU/llm-features/mosi/'
mosi_path = 'mosi_clip_large.pkl'


mosi_large_full_train_50 = 'mosi_large_full_train_50.pkl'
mosi_large_missing_train_text_50 = 'mosi_large_missing_train_text_50.pkl'
mosi_large_missing_train_visual_50 = 'mosi_large_missing_train_visual_50.pkl'


mosi_large_full_test_50 = 'mosi_large_full_test_50.pkl'
mosi_large_missing_test_text_50 = 'mosi_large_missing_test_text_50.pkl'
mosi_large_missing_test_visual_50 = 'mosi_large_missing_test_visual_50.pkl'


# with open('/home/zhaoxianbing/CMU/llm-features/mosei/mosei_meta.pkl', 'rb') as handle:
#     mosei_meta = pickle.load(handle)

with open(path + mosi_path, "rb") as handle:
    data = pickle.load(handle)

train_full_ratio = []
train_missing_ratio = []

dev_full = []

test_full_ratio = []
test_missing_ratio = []


ratio = 50

train_count = 0
train_full_nums = int(float(ratio) / 100 * len(data['train']))

test_count = 0
test_full_nums = int(float(ratio) / 100 * len(data['dev']))

test_count = 0
test_full_nums = int(float(ratio) / 100 * len(data['test']))

for idx in data['train']:
    (words, visual), label_id, segment = idx
    temp = (words, visual), label_id, segment
    # train.append(temp)
    if train_count <= train_full_nums:
        train_full_ratio.append(temp)
    else:
        train_missing_ratio.append(temp)
    print('train', train_count)
    train_count += 1

dev_count = 0
for idx in data['dev']:
    (words, visual), label_id, segment = idx
    temp = (words, visual), label_id, segment
    dev_full.append(temp)
    print('dev', dev_count)
    dev_count += 1

test_count = 0
for idx in data['test']:
    (words, visual), label_id, segment = idx
    temp = (words, visual), label_id, segment
    if test_count <= test_full_nums:
        test_full_ratio.append(temp)
    else:
        test_missing_ratio.append(temp)
    print('test', test_count)
    test_count += 1


# print(train_count, dev_count, test_count)
new_data = {"train_full": train_full_ratio, 'train_missing': train_missing_ratio, "dev": dev_full, "test_full": test_full_ratio, 'test_missing': test_missing_ratio}
with open(f'/home/zhaoxianbing/CMU/llm-features/mosi/mosi_clip_large_{ratio}.pkl', 'wb') as f:
    pickle.dump(new_data, f)

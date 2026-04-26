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
import clip

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from PIL import Image
import requests
from transformers import AutoProcessor, CLIPVisionModelWithProjection

# DEVICE = torch.device("cuda:0")

path = '/home/zhaoxianbing/CMU/datasets/'
mosi_path = 'mosi.pkl'


# with open('/home/zhaoxianbing/CMU/llm-features/mosei/mosei_meta.pkl', 'rb') as handle:
#     mosei_meta = pickle.load(handle)

with open(path + mosi_path, "rb") as handle:
    data = pickle.load(handle)

print(data.keys())

train = []
dev = []
test = []

frames_root = '/disk4/CMU-RAW/MOSI/Raw/Video/Frames'

device ="cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/16", device="cpu")
model = model.to(device)
model_path = '/home/zhaoxianbing/CMU/pretrain/FaRL/FaRL-Base-Patch16-LAIONFace20M-ep16.pth'
farl_state=torch.load(model_path)
model.load_state_dict(farl_state["state_dict"],strict=False)

# model = model.to(device)


train_count = 0
for idx in data['train']:
    (words, visual, acoustic), label_id, segment = idx
    dir_prefix, num = segment[0: -1].split('[')[0], segment[0: -1].split('[')[1]
    dir_name = dir_prefix + '_' + str(int(num) + 1)
    frame_path = os.path.join(frames_root, dir_name)
    ret = os.path.exists(frame_path)
    if ret == False:
        print(frame_path)
        continue
    images = os.listdir(frame_path)
    images.sort()
    temp = []
    for image_file in images:
        image_path = os.path.join(frame_path, image_file)
        image1 = Image.open(image_path)
        image = preprocess(image1).unsqueeze(0).to(device)     
        image_features = model.encode_image(image)
        image_features = image_features.cpu().detach()
        temp.append(image_features)

    image_embeds = torch.cat(temp, dim=0)
    image_embeds = image_embeds.cpu().detach().numpy()
    temp = (words, image_embeds), label_id, segment
    train.append(temp)
    print('train', train_count)
    train_count += 1

dev_count = 0
for idx in data['dev']:
    (words, visual, acoustic), label_id, segment = idx
    dir_prefix, num = segment[0: -1].split('[')[0], segment[0: -1].split('[')[1]
    dir_name = dir_prefix + '_' + str(int(num) + 1)
    frame_path = os.path.join(frames_root, dir_name)
    ret = os.path.exists(frame_path)
    if ret == False:
        print(frame_path)
        continue
    images = os.listdir(frame_path)
    images.sort()
    temp = []
    for image_file in images:
        image_path = os.path.join(frame_path, image_file)
        image1 = Image.open(image_path)
        image = preprocess(image1).unsqueeze(0).to(device)     
        image_features = model.encode_image(image)
        image_features = image_features.cpu().detach()
        temp.append(image_features)

    image_embeds = torch.cat(temp, dim=0)
    image_embeds = image_embeds.cpu().detach().numpy()
    temp = (words, image_embeds), label_id, segment
    dev.append(temp)
    print('dev', dev_count)
    dev_count += 1

test_count = 0
for idx in data['test']:
    (words, visual, acoustic), label_id, segment = idx
    dir_prefix, num = segment[0: -1].split('[')[0], segment[0: -1].split('[')[1]
    dir_name = dir_prefix + '_' + str(int(num) + 1)
    frame_path = os.path.join(frames_root, dir_name)
    ret = os.path.exists(frame_path)
    if ret == False:
        print(frame_path)
        continue
    images = os.listdir(frame_path)
    images.sort()
    temp = []
    for image_file in images:
        image_path = os.path.join(frame_path, image_file)
        image1 = Image.open(image_path)
        image = preprocess(image1).unsqueeze(0).to(device)     
        image_features = model.encode_image(image)
        image_features = image_features.cpu().detach()
        temp.append(image_features)

    image_embeds = torch.cat(temp, dim=0)
    image_embeds = image_embeds.cpu().detach().numpy()
    temp = (words, image_embeds), label_id, segment
    test.append(temp)
    print('test', test_count)
    test_count += 1


print(train_count, dev_count, test_count)
new_data = {"train": train, "dev": dev, "test": test}
with open('/home/zhaoxianbing/CMU/llm-features/mosi/mosi_farls_clip.pkl', 'wb') as f:
    pickle.dump(new_data, f)

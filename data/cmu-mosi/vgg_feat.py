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
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
import pandas as pd
import os
import numpy as np
from PIL import Image
import cv2


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from PIL import Image
import requests
from transformers import AutoProcessor, CLIPVisionModelWithProjection

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

path = '/disk4/cmu-features/mosi/'
mosi_path = 'mosi_noalign_new.pkl'

frames_root = '/disk4/CMU-RAW/MOSI/Raw/Video/Frames'
# with open('/home/zhaoxianbing/CMU/llm-features/mosei/mosei_meta.pkl', 'rb') as handle:
#     mosei_meta = pickle.load(handle)

with open(path + mosi_path, "rb") as handle:
    data = pickle.load(handle)



print(data.keys())

train = []
dev = []
test = []


mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)



train_count = -1
for idx in data['train']:
    train_count += 1
    (words, visual, acoustic), label_id, segment = idx
    frame_path = os.path.join(frames_root, segment)
    ret = os.path.exists(frame_path)
    if ret == False:
        print(frame_path)
        continue
    images = os.listdir(frame_path)
    images.sort()
    aligned = []
    for image_file in images:
        image_path = os.path.join(frame_path, image_file)
        image = Image.open(image_path)
        x_aligned, prob = mtcnn(image, return_prob=True)
        if x_aligned is None:
            continue
        aligned.append(x_aligned)
    if len(aligned) == 0:
        continue
    aligned = torch.stack(aligned).to(device)
    embeddings = resnet(aligned).detach().cpu().numpy()
    visual = embeddings
    temp = (words, visual), label_id, segment
    train.append(temp)
    print('train', train_count)


dev_count = -1
for idx in data['dev']:
    dev_count += 1
    (words, visual, acoustic), label_id, segment = idx
    frame_path = os.path.join(frames_root, segment)
    ret = os.path.exists(frame_path)
    if ret == False:
        print(frame_path)
        continue
    images = os.listdir(frame_path)
    images.sort()
    aligned = []
    for image_file in images:
        image_path = os.path.join(frame_path, image_file)
        image = Image.open(image_path)
        x_aligned, prob = mtcnn(image, return_prob=True)
        if x_aligned is None:
            continue
        aligned.append(x_aligned)
    if len(aligned) == 0:
        continue
    aligned = torch.stack(aligned).to(device)
    embeddings = resnet(aligned).detach().cpu().numpy()
    visual = embeddings
    temp = (words, visual), label_id, segment
    dev.append(temp)
    print('dev', dev_count)


test_count = -1
for idx in data['test']:
    test_count += 1
    (words, visual, acoustic), label_id, segment = idx
    frame_path = os.path.join(frames_root, segment)
    ret = os.path.exists(frame_path)
    if ret == False:
        print(frame_path)
        continue
    images = os.listdir(frame_path)
    images.sort()
    aligned = []
    for image_file in images:
        image_path = os.path.join(frame_path, image_file)
        image = Image.open(image_path)
        x_aligned, prob = mtcnn(image, return_prob=True)
        if x_aligned is None:
            continue
        aligned.append(x_aligned)
    if len(aligned) == 0:
        continue
    aligned = torch.stack(aligned).to(device)
    embeddings = resnet(aligned).detach().cpu().numpy()
    visual = embeddings
    temp = (words, visual), label_id, segment
    temp = (words, visual), label_id, segment
    test.append(temp)
    print('test', test_count)


print(train_count, dev_count, test_count)
new_data = {"train": train, "dev": dev, "test": test}
with open('/disk4/cmu-features/mosi/mosi_vgg.pkl', 'wb') as f:
    pickle.dump(new_data, f)

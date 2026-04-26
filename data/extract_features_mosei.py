from lib2to3.pgen2 import token
import pickle
from types import new_class
from transformers import BertTokenizer, BertModel
import os
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from PIL import Image
import requests
from transformers import AutoProcessor, CLIPVisionModelWithProjection

DEVICE = torch.device("cuda:0")

path = '/home/zhaoxianbing/CMU/datasets/'
mosi_path = 'mosei.pkl'


with open('/home/zhaoxianbing/CMU/llm-features/mosei/mosei_meta.pkl', 'rb') as handle:
    mosei_meta = pickle.load(handle)

with open(path + mosi_path, "rb") as handle:
    data = pickle.load(handle)

print(data.keys())

train = []
dev = []
test = []


large_model = '/home/zhaoxianbing/CMU/pretrain/clip-vit-large-patch14'
base_model = '/home/zhaoxianbing/CMU/pretrain/clip-vit-base-patch32'

frames_root = '/disk4/CMU-RAW/MOSEI/Raw/Videos/Segmented-1/Frames'
model = CLIPVisionModelWithProjection.from_pretrained(large_model)
processor = AutoProcessor.from_pretrained(large_model)
model = model.to(DEVICE)


i = 0
in_key = 0
out_key = 0
image_0 = 0

train_count = 0
for idx in data['train']:
    (words, visual, acoustic), label_id, segment = idx
    key1 = '__'.join(segment)
    key2 = key1.replace('__0.0__', '__0__')
    if key1 in mosei_meta.keys():
        in_key += 1
        num = mosei_meta[key1]
    elif key2 in mosei_meta.keys():
        in_key += 1
        num = mosei_meta[key2]
    else:
        out_key += 1
    frame_dir = segment[0] + '_' + num
    frame_path = os.path.join(frames_root, frame_dir)
    ret = os.path.exists(frame_path)
    if ret == False:
        continue
    images = os.listdir(frame_path)
    images.sort()
    temp = []
    for image_file in images:
        image_path = os.path.join(frame_path, image_file)
        image1 = Image.open(image_path)
        inputs = processor(images=image1, return_tensors="pt").to(DEVICE)
        outputs = model(**inputs)
        image_embeds = outputs.image_embeds.cpu().detach()
        temp.append(image_embeds)
    image_features = torch.cat(temp, dim=0)
    image_features = image_features.cpu().detach().numpy()
    visual = image_features
    temp = (words, words, visual), label_id, segment
    train.append(temp)
    print('train', train_count)
    train_count += 1

dev_count = 0
for idx in data['dev']:
    (words, visual, acoustic), label_id, segment = idx
    key1 = '__'.join(segment)
    key2 = key1.replace('__0.0__', '__0__')
    if key1 in mosei_meta.keys():
        in_key += 1
        num = mosei_meta[key1]
    elif key2 in mosei_meta.keys():
        in_key += 1
        num = mosei_meta[key2]
    else:
        out_key += 1
    frame_dir = segment[0] + '_' + num
    frame_path = os.path.join(frames_root, frame_dir)
    ret = os.path.exists(frame_path)
    if ret == False:
        continue
    images = os.listdir(frame_path)
    images.sort()
    temp = []
    for image_file in images:
        image_path = os.path.join(frame_path, image_file)
        image1 = Image.open(image_path)
        inputs = processor(images=image1, return_tensors="pt").to(DEVICE)
        outputs = model(**inputs)
        image_embeds = outputs.image_embeds.cpu().detach()
        temp.append(image_embeds)
    image_features = torch.cat(temp, dim=0)
    image_features = image_features.cpu().detach().numpy()
    visual = image_features
    temp = (words, visual), label_id, segment
    dev.append(temp)
    print('dev', dev_count)
    dev_count += 1

test_count = 0
for idx in data['test']:
    (words, visual, acoustic), label_id, segment = idx
    key1 = '__'.join(segment)
    key2 = key1.replace('__0.0__', '__0__')
    if key1 in mosei_meta.keys():
        in_key += 1
        num = mosei_meta[key1]
    elif key2 in mosei_meta.keys():
        in_key += 1
        num = mosei_meta[key2]
    else:
        out_key += 1
    frame_dir = segment[0] + '_' + num
    frame_path = os.path.join(frames_root, frame_dir)
    ret = os.path.exists(frame_path)
    if ret == False:
        continue
    images = os.listdir(frame_path)
    images.sort()
    temp = []
    for image_file in images:
        image_path = os.path.join(frame_path, image_file)
        image1 = Image.open(image_path)
        inputs = processor(images=image1, return_tensors="pt").to(DEVICE)
        outputs = model(**inputs)
        image_embeds = outputs.image_embeds.cpu().detach()
        temp.append(image_embeds)
    image_features = torch.cat(temp, dim=0)
    image_features = image_features.cpu().detach().numpy()
    visual = image_features
    temp = (words, visual), label_id, segment
    test.append(temp)
    print('test', test_count)
    test_count += 1


print(train_count, dev_count, test_count)
new_data = {"train": train, "dev": dev, "test": test}
with open('/home/zhaoxianbing/CMU/llm-features/mosei/mosei_clip_large.pkl', 'wb') as f:
    pickle.dump(new_data, f)

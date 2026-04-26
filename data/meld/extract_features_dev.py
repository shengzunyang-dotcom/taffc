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

DEVICE = torch.device("cuda:0")

path = '/home/zhaoxianbing/CMU/datasets/'
mosi_path = 'mosi.pkl'


# with open('/home/zhaoxianbing/CMU/llm-features/mosei/mosei_meta.pkl', 'rb') as handle:
#     mosei_meta = pickle.load(handle)

# with open(path + mosi_path, "rb") as handle:
#     data = pickle.load(handle)

# print(data.keys())

from meld_meta import get_data
data = get_data()

train = []
dev = []
test = []


# large_model = '/home/zhaoxianbing/CMU/pretrain/clip-vit-large-patch14'
# base_model = '/home/zhaoxianbing/CMU/pretrain/clip-vit-base-patch32'
# frames_root = '/disk4/CMU-RAW/MOSI/Raw/Video/Frames'
# model = CLIPVisionModelWithProjection.from_pretrained(large_model)
# processor = AutoProcessor.from_pretrained(large_model)
# model = model.to(DEVICE)

device ="cuda" if torch.cuda.is_available() else "cpu"
print(device)
_, preprocess = clip.load("ViT-B/16", device="cpu")
# model = model.to(device)
# farl_state=torch.load("/home/zhaoxianbing/CMU/pretrain/FaRL/FaRL-Base-Patch16-LAIONFace20M-ep16.pth")

# image = preprocess(Image.open("demo.jpg")).unsqueeze(0).to(device)
# text = clip.tokenize(["a diagram", "b dog", "c cat"]).to(device)

train_frame_path = '/disk3/multimodal-dataset/meld/MELD.Raw/train_splits/frames/'
dev_frame_path = '/disk3/multimodal-dataset/meld/MELD.Raw/dev_splits/frames/'
test_frame_path = '/disk3/multimodal-dataset/meld/MELD.Raw/test_splits/frames/'




dev_count = 0
for idx in data['dev']:
    (sentence, video_file, sent_label, emo_label) = idx
    frame_path = dev_frame_path + video_file
    image_files = os.listdir(frame_path)
    if len(image_files) < 4:
        continue
    image_files.sort()
    image_temp = []
    for image_file in image_files:
        image_path = os.path.join(frame_path, image_file)
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        # tokenized = clip.tokenize([sentence]).to(device)
        image = image.cpu().detach()
        image_temp.append(image)
    images = torch.cat(image_temp, dim=0)
    images = images.cpu().detach().numpy()
    temp = (sentence, video_file, images), sent_label, emo_label
    dev.append(temp)
    print('dev', dev_count)
    dev_count += 1



print(dev_count)
new_data = {"dev": dev}
with open('/home/zhaoxianbing/CMU/llm-features/meld/meld_farl_dev.pkl', 'wb') as f:
    pickle.dump(new_data, f)

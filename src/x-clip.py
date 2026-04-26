import av
import torch
import numpy as np
from transformers import AutoProcessor, AutoModel
from huggingface_hub import hf_hub_download
np.random.seed(0)
def read_video_pyav(container, indices):
         '''
         Decode the video with PyAV decoder.
         Args:
             container (`av.container.input.InputContainer`): PyAV container.
             indices (`List[int]`): List of frame indices to decode.
         Returns:
             result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
         '''
         frames = []
         container.seek(0)
         start_index = indices[0]
         end_index = indices[-1]
         for i, frame in enumerate(container.decode(video=0)):
             if i > end_index:
                 break
             if i >= start_index and i in indices:
                 frames.append(frame)
         return np.stack([x.to_ndarray(format="rgb24") for x in frames])
def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
         converted_len = int(clip_len * frame_sample_rate)
         end_idx = np.random.randint(converted_len, seg_len)
         start_idx = end_idx - converted_len
         indices = np.linspace(start_idx, end_idx, num=clip_len)
         indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
         return indices
# video clip consists of 300 frames (10 seconds at 30 FPS)
file_path = '../data/eating_spaghetti.mp4'
container = av.open(file_path)
# sample 8 frames
indices = sample_frame_indices(clip_len=8, frame_sample_rate=1, seg_len=container.streams.video[0].frames)
video = read_video_pyav(container, indices)

processor = AutoProcessor.from_pretrained("/home/zhaoxianbing/CMU/pretrain/xclip-base-patch32")
model = AutoModel.from_pretrained("/home/zhaoxianbing/CMU/pretrain/xclip-base-patch32")

print(type(video))
a = list(video)
print(type(list(video)), len(a), a[0].shape)
inputs = processor(videos=list(video), return_tensors="pt")

print(type(inputs), inputs.keys(), inputs['pixel_values'].shape)

video_features = model.get_video_features(**inputs)
print(video_features.shape)

import os
import torch
import torchvision
print(torch.__version__, torchvision.__version__)
from feat import Detector
detector = Detector()
test_video_path = "/disk4/CMU-RAW/MOSI/Raw/Video/Segmented/vyB00TXsimI_12.mp4"
video_prediction = detector.detect_video(test_video_path, skip_frames=1)
print(video_prediction)
print(video_prediction.shape)


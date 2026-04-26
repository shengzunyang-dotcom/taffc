import os 
# - * - coding: utf-8 - * -
import librosa
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft

plt.figure(dpi=600) # 将显示的所有图分辨率调高
matplotlib.rc("font",family='SimHei') # 显示中文
matplotlib.rcParams['axes.unicode_minus']=False # 显示符号


def displayWaveform(name, number): # 显示语音时域波形
    """
    display waveform of a given speech sample
    :param sample_name: speech sample name
    :param fs: sample frequency
    :return:
    """
    samples, sr = librosa.load(name, sr=16000)
    # samples = samples[6000:16000]

    print(len(samples), sr)
    time = np.arange(0, len(samples)) * (1.0 / sr)

    plt.plot(time, samples)

    plt.savefig(f"./show/{number}.png", dpi=300)
    # plt.show()


if __name__ == '__main__':
    video = 'LSi-o-IrDMs_26' #d6hH302o4v8_21 yvsjCA6Y5Fc_5 zhpQhgha_KU[18] d3_k5Xpfmik_18
    number = 'tp-26'
    #cmd = f'ffmpeg -i /disk4/CMU-RAW/MOSI/Raw/Video/Segmented/{video}.mp4 -r 1 -f image2 ./tmmda/{number}-%d.png'
    # cmd = f'ffmpeg -i /data/zhaoxianbing/multimodal-dataset/cmu/CMU-RAW/MOSI/Raw/Video/Segmented/{video}.mp4 -r 1 -f image2 ./show/{number}-%d.png'
    # os.system(cmd)
    #cmd = f'ffmpeg -i /disk4/CMU-RAW/MOSI/Raw/Video/Segmented/{video}.mp4 -f mp3 ./tmmda/{number}.mp3'
    # cmd = f'ffmpeg -i /data/zhaoxianbing/multimodal-dataset/cmu/CMU-RAW/MOSI/Raw/Video/Segmented/{video}.mp4 -f mp3 -c:a libmp3lame ./show/{number}.mp3'
    # os.system(cmd)
    #name = f'./video/{number}'
    name = '/data/zhaoxianbing/multimodal-dataset/cmu/CMU-RAW/MOSI/Raw/Audio/WAV_16000/Segmented/0h-zjBukYpk_2.wav'
    displayWaveform(name, number)
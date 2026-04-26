import torchaudio
import torch
from transformers import Wav2Vec2ForPreTraining, Wav2Vec2Processor
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0")


from moviepy.video.io.VideoFileClip import VideoFileClip
from pydub import AudioSegment

def extract_audio(input_file, output_file, target_sample_rate=16000):
    # 使用 moviepy 打开视频文件
    video_clip = VideoFileClip(input_file)
    audio = video_clip.audio
    audio = audio.set_frame_rate(target_sample_rate)
    audio.write_audiofile(output_file, codec='pcm_s16le', ffmpeg_params=['-ac', '1'])  # 采样率设置为16,000 Hz
    video_clip.close()
input_file_path = "path/to/your/video/file.mp4"
output_file_path = "path/to/your/output/audio/file.wav"
extract_audio(input_file_path, output_file_path, target_sample_rate=16000)



model = Wav2Vec2ForPreTraining.from_pretrained("facebook/wav2vec2-base-960h").to(device)
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h").to(device)
waveform, sample_rate = torchaudio.load("path/to/your/audio/file.wav")
# 使用处理器对音频数据进行处理
inputs = processor(waveform.squeeze().numpy(), return_tensors="pt", sampling_rate=sample_rate)
inputs = inputs.to(device)
with torch.no_grad():
    features = model(**inputs).last_hidden_state
    

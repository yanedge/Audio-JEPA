

# pip install -U kaggle yt_dlp transformers
# kaggle datasets download -d gev1994/audioset-encodec-3k-bitrates -p ./data --unzip

import json
from scipy import signal
from transformers import EncodecModel, AutoProcessor
import os
import glob
import numpy as np
import torch
import soundfile as sf
import yt_dlp

dataset = "eval_segments"
index = 10

current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(current_dir, './output')
os.makedirs(output_dir, exist_ok=True)

segments = os.path.join(current_dir, f'data/{dataset}.json')
segments = json.load(open(segments, 'r'))

seg = segments[index]
print(f'segment: {seg}')

encodec_file = os.path.join(current_dir, f"data/{dataset}/" + seg['file_id'])
youtube_id = seg["video_id"][1:]

# Download audio from YouTube

yt_audio_path = os.path.join(output_dir, f"youtube/Y{youtube_id}")
if not os.path.exists(yt_audio_path):
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'postprocessor_args': [
            '-ar', '32000',
            '-ac', '1',
            '-ss', str(seg['start_seconds']),
            '-to', str(seg['end_seconds']),
        ],
        'outtmpl': yt_audio_path,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([f'https://www.youtube.com/watch?v={youtube_id}'])
        print(f'Downloaded audio for YouTube ID {youtube_id} to {yt_audio_path}')

print(f'Decoding {encodec_file}')

# Decode the first file
if os.path.exists(encodec_file):
    
    # Load the model and processor
    model = EncodecModel.from_pretrained("facebook/encodec_24khz")

    # Files are saved as numpy arrays with shape (1, num_codebooks, time_steps), dtype uint16
    codes_np = np.load(encodec_file)
    print(f'Loaded numpy array: shape={codes_np.shape}, dtype={codes_np.dtype}')

    # model.decode expects (batch, nb_chunks, num_codebooks, chunk_length)
    codes = torch.from_numpy(codes_np.astype(np.int64)).unsqueeze(1)  # (1,1,4,750)
    audio_scales = [torch.tensor([0.9])]

    with torch.no_grad():
        decoded = model.decode(codes, audio_scales)

    # Save as WAV file
    audio = decoded.audio_values[0, 0].numpy()  # (samples,)
    audio = signal.resample_poly(audio, 32000, 24000)
    #audio = audio * 0.8
    dec_dir = os.path.join(output_dir, 'decoded')
    os.makedirs(dec_dir, exist_ok=True)
    output_file = os.path.join(dec_dir, os.path.basename(encodec_file).replace('.encodec', '.wav'))
    sf.write(output_file, audio, samplerate=32000)
    print(f'Saved decoded audio to {output_file}')
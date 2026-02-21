import json
import yaml
import sys
import os
from scipy import signal
from transformers import EncodecModel
import torch
import glob
import numpy as np
import soundfile as sf
import h5py

# Add project root to path so we can import src module
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from src.data.components.mel_spec import MelSpecTransform

datasets = ["balanced_train_segments", "eval_segments", "unbalanced_train_segments"]

config_file = os.path.join(project_root, 'configs/data/asmfe.yaml')
with open(config_file, 'r') as f:
    cfg = yaml.safe_load(f)
n_mels = cfg.get('n_mels', 96)
clip_length = cfg.get('clip_length', 10)

current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(current_dir, './output')
os.makedirs(output_dir, exist_ok=True)

# Load the model and processor
model = EncodecModel.from_pretrained("facebook/encodec_24khz")

def dec_encodec(encodec_file, sr=32000):

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
    audio = signal.resample_poly(audio, sr, 24000)
    # Convert to torch tensor and add channel dimension: (1, samples)
    return torch.from_numpy(audio).float().unsqueeze(0)

def conv_to_mfe(audio, sr):
    mel_spec_transform = MelSpecTransform(sr, n_mels=n_mels, clip_length=clip_length)
    mfe = mel_spec_transform(audio)
    return mfe

if __name__ == "__main__":
    for dataset in datasets:
        segments = os.path.join(current_dir, f'data/{dataset}.json')
        segments = json.load(open(segments, 'r'))

        num_samples = len(segments) // 1000

        rec = []

        for s in range(num_samples):
            seg = segments[s]
            filename = seg['file_id'].split('/')[-1]  # Get just the filename
            
            # Try multiple possible paths depending on dataset structure
            if dataset == 'unbalanced_train_segments':
                # For unbalanced, files are nested: data/unbalanced_train_segments/{part}/storage2/audioset_proc/encodec/unbalanced_train_segments/{part}/{filename}
                part = seg['file_id'].split('/')[-2]
                encodec_file = os.path.join(current_dir, f"data/{dataset}/{part}/storage2/audioset_proc/encodec/{dataset}/{part}/{filename}")
            else:
                # For balanced_train_segments and eval_segments, files are directly in dataset folders
                encodec_file = os.path.join(current_dir, f"data/{dataset}/{filename}")
            
            if os.path.exists(encodec_file):
                print(f'Processing {encodec_file}')
                audio = dec_encodec(encodec_file)
                mfe = conv_to_mfe(audio, sr=32000)
                print(f'Mel spectrogram shape: {mfe.shape}')
                res = {
                    "mfe": mfe,
                    "label": seg['labels']
                }
            else:
                raise FileNotFoundError(f'Encodec file not found: {encodec_file}')
            
            rec.append(res)

        output_file = os.path.join(output_dir, f'{dataset}_mfe.h5')
        with h5py.File(output_file, 'w') as hf:
            for i, item in enumerate(rec):
                grp = hf.create_group(f'sample_{i}')
                grp.create_dataset('mfe', data=item['mfe'])
                grp.create_dataset('label', data=item['label'])
        print(f'Saved to {output_file}')
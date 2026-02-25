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
import matplotlib.pyplot as plt

# Add project root to path so we can import src module
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from src.data.components.mel_spec import MelSpecTransform

datasets = [
    "balanced_train_segments",
    "eval_segments",
    #"unbalanced_train_segments"
    ]

config_file = os.path.join(project_root, 'configs/data/asmfe.yaml')
with open(config_file, 'r') as f:
    cfg = yaml.safe_load(f)
# Fix: read from nested transforms config
n_mels = cfg['transforms'][0]['n_mels']
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

def labels_to_binary(labels_str, label_to_idx, classes_num=527):
    """Convert comma-separated label string to binary vector"""
    binary = np.zeros(classes_num, dtype=np.float32)
    for label in labels_str.split(','):
        label = label.strip()
        if label in label_to_idx:
            binary[label_to_idx[label]] = 1.0
    return binary

def save_mfe(audio, mfe, id, sr):

    wav_file = os.path.join(output_dir, f"{id}.wav")
    sf.write(wav_file, audio.cpu().numpy().squeeze(), samplerate=sr)
    print(f'Audio saved to {wav_file}')

    plt.figure(figsize=(10, 4))
    plt.imshow(mfe.squeeze().numpy().T, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(label='Amplitude')
    plt.title(f'Mel Spectrogram - {dataset}')
    plt.xlabel('Time')
    plt.ylabel('Mel Frequency')
    plt.tight_layout()
    png_file = os.path.join(output_dir, f'{id}_mfe.png')
    plt.savefig(png_file)
    plt.close()
    print(f'PNG saved to {png_file}')

if __name__ == "__main__":
    # Fix: load ontology once
    with open(os.path.join(current_dir, '../AudioSet/ontology.json'), 'r') as f:
        ontology = json.load(f)
    label_to_idx = {entry['id']: i for i, entry in enumerate(ontology)}

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
                encodec_file = os.path.join(current_dir, f"data/{dataset}/{dataset}/{filename}")
            
            if os.path.exists(encodec_file):
                print(f'Processing {dataset} {encodec_file.split("/")[-1]}')
                audio = dec_encodec(encodec_file)
                mfe = conv_to_mfe(audio, sr=32000)
                print(f'Mel spectrogram shape: {mfe.shape}')
                rec.append({
                    "mfe": mfe,
                    "label": seg['labels'],
                    "filename": filename
                })
            else:
                raise FileNotFoundError(f'Encodec file not found: {encodec_file}')
            
        output_file = os.path.join(output_dir, f'{dataset}_mfe.h5')
        with h5py.File(output_file, 'w') as hf:
            mfes = np.stack([item['mfe'].squeeze().numpy() for item in rec])                      # (N, time, n_mels)
            filenames = np.array([item['filename'] for item in rec], dtype='S64')                 # (N,)
            targets = np.stack([labels_to_binary(item['label'], label_to_idx) for item in rec])  # (N, classes_num)
            labels = np.array([item['label'] for item in rec], dtype='S256')                      # (N,) raw label strings

            hf.create_dataset('mfe', data=mfes)
            hf.create_dataset('audio_name', data=filenames)
            hf.create_dataset('target', data=targets)
            hf.create_dataset('label', data=labels)

        print(f'Saved {len(rec)} samples to {output_file}')

        save_mfe(audio, mfe, dataset, 32000)

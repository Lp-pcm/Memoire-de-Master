import os
import torch
import librosa
import sys 
import json
import torch.nn as nn

# Get the path to the parent directory containing "beats"
current_dir = os.path.dirname(os.path.abspath(__file__))
beats_parent_dir = os.path.join(current_dir, "beats")  # Adjust as per your directory structure
sys.path.insert(0, beats_parent_dir)

from BEATs import BEATs, BEATsConfig

# Define paths
audio_path = "/Users/leo-polde/Documents/Louis-Lumière/MÉMOIRE/PYTHON/BEATs/beats_modf_PPM/AUDIO_FILES/Family Cooking Hum Of Voices Radio In Background Bolivia Digiffects DIGIA15-23.wav"
beats_ckpt_relative_path = "models/Fine-tuned_BEATs_iter3+_AS2M_cpt2.pt"
beats_ckpt_full_path = os.path.abspath(beats_ckpt_relative_path)
head_ckpt_path = "/Users/leo-polde/Documents/Louis-Lumière/MÉMOIRE/PYTHON/BEATs/beats_modf_PPM/models/final_head_model.pth"
ontology_path = "utils/ontology_FD.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the fine-tuned BEATs checkpoint
checkpoint = torch.load(beats_ckpt_full_path, map_location=device)
cfg = BEATsConfig(checkpoint['cfg'])
cfg.finetuned_model = False  # Important: we just want the backbone embeddings

BEATs_model = BEATs(cfg)
BEATs_model.load_state_dict(checkpoint['model'], strict=False)
BEATs_model.eval()
BEATs_model.to(device)

# Load the trained head
num_classes = 6  # Adapt to your problem
head = nn.Linear(cfg.encoder_embed_dim, num_classes).to(device)
head_ckpt = torch.load(head_ckpt_path, map_location=device)
head.load_state_dict(head_ckpt['head_state_dict'])
head.eval()

# Load the ontology FD
with open(ontology_path, "r") as f:
    label_to_idx = json.load(f)
idx_to_label = {idx: label for label, idx in label_to_idx.items()}

# Load and preprocess audio
sr = 16000  # Target sampling rate
audio, orig_sr = librosa.load(audio_path, sr=None) # rajouter offset=2, duration=8 si on veut prendre 8 seconde 2 secondes après le débit du fichier
y_beats = librosa.resample(audio, orig_sr=orig_sr, target_sr=sr)

# Convert audio to tensor
input_tensor = torch.tensor(y_beats).unsqueeze(0).to(device)  # [1, time]
padding_mask = torch.zeros(1, input_tensor.size(1)).bool().to(device)

# Extract features and predict
with torch.no_grad():
    features = BEATs_model.extract_features(input_tensor, padding_mask=padding_mask)[0]  # [1, time, 768]
    features_mean = features.mean(dim=1)  # [1, 768]
    outputs = head(features_mean)  # [1, num_classes]
    _, predicted_idx = torch.max(outputs, 1)
    predicted_idx = predicted_idx.item()

# Display prediction
predicted_class = idx_to_label[predicted_idx]
print(f"🎧 Prédiction : {predicted_class}")
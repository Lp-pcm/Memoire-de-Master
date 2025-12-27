import os
import torch
import librosa
import sys
import json
import csv
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

# Get the path to the parent directory containing "beats"
current_dir = os.path.dirname(os.path.abspath(__file__))
beats_parent_dir = os.path.join(current_dir, "beats")
sys.path.insert(0, beats_parent_dir)

from BEATs import BEATs, BEATsConfig

# === CONFIGURATION ===
audio_dir = "/Users/leo-polde/Documents/Louis-Lumière/MÉMOIRE/PYTHON/BEATs/beats_modf_PPM/AUDIO_FILES/ANO_EVALUATION"
class_mapping_path = "utils/class_mapping_fooddrink_EVAL_full_files.csv"
ontology_path = "utils/ontology_FD.json"
output_csv = "predictions_test_set_full_files.csv"

beats_ckpt_full_path = os.path.abspath("old_models/Fine-tuned_BEATs_iter3+_AS2M_cpt2.pt")
head_ckpt_paths = [
    "New_models/best_model_B16_LR1e-02_F0.pth",
    "New_models/best_model_B16_LR1e-02_F1.pth",
    "New_models/best_model_B16_LR1e-02_F2.pth",
    "New_models/best_model_B16_LR1e-02_F3.pth",
    "New_models/best_model_B16_LR1e-02_F4.pth",
    "New_models/best_model_B16_LR1e-02_F5.pth",
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === CHARGEMENT DES ONTOLOGIES ET MAPPINGS ===
with open(ontology_path, "r") as f:
    label_to_idx = json.load(f)
idx_to_label = {idx: label for label, idx in label_to_idx.items()}
num_classes = len(idx_to_label)

# Mapping true class
df_mapping = pd.read_csv(class_mapping_path)
filename_to_trueclass = dict(zip(df_mapping["filename"], df_mapping["class"]))

# === CHARGEMENT DU MODÈLE BEATs ===
checkpoint = torch.load(beats_ckpt_full_path, map_location=device)
cfg = BEATsConfig(checkpoint['cfg'])
cfg.finetuned_model = False
BEATs_model = BEATs(cfg)
BEATs_model.load_state_dict(checkpoint['model'], strict=False)
BEATs_model.eval().to(device)

# === TRAITEMENT PAR FICHIER ===
results = []

for filename in sorted(os.listdir(audio_dir)):
    if not filename.endswith(".wav"):
        continue
    file_path = os.path.join(audio_dir, filename)
    print(f"\n🔊 Traitement de : {filename}")

    # Lecture et prétraitement
    audio, orig_sr = librosa.load(file_path, sr=None)
    y_beats = librosa.resample(audio, orig_sr=orig_sr, target_sr=16000)
    input_tensor = torch.tensor(y_beats).unsqueeze(0).to(device)
    padding_mask = torch.zeros(1, input_tensor.size(1)).bool().to(device)

    # Extraction des features
    with torch.no_grad():
        features = BEATs_model.extract_features(input_tensor, padding_mask=padding_mask)[0]
        features_mean = features.mean(dim=1)

    # Prédictions des 6 têtes
    all_probs = []
    for i, head_path in enumerate(head_ckpt_paths):
        head = nn.Linear(cfg.encoder_embed_dim, num_classes).to(device)
        state_dict = torch.load(head_path, map_location=device)
        head.load_state_dict(state_dict['head'])
        head.eval()
        with torch.no_grad():
            logits = head(features_mean)
            probs = F.softmax(logits, dim=1).squeeze()
            all_probs.append(probs)

    avg_probs = torch.stack(all_probs).mean(dim=0)
    top1_idx = avg_probs.argmax().item()
    top1_label = idx_to_label[top1_idx]

    true_class = filename_to_trueclass.get(filename, "UNKNOWN")

    result = {
        "filename": filename,
        "true_class": true_class,
        "predicted_class": top1_label,
    }
    for i, prob in enumerate(avg_probs):
        result[f"prob_class_{i}"] = prob.item()

    results.append(result)

# === ÉCRITURE DU FICHIER CSV ===
fieldnames = ["filename", "true_class", "predicted_class"] + [f"prob_class_{i}" for i in range(num_classes)]

with open(output_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in results:
        writer.writerow(row)

print(f"\n✅ Prédictions sauvegardées dans : {output_csv}")
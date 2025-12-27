# Nouveau script : exploration des hyperparamètres + support mps (Apple M1)

import os
import glob
import json
import csv
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from BEATs import BEATs, BEATsConfig
from datetime import datetime
from tqdm import tqdm

# === CHEMINS ===
save_dir = "/Users/leo-polde/Documents/Louis-Lumière/MÉMOIRE/PYTHON/BEATs/beats_modf_PPM/New_models"
output_dir_ontology = "/Users/leo-polde/Documents/Louis-Lumière/MÉMOIRE/PYTHON/BEATs/beats_modf_PPM/utils"
df = pd.read_csv("utils/class_mapping_fooddrink_train.csv")
DATA_DIR = "/Users/leo-polde/Documents/Louis-Lumière/MÉMOIRE/PYTHON/BEATs/beats_modf_PPM/utils/ANO_TRAINSET_FOODDRINK_4sec"
checkpoint_path = "/Users/leo-polde/Documents/Louis-Lumière/MÉMOIRE/PYTHON/BEATs/beats_modf_PPM/old_models/Fine-tuned_BEATs_iter3+_AS2M_cpt2.pt"
results_dir = "/Users/leo-polde/Documents/Louis-Lumière/MÉMOIRE/PYTHON/BEATs/beats_modf_PPM/utils/RESULTS/New_results"
os.makedirs(results_dir, exist_ok=True)


# === DEVICE SETUP ===
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"✅ Utilisation de : {device}")

# === AUDIO DATASET ===
class AudioDataset(Dataset):
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        waveform, sr = torchaudio.load(self.file_paths[idx])
        waveform = waveform / waveform.abs().max()
        waveform = waveform.squeeze(0)
        label = self.labels[idx]
        return waveform, label

# === LABEL MAPPING ===
label_to_idx = {label: idx for idx, label in enumerate(df['class'].unique())}
df['label_idx'] = df['class'].map(label_to_idx)

ontology_path = os.path.join(output_dir_ontology, 'ontology_FD.json')
with open(ontology_path, 'w') as f:
    json.dump(label_to_idx, f)

X = df['filename'].values
y = df['class'].values

# === CROSS-VALIDATION ===
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
skf = StratifiedKFold(n_splits=6, shuffle=True, random_state=42)

# === HYPERPARAM SEARCH ===
learning_rates = [1e-2]
batch_sizes = [16]
num_epochs = 50
early_stop_patience = 5
num_classes = 6

# Pour suivre la meilleure config
best_overall_acc = 0.0
best_config = ""

for lr in learning_rates:
    for batch_size in batch_sizes:
        print(f"\n=== LRate: {lr}, BatchSize: {batch_size} ===")

        # Écrire un fichier CSV pour cette combinaison
        csv_filename = f"BatchSize={batch_size}_LRate={lr:.0e}.csv"
        csv_output_path = os.path.join(results_dir, csv_filename)
        with open(csv_output_path, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['filename'] + list(label_to_idx.keys()))
            

            # Charger le modèle
            checkpoint = torch.load(checkpoint_path, map_location=device)
            cfg = BEATsConfig(checkpoint['cfg'])
            cfg.finetuned_model = False
            base_model = BEATs(cfg)
            base_model.load_state_dict(checkpoint['model'], strict=False)
            base_model.to(device).eval()
            for param in base_model.parameters():
                param.requires_grad = False

            for fold, (train_idx, val_idx) in enumerate(skf.split(X_trainval, y_trainval)):
                print(f"\n--- Fold {fold + 1} ---")

                # Chargement d'une Tête de classification vierge
                head = nn.Linear(cfg.encoder_embed_dim, num_classes).to(device)
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(head.parameters(), lr=lr)
                best_val_acc = 0.0
                no_improve_epochs = 0

                X_train_paths = [os.path.join(DATA_DIR, fname) for fname in X_trainval[train_idx]]
                y_train_idx = [label_to_idx[label] for label in y_trainval[train_idx]]
                X_val_paths = [os.path.join(DATA_DIR, fname) for fname in X_trainval[val_idx]]
                y_val_idx = [label_to_idx[label] for label in y_trainval[val_idx]]

                train_loader = DataLoader(AudioDataset(X_train_paths, y_train_idx), batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(AudioDataset(X_val_paths, y_val_idx), batch_size=batch_size, shuffle=False)

                for epoch in tqdm(range(num_epochs), desc=f"Fold {fold+1} [LR={lr:.0e}, BS={batch_size}]", leave=False):

                    # Phase d'entraînement
                    head.train()
                    running_loss = 0.0

                    for inputs, labels in train_loader:
                        inputs = inputs.to(device)
                        labels = labels.to(device).long()
                        optimizer.zero_grad()
                        with torch.no_grad():
                            feat = base_model.extract_features(inputs)[0]
                            feats = feat.mean(dim=1) if feat.size(1) > 1 else feat.squeeze(1)
                        outputs = head(feats)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()
                        running_loss += loss.item()

                    # Evaluation sur un jeu de validation
                    head.eval()
                    correct, total = 0, 0
                    with torch.no_grad():
                        for inputs, labels in val_loader:
                            inputs = inputs.to(device)
                            labels = labels.to(device).long()
                            feats = base_model.extract_features(inputs)[0].mean(dim=1)
                            outputs = head(feats)
                            _, preds = torch.max(outputs, 1)
                            total += labels.size(0)
                            correct += (preds == labels).sum().item()

                    val_acc = 100 * correct / total
                    print(f"Epoch {epoch+1}: val_acc = {val_acc:.2f}%")

                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        no_improve_epochs = 0
                        torch.save({'head': head.state_dict()}, os.path.join(save_dir, f"best_model_B{batch_size}_LR{lr:.0e}_F{fold}.pth"))
                    else:
                        no_improve_epochs += 1
                        if no_improve_epochs >= early_stop_patience:
                            print("⏹️ Early stopping\n")
                            break

            # === Prédiction avec ensemble des têtes ===
            test_paths = [os.path.join(DATA_DIR, fname) for fname in X_test]
            all_heads = []

            # Charger toutes les têtes sauvegardées
            for fold in range(skf.get_n_splits()):
                head = nn.Linear(cfg.encoder_embed_dim, num_classes).to(device)
                head_path = os.path.join(save_dir, f"best_model_B{batch_size}_LR{lr:.0e}_F{fold}.pth")
                head.load_state_dict(torch.load(head_path, map_location=device)['head'])
                head.eval()
                all_heads.append(head)

            correct, total = 0, 0

            for path, label_str in zip(test_paths, y_test):
                label = label_to_idx[label_str]
                waveform, _ = torchaudio.load(path)
                waveform = waveform.squeeze(0).to(device)

                with torch.no_grad():
                    feat = base_model.extract_features(waveform.unsqueeze(0))[0]
                    feat = feat.mean(dim=1) if feat.size(1) > 1 else feat.squeeze(1)

                    logits_ensemble = torch.zeros(num_classes).to(device)
                    for head in all_heads:
                        logits = head(feat)
                        logits_ensemble += logits.squeeze(0)

                    probs = torch.softmax(logits_ensemble / len(all_heads), dim=0).cpu().numpy()
                    pred = np.argmax(probs)

                writer.writerow([os.path.basename(path)] + [f"{p:.4f}" for p in probs])
                correct += (pred == label)
                total += 1

            final_acc = 100 * correct / total
            print(f"🎯 Accuracy finale (moyenne des {len(all_heads)} têtes) : {final_acc:.2f}%")

            if final_acc > best_overall_acc:
                best_overall_acc = final_acc
                best_config = f"LR={lr:.0e}, BS={batch_size}"





print("\n==============================")
print(f"🏆 Meilleure config : {best_config} avec {best_overall_acc:.2f}% de précision")
print("==============================")



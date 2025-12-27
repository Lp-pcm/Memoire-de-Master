from BEATs import BEATs, BEATsConfig
import torch
import torch.nn as nn
import torch

# load the fine-tuned checkpoints
checkpoint = torch.load("/Users/leo-polde/Documents/Louis-Lumière/MÉMOIRE/PYTHON/BEATs/beats_modf_PPM/models/Fine-tuned_BEATs_iter3+_AS2M_cpt2.pt", map_location="cpu")

# Charger le modèle
cfg = BEATsConfig(checkpoint['cfg'])
BEATs_model = BEATs(cfg)
BEATs_model.load_state_dict(checkpoint['model'])
BEATs_model.eval()

cfg.predictor_class = 527  # nombre de classes d'origine AudioSet
cfg.finetuned_model = True


print("Les keys du state_dict sont :", list(checkpoint['model'].keys())[:20])

missing_keys, unexpected_keys = BEATs_model.load_state_dict(checkpoint['model'], strict=False)
print("Missing keys:", missing_keys)
print("Unexpected keys:", unexpected_keys)

print("✅ Modèle pré-entraîné chargé !")
print(BEATs_model.predictor)  # Ça devrait être nn.Linear(..., 527)
num_classes = 6
BEATs_model.predictor = nn.Linear(cfg.encoder_embed_dim, num_classes)
print(f"✅ La tête du modèle a été remplacée par {num_classes} classes.")
print("La nouvelle dimension du prédicteur est :", BEATs_model.predictor)

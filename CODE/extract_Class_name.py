import os
import shutil

def extract_catid_from_filename(filename):
    """Extrait le CATID depuis le nom de fichier (avant le premier '_')."""
    base = os.path.basename(filename)
    catid = base.split('_')[0]
    return catid if '_' in base else 'UnknownCatID'

def anonymize_dataset(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    mapping = []

    for idx, filename in enumerate(os.listdir(input_dir)):
        if filename.endswith('.wav'):
            old_path = os.path.join(input_dir, filename)
            new_name = f"{idx:05d}.wav"
            new_path = os.path.join(output_dir, new_name)

            shutil.copy2(old_path, new_path)

            # Stocke l'info de correspondance
            class_name = extract_catid_from_filename(filename)
            mapping.append((new_name, class_name))

    return mapping

# Utilisation
input_dir = "/Users/leo-polde/Documents/Louis-Lumière/MÉMOIRE/PYTHON/BEATs/beats_modf/utils/DATASET_KITCHEN_4sec"
output_dir = "/Users/leo-polde/Documents/Louis-Lumière/MÉMOIRE/PYTHON/BEATs/beats_modf/utils/ANO_DATASET_KITCHEN"
mapping = anonymize_dataset(input_dir, output_dir)

# Sauvegarde le mapping dans un CSV pour créer ton dataset plus tard
import csv
with open("class_mapping.csv", "w", newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["filename", "class"])
    writer.writerows(mapping)

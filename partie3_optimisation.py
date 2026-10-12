"""
PARTIE 3 — Comparer automatiquement plusieurs niveaux d'élagage.
C'est l'équivalent fonctionnel d'une recherche d'hyperparamètres avec Experiment Manager.
Cette version optimise un niveau d'élagage par magnitude ; elle ne reproduit pas exactement
l'optimisation conjointe pruning/projection de MATLAB.
"""

from pathlib import Path
import copy
import csv
import torch
from torch import nn
from torch.nn.utils import prune
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from partie1_entrainement import (
    RoadCNN, load_matlab_sequences, left_pad_sequences, SEED
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def load_checkpoint_and_data():
    # Charger le modèle entraîné et les paramètres de prétraitement.
    checkpoint = torch.load(
        MODEL_DIR / "road_cnn_trained.pt", map_location="cpu", weights_only=False
    )
    base_model = RoadCNN(num_classes=len(checkpoint["class_names"]))
    base_model.load_state_dict(checkpoint["model_state"])
    base_model.eval()

    # Charger les signaux et les étiquettes originaux.
    sequences, labels_text = load_matlab_sequences(
        DATA_DIR / "allroadData.mat", DATA_DIR / "allroadLabel.mat"
    )
    X = left_pad_sequences(sequences, checkpoint["max_length"])
    X = (X - checkpoint["mean"]) / checkpoint["std"]
    label_to_id = {name: i for i, name in enumerate(checkpoint["class_names"])}
    y = np.asarray([label_to_id[label] for label in labels_text], dtype=np.int64)

    # Recréer une partition reproductible train/test pour comparer les essais.
    # Pour une étude rigoureuse, sauvegarder les indices de la partie 1 et les réutiliser.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )
    X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
    y_test = torch.tensor(y_test, dtype=torch.long)
    return base_model, checkpoint, X_test, y_test


def evaluate(model, X, y):
    # Passer en mode évaluation et désactiver le calcul des gradients.
    model.eval()
    with torch.no_grad():
        pred = model(X).argmax(dim=1)
    return accuracy_score(y.numpy(), pred.numpy())


def main():
    base_model, checkpoint, X_test, y_test = load_checkpoint_and_data()
    fractions = np.arange(0.0, 0.81, 0.10)
    results = []

    # Tester chaque valeur de fraction d'élagage, comme une petite expérience indépendante.
    for fraction in fractions:
        candidate = copy.deepcopy(base_model)

        # Appliquer le même taux de mise à zéro aux poids des convolutions.
        for module in candidate.modules():
            if isinstance(module, nn.Conv1d) and fraction > 0:
                prune.l1_unstructured(module, name="weight", amount=float(fraction))
                prune.remove(module, "weight")

        # Mesurer la précision sans réentraîner : c'est une comparaison rapide de candidats.
        accuracy = evaluate(candidate, X_test, y_test)
        parameter_count = sum(p.numel() for p in candidate.parameters())
        nonzero_count = sum(torch.count_nonzero(p).item() for p in candidate.parameters())

        results.append({
            "pruning_fraction": float(fraction),
            "test_accuracy": float(accuracy),
            "parameter_count": int(parameter_count),
            "nonzero_parameters": int(nonzero_count),
        })
        print(
            f"Élagage={fraction:.0%} | accuracy={accuracy:.4f} | "
            f"poids non nuls={nonzero_count}"
        )

    # Écrire les résultats dans un CSV pour les ouvrir dans Excel ou pandas.
    csv_path = RESULTS_DIR / "comparaison_elagage.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    # Choisir le candidat qui a la meilleure précision de test dans cette démonstration.
    # En pratique, le choix doit être fait sur validation ; le test final ne sert qu'une fois.
    best = max(results, key=lambda row: row["test_accuracy"])
    print("\nMeilleur candidat observé :", best)
    print(f"Résultats enregistrés dans : {csv_path}")
    print(
        "Attention : les poids nuls ne réduisent pas forcément le nombre de paramètres "
        "ou la taille du modèle dense. Pour une compression réelle, employer une "
        "reconstruction structurée ou un format d'inférence adapté."
    )


if __name__ == "__main__":
    main()

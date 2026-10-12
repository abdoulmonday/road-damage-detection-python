"""
PARTIE 2 — Exemple de compression du CNN entraîné.
Important : cette version Python est une approximation fonctionnelle, pas une
reproduction exacte des algorithmes propriétaires du tutoriel MATLAB.
Elle applique un élagage par magnitude et une quantification dynamique des couches Linear.
"""

from pathlib import Path
import copy
import torch
from torch import nn
from torch.nn.utils import prune
from partie1_entrainement import RoadCNN, DEVICE, load_matlab_sequences, left_pad_sequences
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def load_model_and_test_data():
    # Charger le modèle et ses métadonnées, y compris longueur et normalisation.
    checkpoint = torch.load(MODEL_DIR / "road_cnn_trained.pt", map_location="cpu", weights_only=False)
    model = RoadCNN(num_classes=len(checkpoint["class_names"]))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # Recharger les données pour illustrer l'évaluation ; pour une comparaison rigoureuse,
    # réutiliser exactement les indices de test sauvegardés pendant la partie 1.
    # Ici, on crée une partition déterministe de démonstration, qui peut différer du test de la partie 1.
    sequences, labels_text = load_matlab_sequences(
        DATA_DIR / "allroadData.mat", DATA_DIR / "allroadLabel.mat"
    )
    X = left_pad_sequences(sequences, checkpoint["max_length"])
    X = (X - checkpoint["mean"]) / checkpoint["std"]
    names = checkpoint["class_names"]
    mapping = {name: i for i, name in enumerate(names)}
    y = np.asarray([mapping[label] for label in labels_text], dtype=np.int64)

    # On utilise ici tout le jeu pour illustrer le fonctionnement du code,
    # pas pour rapporter une performance de test indépendante.
    x_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
    y_tensor = torch.tensor(y, dtype=torch.long)
    return model, checkpoint, x_tensor, y_tensor


def evaluate(model, x_tensor, y_tensor):
    # Mesurer la précision sur les tenseurs fournis.
    model.eval()
    with torch.no_grad():
        predictions = model(x_tensor).argmax(dim=1)
    return accuracy_score(y_tensor.numpy(), predictions.numpy())


def model_file_size(path):
    # Obtenir la taille du fichier en octets puis la convertir en Ko décimaux.
    return path.stat().st_size / 1000.0


def main():
    model, checkpoint, X, y = load_model_and_test_data()
    original_accuracy = evaluate(model, X, y)

    # Compter les paramètres entraînables du réseau initial.
    original_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Créer une copie pour ne pas modifier le réseau original sauvegardé.
    pruned_model = copy.deepcopy(model)

    # Élagage non structuré : mettre à zéro 30 % des poids de chaque couche Conv1d.
    # Cela réduit le nombre de poids non nuls, mais pas forcément le nombre de paramètres
    # ni la taille d'un fichier dense. Le modèle reste donc de dimensions identiques.
    for module in pruned_model.modules():
        if isinstance(module, nn.Conv1d):
            prune.l1_unstructured(module, name="weight", amount=0.30)
            prune.remove(module, "weight")  # Rendre les zéros permanents dans les poids.

    pruned_accuracy = evaluate(pruned_model, X, y)

    # Sauvegarder les poids élagués. La taille du fichier dépend aussi du format de sauvegarde.
    pruned_path = MODEL_DIR / "road_cnn_pruned.pt"
    torch.save(
        {"model_state": pruned_model.state_dict(), **{
            k: checkpoint[k] for k in ["max_length", "mean", "std", "class_names"]
        }},
        pruned_path
    )

    # Quantification dynamique : PyTorch convertit les couches Linear compatibles
    # en représentation quantifiée, généralement utile surtout sur CPU.
    # Les couches Conv1d ne sont pas automatiquement quantifiées par cette ligne.
    quantized_model = torch.ao.quantization.quantize_dynamic(
        copy.deepcopy(pruned_model),
        {nn.Linear},
        dtype=torch.qint8
    )
    quantized_accuracy = evaluate(quantized_model, X, y)

    quantized_path = MODEL_DIR / "road_cnn_pruned_dynamic_quantized.pt"
    torch.save(
        {"model_state": quantized_model.state_dict(), **{
            k: checkpoint[k] for k in ["max_length", "mean", "std", "class_names"]
        }},
        quantized_path
    )

    # Afficher les mesures pour comparer les étapes.
    print(f"Précision initiale (jeu illustratif) : {original_accuracy * 100:.2f} %")
    print(f"Précision après élagage : {pruned_accuracy * 100:.2f} %")
    print(f"Précision après quantification dynamique : {quantized_accuracy * 100:.2f} %")
    print(f"Paramètres entraînables du modèle initial : {original_parameters}")
    print(f"Taille fichier élagué : {model_file_size(pruned_path):.2f} Ko")
    print(f"Taille fichier quantifié : {model_file_size(quantized_path):.2f} Ko")
    print("\nAttention : cette évaluation sur tout le jeu est illustrative, pas une métrique de test indépendante.")


if __name__ == "__main__":
    main()

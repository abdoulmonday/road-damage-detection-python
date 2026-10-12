"""
PARTIE 1 — Entraîner un CNN 1D pour classer les signaux routiers.
Données attendues dans data/ :
  allroadData.mat  -> variable MATLAB allroadData
  allroadLabel.mat -> variable MATLAB allroadLabel
Classes : CR (fissurée) et UNCR (non fissurée).
"""

from pathlib import Path
import json
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

# Fixer les graines aléatoires aide à rendre les résultats plus reproductibles.
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Définir les dossiers du projet à partir de l'emplacement de ce script.
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Choisir le processeur graphique s'il est disponible, sinon utiliser le CPU.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_mat_variable(mat_dict, expected_name):
    """Trouver une variable MATLAB en ignorant les clés internes commençant par __."""
    if expected_name in mat_dict:
        return mat_dict[expected_name]
    public_keys = [key for key in mat_dict if not key.startswith("__")]
    if len(public_keys) == 1:
        return mat_dict[public_keys[0]]
    raise KeyError(
        f"Variable '{expected_name}' introuvable. Variables disponibles : {public_keys}"
    )


def load_matlab_sequences(data_path, labels_path):
    """Charger les signaux MATLAB et leurs étiquettes, puis les convertir en Python."""
    data_mat = sio.loadmat(data_path, simplify_cells=True)
    labels_mat = sio.loadmat(labels_path, simplify_cells=True)

    raw_data = find_mat_variable(data_mat, "allroadData")
    raw_labels = find_mat_variable(labels_mat, "allroadLabel")

    # Une cellule MATLAB contenant plusieurs signaux devient généralement un tableau
    # d'objets NumPy lorsque simplify_cells=True.
    sequences = []
    for item in np.asarray(raw_data, dtype=object).reshape(-1):
        signal = np.asarray(item, dtype=np.float32).squeeze()
        if signal.ndim != 1:
            raise ValueError(f"Un signal n'est pas un vecteur 1D : forme {signal.shape}")
        sequences.append(signal)

    # Convertir les catégories MATLAB (chaînes ou objets) en chaînes Python.
    labels = []
    for item in np.asarray(raw_labels, dtype=object).reshape(-1):
        if isinstance(item, bytes):
            label = item.decode("utf-8")
        else:
            label = str(item)
        labels.append(label.strip())

    if len(sequences) != len(labels):
        raise ValueError(
            f"Nombre de signaux ({len(sequences)}) différent du nombre "
            f"d'étiquettes ({len(labels)}). Vérifier les fichiers MAT."
        )

    # Le tutoriel utilise les classes CR et UNCR. Refuser les étiquettes inattendues
    # évite de lancer silencieusement un entraînement sur des données mal lues.
    allowed = {"CR", "UNCR"}
    observed = set(labels)
    if not observed.issubset(allowed):
        raise ValueError(
            f"Étiquettes inattendues : {observed}. "
            "Inspectez allroadLabel.mat et adaptez le décodage si nécessaire."
        )

    return sequences, np.asarray(labels)


def left_pad_sequences(sequences, target_length):
    """Compléter à gauche les signaux courts et conserver les target_length dernières valeurs."""
    padded = np.zeros((len(sequences), target_length), dtype=np.float32)
    for index, signal in enumerate(sequences):
        signal = np.asarray(signal, dtype=np.float32)
        signal = signal[-target_length:]
        padded[index, -len(signal):] = signal
    return padded


class RoadDataset(Dataset):
    """Petit Dataset PyTorch : un signal et son numéro de classe par observation."""

    def __init__(self, signals, labels):
        self.signals = torch.tensor(signals, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # Conv1d attend une forme (canaux, longueur) pour chaque signal.
        return self.signals[index].unsqueeze(0), self.labels[index]


class RoadCNN(nn.Module):
    """CNN 1D compact, inspiré de l'architecture du tutoriel MATLAB."""

    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),
            nn.Conv1d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        # x a la forme (taille_du_lot, 1, longueur_du_signal).
        features = self.features(x)
        return self.classifier(features)


def main():
    # Vérifier que les deux fichiers de données existent avant de commencer.
    data_file = DATA_DIR / "allroadData.mat"
    labels_file = DATA_DIR / "allroadLabel.mat"
    if not data_file.exists() or not labels_file.exists():
        raise FileNotFoundError(
            "Placez allroadData.mat et allroadLabel.mat dans le dossier data/."
        )

    # Lire les signaux et les étiquettes du jeu de données original.
    sequences, labels_text = load_matlab_sequences(data_file, labels_file)
    print(f"Nombre d'observations : {len(sequences)}")
    print("Répartition des classes :", {c: int(np.sum(labels_text == c)) for c in np.unique(labels_text)})

    # Fixer une longueur commune : le plus long signal définit la longueur retenue.
    # Les signaux plus courts seront complétés à gauche par des zéros.
    max_length = max(len(signal) for signal in sequences)
    X = left_pad_sequences(sequences, max_length)

    # Transformer les noms de classe en nombres, en conservant la correspondance.
    class_names = sorted(np.unique(labels_text).tolist())
    class_to_id = {name: index for index, name in enumerate(class_names)}
    y = np.asarray([class_to_id[name] for name in labels_text], dtype=np.int64)

    # Séparer d'abord 80 % pour l'entraînement et 20 % pour le reste.
    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )
    # Séparer les 20 % restants en validation (10 %) et test (10 %).
    X_val, X_test, y_val, y_test = train_test_split(
        X_rest, y_rest, test_size=0.50, random_state=SEED, stratify=y_rest
    )

    # Estimer moyenne et écart-type uniquement sur les données d'entraînement.
    # Cela évite d'utiliser indirectement les données de test pour le prétraitement.
    train_mean = float(X_train.mean())
    train_std = float(X_train.std())
    if train_std == 0:
        train_std = 1.0

    # Appliquer la même normalisation aux trois ensembles.
    X_train = (X_train - train_mean) / train_std
    X_val = (X_val - train_mean) / train_std
    X_test = (X_test - train_mean) / train_std

    # Créer les chargeurs qui fournissent les observations par lots.
    train_loader = DataLoader(RoadDataset(X_train, y_train), batch_size=16, shuffle=True)
    val_loader = DataLoader(RoadDataset(X_val, y_val), batch_size=16, shuffle=False)
    test_loader = DataLoader(RoadDataset(X_test, y_test), batch_size=16, shuffle=False)

    # Construire le réseau et le déplacer sur CPU ou GPU.
    model = RoadCNN(num_classes=len(class_names)).to(DEVICE)

    # Calculer des poids de classe inversément proportionnels aux fréquences.
    class_counts = np.bincount(y_train, minlength=len(class_names))
    weights = len(y_train) / (len(class_names) * np.maximum(class_counts, 1))
    class_weights = torch.tensor(weights, dtype=torch.float32, device=DEVICE)

    # CrossEntropyLoss compare les scores du réseau aux classes attendues.
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    # Adam ajuste les paramètres du réseau pour réduire la fonction de perte.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    best_val_accuracy = -1.0
    best_state = None
    patience = 20
    epochs_without_improvement = 0
    max_epochs = 200

    # Une époque correspond à un passage complet sur les données d'entraînement.
    for epoch in range(max_epochs):
        model.train()  # Active le mode entraînement (notamment BatchNorm).
        total_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()           # Effacer les gradients précédents.
            logits = model(batch_x)         # Calculer les scores des classes.
            loss = criterion(logits, batch_y)  # Calculer l'erreur de classification.
            loss.backward()                 # Calculer les gradients.
            optimizer.step()                # Mettre à jour les poids du réseau.
            total_loss += loss.item() * len(batch_y)

        # Mesurer l'accuracy de validation sans modifier les paramètres.
        model.eval()
        val_true, val_pred = [], []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                logits = model(batch_x.to(DEVICE))
                predictions = logits.argmax(dim=1).cpu().numpy()
                val_pred.extend(predictions.tolist())
                val_true.extend(batch_y.numpy().tolist())
        val_accuracy = accuracy_score(val_true, val_pred)
        mean_loss = total_loss / len(train_loader.dataset)
        print(
            f"Époque {epoch + 1:03d}/{max_epochs} | "
            f"perte={mean_loss:.4f} | val_accuracy={val_accuracy:.4f}"
        )

        # Garder en mémoire la meilleure version d'après la validation.
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Arrêter si la validation ne s'améliore plus pendant la patience définie.
        if epochs_without_improvement >= patience:
            print("Arrêt anticipé : la validation ne s'améliore plus.")
            break

    # Recharger les poids qui ont donné la meilleure précision de validation.
    model.load_state_dict(best_state)
    model.to(DEVICE)
    model.eval()

    # Évaluer une seule fois sur le jeu de test indépendant.
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            logits = model(batch_x.to(DEVICE))
            y_pred.extend(logits.argmax(dim=1).cpu().numpy().tolist())
            y_true.extend(batch_y.numpy().tolist())

    test_accuracy = accuracy_score(y_true, y_pred)
    print(f"\nPrécision de test : {test_accuracy * 100:.2f} %")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    # Sauvegarder les poids et toutes les informations nécessaires à l'inférence.
    checkpoint = {
        "model_state": model.cpu().state_dict(),
        "max_length": max_length,
        "mean": train_mean,
        "std": train_std,
        "class_names": class_names,
        "class_to_id": class_to_id,
        "test_accuracy": test_accuracy,
    }
    torch.save(checkpoint, MODEL_DIR / "road_cnn_trained.pt")

    # Enregistrer aussi les paramètres de prétraitement dans un fichier JSON lisible.
    with open(MODEL_DIR / "metadata.json", "w", encoding="utf-8") as file:
        json.dump(
            {"max_length": max_length, "mean": train_mean, "std": train_std,
             "class_names": class_names, "test_accuracy": test_accuracy},
            file, ensure_ascii=False, indent=2
        )

    # Visualiser la matrice de confusion du jeu de test.
    ConfusionMatrixDisplay.from_predictions(
        [class_names[i] for i in y_true],
        [class_names[i] for i in y_pred],
        display_labels=class_names,
        cmap="Blues"
    )
    plt.title("Matrice de confusion — modèle Python")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "matrice_confusion_partie1.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()

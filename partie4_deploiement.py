"""
PARTIE 4 — Exporter le réseau PyTorch au format ONNX et vérifier l'inférence.
ONNX remplace fonctionnellement l'export Simulink pour un déploiement portable,
mais ne crée pas un modèle Simulink.
"""

from pathlib import Path
import torch
from partie1_entrainement import RoadCNN

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)


def main():
    # Charger le checkpoint entraîné avec les informations de prétraitement.
    checkpoint = torch.load(
        MODEL_DIR / "road_cnn_trained.pt", map_location="cpu", weights_only=False
    )

    # Reconstruire l'architecture en utilisant le même nombre de classes.
    model = RoadCNN(num_classes=len(checkpoint["class_names"]))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # Créer un signal fictif uniquement pour indiquer à l'exporteur la forme de l'entrée.
    # Ce signal ne sert pas à entraîner le modèle.
    example_input = torch.zeros(1, 1, checkpoint["max_length"], dtype=torch.float32)

    # Exporter l'architecture et ses poids dans un fichier ONNX.
    # L'axe 0 est dynamique : le modèle peut recevoir des lots de tailles différentes.
    torch.onnx.export(
        model,
        (example_input,),
        MODEL_DIR / "road_damage_cnn.onnx",
        input_names=["accelerometer_signal"],
        output_names=["class_scores"],
        dynamic_axes={
            "accelerometer_signal": {0: "batch_size"},
            "class_scores": {0: "batch_size"},
        },
        opset_version=17,
        dynamo=False,
    )

    print("Modèle exporté :", MODEL_DIR / "road_damage_cnn.onnx")
    print("Classes :", checkpoint["class_names"])
    print("Longueur attendue du signal :", checkpoint["max_length"])
    print("Moyenne de normalisation :", checkpoint["mean"])
    print("Écart-type de normalisation :", checkpoint["std"])
    print(
        "Pour l'inférence, appliquer exactement le même remplissage à gauche "
        "et la même normalisation avant d'envoyer un signal au modèle."
    )

    # Vérifier que le fichier ONNX peut être chargé et exécuté.
    try:
        import onnxruntime as ort
        import numpy as np

        session = ort.InferenceSession(
            str(MODEL_DIR / "road_damage_cnn.onnx"),
            providers=["CPUExecutionProvider"]
        )
        ort_output = session.run(
            ["class_scores"],
            {"accelerometer_signal": np.zeros(
                (1, 1, checkpoint["max_length"]), dtype=np.float32
            )}
        )[0]
        print("Vérification ONNX réussie. Forme de la sortie :", ort_output.shape)
    except ImportError:
        print("onnxruntime n'est pas installé ; vérification d'exécution ignorée.")


if __name__ == "__main__":
    main()

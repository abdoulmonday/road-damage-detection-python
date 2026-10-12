# Détection des dommages routiers — équivalent Python des 4 tutoriels MATLAB

Ce projet illustre une chaîne de travail avec PyTorch :
1. entraîner un CNN 1D sur les signaux d'accéléromètre ;
2. appliquer un élagage par magnitude et une quantification dynamique ;
3. comparer plusieurs niveaux d'élagage ;
4. exporter le réseau au format ONNX.

## Limites importantes

Ce projet est un équivalent pédagogique, pas une reproduction numérique exacte des tutoriels MATLAB.
- L'architecture PyTorch est inspirée du CNN MATLAB, mais n'est pas identique couche par couche.
- Le prétraitement est explicite : remplissage à gauche, puis centrage-réduction estimé sur le train.
- Le pruning utilisé ici est non structuré. Il met des poids à zéro, mais ne supprime pas automatiquement les dimensions du réseau.
- La quantification dynamique PyTorch de cet exemple concerne les couches `Linear`, pas les couches `Conv1d`.
- La projection PCA des couches du tutoriel MATLAB n'est pas reproduite exactement.
- ONNX est un format portable de déploiement, pas un remplacement direct de Simulink.
- Pour comparer scientifiquement les résultats, il faut conserver les mêmes indices train/validation/test dans tous les scripts. Le script de partie 2 et celui de partie 3 indiquent les endroits à adapter pour réutiliser les indices de la partie 1.

## Préparer les données

1. Télécharger le ZIP original :
   https://ssd.mathworks.com/supportfiles/wavelet/crackDetection/transverse_crack.zip
2. Décompresser le ZIP.
3. Copier `allroadData.mat` et `allroadLabel.mat` dans le dossier `data/`.
4. Vérifier que les fichiers contiennent bien les variables MATLAB `allroadData` et `allroadLabel`.

Les données originales décrites par MathWorks contiennent des séries d'accélération verticale et les étiquettes `CR` et `UNCR`.

## Installer

```bash
python -m venv .venv
```

Activation Windows :
```bash
.venv\Scripts\activate
```

Activation macOS/Linux :
```bash
source .venv/bin/activate
```

Installer les bibliothèques :
```bash
python -m pip install -r requirements.txt
```

## Exécuter dans l'ordre

```bash
python partie1_entrainement.py
python partie2_compression.py
python partie3_optimisation.py
python partie4_deploiement.py
```

## Explication des lignes : Partie 1

- `Path(__file__).resolve().parent` trouve le dossier où se trouve le script.
- `np.random.seed` et `torch.manual_seed` fixent les graines aléatoires.
- `scipy.io.loadmat` lit les fichiers `.mat`.
- `simplify_cells=True` simplifie les structures MATLAB, notamment les cellules.
- `np.asarray(..., dtype=object).reshape(-1)` met les éléments dans une liste unidimensionnelle.
- `np.asarray(item, dtype=np.float32).squeeze()` convertit un signal en vecteur numérique.
- `left_pad_sequences` ajoute des zéros au début des signaux courts.
- `train_test_split(..., stratify=y)` conserve autant que possible les proportions de classes dans les partitions.
- La moyenne et l'écart-type sont calculés sur le train uniquement, puis appliqués à validation et test.
- `Dataset` et `DataLoader` préparent les lots d'observations.
- `nn.Conv1d` extrait des motifs temporels locaux.
- `nn.BatchNorm1d` stabilise les activations.
- `nn.ReLU` introduit une non-linéarité.
- `nn.MaxPool1d` réduit la longueur temporelle.
- `nn.AdaptiveAvgPool1d(1)` agrège les caractéristiques dans le temps.
- `nn.Linear` combine les caractéristiques pour produire deux scores de classe.
- `CrossEntropyLoss` calcule l'erreur entre les scores et les étiquettes.
- `optimizer.zero_grad()` efface les gradients de l'itération précédente.
- `loss.backward()` calcule les gradients.
- `optimizer.step()` met à jour les poids.
- `model.eval()` active le mode évaluation.
- `torch.no_grad()` évite de calculer les gradients pendant la validation et le test.
- `argmax(dim=1)` choisit la classe au score le plus élevé.
- `torch.save` sauvegarde le réseau et les paramètres de prétraitement.

## Explication des lignes : Partie 2

- `copy.deepcopy(model)` crée une copie indépendante du réseau.
- `prune.l1_unstructured(..., amount=0.30)` met à zéro 30 % des poids de convolution de plus faible magnitude.
- `prune.remove` rend les zéros permanents dans le tenseur.
- `quantize_dynamic(..., {nn.Linear}, dtype=torch.qint8)` quantifie dynamiquement les couches linéaires prises en charge.
- `stat().st_size` renvoie la taille du fichier en octets.

Un fichier peut ne pas devenir plus petit après un élagage non structuré : les zéros occupent encore des positions dans les tenseurs denses. Il faut une représentation sparse ou une reconstruction structurée pour convertir ces zéros en véritable gain de mémoire d'inférence.

## Explication des lignes : Partie 3

- `np.arange(0.0, 0.81, 0.10)` crée les niveaux d'élagage testés.
- La boucle `for fraction in fractions` lance une expérience par niveau.
- `copy.deepcopy` repart du même modèle initial pour que les essais soient comparables.
- `accuracy_score` mesure la proportion de prédictions correctes.
- `torch.count_nonzero` compte les paramètres non nuls.
- `csv.DictWriter` enregistre les résultats dans un fichier CSV.

Pour une sélection correcte, utilisez l'ensemble de validation pour choisir le modèle, puis n'évaluez le candidat final qu'une seule fois sur le test indépendant.

## Explication des lignes : Partie 4

- `model.eval()` prépare le réseau pour l'inférence.
- `torch.zeros(1, 1, max_length)` crée un exemple de forme `(lot, canaux, longueur)`.
- `torch.onnx.export` convertit le graphe PyTorch et ses poids en fichier ONNX.
- `input_names` et `output_names` nomment les entrées et sorties.
- `dynamic_axes` autorise des tailles de lots variables.
- `onnxruntime.InferenceSession` ouvre le fichier ONNX.
- `session.run` exécute une prédiction de test.

L'exemple fictif de zéros sert uniquement à vérifier l'exportation ; ce n'est pas une mesure de performance du modèle.

## Interpréter les résultats

Ne vous attendez pas nécessairement à retrouver exactement les 97,06 % du tutoriel MATLAB. Les résultats dépendent de l'architecture, du prétraitement, de la partition train/test, des graines aléatoires et des détails d'entraînement. Le jeu de données est petit : il faut rapporter les métriques du test indépendant et idéalement répéter l'expérience avec plusieurs graines.

# KuaiRec 2.0 Recommender System

Ce projet implémente un système de recommandation à deux étapes pour le dataset KuaiRec 2.0, comparant un modèle de filtrage collaboratif de référence avec un modèle hybride qui incorpore des caractéristiques secondaires.

## Installation

```bash
# Créer un environnement virtuel Python
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows

# Installer les dépendances
pip install -r requirements.txt
```

## Exécution standard (entraînement complet)

```bash
python main.py --matrix big_matrix.csv --epochs 200 --test_neg_ratio 99
```

## Choix du modèle à entraîner/évaluer

L'argument `--model` permet de choisir quel modèle exécuter :

- `--model baseline` : entraîne et évalue uniquement le modèle de base (LightFM sans features)
- `--model hybrid` : entraîne et évalue uniquement le modèle hybride (LightFM avec features)
- `--model all` : entraîne et compare les deux modèles (par défaut)

Exemple :

```bash
python main.py --matrix big_matrix.csv --model hybrid --epochs 200
```

## Configuration des threads

```bash
# Spécifier manuellement le nombre de threads (utile sur serveurs avec beaucoup de cœurs)
python main.py --threads 8 --epochs 200
```

## Résolution des problèmes courants

- **Erreur de mémoire**: Réduire `test_neg_ratio`
- **Performance lente**: Augmenter le nombre de threads
- **Précision insuffisante**: Augmenter `epochs` et `test_neg_ratio`

## Exécution sur différents matériels

### CPU multi-cœurs
- Augmenter les threads selon les capacités du CPU (4-8 threads généralement optimal)
- Utiliser `--threads X` où X est ~80% des cœurs disponibles

### Serveur de calcul
- Utiliser toute la mémoire disponible pour des ratios de test plus élevés
- Augmenter les threads selon les capacités du serveur

### Machine virtuelle / Cloud
- Choisir une instance avec plus de vCPUs
- Ajouter swap si la mémoire est limitée
- Réduire `test_neg_ratio` pour économiser la mémoire

## Requirements

- Python 3.6+
- Dependencies listed in `requirements.txt`

## Setup

1. Clone the repository
2. Set up a virtual environment (recommended)
3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset Structure

The dataset files should be organized in the following structure:

```
sys-recommenders/
  KuaiRec2.0/
    data/
      small_matrix.csv         # Interaction data for prototyping (~4.7M rows)
      big_matrix.csv           # Complete interaction data (~12.5M rows)
      item_categories.csv      # Item category data
      user_features.csv        # User feature data
```

## Running the Pipeline

To run the pipeline with the small matrix (for prototyping):

```bash
python main.py --matrix small_matrix.csv
```

To run with the big matrix (for final evaluation):

```bash
python main.py --matrix big_matrix.csv --model all --epochs 200 --test_neg_ratio 99
```

### Command Line Arguments

- `--matrix`: Specifies which interaction matrix to use (default: `small_matrix.csv`)
- `--epochs`: Number of training epochs (default: `200`)
- `--eval_every`: Evaluate model every N epochs (default: `10`)
- `--test_neg_ratio`: Negative ratio for evaluation (default: `99`)
- `--model`: Model to train (`baseline`, `hybrid`, `all`)
- `--threads`: Number of threads to use
- `--data_dir`: Custom data directory path (optional, overrides default path)

## Project Structure

- `loaddata.py`: Data loading utilities
- `preprocess.py`: Data preprocessing functions
- `evaluation.py`: Evaluation metrics implementation
- `

# KuaiRec 2.0 Recommender System

Ce projet implémente un système de recommandation basé sur LightFM pour le dataset KuaiRec 2.0. Seul le **modèle de base** (factorisation matricielle avec perte WARP) est désormais pris en charge. Les résultats sont enregistrés au format CSV.

---

## Installation

```bash
# 1. Créer et activer un environnement virtuel Python
python3 -m venv venv
source venv/bin/activate     # Linux/Mac
# ou
venv\Scripts\activate        # Windows

# 2. Installer les dépendances
pip install -r requirements.txt
```

## Structure du dataset
```
KuaiRec2.0/
└── data/
    ├── small_matrix.csv      # ~4.7M interactions (prototype)
    ├── big_matrix.csv        # ~12.5M interactions (évaluation finale)
    ├── item_categories.csv   # Catégories de vidéos
    └── user_features.csv     # Caractéristiques des utilisateurs
```

## Usage
Exécution standard
Entraînement complet sur big_matrix.csv et sauvegarde des résultats :

```bash
python main.py \
  --matrix big_matrix.csv \
  --epochs 200 \
  --eval_every 10 \
  --test_neg_ratio 99
```
À la fin, un fichier results_<matrix>_<YYYYMMDD_HHMMSS>.csv est créé, contenant pour chaque époque :

- Precision@5, Recall@5, F1@5

- Precision@10, Recall@10, F1@10

- Precision@20, Recall@20, F1@20

- Precision@50, Recall@50, F1@50


Arguments disponibles
```bash
--matrix        Fichier d'interactions à utiliser (small_matrix.csv ou big_matrix.csv)
--epochs        Nombre d’époques d’entraînement (défaut : 300)
--eval_every    Fréquence d’évaluation en époques (défaut : 10)
--patience      Patience pour l’arrêt anticipé (défaut : 8)
--test_neg_ratio Ratio négatif pour évaluation (défaut : 49)
--fast          Mode rapide (échantillonnage réduit)
--threads       Nombre de threads à utiliser (par défaut : détection automatique)
--data_dir      Chemin vers le dossier de données (optionnel)
--model         Doit être “baseline” (modèle hybride retiré)

```
Configuration des threads
Pour forcer un nombre de threads :

```bash
python main.py --threads 8 --epochs 200
```


## Structure du projet
```
├── loaddata.py
├── preprocess.py
├── evaluation.py
├── main.py
├── requirements.txt
└── README.md

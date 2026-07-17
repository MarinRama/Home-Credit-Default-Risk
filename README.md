# 🏦 Home Credit Default Risk - Credit Scoring MLOps

Projet réalisé dans le cadre de la formation **Machine Learning Engineer** d'OpenClassrooms.

## 📋 Description

L'objectif de ce projet est de construire un modèle de **scoring de crédit** capable de prédire si un client rencontrera des difficultés de remboursement.

Le projet met en œuvre une démarche MLOps complète :

- Analyse exploratoire des données (EDA)
- Prétraitement et Feature Engineering
- Entraînement et comparaison de modèles
- Optimisation du seuil de décision selon un coût métier
- Suivi des expérimentations avec MLflow
- Déploiement du modèle via une API FastAPI

---

## 📂 Structure du projet

```text
.
├── app/                    # API FastAPI
├── data/
│   ├── raw/                # Données brutes (non versionnées)
│   ├── processed/          # Données préparées (non versionnées)
├── models/                 # Modèles entraînés
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_feature_engineering.ipynb
│   └── 04_model_training_mlflow.ipynb
├── tests/                  # Tests unitaires
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📊 Jeu de données

Le projet utilise les données de la compétition Kaggle :

**Home Credit Default Risk**

➡️ https://www.kaggle.com/competitions/home-credit-default-risk

Les données ne sont **pas incluses** dans ce dépôt en raison de leur taille.

Après téléchargement, placez les fichiers CSV dans :

```text
data/raw/
```

---

## 🚀 Installation

Cloner le dépôt :

```bash
git clone https://github.com/MarinRama/Home-Credit-Default-Risk.git
cd Home-Credit-Default-Risk
```

Créer un environnement virtuel :

### macOS / Linux

```bash
python -m venv .venv
source .venv/bin/activate
```

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

Installer les dépendances :

```bash
pip install -r requirements.txt
```

---

## 📈 Entraînement du modèle

Les différentes étapes sont détaillées dans les notebooks :

1. Analyse exploratoire
2. Prétraitement
3. Feature Engineering
4. Entraînement
5. Optimisation du seuil métier
6. Suivi des expériences avec MLflow

---

## 📊 MLflow

Lancer MLflow :

```bash
mlflow ui
```

Puis ouvrir :

```
http://127.0.0.1:5000
```

---

## 🌐 API

L'API est développée avec **FastAPI**.

Lancer l'application :

```bash
uvicorn app.main:app --reload
```

Documentation interactive :

```
http://127.0.0.1:8000/docs
```

---

## 🧪 Tests

Lancer les tests :

```bash
pytest
```

---

## 🛠️ Technologies utilisées

- Python
- Pandas
- NumPy
- Scikit-learn
- LightGBM
- MLflow
- FastAPI
- SQLAlchemy
- Pytest
- Git / GitHub

---

## 📌 Objectif métier

Le modèle prédit la probabilité qu'un client rencontre des difficultés de remboursement.

Le seuil de décision est optimisé afin de **minimiser un coût métier**, où un **faux négatif coûte 10 fois plus qu'un faux positif**.

---

## 👤 Auteur

**Marin Ramananjaona**

Projet réalisé dans le cadre de la formation **Machine Learning Engineer** chez **OpenClassrooms**.
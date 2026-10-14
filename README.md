# Home Credit Default Risk — Scoring et MLOps

Projet réalisé dans le cadre de la formation Machine Learning Engineer
d’OpenClassrooms.

Ce dépôt contient le développement du modèle de scoring du projet
« Initiez-vous au MLOps », puis son déploiement et son monitoring
pour le projet « Confirmez vos compétences en MLOps ».

## Objectif

Servir le modèle de scoring via une API, automatiser les tests et le
déploiement, enregistrer les appels et analyser les données de production
simulées.

Le modèle retourne un score de risque et une classe prédite.
Le score n’est pas présenté comme une probabilité calibrée.

## Solution retenue

- Modèle LightGBM issu du projet précédent.
- API FastAPI servie par Uvicorn.
- Conteneurisation avec Docker et Docker Compose.
- Tests automatisés avec pytest.
- CI/CD avec GitHub Actions.
- Stockage local des événements dans SQLite.
- Rapport HTML de monitoring avec pandas, SciPy et Matplotlib.

Le déploiement fonctionne localement dans Docker.
La CI/CD réalise un déploiement temporaire de vérification dans
GitHub Actions. Elle ne fournit pas un hébergement public permanent.

## Structure

```text
.
├── src/
│   ├── api/main.py                 # Routes et journalisation des appels
│   ├── scoring/predictor.py        # Validation et prédiction
│   └── monitoring/storage.py      # Stockage SQLite
├── models/                        # Modèle, seuil et contrat des variables
├── scripts/                       # Simulation, export et analyse
├── tests/                         # Tests automatisés et requête exemple
├── notebooks/                     # Travaux du projet précédent
├── reports/
│   ├── monitoring_report.html
│   ├── performance/
│   └── screenshots/
├── data/                          # Données locales non versionnées
├── .github/workflows/ci-cd.yml
├── Dockerfile
├── compose.yaml
├── requirements-api.txt
├── requirements-dev.txt
├── requirements-monitoring.txt
└── requirements.txt               # Environnement du projet précédent
```

## Lancer l’API avec Docker

Prérequis : Git, Docker et Docker Compose.

```bash
git clone https://github.com/MarinRama/Home-Credit-Default-Risk.git
cd Home-Credit-Default-Risk
docker compose up -d --build --wait
```

Vérifier le service :

```bash
curl -sS http://127.0.0.1:8001/health
```

Documentation interactive :
http://127.0.0.1:8001/docs

Envoyer une requête exemple :

```bash
curl -sS -i \
  -X POST http://127.0.0.1:8001/predict \
  -H "Content-Type: application/json" \
  --data-binary @tests/fixtures/valid_request.json
```

Consulter les logs :

```bash
docker compose logs --tail=80 api
```

Arrêter le service :

```bash
docker compose down
```

Le volume `production_data` conserve SQLite après la suppression
du conteneur. La commande `docker compose down -v` supprime également
ce volume et ses données.

## Contrat de l’API

### Routes

- `GET /health` : disponibilité du service et chargement du modèle.
- `GET /model-info` : informations sur le modèle et ses variables.
- `POST /predict` : score et classe prédite.

### Entrée

La requête contient un objet `features` avec les 541 variables préparées
attendues par le modèle.

La liste et l’ordre des variables sont définis dans
`models/feature_schema.json`.

L’API reçoit des données déjà préparées : elle ne reconstruit pas
les 541 variables à partir des seuls renseignements bruts d’un client.

Toutes les clés sont obligatoires. Une valeur manquante peut être
représentée par `null`. Une observation entièrement vide est rejetée.

Les contrôles portent notamment sur :

- les variables absentes ou inconnues ;
- les types incorrects et valeurs non finies ;
- les contraintes définies sur certains montants, nombres de personnes
  et scores externes.

### Sortie

```json
{
  "score": 0.0884589283752367,
  "prediction": 0,
  "threshold": 0.5000000000000001,
  "model_version": "1.0.0"
}
```

- `prediction = 1` lorsque le score est supérieur ou égal au seuil.
- La classe 1 correspond à un défaut prédit.
- La classe 0 correspond à une absence de défaut prédite.
- La version retournée est celle définie dans les métadonnées du service.

Le seuil est repris du projet précédent. Le contexte métier attribue
un coût dix fois supérieur aux faux négatifs par rapport aux faux positifs.

### Erreurs

- `422` : requête ou variables invalides.
- `503` : modèle indisponible.
- `500` : erreur interne de prédiction.

Le modèle est chargé au démarrage, puis réutilisé.
La validation et la préparation des données sont réalisées une seule
fois par prédiction.

## Tests

Pour travailler localement, créer un environnement Python.
Le conteneur et la CI utilisent Python 3.11.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-api.txt -r requirements-dev.txt
python -m pytest tests -v --cov=src --cov-report=term-missing
```

LightGBM nécessite une bibliothèque OpenMP disponible sur le système.
Le Dockerfile installe `libgomp1` pour l’environnement Linux.

Dernière validation locale après optimisation :
50 tests réussis, couverture de 95 %.

Les tests couvrent les prédictions, les entrées invalides,
les erreurs API et le stockage des événements.

## CI/CD

Le workflow `.github/workflows/ci-cd.yml` enchaîne :

1. Exécution des tests.
2. Construction de l’image Docker.
3. Démarrage du conteneur et vérification de l’API.

Il est déclenché notamment par les push sur `main` et les pull requests
vers `main`.

L’image construite est utilisée par le job de déploiement.
Des rapports de tests et de déploiement sont conservés comme artefacts.

Les ressources de ce déploiement simulé sont supprimées en fin de job.
La persistance locale repose sur le volume Docker Compose.

## Stockage des appels

Dans Docker, la base se trouve à :

```text
/app/data/production/predictions.sqlite3
```

La table `prediction_events` contient :

- identifiant de requête et horodatage ;
- version du modèle et statut HTTP ;
- entrées et sorties JSON ;
- temps d’inférence et temps de traitement ;
- type d’erreur éventuel.

Les entrées et sorties peuvent être absentes pour certains appels invalides.

Chaque réponse de prédiction fournit un en-tête `X-Request-ID`.
L’en-tête `X-Monitoring-Status` indique `stored` ou `failed`.

Si le stockage échoue, l’API conserve sa réponse de prédiction,
signale cet échec dans l’en-tête et écrit une erreur dans les logs.
La disponibilité du scoring est ainsi privilégiée, au prix d’une
possible perte de traçabilité.

La base et les données de simulation ne sont pas versionnées.
SQLite convient au PoC local ; les besoins de concurrence,
de conservation et d’accès devront être réévalués pour une exploitation réelle.

Le service local est exposé sur l’interface de boucle locale.
Une exposition publique nécessiterait notamment une authentification,
du chiffrement en transit et une gestion adaptée des accès aux données.

## Rapport de monitoring et analyse du drift

Le notebook exécuté demandé pour la remise est :

`notebooks/Ramananjaona_Marin_6_analyse_data_drift_092026.ipynb`

Une version HTML du rapport est également disponible :

`reports/monitoring_report.html`

L’analyse contient :

- la comparaison des distributions et des taux de valeurs manquantes ;
- la distribution des scores ;
- le taux d’erreur et les durées d’exécution ;
- les alertes, leur interprétation et les limites.

Les sorties enregistrées dans le notebook permettent de consulter les
résultats sans disposer des données locales. Sa réexécution nécessite
les fichiers de référence et les exports décrits ci-dessous.

### Données utilisées

- Référence : 5 000 clients.
- Production stable : 1 000 autres clients.
- Production avec dérive : les mêmes 1 000 clients, modifiés artificiellement.

Les clients de référence et de production stable sont distincts.
Leur source est le fichier préparé du projet précédent :

```text
data/processed/application_train_enriched.parquet
```

Les données d’origine proviennent de la compétition
Home Credit Default Risk :
https://www.kaggle.com/competitions/home-credit-default-risk

Les données préparées ne sont pas incluses dans Git.
Pour reproduire la simulation, ce fichier doit être disponible localement.
Il contient `SK_ID_CURR`, `TARGET` et les variables préparées.
L’identifiant et la cible sont exclus des entrées du modèle.

### Reproduire l’analyse

Installer les dépendances de monitoring dans l’environnement local :

```bash
python -m pip install -r requirements-monitoring.txt
```

Avec l’API Docker démarrée :

```bash
python -m scripts.prepare_monitoring_data
python -m scripts.create_drift_scenario
python -m scripts.simulate_production --scenario stable --limit 1000
python -m scripts.simulate_production --scenario drift --limit 1000
```

Relever les deux identifiants d’exécution affichés, puis remplacer
les valeurs entre chevrons dans la commande suivante :

```bash
python -m scripts.export_monitoring_runs \
  --stable-run <identifiant_stable> \
  --drift-run <identifiant_drift>

python -m scripts.generate_monitoring_report
```

Ouvrir ensuite `reports/monitoring_report.html` dans un navigateur.

L’analyse est exécutée automatiquement par le script lorsqu’il est lancé.
Son exécution périodique n’est pas configurée.

### Méthode et seuils

L’étude porte sur `EXT_SOURCE_1`, `EXT_SOURCE_2` et `EXT_SOURCE_3`.

- Distance KS sur les valeurs renseignées : alerte si supérieure à 0,10.
- Variation absolue du taux de valeurs manquantes : alerte au-delà de 5 points.
- Latence client p95 : alerte au-delà de 100 ms.
- Taux d’erreur : alerte au-delà de 1 %.

Ces seuils sont des choix de démonstration, pas des normes.
Ils doivent être calibrés selon les distributions habituelles
et les objectifs métier.

### Résultats de l’étude livrée

- Aucune alerte sur les trois variables du scénario stable.
- `EXT_SOURCE_2` : distance KS de 0,650 dans le scénario modifié.
- `EXT_SOURCE_3` : hausse de 17,38 points des valeurs manquantes
  par rapport à la référence.
- `EXT_SOURCE_1`, variable témoin : aucune alerte.
- Deux décisions modifiées sur 1 000 clients entre les scénarios.
- Aucun appel en erreur dans les deux exécutions sélectionnées.

Une alerte doit conduire à rechercher sa cause : changement de population,
problème de source ou de préparation des données.
Elle ne justifie pas à elle seule un réentraînement.

### Limites

Les changements sont artificiels et les éventuelles variables dérivées
ne sont pas recalculées.

L’étude ne couvre pas toutes les variables du modèle.
Le taux d’erreur affiché porte sur les exécutions sélectionnées,
pas sur tout l’historique SQLite.

La dérive des données ne démontre pas une perte de performance prédictive.
Sans résultats réels de remboursement, le rapport ne mesure pas
la précision du modèle en production.

## Optimisation des performances

Le profiling a identifié une double validation et construction
du DataFrame pour chaque requête.

Cette répétition a été supprimée, sans modifier le modèle ni son seuil.

Comparaison sur les mêmes 1 000 clients :

| Mesure | Avant | Après | Réduction |
|---|---:|---:|---:|
| Traitement serveur médian | 3,646 ms | 2,287 ms | 37,3 % |
| Latence client médiane | 6,125 ms | 4,463 ms | 27,1 % |
| Latence client p95 | 7,594 ms | 5,988 ms | 21,1 % |

Écart maximal des scores : 0.
Aucune décision modifiée.

Ces résultats correspondent à des appels locaux séquentiels.
Ils ne constituent pas une garantie de performance sous charge concurrente.
L’optimisation concerne le service autour du modèle.

Rapport détaillé et mesures :
`reports/performance/optimization_report.md`
et `reports/performance/comparison.json`.

## Projet précédent

Les notebooks conservent les travaux d’exploration, de préparation,
d’entraînement et de suivi des expérimentations avec MLflow.

`requirements.txt` conserve les dépendances de cet environnement.
Son installation n’est pas nécessaire pour lancer l’API avec Docker.

Les fichiers locaux de suivi MLflow ne sont pas inclus dans Git.

## Preuves et livrables

- Code API et scoring : `src/`.
- Tests : `tests/`.
- Conteneurisation : `Dockerfile` et `compose.yaml`.
- Automatisation : `.github/workflows/ci-cd.yml`.
- Monitoring : `reports/monitoring_report.html`.
- Optimisation : `reports/performance/`.
- Captures : `reports/screenshots/`.
- Historique de construction : commits Git.

## Auteur

Marin Ramananjaona
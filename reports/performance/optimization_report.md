# Rapport d’optimisation du service de scoring

## Objectif

Réduire le temps de réponse de l’API après son déploiement,
en conservant les validations d’entrée et les prédictions du modèle.

## Diagnostic

Les premières mesures de monitoring indiquaient une inférence médiane
de 0,578 ms pour un traitement serveur médian de 3,646 ms.

L’inspection du code a identifié une double préparation des données :
la route API appelait prepare_input, puis la méthode predict
appelait à nouveau cette même fonction.

Un profiling cProfile réalisé dans Docker sur 200 prédictions,
après 20 appels d’échauffement, a confirmé :

- 400 appels à prepare_input ;
- 1,679 s cumulée dans prepare_input sur 1,904 s au total ;
- 1,509 s consacrée à la construction des DataFrames,
  incluse dans le temps de préparation ;
- 0,222 s dans predict_proba.

Ces temps instrumentés servent au diagnostic et ne sont pas
directement comparables aux latences HTTP sans profiling.

## Modification réalisée

La validation et la préparation sont désormais réalisées une seule fois,
dans CreditPredictor.predict.

Une exception InputValidationError permet à l’API de conserver
les réponses HTTP 422 pour les entrées invalides.
Les erreurs de prédiction restent traitées comme des erreurs HTTP 500.

Le modèle, les variables d’entrée et le seuil de décision
n’ont pas été modifiés.

## Protocole de comparaison

- Même fichier de 1 000 clients, vérifié par son empreinte.
- Même ordre d’envoi des clients.
- Appels HTTP séquentiels sur la même machine.
- API exécutée avec Docker Compose.
- Stockage SQLite actif avant et après.
- Reconstruction de l’image après modification.
- Comparaison des scores, des décisions et des durées.

Exécution avant :
61dec5cd-789a-4112-be00-bed1d51f1ddc

Exécution après :
458e1b3d-1a3e-459d-b5f2-d992df3de378

## Résultats

| Mesure | Avant | Après | Réduction |
|---|---:|---:|---:|
| Inférence médiane | 0,5778 ms | 0,5647 ms | 2,3 % |
| Traitement serveur médian | 3,6460 ms | 2,2874 ms | 37,3 % |
| Latence client médiane | 6,1250 ms | 4,4625 ms | 27,1 % |
| Latence client p95 | 7,5937 ms | 5,9878 ms | 21,1 % |

La modification vise la préparation des données.
La variation du temps d’inférence ne démontre pas une accélération
du modèle lui-même.

## Vérification de non-régression

- 50 tests automatisés réussis.
- Couverture du code source : 95 %.
- 1 000 réponses HTTP 200 après optimisation.
- Stockage confirmé pour les 1 000 appels.
- Écart maximal des scores avant/après : 0.
- Nombre de décisions modifiées : 0.
- Seuils de décision identiques.

L’absence de changement des prédictions est démontrée sur cet échantillon.
Elle ne constitue pas une évaluation générale de la précision
ou des biais du modèle.

## Configuration retenue et justification

- Machine locale : MacBook Air Apple M5, 16 Go de mémoire.
- Conteneur Linux ARM64, Python 3.11, base Debian Bookworm slim.
- FastAPI et Uvicorn pour servir le modèle.
- LightGBM pour conserver le modèle du projet précédent.
- pandas pour préparer les variables dans l’ordre attendu.
- SQLite avec volume Docker persistant pour le stockage local.
- Versions des dépendances API définies dans requirements-api.txt.

L’optimisation du code suffit à réduire la latence observée.
Aucune conversion du modèle, quantification ou modification matérielle
n’a été introduite.

## Limites

La comparaison porte sur deux exécutions locales séquentielles.
La charge de la machine n’est pas strictement contrôlée.
Les pourcentages décrivent le gain observé dans ces conditions,
sans garantir les mêmes résultats sous charge concurrente ou dans le cloud.

Le traitement serveur est mesuré avant l’écriture SQLite.
La latence client inclut l’aller-retour HTTP et la réception de la réponse.

## Déploiement

La version optimisée fonctionne localement dans Docker et a été validée et déployée temporairement
par GitHub Actions après fusion dans main. Les trois étapes
du pipeline ont réussi : tests, construction Docker et vérification
du déploiement.
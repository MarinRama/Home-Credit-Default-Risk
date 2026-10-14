import base64
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


ROOT = Path(__file__).resolve().parents[1]

VARIABLES = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]

KS_THRESHOLD = 0.10
MISSING_THRESHOLD = 0.05
LATENCY_THRESHOLD_MS = 100
ERROR_THRESHOLD_PERCENT = 1


def table(frame):
    return frame.to_html(
        index=False,
        border=0,
        float_format=lambda value: f"{value:.3f}",
        escape=True,
    )


def main():
    reference = pd.read_parquet(
        ROOT / "data/reference/reference_features.parquet"
    )
    manifest = json.loads(
        (ROOT / "data/production/exports/monitoring_exports.json")
        .read_text(encoding="utf-8")
    )

    features = {}
    events = {}

    for scenario in ["stable", "drift"]:
        features[scenario] = pd.read_parquet(
            ROOT / manifest[scenario]["features"]
        )
        events[scenario] = pd.read_parquet(
            ROOT / manifest[scenario]["events"]
        )

    # Analyse des distributions et des valeurs manquantes.
    drift_rows = []

    for scenario, current in features.items():
        for variable in VARIABLES:
            ref_values = reference[variable].dropna()
            current_values = current[variable].dropna()

            missing_ref = reference[variable].isna().mean()
            missing_current = current[variable].isna().mean()
            missing_delta = missing_current - missing_ref

            if ref_values.empty or current_values.empty:
                distance = float("nan")
                distribution_status = "NON CALCULABLE"
            else:
                # Utilisation de la distance, sans interpréter la p-value.
                distance = float(
                    ks_2samp(
                        ref_values,
                        current_values,
                        method="asymp",
                    ).statistic
                )
                distribution_status = (
                    "ALERTE" if distance > KS_THRESHOLD else "OK"
                )

            drift_rows.append({
                "Scénario": scenario,
                "Variable": variable,
                "Distance KS": distance,
                "Distribution": distribution_status,
                "Manquants référence (%)": 100 * missing_ref,
                "Manquants production (%)": 100 * missing_current,
                "Écart (points)": 100 * missing_delta,
                "Valeurs manquantes": (
                    "ALERTE"
                    if abs(missing_delta) > MISSING_THRESHOLD
                    else "OK"
                ),
            })

    drift_table = pd.DataFrame(drift_rows)

    # Métriques opérationnelles des exécutions sélectionnées.
    operation_rows = []

    for scenario, frame in events.items():
        error_rate = 100 * (frame["status_code"] >= 400).mean()
        latency_p95 = frame["client_latency_ms"].quantile(0.95)

        operation_rows.append({
            "Scénario": scenario,
            "Appels": len(frame),
            "Erreurs (%)": error_rate,
            "Inférence médiane (ms)": frame["inference_ms"].median(),
            "Traitement médian (ms)": frame["processing_ms"].median(),
            "Latence médiane (ms)": frame["client_latency_ms"].median(),
            "Latence p95 (ms)": latency_p95,
            "Alerte erreurs": (
                "ALERTE" if error_rate > ERROR_THRESHOLD_PERCENT else "OK"
            ),
            "Alerte latence": (
                "ALERTE" if latency_p95 > LATENCY_THRESHOLD_MS else "OK"
            ),
            "Score moyen": frame["score"].mean(),
            "Classe 1 (%)": 100 * frame["prediction"].mean(),
        })

    operation_table = pd.DataFrame(operation_rows)

    # Les mêmes clients figurent dans les deux scénarios.
    # Leur position permet de comparer les décisions individuelles.
    paired = events["stable"][
        ["row_position", "score", "prediction"]
    ].merge(
        events["drift"][["row_position", "score", "prediction"]],
        on="row_position",
        suffixes=("_stable", "_drift"),
        validate="one_to_one",
    )

    if len(paired) != len(events["stable"]) or len(paired) != len(events["drift"]):
        raise ValueError("Les scénarios ne contiennent pas les mêmes positions.")

    decision_changes = int(
        (paired["prediction_stable"] != paired["prediction_drift"]).sum()
    )
    mean_absolute_score_change = (
        paired["score_drift"] - paired["score_stable"]
    ).abs().mean()

    # Graphiques intégrés dans le HTML : aucun fichier image séparé.
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    bins = np.linspace(0, 1, 26)

    for ax, variable in zip(
        axes[0],
        ["EXT_SOURCE_2", "EXT_SOURCE_3"],
    ):
        ax.hist(
            reference[variable].dropna(),
            bins=bins,
            density=True,
            histtype="step",
            label="Référence",
        )
        for scenario in ["stable", "drift"]:
            ax.hist(
                features[scenario][variable].dropna(),
                bins=bins,
                density=True,
                histtype="step",
                label=scenario,
            )
        ax.set_title(f"{variable} — valeurs renseignées")
        ax.set_ylabel("Densité")
        ax.legend()

    for scenario in ["stable", "drift"]:
        axes[1, 0].hist(
            events[scenario]["score"],
            bins=bins,
            density=True,
            histtype="step",
            label=scenario,
        )

    axes[1, 0].set_title("Distribution des scores")
    axes[1, 0].set_xlabel("Score")
    axes[1, 0].set_ylabel("Densité")
    axes[1, 0].legend()

    for scenario in ["stable", "drift"]:
        axes[1, 1].plot(
            events[scenario]["row_position"],
            events[scenario]["client_latency_ms"],
            label=scenario,
            alpha=0.7,
        )

    axes[1, 1].axhline(
        LATENCY_THRESHOLD_MS,
        color="red",
        linestyle="--",
        label="Seuil indicatif : 100 ms",
    )
    axes[1, 1].set_title("Latence au fil des appels")
    axes[1, 1].set_xlabel("Position de la requête")
    axes[1, 1].set_ylabel("Latence client (ms)")
    axes[1, 1].legend()

    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=140)
    plt.close(fig)
    picture = base64.b64encode(buffer.getvalue()).decode("ascii")

    generated_at = datetime.now(timezone.utc).isoformat()

    html = f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>Rapport de monitoring du scoring</title>
<style>
body {{ font-family: sans-serif; margin: 35px; color: #222; }}
h1, h2 {{ color: #163b60; }}
table {{ border-collapse: collapse; font-size: 13px; }}
th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
th {{ background: #eaf0f6; }}
img {{ width: 100%; max-width: 1200px; }}
.scroll {{ overflow-x: auto; }}
li {{ margin-bottom: 8px; }}
</style>
</head>
<body>
<h1>Rapport de monitoring du modèle de scoring</h1>
<p>Généré le {generated_at} — heure UTC.</p>

<h2>Périmètre et méthode</h2>
<p>Référence : {len(reference)} clients.
Production : {len(events["stable"])} appels stables et
{len(events["drift"])} appels avec dérive.
Les entrées, scores et durées serveur proviennent de SQLite.
Les latences client proviennent des journaux d'appels.</p>
<p>L'analyse du drift porte sur trois variables sélectionnées,
sur les 541 variables du modèle. EXT_SOURCE_1 est un témoin non modifié.
La distance KS compare les valeurs renseignées ; les valeurs manquantes
font l'objet d'un contrôle distinct.</p>
<p>Seuils indicatifs du PoC : KS &gt; {KS_THRESHOLD},
variation absolue des valeurs manquantes &gt; 5 points,
latence p95 &gt; {LATENCY_THRESHOLD_MS} ms, erreurs &gt; 1 %.
Ces seuils sont des choix de démonstration, à calibrer sur des
fenêtres stables et des objectifs métier avant un usage réel.</p>

<h2>Dérive et qualité des données</h2>
<div class="scroll">{table(drift_table)}</div>

<h2>Fonctionnement de l'API et scores</h2>
<div class="scroll">{table(operation_table)}</div>
<p>Inférence : calcul du modèle.
Traitement serveur : traitement mesuré avant l'écriture SQLite.
Latence client : durée de l'appel HTTP, incluant l'aller-retour
et la réception de la réponse.</p>
<p>Les taux d'erreur concernent uniquement les deux exécutions
sélectionnées, pas l'ensemble de l'historique de l'API.
Les appels sont séquentiels et locaux : ce n'est pas un test de charge.</p>

<h2>Comparaison des mêmes clients</h2>
<p>Décisions différentes : {decision_changes} sur {len(paired)} clients.
Variation absolue moyenne du score :
{mean_absolute_score_change:.5f}.</p>

<h2>Distributions et latences</h2>
<img src="data:image/png;base64,{picture}" alt="Graphiques de monitoring">

<h2>Interprétation et actions</h2>
<ul>
<li>Une alerte de distribution nécessite de vérifier la population
reçue et la préparation des variables.</li>
<li>Une hausse des valeurs manquantes nécessite de vérifier
la disponibilité de la source et les transformations de données.</li>
<li>Une alerte de latence ou d'erreurs nécessite d'examiner
les logs et les ressources du service.</li>
<li>Une dérive ne déclenche pas automatiquement un réentraînement :
il faut d'abord en rechercher la cause et évaluer son impact.</li>
</ul>

<h2>Limites</h2>
<ul>
<li>Le scénario modifie artificiellement EXT_SOURCE_2 et EXT_SOURCE_3.
Les variables dérivées éventuelles ne sont pas recalculées.</li>
<li>La référence et le scénario stable proviennent du même fichier,
avec des clients distincts. Ce découpage n'est pas une nouvelle
évaluation indépendante du modèle.</li>
<li>La classe 1 désigne une prédiction du modèle, pas un défaut observé.
Sans résultats réels de remboursement, ce rapport ne mesure
pas la performance prédictive en production.</li>
<li>Une absence d'alerte sur ces trois variables ne garantit pas
l'absence de dérive sur les autres variables.</li>
</ul>
</body>
</html>
"""

    output = ROOT / "reports" / "monitoring_report.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")

    print(drift_table.to_string(index=False))
    print("\n", operation_table.to_string(index=False))
    print(f"\nDécisions modifiées : {decision_changes}/{len(paired)}")
    print(f"Rapport créé : {output}")


if __name__ == "__main__":
    main()
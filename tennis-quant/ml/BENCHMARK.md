# Tennis Quant Engine — Benchmark de référence

Date de validation CI : 2026-09-21

## Protocole

- Tours séparés : ATP et WTA.
- Historique d'entraînement : 2005–2026.
- Années totalement hors échantillon : 2022, 2023, 2024, 2025.
- Calibration : fenêtre chronologiquement antérieure de 180 jours.
- Validation : walk-forward uniquement.
- Sélection du champion : log loss, puis Brier Score.
- Source R&D : archive Sackmann, CC BY-NC-SA 4.0, usage recherche/non-commercial.

## ATP

Échantillon hors échantillon : **10 514 matchs**.

| Modèle | Log Loss | Brier | Accuracy | ECE-10 |
|---|---:|---:|---:|---:|
| Ranking seul | 0.631818 | 0.221020 | 63.67% | 0.00853 |
| Elo seul | 0.666571 | 0.237005 | 59.18% | 0.01692 |
| **Logistique complète** | **0.618942** | **0.215459** | **65.01%** | **0.00956** |
| HistGradientBoosting | 0.619241 | 0.215538 | 64.83% | 0.01084 |

**Champion ATP : full_logit**

La logistique complète améliore nettement le ranking seul en log loss et Brier Score. Le gradient boosting est extrêmement proche, mais ne le bat pas sur les métriques probabilistes prioritaires.

## WTA

Échantillon hors échantillon : **4 485 matchs**.

| Modèle | Log Loss | Brier | Accuracy | ECE-10 |
|---|---:|---:|---:|---:|
| Ranking seul | 0.625922 | 0.217873 | 65.24% | 0.01250 |
| Elo seul | 0.667941 | 0.237147 | 59.87% | 0.02130 |
| **Logistique complète** | **0.613412** | **0.212604** | 66.06% | **0.01075** |
| HistGradientBoosting | 0.615539 | 0.213350 | **66.58%** | 0.01348 |

**Champion WTA : full_logit**

Le gradient boosting gagne légèrement en accuracy brute, mais la logistique complète est meilleure en log loss, Brier et calibration. Elle reste donc le choix de production de la V1.

## Interprétation

Ces résultats valident la **qualité probabiliste** du moteur, pas sa rentabilité de pari.

Aucun ROI, CLV ou profit n'est revendiqué sans :
1. vraies cotes pré-match horodatées ;
2. règles de sélection fixées avant le test ;
3. simulation de mise sans look-ahead ;
4. comparaison à la probabilité marché corrigée de la marge ;
5. suivi du closing line value.

Le prochain jalon est le backtest économique sur cotes autorisées.

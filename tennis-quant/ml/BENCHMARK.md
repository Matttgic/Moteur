# Tennis Quant Engine — Benchmark de référence

Date de validation : 2026-10-02 (Elo à somme nulle, voir « Historique »)

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
| Ranking seul | 0.631818 | 0.221020 | 63.67% | **0.00853** |
| Elo seul | 0.620325 | 0.216178 | 64.21% | 0.01120 |
| **Logistique complète** | **0.613231** | 0.213034 | 65.06% | 0.01033 |
| HistGradientBoosting | 0.613342 | **0.212956** | **65.40%** | 0.01624 |

**Champion ATP : full_logit**

La logistique complète améliore nettement le ranking seul en log loss et Brier Score. Le gradient boosting a une accuracy et un Brier très légèrement supérieurs, mais une log loss et une calibration moins bonnes : la logistique reste le champion.

## WTA

Échantillon hors échantillon : **4 485 matchs**.

| Modèle | Log Loss | Brier | Accuracy | ECE-10 |
|---|---:|---:|---:|---:|
| Ranking seul | 0.625922 | 0.217873 | 65.24% | **0.01250** |
| Elo seul | 0.613415 | 0.212733 | 65.93% | 0.01677 |
| **Logistique complète** | **0.606547** | **0.209587** | **67.18%** | 0.01628 |
| HistGradientBoosting | 0.609242 | 0.210583 | 66.60% | 0.01508 |

**Champion WTA : full_logit**

La logistique complète est la meilleure en log loss, Brier et accuracy. Son ECE est un peu plus élevé que celui du ranking seul, mais reste sous 2 %.

## Interprétation

Ces résultats valident la **qualité probabiliste** du moteur, pas sa rentabilité de pari.

Aucun ROI, CLV ou profit n'est revendiqué sans :
1. vraies cotes pré-match horodatées ;
2. règles de sélection fixées avant le test ;
3. simulation de mise sans look-ahead ;
4. comparaison à la probabilité marché corrigée de la marge ;
5. suivi du closing line value.

Le prochain jalon est le backtest économique sur cotes autorisées.

## Historique

### 2026-10-02 — Elo à somme nulle

La mise à jour Elo du perdant utilisait le score attendu du **vainqueur** (`perdant += K × (0 − E_vainqueur)`)
au lieu du sien (`1 − E_vainqueur`). L'Elo n'était donc pas à somme nulle : après un favori victorieux,
le perdant perdait presque K points ; après une surprise, le favori battu ne perdait presque rien.
Conséquence : l'Elo seul faisait moins bien que le simple classement (59,2 % contre 63,7 % ATP).

| Log loss hors échantillon (full_logit) | Avant | Après |
|---|---:|---:|
| ATP | 0.618942 | 0.613231 |
| WTA | 0.613412 | 0.606547 |

| Elo seul | Avant | Après |
|---|---:|---:|
| ATP accuracy | 59.18 % | 64.21 % |
| WTA accuracy | 59.87 % | 65.93 % |

Corrigé à l'identique dans `ml/tennis_quant_ml/features.py` (entraînement) et `lib/player-state.ts`
(mise à jour live). Spécifications du modèle (`lib/model-specs/*-full.json`) et états de départ
(`generated/`) régénérés. Les états joueurs déjà en base Supabase ont été calculés avec l'ancienne
règle : voir `PRODUCTION.md`.

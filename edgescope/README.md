# EdgeScope

MVP de scanner personnel de value betting.

- Source initiale: The Odds API
- Référence: Pinnacle
- Books FR: Betclic, NetBet, PMU, Unibet, Winamax
- Marchés: h2h/1X2, spreads/handicaps, totals
- De-vig proportionnel et edge = p_fair × odds_book - 1
- Données obsolètes >180 s rejetées
- Edge >25% rejeté comme outlier potentiel
- UI mobile, refresh 60 s
- Projet Supabase: tapotyftgyacbnjmeovn

Aucune donnée fictive n'est affichée en production. Sans clé API, l'interface indique explicitement que la source manque.

## Démarrage
Créer `.env.local` à partir de `.env.example` et renseigner `THE_ODDS_API_KEY`.

## Suite
Persistance/CLV, Telegram, historique, performance, backtests et exchange sont prévus dans les phases suivantes.

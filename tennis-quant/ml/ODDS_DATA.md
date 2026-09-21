# Historical odds input

The project deliberately does **not** scrape sources whose terms prohibit automated collection, AI/data-training use, or commercial reuse.

## Preferred normalized CSV schema

```text
tour,match_date,tournament,player_1,player_2,odds_1,odds_2,closing_odds_1,closing_odds_2,bookmaker
ATP,2025-01-07,Example Open,Player One,Player Two,1.72,2.18,1.68,2.25,provider
```

Required:
- `tour`
- `match_date`
- `tournament`
- `player_1`
- `player_2`
- `odds_1`
- `odds_2`

Optional:
- `closing_odds_1`
- `closing_odds_2`
- `bookmaker`

The importer also understands legacy winner/loser layouts when a user already possesses a file they are entitled to use, including `Winner/Loser` plus one supported odds pair such as `AvgW/AvgL`, `MaxW/MaxL`, `B365W/B365L`, or `PSW/PSL`.

## Backtest rules

The economic backtest:
1. uses only out-of-sample model probabilities;
2. removes bookmaker margin before computing edge;
3. uses fixed thresholds by default: edge >= 4 percentage points and EV >= 2%;
4. stakes a fixed 1 unit per qualifying bet;
5. reports profit, ROI, hit rate, average odds, average edge, expected value, drawdown and CLV when closing odds exist.

Do not tune the edge/EV thresholds on the same period later presented as an unbiased test.

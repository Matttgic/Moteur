import os
import json
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.stats import poisson

# --- 1. CONFIGURATION & SÉCURITÉ ---
# La clé est récupérée depuis les secrets GitHub
API_KEY = os.environ.get("API_KEY")

# Si on lance en local pour tester sans variable d'env, on peut mettre une clé par défaut (optionnel)
if not API_KEY:
    print("ATTENTION : Aucune clé API trouvée dans les variables d'environnement.")
    # API_KEY = "TA_CLE_ICI_POUR_TEST_LOCAL" # Décommente ça seulement pour tester sur ton PC

BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {'x-apisports-key': API_KEY}

LEAGUES = {
    "Premier League": 39,
    "Ligue 1": 61
}
CURRENT_SEASON = 2025

# --- 2. CLASSE ETL (Extraction & Cache) ---
class FootballDataConnector:
    def __init__(self):
        if not os.path.exists('data_cache'):
            os.makedirs('data_cache')

    def _get_from_api(self, endpoint, params):
        if not API_KEY: return []
        url = f"{BASE_URL}{endpoint}"
        try:
            response = requests.get(url, headers=HEADERS, params=params)
            if response.status_code == 200:
                return response.json()['response']
            else:
                print(f"Erreur API {response.status_code}: {response.text}")
                return []
        except Exception as e:
            print(f"Exception lors de l'appel API : {e}")
            return []

    def get_season_fixtures(self, league_id, season):
        filename = f"data_cache/fixtures_{league_id}_{season}.json"
        
        # Vérification Cache (24h)
        if os.path.exists(filename):
            file_age = time.time() - os.path.getmtime(filename)
            if file_age < 86400: 
                print(f"Chargement cache: Ligue {league_id}")
                with open(filename, 'r') as f:
                    return pd.DataFrame(json.load(f))

        # Appel API
        print(f"Appel API: Ligue {league_id} - Saison {season}")
        data = self._get_from_api("/fixtures", {"league": league_id, "season": season})
        
        if data:
            df = pd.json_normalize(data)
            df.to_json(filename, orient='records', indent=4)
            return df
        return pd.DataFrame()

# --- 3. FEATURES ENGINEERING (Elo & Stats) ---
class EloTracker:
    def __init__(self, k_factor=20, home_advantage=100):
        self.ratings = {}
        self.k = k_factor
        self.home_adv = home_advantage
        self.default_rating = 1500

    def get_rating(self, team):
        return self.ratings.get(team, self.default_rating)

    def expected_result(self, rating_a, rating_b):
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    def update_ratings(self, home_team, away_team, goal_diff):
        if goal_diff > 0: actual = 1
        elif goal_diff == 0: actual = 0.5
        else: actual = 0
        
        r_home = self.get_rating(home_team)
        r_away = self.get_rating(away_team)
        
        p_home = self.expected_result(r_home + self.home_adv, r_away)
        margin_mult = np.log(abs(goal_diff) + 1) if goal_diff != 0 else 1
        delta = self.k * margin_mult * (actual - p_home)
        
        self.ratings[home_team] = r_home + delta
        self.ratings[away_team] = r_away - delta

def process_features(df):
    if df.empty: return df
    
    # Nettoyage
    df['fixture.date'] = pd.to_datetime(df['fixture.date'])
    df = df.sort_values('fixture.date')
    finished = df[df['fixture.status.short'] == 'FT'].copy()
    
    # Elo
    elo_tracker = EloTracker()
    elo_home_vals = []
    elo_away_vals = []
    
    for index, row in finished.iterrows():
        h_team = row['teams.home.name']
        a_team = row['teams.away.name']
        h_goals = row['goals.home']
        a_goals = row['goals.away']
        
        elo_home_vals.append(elo_tracker.get_rating(h_team))
        elo_away_vals.append(elo_tracker.get_rating(a_team))
        
        if pd.notna(h_goals) and pd.notna(a_goals):
            elo_tracker.update_ratings(h_team, a_team, h_goals - a_goals)

    finished['elo_home'] = elo_home_vals
    finished['elo_away'] = elo_away_vals
    
    # Stats Glissantes (Correction avec transform)
    matches_long = []
    for idx, row in finished.iterrows():
        matches_long.append({
            'date': row['fixture.date'], 'team': row['teams.home.name'],
            'goals_for': row['goals.home'], 'goals_against': row['goals.away'], 'is_home': 1
        })
        matches_long.append({
            'date': row['fixture.date'], 'team': row['teams.away.name'],
            'goals_for': row['goals.away'], 'goals_against': row['goals.home'], 'is_home': 0
        })
        
    df_long = pd.DataFrame(matches_long).sort_values('date')
    
    df_long['avg_goals_for_5'] = df_long.groupby('team')['goals_for'].transform(lambda x: x.rolling(window=5).mean().shift())
    df_long['avg_goals_against_5'] = df_long.groupby('team')['goals_against'].transform(lambda x: x.rolling(window=5).mean().shift())
    
    # Mapping rapide
    df_long['date_team'] = list(zip(df_long['date'], df_long['team']))
    stats_map = df_long.set_index('date_team')[['avg_goals_for_5', 'avg_goals_against_5']].to_dict('index')
    
    finished['date_team_home'] = list(zip(finished['fixture.date'], finished['teams.home.name']))
    finished['date_team_away'] = list(zip(finished['fixture.date'], finished['teams.away.name']))
    
    finished['home_att_5'] = finished['date_team_home'].map(lambda x: stats_map.get(x, {}).get('avg_goals_for_5', np.nan))
    finished['home_def_5'] = finished['date_team_home'].map(lambda x: stats_map.get(x, {}).get('avg_goals_against_5', np.nan))
    finished['away_att_5'] = finished['date_team_away'].map(lambda x: stats_map.get(x, {}).get('avg_goals_for_5', np.nan))
    finished['away_def_5'] = finished['date_team_away'].map(lambda x: stats_map.get(x, {}).get('avg_goals_against_5', np.nan))
    
    finished.drop(columns=['date_team_home', 'date_team_away'], inplace=True)
    return finished.dropna(subset=['home_att_5', 'away_att_5'])

# --- 4. MODÈLES & STRATÉGIE ---
def calculate_poisson_probs(avg_home_goals, avg_away_goals):
    lamb_home = max(0.1, avg_home_goals)
    lamb_away = max(0.1, avg_away_goals)
    prob_home, prob_draw, prob_away = 0, 0, 0
    
    for h in range(6):
        for a in range(6):
            p = poisson.pmf(h, lamb_home) * poisson.pmf(a, lamb_away)
            if h > a: prob_home += p
            elif h == a: prob_draw += p
            else: prob_away += p
            
    total = prob_home + prob_draw + prob_away
    return prob_home/total, prob_draw/total, prob_away/total

def calculate_elo_probs(elo_home, elo_away):
    diff = elo_home - elo_away + 100
    return 1 / (1 + 10 ** (-diff / 400))

class StrategyEngine:
    def __init__(self, bankroll=1000, kelly_fraction=0.25, min_value=0.05):
        self.bankroll = bankroll
        self.kelly_fraction = kelly_fraction
        self.min_value = min_value

    def get_real_odds(self, fixture_id):
        # SIMULATION. En prod, remplacer par un appel API /odds
        return {
            "home": round(np.random.uniform(1.5, 3.0), 2),
            "draw": round(np.random.uniform(3.0, 4.0), 2),
            "away": round(np.random.uniform(2.5, 5.0), 2)
        }

    def calculate_kelly_stake(self, probability, odds):
        if probability <= 0 or odds <= 1: return 0
        b = odds - 1
        q = 1 - probability
        f = (b * probability - q) / b
        return min(max(0, f * self.kelly_fraction), 0.05)

    def analyze_matches(self, df, league_name):
        opportunities = []
        for idx, row in df.iterrows():
            # Prédictions
            exp_goals_h = (row['home_att_5'] + row['away_def_5']) / 2
            exp_goals_a = (row['away_att_5'] + row['home_def_5']) / 2
            p_h, p_d, p_a = calculate_poisson_probs(exp_goals_h, exp_goals_a)
            
            # Stratégie
            odds = self.get_real_odds(idx)
            edge = p_h - (1 / odds['home'])
            
            if edge > self.min_value:
                stake_pct = self.calculate_kelly_stake(p_h, odds['home'])
                amount = round(self.bankroll * stake_pct, 2)
                if amount > 0:
                    opportunities.append({
                        "date": str(row['fixture.date']),
                        "league": league_name,
                        "match": f"{row['teams.home.name']} vs {row['teams.away.name']}",
                        "bet_type": "Home Win",
                        "model_prob": round(p_h * 100, 1),
                        "bookmaker_odds": odds['home'],
                        "value_edge": round(edge * 100, 1),
                        "kelly_stake_pct": round(stake_pct * 100, 2),
                        "suggested_wager": amount
                    })
        return opportunities

# --- 5. EXÉCUTION PRINCIPALE ---
if __name__ == "__main__":
    print("--- Démarrage de l'Algo ---")
    connector = FootballDataConnector()
    strategy = StrategyEngine()
    all_bets = []

    for name, id_league in LEAGUES.items():
        print(f"Traitement : {name}")
        # 1. Get Data
        df = connector.get_season_fixtures(id_league, CURRENT_SEASON)
        if df.empty: continue
        
        # 2. Features
        df_processed = process_features(df)
        if df_processed.empty: continue
        
        # 3. Strategy
        bets = strategy.analyze_matches(df_processed, name)
        all_bets.extend(bets)
        print(f"> {len(bets)} opportunités trouvées.")

    # 4. Export JSON
    final_output = {
        "generated_at": datetime.now().isoformat(),
        "strategy": "Poisson + Elo / Kelly 0.25",
        "total_opportunities": len(all_bets),
        "bets": all_bets
    }

    with open('predictions_anj.json', 'w') as f:
        json.dump(final_output, f, indent=2)
    
    print("\nSUCCÈS : Fichier predictions_anj.json généré.")

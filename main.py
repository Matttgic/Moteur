import os
import json
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.stats import poisson

# --- 1. CONFIGURATION ---
API_KEY = os.environ.get("API_KEY")
if not API_KEY:
    # Fallback pour test local (à retirer en prod)
    # API_KEY = "TA_CLE_ICI"
    pass

BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {'x-apisports-key': API_KEY}

# Configuration ANJ
BOOKMAKER_ID = 16  # 16 = Betclic (Souvent dispo et cotes FR)
# Autres options : 8 (Unibet), 52 (Winamax - parfois restreint via API)

LEAGUES = {
    "Premier League": 39,
    "Ligue 1": 61
}
CURRENT_SEASON = 2025 # Assure-toi que c'est la saison active

# --- 2. ETL (EXTRACTION) ---
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
            print(f"Erreur API {response.status_code}: {response.text}")
            return []
        except Exception as e:
            print(f"Exception API : {e}")
            return []

    def get_season_fixtures(self, league_id, season):
        # On récupère TOUTE la saison (passés et futurs)
        filename = f"data_cache/fixtures_{league_id}_{season}.json"
        
        # Cache de 12h seulement pour avoir les mises à jour récentes
        if os.path.exists(filename):
            if (time.time() - os.path.getmtime(filename)) < 43200: 
                print(f"Chargement cache: Ligue {league_id}")
                with open(filename, 'r') as f:
                    return pd.DataFrame(json.load(f))

        print(f"Appel API (Fixtures): Ligue {league_id}")
        data = self._get_from_api("/fixtures", {"league": league_id, "season": season})
        if data:
            df = pd.json_normalize(data)
            df.to_json(filename, orient='records', indent=4)
            return df
        return pd.DataFrame()

    def get_real_odds(self, fixture_id):
        """Récupère les cotes réelles pour un match spécifique."""
        time.sleep(0.2) # Petite pause pour respecter le rate limit (max 10 req/sec)
        params = {"fixture": fixture_id, "bookmaker": BOOKMAKER_ID}
        data = self._get_from_api("/odds", params)
        
        if data and len(data) > 0:
            # On cherche les cotes "Match Winner" (id=1 chez API-Football)
            for market in data[0]['bookmakers'][0]['bets']:
                if market['id'] == 1: # 1 = Winner
                    odds = {v['value']: v['odd'] for v in market['values']}
                    return {
                        "home": odds.get("Home"),
                        "draw": odds.get("Draw"),
                        "away": odds.get("Away")
                    }
        return None

# --- 3. FEATURES (ELO & STATS) ---
class EloTracker:
    def __init__(self):
        self.ratings = {}
        self.default_rating = 1500

    def get_rating(self, team):
        return self.ratings.get(team, self.default_rating)

    def update_ratings(self, home, away, goal_diff):
        k_factor = 20
        # Formule Elo simplifiée
        r_h = self.get_rating(home)
        r_a = self.get_rating(away)
        exp_h = 1 / (1 + 10 ** ((r_a - r_h - 100) / 400)) # +100 avantage domicile
        
        actual = 1 if goal_diff > 0 else (0.5 if goal_diff == 0 else 0)
        delta = k_factor * (actual - exp_h)
        
        self.ratings[home] = r_h + delta
        self.ratings[away] = r_a - delta

def process_and_split(df):
    """Sépare l'historique (pour apprendre) du futur (pour parier)."""
    if df.empty: return pd.DataFrame(), pd.DataFrame()
    
    # Nettoyage
    df['fixture.date'] = pd.to_datetime(df['fixture.date'])
    df = df.sort_values('fixture.date')

    # 1. On sépare Passé (FT) et Futur (NS = Not Started)
    finished = df[df['fixture.status.short'] == 'FT'].copy()
    upcoming = df[df['fixture.status.short'].isin(['NS', 'TBD'])].copy()
    
    # 2. Calcul Elo & Stats sur le Passé
    elo = EloTracker()
    stats_memory = {} # Pour stocker la forme récente

    # Format long pour stats glissantes
    history_records = []
    
    for _, row in finished.iterrows():
        h, a = row['teams.home.name'], row['teams.away.name']
        gh, ga = row['goals.home'], row['goals.away']
        
        # Mise à jour Elo
        elo.update_ratings(h, a, gh - ga)
        
        # Enregistrement pour Stats
        history_records.append({'date': row['fixture.date'], 'team': h, 'gf': gh, 'ga': ga})
        history_records.append({'date': row['fixture.date'], 'team': a, 'gf': ga, 'ga': gh})

    # Calcul des Moyennes Glissantes (Derniers 5 matchs)
    df_hist = pd.DataFrame(history_records).sort_values('date')
    if not df_hist.empty:
        df_hist['avg_gf'] = df_hist.groupby('team')['gf'].transform(lambda x: x.rolling(5).mean())
        df_hist['avg_ga'] = df_hist.groupby('team')['ga'].transform(lambda x: x.rolling(5).mean())
        
        # On garde la DERNIÈRE stat connue pour chaque équipe
        last_stats = df_hist.groupby('team').last()[['avg_gf', 'avg_ga']].to_dict('index')
    else:
        last_stats = {}

    # 3. Préparation des matchs FUTURS
    # On leur injecte l'Elo ACTUEL et les stats ACTUELLES
    upcoming['elo_home'] = upcoming['teams.home.name'].apply(lambda x: elo.get_rating(x))
    upcoming['elo_away'] = upcoming['teams.away.name'].apply(lambda x: elo.get_rating(x))
    
    # Injection des stats (avec valeurs par défaut si début de saison)
    def get_stat(team, type_stat):
        return last_stats.get(team, {}).get(type_stat, 1.2) # 1.2 but moyen par défaut

    upcoming['home_att'] = upcoming['teams.home.name'].apply(lambda x: get_stat(x, 'avg_gf'))
    upcoming['away_def'] = upcoming['teams.away.name'].apply(lambda x: get_stat(x, 'avg_ga'))
    upcoming['away_att'] = upcoming['teams.away.name'].apply(lambda x: get_stat(x, 'avg_gf'))
    upcoming['home_def'] = upcoming['teams.home.name'].apply(lambda x: get_stat(x, 'avg_ga'))

    return upcoming

# --- 4. STRATÉGIE (VALUE BET) ---
class StrategyEngine:
    def __init__(self, connector, bankroll=1000):
        self.connector = connector
        self.bankroll = bankroll

    def calculate_kelly(self, proba, odds):
        if not odds or proba == 0: return 0
        b = odds - 1
        q = 1 - proba
        f = (b * proba - q) / b
        return max(0, f * 0.25) # Kelly Quart (Sécurité)

    def analyze(self, df, league_name):
        bets = []
        # On ne prend que les matchs dans les 7 prochains jours pour éviter les cotes vides
        now = pd.Timestamp.now(tz='UTC')
        mask = (df['fixture.date'] > now) & (df['fixture.date'] < now + pd.Timedelta(days=7))
        target_matches = df[mask]

        print(f"Analyse de {len(target_matches)} matchs à venir ({league_name})...")

        for idx, row in target_matches.iterrows():
            # 1. Prédiction Poisson
            lamb_h = (row['home_att'] + row['away_def']) / 2
            lamb_a = (row['away_att'] + row['home_def']) / 2
            
            prob_home = 0
            for h in range(6):
                for a in range(6):
                    if h > a: prob_home += poisson.pmf(h, lamb_h) * poisson.pmf(a, lamb_a)
            
            # Normalisation (approx)
            prob_home = min(0.99, max(0.01, prob_home / 0.95)) # Ajustement simple

            # 2. VRAIES COTES
            odds = self.connector.get_real_odds(row['fixture.id'])
            
            if odds and odds['home']:
                bookie_odd = float(odds['home'])
                
                # 3. Value ?
                min_edge = 0.05 # 5% de marge
                edge = prob_home - (1 / bookie_odd)
                
                if edge > min_edge:
                    stake = self.calculate_kelly(prob_home, bookie_odd)
                    amount = round(self.bankroll * stake, 2)
                    
                    if amount > 5: # Mise mini 5€
                        bets.append({
                            "date": str(row['fixture.date']),
                            "league": league_name,
                            "match": f"{row['teams.home.name']} vs {row['teams.away.name']}",
                            "bet_type": "Home Win",
                            "model_prob": round(prob_home * 100, 1),
                            "bookmaker_odds": bookie_odd,
                            "value_edge": round(edge * 100, 1),
                            "suggested_wager": amount
                        })
        return bets

# --- 5. MAIN ---
if __name__ == "__main__":
    print("--- Démarrage Algo (Mode Production) ---")
    connector = FootballDataConnector()
    strategy = StrategyEngine(connector)
    all_bets = []

    for name, lid in LEAGUES.items():
        df_raw = connector.get_season_fixtures(lid, CURRENT_SEASON)
        # On ne traite que les matchs futurs
        df_upcoming = process_and_split(df_raw)
        
        if not df_upcoming.empty:
            league_bets = strategy.analyze(df_upcoming, name)
            all_bets.extend(league_bets)
            print(f"> {name}: {len(league_bets)} Value Bets détectés.")

    # Export
    output = {
        "generated_at": datetime.now().isoformat(),
        "bets": sorted(all_bets, key=lambda x: x['value_edge'], reverse=True)
    }
    
    with open('predictions_anj.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\n✅ Terminé. Fichier JSON mis à jour avec les cotes réelles.")

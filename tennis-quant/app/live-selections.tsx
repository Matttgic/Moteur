"use client";

import { useCallback, useEffect, useState } from "react";

type Tour = "ATP" | "WTA";

type BetRow = {
  matchId: number | null;
  tournament: string | null;
  surface: string | null;
  scheduledTime: string | null;
  playerA: { name: string; ranking: number };
  playerB: { name: string; ranking: number };
  model: {
    mode: "full_logit" | "rank_only_fallback";
    probabilityA: number;
    probabilityB: number;
    fairOddsA: number;
    fairOddsB: number;
    quality: number;
    reason: string;
  };
  market: {
    bookmaker: string;
    oddsA: number;
    oddsB: number;
    sourceCount?: number;
    medianSelectedOdds?: number | null;
    priceSpreadRatio?: number | null;
  };
  decision: {
    side: "A" | "B";
    player: string;
    odds: number;
    modelProbability: number;
    marketProbability: number;
    edge: number;
    ev: number;
    fairOdds: number;
    tier: "PREMIUM" | "VALUE" | "LEAN" | "NO_BET";
    stakeUnits: number;
    bet: boolean;
    guard?: {
      blocked: boolean;
      reasons: string[];
    };
  };
};

type SelectionResponse = {
  generatedAt: string;
  tour: Tour;
  status: "BET_OPPORTUNITIES_FOUND" | "NO_BET_TODAY";
  modelPolicy: {
    fullModelEnabledForLive: boolean;
    fullModelMatches: number;
    fallbackMatches: number;
    historySync: {
      fresh: boolean;
      windowTo: string | null;
      reason: string;
    };
  };
  bets: BetRow[];
  sourceSummary: {
    liveAccepted: number;
    oddsFixtures: number;
    analyzed: number;
    rejected: number;
  };
};

const pct = (value: number) => `${(value * 100).toFixed(1)}%`;
const signedPct = (value: number) =>
  `${value >= 0 ? "+" : ""}${(value * 100).toFixed(1)}%`;

function matchLabel(row: BetRow) {
  const opponent =
    row.decision.side === "A" ? row.playerB.name : row.playerA.name;
  return `${row.decision.player} vs ${opponent}`;
}

function formatTime(value: string | null) {
  if (!value) return "Horaire inconnu";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;

  return date.toLocaleString("fr-FR", {
    timeZone: "Europe/Paris",
    weekday: "short",
    day: "2-digit",
    month: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export default function LiveSelections() {
  const [tour, setTour] = useState<Tour>("ATP");
  const [data, setData] = useState<SelectionResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async (selectedTour: Tour) => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch(
        `/api/selections/today?tour=${selectedTour.toLowerCase()}`,
        { cache: "no-store" },
      );

      const payload = await response.json();

      if (!response.ok) {
        throw new Error(payload?.message ?? payload?.error ?? "Erreur moteur");
      }

      setData(payload);
    } catch (err) {
      setData(null);
      setError(err instanceof Error ? err.message : "Erreur inconnue");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load(tour);
  }, [load, tour]);

  return (
    <section className="liveSection" aria-label="Paris du jour">
      <div className="liveHeader">
        <div>
          <p className="eyebrow">PARIS DU JOUR · DONNÉES LIVE</p>
          <h2>Sélections du moteur</h2>
          <p>
            Le moteur peut répondre <strong>NO BET</strong>. Les paris à miser
            exigent désormais le <strong>FULL ML</strong>. Le fallback ranking
            reste visible dans l'analyse mais n'est jamais proposé comme pari.
          </p>
        </div>

        <div className="liveControls">
          <div className="tourTabs" role="tablist" aria-label="Circuit">
            {(["ATP", "WTA"] as const).map((item) => (
              <button
                key={item}
                type="button"
                className={tour === item ? "tourTab active" : "tourTab"}
                onClick={() => setTour(item)}
                aria-pressed={tour === item}
              >
                {item}
              </button>
            ))}
          </div>
          <button
            type="button"
            className="refreshButton"
            onClick={() => void load(tour)}
            disabled={loading}
          >
            {loading ? "Chargement…" : "Actualiser"}
          </button>
        </div>
      </div>

      {error ? (
        <div className="liveState liveError">
          <strong>Impossible de charger les sélections.</strong>
          <span>{error}</span>
        </div>
      ) : loading && !data ? (
        <div className="liveState">
          <strong>Analyse {tour} en cours…</strong>
          <span>Matchs, états joueurs et cotes françaises sont croisés.</span>
        </div>
      ) : data ? (
        <>
          <div className="liveMeta">
            <span>
              Historique :{" "}
              <strong>{data.modelPolicy.historySync.fresh ? "FRAIS" : "À RAFRAÎCHIR"}</strong>
            </span>
            <span>
              Full ML : <strong>{data.modelPolicy.fullModelMatches}</strong>
            </span>
            <span>
              Fallback : <strong>{data.modelPolicy.fallbackMatches}</strong>
            </span>
            <span>
              Matchs analysés : <strong>{data.sourceSummary.analyzed}</strong>
            </span>
            <span>
              Cotes trouvées : <strong>{data.sourceSummary.oddsFixtures}</strong>
            </span>
          </div>

          {data.bets.length ? (
            <div className="cards liveCards">
              {data.bets.map((row) => (
                <article
                  className="matchCard liveBetCard"
                  key={`${row.matchId}-${row.decision.player}`}
                >
                  <div className="matchTop">
                    <div>
                      <span className="pill">{tour}</span>
                      <span className="muted">{row.surface ?? "Surface inconnue"}</span>
                    </div>
                    <span className={`tier tier-${row.decision.tier.toLowerCase()}`}>
                      {row.decision.tier}
                    </span>
                  </div>

                  <p className="liveTournament">
                    {row.tournament ?? "Tournoi"} · {formatTime(row.scheduledTime)}
                  </p>

                  <h3>{matchLabel(row)}</h3>

                  <div className="pickHero">
                    <span>PARI SÉLECTIONNÉ</span>
                    <strong>{row.decision.player}</strong>
                    <div>
                      <b>{row.market.bookmaker}</b>
                      <b>Cote {row.decision.odds.toFixed(2)}</b>
                    </div>
                  </div>

                  <div className="decision liveDecision">
                    <div>
                      <span>Proba modèle</span>
                      <strong>{pct(row.decision.modelProbability)}</strong>
                    </div>
                    <div>
                      <span>Cote juste</span>
                      <strong>{row.decision.fairOdds.toFixed(2)}</strong>
                    </div>
                    <div>
                      <span>Edge</span>
                      <strong>{signedPct(row.decision.edge)}</strong>
                    </div>
                    <div>
                      <span>EV</span>
                      <strong>{signedPct(row.decision.ev)}</strong>
                    </div>
                  </div>

                  <div className="liveTags">
                    <span>
                      {row.model.mode === "full_logit"
                        ? "FULL ML"
                        : "RANK FALLBACK"}
                    </span>
                    <span>Mise {row.decision.stakeUnits.toFixed(2)}u</span>
                    {typeof row.market.sourceCount === "number" ? (
                      <span>{row.market.sourceCount} book(s) FR</span>
                    ) : null}
                  </div>
                </article>
              ))}
            </div>
          ) : (
            <div className="liveState noBetState">
              <strong>NO BET {tour} pour l'instant</strong>
              <span>
                Aucun match ne passe tous les critères. Le moteur ne force pas
                de pari lorsqu'il ne trouve pas d'edge suffisamment propre.
              </span>
            </div>
          )}

          <p className="liveUpdated">
            Calcul actualisé : {formatTime(data.generatedAt)} · Les probabilités
            restent incertaines et ne garantissent aucun gain.
          </p>
        </>
      ) : null}
    </section>
  );
}

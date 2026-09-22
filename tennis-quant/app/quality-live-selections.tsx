"use client";

import { useCallback, useEffect, useState } from "react";

type Tour = "ATP" | "WTA";

type Quality = {
  score: number;
  grade: "A+" | "A" | "B" | "C" | "D";
  actionable: boolean;
  threshold: number;
  components: {
    model: number;
    marketDepth: number;
    sharpConfirmation: number;
    edge: number;
    ev: number;
    priceStability: number;
    oddsRisk: number;
  };
  signals: string[];
};

type BetRow = {
  matchId: number | null;
  tournament: string | null;
  surface: string | null;
  scheduledTime: string | null;
  playerA: { name: string; ranking: number };
  playerB: { name: string; ranking: number };
  model: {
    mode: "full_logit" | "rank_only_fallback";
    quality: number;
  };
  market: {
    bookmaker: string;
    sourceCount?: number;
    priceSpreadRatio?: number | null;
  };
  decision: {
    player: string;
    odds: number;
    modelProbability: number;
    edge: number;
    ev: number;
    fairOdds: number;
    tier: "PREMIUM" | "VALUE" | "LEAN" | "NO_BET";
    stakeUnits: number;
    sharpProbability?: number | null;
    quality: Quality;
  };
};

type SelectionResponse = {
  generatedAt: string;
  tour: Tour;
  status: "BET_OPPORTUNITIES_FOUND" | "NO_BET_TODAY" | "DATA_UNAVAILABLE";
  bets: BetRow[];
  sourceSummary: {
    oddsProvider?: string;
    liveModelEligible?: number;
    oddsFixtures: number;
    matchedOddsFixtures?: number;
    analyzed: number;
  };
  qualityPolicy: {
    actionableThreshold: number;
    qualityRejected: number;
    meaning: string;
  };
  snapshot?: {
    ageMinutes: number;
    stale: boolean;
  };
};

const pct = (value: number) => `${(value * 100).toFixed(1)}%`;

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

function signalLabel(signal: string) {
  const labels: Record<string, string> = {
    strong_model_quality: "modèle solide",
    deep_market_confirmation: "marché confirmé",
    sharp_market_confirms_edge: "Pinnacle confirme",
    no_sharp_reference: "sans référence Pinnacle",
    wide_cross_book_spread: "dispersion des cotes",
    longshot_price: "cote longue",
    extreme_model_market_gap: "écart modèle/marché élevé",
    below_actionable_quality_threshold: "sous le seuil qualité",
  };
  return labels[signal] ?? signal.replaceAll("_", " ");
}

export default function QualityLiveSelections() {
  const [tour, setTour] = useState<Tour>("ATP");
  const [data, setData] = useState<SelectionResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async (selectedTour: Tour) => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(
        `/api/selections/quality-today?tour=${selectedTour.toLowerCase()}`,
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
    <section className="liveSection" aria-label="Sélections qualité">
      <div className="liveHeader">
        <div>
          <p className="eyebrow">BET QUALITY SCORE · 0–100</p>
          <h2>Paris robustes du moteur</h2>
          <p>
            Le score mesure la solidité de l&apos;exécution, pas la probabilité de
            gagner. Un VALUE/PREMIUM doit atteindre <strong>70/100</strong> pour
            être proposé.
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
            {loading ? "Chargement…" : "Rafraîchir l’affichage"}
          </button>
        </div>
      </div>

      {error ? (
        <div className="liveState liveError">
          <strong>Impossible de charger les sélections qualité.</strong>
          <span>{error}</span>
        </div>
      ) : loading && !data ? (
        <div className="liveState">
          <strong>Calcul qualité {tour}…</strong>
          <span>Le snapshot est relu sans nouvel appel fournisseur.</span>
        </div>
      ) : data ? (
        <>
          <div className="liveMeta">
            <span>
              Seuil : <strong>{data.qualityPolicy.actionableThreshold}/100</strong>
            </span>
            <span>
              Actionnables : <strong>{data.bets.length}</strong>
            </span>
            <span>
              Filtrés qualité : <strong>{data.qualityPolicy.qualityRejected}</strong>
            </span>
            <span>
              Matchs analysés : <strong>{data.sourceSummary.analyzed}</strong>
            </span>
            <span>
              Cotes appariées :{" "}
              <strong>
                {data.sourceSummary.matchedOddsFixtures ??
                  data.sourceSummary.oddsFixtures}
              </strong>
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
                      <span className="muted">
                        {row.surface ?? "Surface inconnue"}
                      </span>
                    </div>
                    <span className={`tier tier-${row.decision.tier.toLowerCase()}`}>
                      {row.decision.tier}
                    </span>
                  </div>

                  <p className="liveTournament">
                    {row.tournament ?? "Tournoi"} · {formatTime(row.scheduledTime)}
                  </p>
                  <h3>
                    {row.decision.player} @ {row.decision.odds.toFixed(2)}
                  </h3>

                  <div className="pickHero">
                    <span>BET QUALITY SCORE</span>
                    <strong>
                      {row.decision.quality.score}/100 · {row.decision.quality.grade}
                    </strong>
                    <div>
                      <b>{row.market.bookmaker}</b>
                      <b>{row.market.sourceCount ?? 0} book(s) FR</b>
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
                      <strong>{pct(row.decision.edge)}</strong>
                    </div>
                    <div>
                      <span>EV</span>
                      <strong>{pct(row.decision.ev)}</strong>
                    </div>
                  </div>

                  <div className="liveTags">
                    {row.decision.quality.signals.slice(0, 4).map((signal) => (
                      <span key={signal}>{signalLabel(signal)}</span>
                    ))}
                  </div>
                </article>
              ))}
            </div>
          ) : data.status === "DATA_UNAVAILABLE" ? (
            <div className="liveState liveError">
              <strong>Cotes {tour} temporairement indisponibles</strong>
              <span>Le filtre qualité n&apos;invente aucune sélection sans marché.</span>
            </div>
          ) : (
            <div className="liveState noBetState">
              <strong>NO BET {tour}</strong>
              <span>
                Aucun VALUE/PREMIUM ne dépasse actuellement le seuil qualité de
                {" "}{data.qualityPolicy.actionableThreshold}/100.
              </span>
            </div>
          )}

          <p className="liveUpdated">
            Calcul serveur : {formatTime(data.generatedAt)}
            {data.snapshot
              ? ` · snapshot ${Math.round(data.snapshot.ageMinutes)} min${data.snapshot.stale ? " · À RAFRAÎCHIR" : ""}`
              : ""}
            {" · "}Ce score n&apos;est ni une probabilité de victoire ni une garantie de gain.
          </p>
        </>
      ) : null}
    </section>
  );
}

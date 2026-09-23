"use client";

import { useCallback, useEffect, useState } from "react";
import { calculateBetQuality } from "@/lib/bet-quality";

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
    sharpProbability?: number | null;
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
  status: "BET_OPPORTUNITIES_FOUND" | "NO_BET_TODAY" | "DATA_UNAVAILABLE";
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
    oddsProvider?: string;
    oddsDiscoveryMode?: string | null;
    liveAccepted: number;
    liveModelEligible?: number;
    providerDiscoveredFixtures?: number;
    oddsFixtures: number;
    matchedOddsFixtures?: number;
    unmatchedOddsFixtures?: number;
    oddsCoverageRatio?: number | null;
    analyzed: number;
    rejected: number;
  };
  snapshot?: {
    mode: "read_only" | "fresh_private_refresh";
    generatedAt: string;
    updatedAt: string;
    ageMinutes: number;
    stale: boolean;
  };
};

const pct = (value: number) => `${(value * 100).toFixed(1)}%`;
const signedPct = (value: number) =>
  `${value >= 0 ? "+" : ""}${(value * 100).toFixed(1)}%`;

function qualityFor(row: BetRow) {
  return calculateBetQuality({
    modelQuality: row.model.quality,
    sourceCount: row.market.sourceCount ?? 0,
    sharpProbability: row.decision.sharpProbability ?? null,
    modelProbability: row.decision.modelProbability,
    edge: row.decision.edge,
    ev: row.decision.ev,
    odds: row.decision.odds,
    priceSpreadRatio: row.market.priceSpreadRatio ?? null,
  });
}

function qualityLabel(score: number) {
  if (score >= 90) return "EXCELLENT";
  if (score >= 80) return "TRÈS SOLIDE";
  if (score >= 70) return "SOLIDE";
  if (score >= 60) return "À SURVEILLER";
  return "FRAGILE";
}

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
            exigent le <strong>FULL ML</strong>. Le Bet Quality Score 0–100
            mesure la robustesse du signal sans modifier encore les seuils de pari.
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
              Cotes fournisseur : <strong>{data.sourceSummary.oddsFixtures}</strong>
            </span>
            <span>
              Couverture :{" "}
              <strong>
                {data.sourceSummary.matchedOddsFixtures ??
                  data.sourceSummary.oddsFixtures}
                /
                {data.sourceSummary.liveModelEligible ??
                  data.sourceSummary.liveAccepted}
              </strong>
            </span>
            {typeof data.sourceSummary.unmatchedOddsFixtures === "number" ? (
              <span>
                Cotes non appariées :{" "}
                <strong>{data.sourceSummary.unmatchedOddsFixtures}</strong>
              </span>
            ) : null}
          </div>

          {data.bets.length ? (
            <div className="cards liveCards">
              {data.bets.map((row) => {
                const quality = qualityFor(row);

                return (
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
                        <span>Quality Score</span>
                        <strong>{quality.score}/100 · {quality.grade}</strong>
                      </div>
                      <div>
                        <span>Robustesse</span>
                        <strong>{qualityLabel(quality.score)}</strong>
                      </div>
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
                      <span>
                        Qualité {quality.actionable ? "≥70" : "<70"}
                      </span>
                      {typeof row.market.sourceCount === "number" ? (
                        <span>{row.market.sourceCount} book(s) FR</span>
                      ) : null}
                      {row.decision.sharpProbability != null ? (
                        <span>Pinnacle confirmé</span>
                      ) : (
                        <span>Sans référence sharp</span>
                      )}
                    </div>
                  </article>
                );
              })}
            </div>
          ) : data.status === "DATA_UNAVAILABLE" ? (
            <div className="liveState liveError">
              <strong>Cotes {tour} temporairement indisponibles</strong>
              <span>
                Les matchs sont bien détectés, mais aucune grille de cotes
                exploitable n'est disponible dans le dernier snapshot. Ce n'est
                pas un NO BET du modèle.
              </span>
            </div>
          ) : (
            <div className="liveState noBetState">
              <strong>NO BET {tour} pour l'instant</strong>
              <span>
                Les données sont disponibles, mais aucun match ne passe tous les
                critères. Le moteur ne force pas de pari sans edge suffisamment
                propre.
              </span>
            </div>
          )}

          {(data.sourceSummary.liveModelEligible ??
            data.sourceSummary.liveAccepted) >
            (data.sourceSummary.matchedOddsFixtures ??
              data.sourceSummary.oddsFixtures) ? (
            <p className="liveUpdated">
              {data.sourceSummary.unmatchedOddsFixtures === 0
                ? `${(data.sourceSummary.liveModelEligible ?? data.sourceSummary.liveAccepted) - (data.sourceSummary.matchedOddsFixtures ?? data.sourceSummary.oddsFixtures)} match(s) n'ont pas encore de marché exploitable chez les bookmakers suivis. Ils seront recontrôlés automatiquement au prochain snapshot.`
                : `${data.sourceSummary.unmatchedOddsFixtures ?? 0} cote(s) fournisseur restent à rapprocher d'un match live ; le moteur les exclut tant que l'appariement n'est pas sûr.`}
            </p>
          ) : null}

          <p className="liveUpdated">
            Bet Quality : A+ ≥90 · A ≥80 · B ≥70 · C ≥60 · D &lt;60. Le score est en validation et ne garantit pas la réussite d'un pari.
          </p>

          <p className="liveUpdated">
            Calcul serveur : {formatTime(data.generatedAt)}
            {data.snapshot
              ? ` · snapshot ${Math.round(data.snapshot.ageMinutes)} min${data.snapshot.stale ? " · À RAFRAÎCHIR" : ""}`
              : ""}
            {" · "}Les probabilités restent incertaines et ne garantissent aucun gain.
          </p>
        </>
      ) : null}
    </section>
  );
}

"use client";

import { useEffect, useMemo, useState } from "react";

type Aggregate = {
  total: number;
  settled: number;
  pending: number;
  wins: number;
  losses: number;
  stakeUnits: number;
  profitUnits: number;
  roi: number | null;
  hitRate: number | null;
  avgOdds: number | null;
  avgEdge: number | null;
  avgEv: number | null;
  avgQualityScore: number | null;
  maxDrawdownUnits: number;
  avgClv: number | null;
  clvSamples: number;
};

type HistoryRow = {
  id: string;
  tour: string;
  scheduledAt: string | null;
  tournament: string | null;
  playerA: string;
  playerB: string;
  selectedPlayer: string;
  selectedSide: "A" | "B";
  bookmaker: string;
  odds: number;
  modelProbability: number | null;
  marketProbability: number | null;
  edge: number | null;
  ev: number | null;
  fairOdds: number | null;
  tier: string;
  stakeUnits: number;
  modelMode: string;
  modelQuality: number | null;
  qualityScore: number | null;
  qualityGrade: string | null;
  qualityComponents?: Record<string, number> | null;
  qualitySignals?: string[] | null;
  closingOdds: number | null;
  clv: number | null;
  result: "WIN" | "LOSS" | null;
  profitUnits: number | null;
  settledAt: string | null;
  createdAt: string;
};

type EconomicSummary = {
  generatedAt: string;
  sampleStatus: "COLLECTING" | "EARLY_SAMPLE" | "VALIDATING" | "ROBUST_SAMPLE";
  overall: Aggregate;
  byTour: Array<{ name: string } & Aggregate>;
  byModel: Array<{ name: string } & Aggregate>;
  byTier: Array<{ name: string } & Aggregate>;
  byQuality?: Array<{ name: string } & Aggregate>;
  last10: Array<{
    result: "WIN" | "LOSS";
    tour: string;
    player: string;
    bookmaker: string;
    odds: number;
    stakeUnits: number;
    profitUnits: number;
    modelMode: string;
    tier: string;
    qualityScore?: number | null;
    qualityGrade?: string | null;
    scheduledAt: string;
  }>;
  history?: HistoryRow[];
};

type HistoryFilter = "ALL" | "PENDING" | "WIN" | "LOSS";

const pct = (value: number | null) =>
  value == null ? "—" : `${(Number(value) * 100).toFixed(1)}%`;

const units = (value: number) =>
  `${Number(value) >= 0 ? "+" : ""}${Number(value).toFixed(2)}u`;

const decimal = (value: number | null, digits = 2) =>
  value == null ? "—" : Number(value).toFixed(digits);

const quality = (score: number | null, grade: string | null) =>
  score == null ? "—" : `${Math.round(Number(score))}/100${grade ? ` · ${grade}` : ""}`;

const formatDate = (value: string | null) => {
  if (!value) return "Date inconnue";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "Date inconnue";

  return new Intl.DateTimeFormat("fr-FR", {
    timeZone: "Europe/Paris",
    day: "2-digit",
    month: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
};

function statusText(status: EconomicSummary["sampleStatus"]) {
  if (status === "ROBUST_SAMPLE") return "ÉCHANTILLON ROBUSTE";
  if (status === "VALIDATING") return "VALIDATION EN COURS";
  if (status === "EARLY_SAMPLE") return "ÉCHANTILLON PRÉCOCE";
  return "COLLECTE EN COURS";
}

function historyStatus(row: HistoryRow) {
  if (row.result === "WIN") {
    return { label: "GAGNÉ", className: "historyStatus historyWin" };
  }

  if (row.result === "LOSS") {
    return { label: "PERDU", className: "historyStatus historyLoss" };
  }

  return { label: "EN ATTENTE", className: "historyStatus historyPending" };
}

export default function EconomicValidation() {
  const [data, setData] = useState<EconomicSummary | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [historyFilter, setHistoryFilter] = useState<HistoryFilter>("ALL");

  useEffect(() => {
    let active = true;

    fetch("/api/economics/summary", { cache: "no-store" })
      .then(async (response) => {
        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload?.error ?? "Economic validation unavailable");
        }
        if (active) setData(payload.summary);
      })
      .catch((err) => {
        if (active) {
          setError(err instanceof Error ? err.message : "Erreur inconnue");
        }
      });

    return () => {
      active = false;
    };
  }, []);

  const history = data?.history ?? [];

  const filteredHistory = useMemo(() => {
    if (historyFilter === "ALL") return history;
    if (historyFilter === "PENDING") {
      return history.filter((row) => row.result == null);
    }
    return history.filter((row) => row.result === historyFilter);
  }, [history, historyFilter]);

  const historyCounts = useMemo(
    () => ({
      ALL: history.length,
      PENDING: history.filter((row) => row.result == null).length,
      WIN: history.filter((row) => row.result === "WIN").length,
      LOSS: history.filter((row) => row.result === "LOSS").length,
    }),
    [history],
  );

  const pendingStakeUnits = useMemo(
    () =>
      history
        .filter((row) => row.result == null)
        .reduce((sum, row) => sum + Number(row.stakeUnits ?? 0), 0),
    [history],
  );

  if (error) {
    return (
      <article className="economicGate">
        <div>
          <p className="eyebrow">ECONOMIC VALIDATION</p>
          <h3>Collecte temporairement indisponible</h3>
        </div>
        <p>
          Le moteur de paris reste actif. Seul le tableau économique n'a pas pu
          charger ses statistiques.
        </p>
        <span>{error}</span>
      </article>
    );
  }

  if (!data) {
    return (
      <article className="economicGate">
        <div>
          <p className="eyebrow">ECONOMIC VALIDATION</p>
          <h3>Chargement du journal réel…</h3>
        </div>
        <p>
          Les picks proposés sont enregistrés puis comparés aux résultats
          officiels pour mesurer la performance économique réelle.
        </p>
        <span>ROI · Drawdown · CLV · Bet Quality</span>
      </article>
    );
  }

  const stats = data.overall;

  return (
    <section className="economicPanel" aria-label="Validation économique" data-history-version="3">
      <div className="economicHeader">
        <div>
          <p className="eyebrow">ECONOMIC VALIDATION · LIVE TRACKING</p>
          <h2>Performance des picks réellement proposés</h2>
          <p>
            Pas de ROI théorique : uniquement les sélections réellement sorties
            par le moteur, avec leur cote et leur Bet Quality Score au moment de la décision.
          </p>
        </div>
        <span className="economicStatus">{statusText(data.sampleStatus)}</span>
      </div>

      <div className="economicStats">
        <article>
          <span>Paris enregistrés</span>
          <strong>{stats.total}</strong>
          <small>{stats.pending} en attente</small>
        </article>
        <article>
          <span>Bilan réglé</span>
          <strong>{stats.wins}-{stats.losses}</strong>
          <small>{stats.settled} paris réglés</small>
        </article>
        <article>
          <span>Profit net</span>
          <strong className={stats.profitUnits >= 0 ? "positive" : "negative"}>
            {units(stats.profitUnits)}
          </strong>
          <small>
            {stats.stakeUnits.toFixed(2)}u réglées · {pendingStakeUnits.toFixed(2)}u en attente
          </small>
        </article>
        <article>
          <span>ROI réel</span>
          <strong className={(stats.roi ?? 0) >= 0 ? "positive" : "negative"}>
            {pct(stats.roi)}
          </strong>
          <small>profit / mises</small>
        </article>
        <article>
          <span>Hit rate</span>
          <strong>{pct(stats.hitRate)}</strong>
          <small>paris gagnants</small>
        </article>
        <article>
          <span>Bet Quality moyen</span>
          <strong>{stats.avgQualityScore == null ? "—" : `${Math.round(stats.avgQualityScore)}/100`}</strong>
          <small>seuil ≥70 requis pour être proposé</small>
        </article>
        <article>
          <span>Cote moyenne</span>
          <strong>{stats.avgOdds?.toFixed(2) ?? "—"}</strong>
          <small>au moment du pick</small>
        </article>
        <article>
          <span>Max drawdown</span>
          <strong>{stats.maxDrawdownUnits.toFixed(2)}u</strong>
          <small>baisse depuis un pic</small>
        </article>
        <article>
          <span>CLV moyen</span>
          <strong>{pct(stats.avgClv)}</strong>
          <small>{stats.clvSamples} cote(s) de clôture</small>
        </article>
      </div>

      <div className="betHistory">
        <div className="betHistoryHeader">
          <div>
            <p className="eyebrow">HISTORIQUE DES PARIS</p>
            <h3>Tous les picks enregistrés</h3>
            <p>
              Les paris en attente restent visibles jusqu'à leur règlement. Les grades A+/A/B/C/D permettent de comparer leur rentabilité réelle.
            </p>
          </div>
          <span>{history.length} pick{history.length > 1 ? "s" : ""}</span>
        </div>

        <div className="historyFilters" role="tablist" aria-label="Filtrer l'historique">
          {([
            ["ALL", "Tous"],
            ["PENDING", "En attente"],
            ["WIN", "Gagnés"],
            ["LOSS", "Perdus"],
          ] as const).map(([key, label]) => (
            <button
              type="button"
              key={key}
              className={historyFilter === key ? "historyFilter active" : "historyFilter"}
              onClick={() => setHistoryFilter(key)}
            >
              {label} <span>{historyCounts[key]}</span>
            </button>
          ))}
        </div>

        {filteredHistory.length ? (
          <div className="historyGrid">
            {filteredHistory.map((row) => {
              const status = historyStatus(row);
              const profitClass =
                row.profitUnits == null
                  ? ""
                  : Number(row.profitUnits) >= 0
                    ? "positive"
                    : "negative";

              return (
                <article className="historyCard" key={row.id}>
                  <div className="historyTop">
                    <div>
                      <span className="pill">{row.tour}</span>
                      <span className="historyDate">{formatDate(row.scheduledAt)}</span>
                    </div>
                    <span className={status.className}>{status.label}</span>
                  </div>

                  <p className="historyTournament">{row.tournament ?? "Tournoi"}</p>

                  <h4>{row.selectedPlayer} <span>@{Number(row.odds).toFixed(2)}</span></h4>
                  <p className="historyMatch">{row.playerA} vs {row.playerB}</p>

                  <div className="historyMetrics">
                    <div>
                      <span>Book</span>
                      <strong>{row.bookmaker}</strong>
                    </div>
                    <div>
                      <span>Quality</span>
                      <strong>{quality(row.qualityScore, row.qualityGrade)}</strong>
                    </div>
                    <div>
                      <span>Proba modèle</span>
                      <strong>{pct(row.modelProbability)}</strong>
                    </div>
                    <div>
                      <span>Edge</span>
                      <strong>{pct(row.edge)}</strong>
                    </div>
                    <div>
                      <span>EV</span>
                      <strong>{pct(row.ev)}</strong>
                    </div>
                    <div>
                      <span>Mise</span>
                      <strong>{decimal(row.stakeUnits)}u</strong>
                    </div>
                    <div>
                      <span>CLV</span>
                      <strong>{pct(row.clv)}</strong>
                    </div>
                  </div>

                  <div className="historyBottom">
                    <div>
                      <span>{row.tier}</span>
                      <span>{row.modelMode === "full_logit" ? "FULL ML" : row.modelMode}</span>
                      {row.qualityGrade ? <span>QUALITY {row.qualityGrade}</span> : null}
                      {row.closingOdds != null ? (
                        <span>Clôture {Number(row.closingOdds).toFixed(2)}</span>
                      ) : null}
                    </div>

                    <strong className={profitClass}>
                      {row.profitUnits == null
                        ? "Résultat en attente"
                        : units(Number(row.profitUnits))}
                    </strong>
                  </div>
                </article>
              );
            })}
          </div>
        ) : (
          <p className="historyEmpty">
            Aucun pari dans ce filtre pour le moment.
          </p>
        )}
      </div>

      {data.last10.length ? (
        <div className="economicRecent">
          <strong>10 derniers paris réglés</strong>
          <div>
            {data.last10.map((row, index) => (
              <span
                className={row.result === "WIN" ? "recentWin" : "recentLoss"}
                key={`${row.player}-${row.scheduledAt}-${index}`}
              >
                {row.result === "WIN" ? "W" : "L"} · {row.player} @{Number(row.odds).toFixed(2)}
                {row.qualityGrade ? ` · Q${row.qualityGrade}` : ""}
              </span>
            ))}
          </div>
        </div>
      ) : (
        <p className="economicCollecting">
          Les premiers picks sont enregistrés. Les lignes ci-dessus restent
          visibles en attendant leur résultat officiel.
        </p>
      )}
    </section>
  );
}

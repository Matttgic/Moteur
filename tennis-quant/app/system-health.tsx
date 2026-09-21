"use client";

import { useEffect, useState } from "react";

type Run = {
  status: "success" | "failed";
  details: Record<string, unknown>;
  window_to?: string | null;
  finished_at?: string | null;
};

type HealthPayload = {
  health: {
    generatedAt: string;
    history: { ATP: Run | null; WTA: Run | null };
    liveProvider: { ATP: Run | null; WTA: Run | null };
    oddsProvider: { ATP: Run | null; WTA: Run | null };
    pickCapture: { ATP: Run | null; WTA: Run | null };
    settlement: Run | null;
    clvRefresh: Run | null;
    lastPick: {
      created_at: string;
      selected_player_name: string;
      bookmaker: string;
      odds: number;
      result: string | null;
    } | null;
    lastFailure: {
      job_name: string;
      tour: string | null;
      details: Record<string, unknown>;
      finished_at: string;
    } | null;
  };
};

function ageHours(value?: string | null) {
  if (!value) return null;
  const timestamp = Date.parse(value);
  if (!Number.isFinite(timestamp)) return null;
  return Math.max(0, (Date.now() - timestamp) / 3_600_000);
}

function okRun(run: Run | null, maxHours: number, useWindow = false) {
  if (!run || run.status !== "success") return false;
  const age = ageHours(useWindow ? run.window_to : run.finished_at);
  return age !== null && age <= maxHours;
}

function Badge({ ok, label }: { ok: boolean; label: string }) {
  return (
    <span className={ok ? "healthBadge healthOk" : "healthBadge healthWarn"}>
      {ok ? "OK" : "À VÉRIFIER"} · {label}
    </span>
  );
}

export default function SystemHealth() {
  const [data, setData] = useState<HealthPayload["health"] | null>(null);

  useEffect(() => {
    let active = true;

    fetch("/api/health/tennis", { cache: "no-store" })
      .then((response) => response.json())
      .then((payload) => {
        if (active && payload?.health) setData(payload.health);
      })
      .catch(() => {});

    return () => {
      active = false;
    };
  }, []);

  if (!data) return null;

  const historyOk =
    okRun(data.history.ATP, 48, true) &&
    okRun(data.history.WTA, 48, true);
  const liveOk =
    okRun(data.liveProvider.ATP, 18) &&
    okRun(data.liveProvider.WTA, 18);
  const oddsOk =
    okRun(data.oddsProvider.ATP, 18) &&
    okRun(data.oddsProvider.WTA, 18);
  const settlementOk = okRun(data.settlement, 36);
  const clvOk = okRun(data.clvRefresh, 2.5);

  return (
    <section className="healthPanel" aria-label="Santé du moteur">
      <div className="healthHeader">
        <div>
          <p className="eyebrow">SYSTEM HEALTH</p>
          <h2>Le moteur tourne-t-il normalement ?</h2>
        </div>
        <span className="healthOverall">
          {[historyOk, liveOk, oddsOk, settlementOk, clvOk].every(Boolean)
            ? "TOUT EST OK"
            : "SURVEILLANCE"}
        </span>
      </div>

      <div className="healthBadges">
        <Badge ok={historyOk} label="Historique frais" />
        <Badge ok={liveOk} label="Live Tennis" />
        <Badge ok={oddsOk} label="Cotes" />
        <Badge ok={settlementOk} label="Règlement" />
        <Badge ok={clvOk} label="CLV" />
      </div>

      <div className="healthDetails">
        <span>
          Dernier pick :{" "}
          <strong>
            {data.lastPick
              ? `${data.lastPick.selected_player_name} @${Number(
                  data.lastPick.odds,
                ).toFixed(2)}`
              : "aucun"}
          </strong>
        </span>
        <span>
          Dernière erreur :{" "}
          <strong>
            {data.lastFailure
              ? `${data.lastFailure.job_name}${data.lastFailure.tour ? ` · ${data.lastFailure.tour}` : ""}`
              : "aucune connue"}
          </strong>
        </span>
      </div>
    </section>
  );
}

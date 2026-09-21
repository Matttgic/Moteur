"use client";

import { useEffect, useMemo, useState } from "react";
import styles from "./edgescope.module.css";

const API = "https://tapotyftgyacbnjmeovn.supabase.co/functions/v1/edgescope-api";

type Opportunity = {
  id: string;
  sport: string;
  event: string;
  market: string;
  selection: string;
  bookmakerTitle: string;
  offeredOdds: number;
  fairOdds: number;
  edge: number;
  confidence: number;
};

type ScanResponse = {
  status: string;
  provider?: string;
  rows?: Opportunity[];
  cached?: boolean;
  coverage?: {
    sports?: number;
    tournaments?: number;
    fixtures?: number;
    mode?: string;
  };
  error?: string;
  message?: string;
};

type Tab = "scanner" | "history" | "performance" | "settings";

export default function EdgeScopeClient() {
  const [pin, setPin] = useState("");
  const [pinInput, setPinInput] = useState("");
  const [authenticated, setAuthenticated] = useState(false);
  const [tab, setTab] = useState<Tab>("scanner");
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState("Prêt à scanner tous les sports actifs.");
  const [scan, setScan] = useState<ScanResponse>({ status: "idle", rows: [] });
  const [history, setHistory] = useState<any[]>([]);
  const [performance, setPerformance] = useState<any>(null);
  const [apiKeyInput, setApiKeyInput] = useState("");
  const [settingsStatus, setSettingsStatus] = useState("");

  const rows = scan.rows ?? [];
  const bestEdge = useMemo(() => rows.length ? Math.max(...rows.map((r) => Number(r.edge || 0))) : null, [rows]);

  useEffect(() => {
    const saved = window.localStorage.getItem("edgescope_pin") || "";
    if (saved) {
      setPin(saved);
      validatePin(saved);
    }
  }, []);

  async function call(action: string, options?: { method?: string; body?: any; extra?: string; timeoutMs?: number }) {
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), options?.timeoutMs ?? 45000);
    try {
      const res = await fetch(`${API}?action=${encodeURIComponent(action)}${options?.extra ?? ""}`, {
        method: options?.method ?? "GET",
        headers: {
          "content-type": "application/json",
          "x-edgescope-pin": pin || pinInput
        },
        body: options?.body ? JSON.stringify(options.body) : undefined,
        cache: "no-store",
        signal: controller.signal
      });
      const data = await res.json().catch(() => ({ status: "error", error: "Réponse serveur invalide." }));
      if (res.status === 401) throw new Error("Code PIN incorrect.");
      if (!res.ok && data.status !== "ok") throw new Error(data.message || data.error || `HTTP ${res.status}`);
      return data;
    } finally {
      window.clearTimeout(timeout);
    }
  }

  async function validatePin(value: string) {
    try {
      const res = await fetch(`${API}?action=settings`, {
        headers: { "x-edgescope-pin": value, "content-type": "application/json" },
        cache: "no-store"
      });
      if (!res.ok) throw new Error();
      setPin(value);
      setAuthenticated(true);
      window.localStorage.setItem("edgescope_pin", value);
    } catch {
      setAuthenticated(false);
      window.localStorage.removeItem("edgescope_pin");
    }
  }

  async function login() {
    if (!pinInput.trim()) return;
    await validatePin(pinInput.trim());
  }

  function logout() {
    setAuthenticated(false);
    setPin("");
    setPinInput("");
    window.localStorage.removeItem("edgescope_pin");
  }

  async function runGlobalScan() {
    setTab("scanner");
    setLoading(true);
    setStatus("Scan global en cours : détection des sports actifs puis comparaison des cotes…");
    try {
      const data: ScanResponse = await call("scan", { extra: "&sport=all", timeoutMs: 60000 });
      setScan(data);
      const count = data.rows?.length ?? 0;
      const cover = data.coverage;
      setStatus(
        count
          ? `${data.cached ? "Cache 5 min · " : ""}${count} opportunité(s) détectée(s) sur ${cover?.sports ?? "?"} sport(s).`
          : `Scan terminé : aucune value ≥ 2,5 % sur ${cover?.sports ?? "?"} sport(s) actuellement couverts.`
      );
    } catch (e: any) {
      setStatus(e?.name === "AbortError" ? "Le scan a dépassé 60 secondes." : `Erreur : ${e?.message ?? "inconnue"}`);
    } finally {
      setLoading(false);
    }
  }

  async function loadHistory() {
    setTab("history");
    setLoading(true);
    try {
      const data = await call("history");
      setHistory(data.rows ?? []);
    } catch (e: any) {
      setStatus(`Erreur historique : ${e?.message ?? "inconnue"}`);
    } finally {
      setLoading(false);
    }
  }

  async function loadPerformance() {
    setTab("performance");
    setLoading(true);
    try {
      const data = await call("performance");
      setPerformance(data.data ?? null);
    } catch (e: any) {
      setStatus(`Erreur performance : ${e?.message ?? "inconnue"}`);
    } finally {
      setLoading(false);
    }
  }

  async function loadSettings() {
    setTab("settings");
    try {
      const data = await call("settings");
      setSettingsStatus(`OddsPapi : ${data.oddspapi ? "configuré" : "à configurer"}`);
    } catch (e: any) {
      setSettingsStatus(e?.message ?? "Erreur");
    }
  }

  async function saveSettings() {
    if (!apiKeyInput.trim()) return;
    setSettingsStatus("Validation de la clé…");
    try {
      const data = await call("settings", { method: "POST", body: { oddspapiKey: apiKeyInput.trim() } });
      if (data.error) {
        setSettingsStatus(data.message || data.error);
        return;
      }
      setApiKeyInput("");
      setSettingsStatus("Clé validée et enregistrée.");
    } catch (e: any) {
      setSettingsStatus(e?.message ?? "Erreur");
    }
  }

  if (!authenticated) {
    return (
      <main className={styles.loginShell}>
        <section className={styles.loginCard}>
          <p className={styles.eyebrow}>ACCÈS PRIVÉ</p>
          <h1>EdgeScope</h1>
          <p>Entre le code PIN du scanner.</p>
          <input
            className={styles.input}
            type="password"
            value={pinInput}
            onChange={(e) => setPinInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && login()}
            placeholder="Code PIN"
          />
          <button className={styles.primaryButton} onClick={login}>Entrer</button>
        </section>
      </main>
    );
  }

  return (
    <main className={styles.shell}>
      <header className={styles.header}>
        <div>
          <p className={styles.eyebrow}>MULTI-SPORTS · VALUE SCANNER</p>
          <h1>EdgeScope</h1>
          <p className={styles.subtitle}>Pinnacle → cote juste → comparaison bookmakers FR → edge.</p>
        </div>
        <span className={styles.badge}>VERCEL</span>
      </header>

      <nav className={styles.nav}>
        <button onClick={() => setTab("scanner")}>Scanner</button>
        <button onClick={loadHistory}>Historique</button>
        <button onClick={loadPerformance}>Performance</button>
        <button onClick={loadSettings}>Réglages</button>
        <button onClick={logout}>Quitter</button>
      </nav>

      {tab === "scanner" && (
        <>
          <section className={styles.hero}>
            <div>
              <h2>Tous les sports, sans sélection manuelle</h2>
              <p>Le moteur détecte automatiquement les sports actifs sur les prochaines 24 h et priorise le tournoi qui joue le plus tôt dans chaque sport pour préserver le quota API.</p>
            </div>
            <button className={styles.primaryButton} disabled={loading} onClick={runGlobalScan}>
              {loading ? "Scan en cours…" : "Scanner tous les sports"}
            </button>
          </section>

          <section className={styles.metrics}>
            <article><span>Opportunités</span><strong>{rows.length}</strong></article>
            <article><span>Sports couverts</span><strong>{scan.coverage?.sports ?? "—"}</strong></article>
            <article><span>Tournois scannés</span><strong>{scan.coverage?.tournaments ?? "—"}</strong></article>
            <article><span>Meilleur edge</span><strong>{bestEdge == null ? "—" : `${(bestEdge * 100).toFixed(2)}%`}</strong></article>
          </section>

          <section className={styles.statusCard}>
            <strong>{loading ? "Analyse en cours" : "État"}</strong>
            <p>{status}</p>
            {scan.coverage?.fixtures != null && <small>{scan.coverage.fixtures} fixture(s) détectée(s) sur 24 h.</small>}
          </section>

          <section className={styles.list}>
            {rows.map((o) => (
              <article className={styles.opportunity} key={o.id}>
                <div className={styles.eventBlock}>
                  <strong>{o.event}</strong>
                  <span>{o.sport} · {o.market}</span>
                </div>
                <div><span className={styles.pill}>{o.bookmakerTitle}</span><strong>{o.selection} @ {Number(o.offeredOdds).toFixed(2)}</strong></div>
                <div><span>Fair</span><strong>{Number(o.fairOdds).toFixed(2)}</strong></div>
                <div><span>Edge</span><strong className={styles.edge}>{(Number(o.edge) * 100).toFixed(2)}%</strong></div>
                <div><span>Score</span><strong>{o.confidence}/100</strong></div>
              </article>
            ))}
          </section>
        </>
      )}

      {tab === "history" && (
        <section>
          <h2>Historique</h2>
          <div className={styles.list}>
            {history.length === 0 ? <div className={styles.statusCard}>Aucun signal enregistré.</div> : history.map((r) => (
              <article className={styles.historyRow} key={r.id}>
                <strong>{r.event_name}</strong>
                <span>{r.sport_key} · {r.selection} · {r.bookmaker_key}</span>
                <b>{(Number(r.edge) * 100).toFixed(2)}%</b>
              </article>
            ))}
          </div>
        </section>
      )}

      {tab === "performance" && (
        <section>
          <h2>Performance</h2>
          <section className={styles.metrics}>
            <article><span>Signaux</span><strong>{performance?.signals ?? 0}</strong></article>
            <article><span>Edge moyen</span><strong>{performance ? `${(Number(performance.avgEdge || 0) * 100).toFixed(2)}%` : "—"}</strong></article>
            <article><span>P&L</span><strong>{Number(performance?.pnl || 0).toFixed(2)} €</strong></article>
            <article><span>ROI</span><strong>{performance?.roi == null ? "—" : `${(Number(performance.roi) * 100).toFixed(2)}%`}</strong></article>
          </section>
        </section>
      )}

      {tab === "settings" && (
        <section>
          <h2>Réglages</h2>
          <div className={styles.settingsCard}>
            <p>{settingsStatus}</p>
            <input className={styles.input} type="password" value={apiKeyInput} onChange={(e) => setApiKeyInput(e.target.value)} placeholder="Nouvelle clé OddsPapi" />
            <button className={styles.primaryButton} onClick={saveSettings}>Valider la clé</button>
          </div>
        </section>
      )}

      <footer className={styles.footer}>18+ · Aucun gain garanti. Le score de confiance mesure la qualité technique du signal, pas la probabilité de gagner.</footer>
    </main>
  );
}

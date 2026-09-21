import benchmark from "@/ml/benchmark_reference.json";
import LiveSelections from "./live-selections";

const pct = (value: number) => `${(value * 100).toFixed(1)}%`;
const tours = ["ATP", "WTA"] as const;

export default function Home() {
  const totalOos =
    benchmark.ATP.test_n +
    benchmark.WTA.test_n;

  return (
    <main className="shell">
      <header className="hero">
        <div>
          <p className="eyebrow">ATP + WTA · PRE-MATCH RESEARCH</p>
          <h1>Tennis Quant Engine</h1>
          <p className="subtitle">
            Probabilité → cote juste → marché sans marge → edge → EV → décision.
          </p>
        </div>
        <span className="status">ML VALIDÉ · ROI EN ATTENTE</span>
      </header>

      <section className="kpis" aria-label="Statut du moteur">
        <article>
          <span>Univers</span>
          <strong>ATP / WTA</strong>
        </article>
        <article>
          <span>Champion ML</span>
          <strong>Full Logit</strong>
        </article>
        <article>
          <span>Matchs OOS</span>
          <strong>{totalOos.toLocaleString("fr-FR")}</strong>
        </article>
        <article>
          <span>Validation</span>
          <strong>Walk-forward</strong>
        </article>
      </section>

      <section className="sectionHeader">
        <div>
          <p className="eyebrow">MODEL VALIDATION</p>
          <h2>Résultats hors échantillon 2022–2025</h2>
        </div>
        <p>
          Les métriques ci-dessous proviennent de matchs futurs jamais utilisés
          pour entraîner le fold correspondant.
        </p>
      </section>

      <div className="validationGrid">
        {tours.map((tour) => {
          const result = benchmark[tour];
          return (
            <article className="validationCard" key={tour}>
              <div className="validationTop">
                <span className="pill">{tour}</span>
                <span className="validationOk">CALIBRÉ</span>
              </div>

              <strong className="validationChampion">
                {result.champion.replace("_", " ").toUpperCase()}
              </strong>
              <span className="validationSample">
                {result.test_n.toLocaleString("fr-FR")} matchs OOS
              </span>

              <div className="metricGrid">
                <div>
                  <span>Accuracy</span>
                  <strong>{pct(result.accuracy)}</strong>
                </div>
                <div>
                  <span>Log Loss</span>
                  <strong>{result.log_loss.toFixed(4)}</strong>
                </div>
                <div>
                  <span>Brier</span>
                  <strong>{result.brier_score.toFixed(4)}</strong>
                </div>
                <div>
                  <span>ECE-10</span>
                  <strong>{result.ece_10.toFixed(4)}</strong>
                </div>
              </div>
            </article>
          );
        })}

        <article className="economicGate">
          <div>
            <p className="eyebrow">ECONOMIC VALIDATION</p>
            <h3>ROI / CLV non validés</h3>
          </div>
          <p>
            Le moteur refuse de revendiquer une rentabilité sans cotes historiques
            autorisées et horodatées. La couche d’import, no-vig, edge, EV, ROI,
            drawdown et CLV est prête.
          </p>
          <span>Prochaine gate : vraies cotes pré-match</span>
        </article>
      </div>

      <LiveSelections />

      <section className="method">
        <div>
          <p className="eyebrow">GOUVERNANCE DU MODÈLE</p>
          <h2>Ce que la V1 contrôle déjà</h2>
        </div>
        <div className="methodGrid">
          <p><strong>1.</strong> Validation temporelle sans mélange futur/passé.</p>
          <p><strong>2.</strong> Calibration séparée avant chaque période de test.</p>
          <p><strong>3.</strong> Comparaison à des modèles plus simples avant promotion.</p>
          <p><strong>4.</strong> Retrait de marge avant le calcul d’edge.</p>
          <p><strong>5.</strong> NO BET reste une sortie de premier rang.</p>
          <p><strong>6.</strong> Aucun ROI affiché avant preuve économique réelle.</p>
        </div>
      </section>

      <footer>
        Outil de recherche quantitative. Les probabilités restent incertaines et
        aucun rendement n’est garanti.
      </footer>
    </main>
  );
}

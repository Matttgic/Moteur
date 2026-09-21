import benchmark from "@/ml/benchmark_reference.json";
import { demoMatches } from "@/lib/demo-data";
import { predictMatch } from "@/lib/model";

const pct = (value: number) => `${(value * 100).toFixed(1)}%`;
const signedPct = (value: number) =>
  `${value >= 0 ? "+" : ""}${(value * 100).toFixed(1)}%`;

const tours = ["ATP", "WTA"] as const;

export default function Home() {
  const rows = demoMatches.map((match) => ({
    match,
    prediction: predictMatch(match)
  }));

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

      <section className="sectionHeader opportunitiesHeader">
        <div>
          <p className="eyebrow">ENGINE PREVIEW</p>
          <h2>Lecture quantitative</h2>
        </div>
        <p>
          Les cartes suivantes utilisent des données fictives pour démontrer le
          workflow produit. Elles ne sont pas des paris du jour.
        </p>
      </section>

      <section className="notice">
        <strong>Données de démonstration.</strong> Aucun joueur, match ou prix
        ci-dessous ne doit être interprété comme une donnée live réelle.
      </section>

      <div className="cards demoCards">
        {rows.map(({ match, prediction }) => {
          const side =
            prediction.recommendation === "BET_A"
              ? prediction.playerA
              : prediction.recommendation === "BET_B"
                ? prediction.playerB
                : "NO BET";
          const chosenEdge =
            prediction.recommendation === "BET_B"
              ? prediction.edgeB
              : prediction.edgeA;
          const chosenEv =
            prediction.recommendation === "BET_B"
              ? prediction.evB
              : prediction.evA;

          return (
            <article
              className="matchCard"
              key={`${match.playerA.name}-${match.playerB.name}`}
            >
              <div className="matchTop">
                <div>
                  <span className="pill">{match.tour}</span>
                  <span className="muted">{match.surface}</span>
                </div>
                <span className={`tier tier-${prediction.tier.toLowerCase()}`}>
                  {prediction.tier}
                </span>
              </div>

              <h3>
                {match.playerA.name} <span>vs</span> {match.playerB.name}
              </h3>

              <div className="probGrid">
                <div>
                  <span>{prediction.playerA}</span>
                  <strong>{pct(prediction.probabilityA)}</strong>
                  <small>Cote juste {prediction.fairOddsA.toFixed(2)}</small>
                </div>
                <div>
                  <span>{prediction.playerB}</span>
                  <strong>{pct(prediction.probabilityB)}</strong>
                  <small>Cote juste {prediction.fairOddsB.toFixed(2)}</small>
                </div>
              </div>

              <div className="decision">
                <div>
                  <span>Décision</span>
                  <strong>{side}</strong>
                </div>
                <div>
                  <span>Edge</span>
                  <strong>{signedPct(chosenEdge)}</strong>
                </div>
                <div>
                  <span>EV</span>
                  <strong>{signedPct(chosenEv)}</strong>
                </div>
                <div>
                  <span>Mise</span>
                  <strong>{prediction.stakeUnits.toFixed(2)}u</strong>
                </div>
              </div>

              <ul>
                {prediction.reasons.map((reason) => (
                  <li key={reason}>{reason}</li>
                ))}
              </ul>
            </article>
          );
        })}
      </div>

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

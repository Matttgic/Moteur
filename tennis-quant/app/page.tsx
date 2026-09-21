import benchmark from "@/ml/benchmark_reference.json";
import LiveSelections from "./live-selections";
import EconomicValidation from "./economic-validation";
import SystemHealth from "./system-health";
import ShadowLab from "./shadow-lab";

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
        <span className="status">ML VALIDÉ · ÉCONOMIE EN COLLECTE</span>
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

      <SystemHealth />

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
                  <span>Vainqueurs trouvés</span>
                  <strong>{pct(result.accuracy)}</strong>
                  <small>Accuracy · plus haut = mieux</small>
                </div>
                <div>
                  <span>Erreurs confiantes</span>
                  <strong>{result.log_loss.toFixed(4)}</strong>
                  <small>Log Loss · plus bas = mieux · 0,693 ≈ 50/50</small>
                </div>
                <div>
                  <span>Qualité des probabilités</span>
                  <strong>{result.brier_score.toFixed(4)}</strong>
                  <small>Brier · plus bas = mieux · 0,25 ≈ 50/50</small>
                </div>
                <div>
                  <span>Calibration</span>
                  <strong>{pct(result.ece_10)}</strong>
                  <small>ECE-10 · écart moyen sur 10 tranches</small>
                </div>
              </div>
            </article>
          );
        })}


      </div>

      <section className="metricGuide">
        <div>
          <strong>Comment lire ces scores ?</strong>
          <p>
            Accuracy mesure le nombre de vainqueurs trouvés. Brier et Log Loss
            jugent surtout si les probabilités annoncées sont crédibles.
            ECE-10 mesure la calibration : un 70 % devrait gagner environ 70 %
            du temps.
          </p>
        </div>
        <div>
          <strong>Ce qui compte pour parier</strong>
          <p>
            Une bonne prédiction ne suffit pas : la vraie question est de savoir
            si notre probabilité bat le prix du bookmaker. C'est pour cela que
            ROI, drawdown et CLV sont suivis séparément ci-dessous.
          </p>
        </div>
      </section>

      <EconomicValidation />

      <ShadowLab />

      <LiveSelections />

      <section className="method">
        <div>
          <p className="eyebrow">GOUVERNANCE DU MODÈLE</p>
          <h2>Ce que le moteur contrôle déjà</h2>
        </div>
        <div className="methodGrid">
          <p><strong>1.</strong> Validation temporelle sans mélange futur/passé.</p>
          <p><strong>2.</strong> Calibration séparée avant chaque période de test.</p>
          <p><strong>3.</strong> Comparaison à des modèles plus simples avant promotion.</p>
          <p><strong>4.</strong> Retrait de marge avant le calcul d’edge.</p>
          <p><strong>5.</strong> NO BET reste une sortie de premier rang.</p>
          <p><strong>6.</strong> ROI, drawdown et CLV mesurés sur les picks réellement sortis.</p>
          <p><strong>7.</strong> Settlement et contrôle santé automatisés par cron.</p>
          <p><strong>8.</strong> Variantes shadow séparées du champion, sans auto-promotion.</p>
        </div>
      </section>

      <footer>
        Outil de recherche quantitative. Les probabilités restent incertaines et
        aucun rendement n’est garanti.
      </footer>
    </main>
  );
}

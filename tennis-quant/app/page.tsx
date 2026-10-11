import benchmark from "@/ml/benchmark_reference.json";
import QualityLiveSelections from "./quality-live-selections";
import EconomicValidation from "./economic-validation";
import QualityEconomics from "./quality-economics";
import SystemHealth from "./system-health";
import ShadowLab from "./shadow-lab";
import LiveModelAudit from "./live-model-audit";

const pct = (value: number) => `${(value * 100).toFixed(1)}%`;
const tours = ["ATP", "WTA"] as const;

export default function Home() {
  const totalOos = benchmark.ATP.test_n + benchmark.WTA.test_n;

  return (
    <main className="shell">
      <nav className="topNav" aria-label="Navigation Tennis Quant">
        <a className="brandLockup" href="#top" aria-label="Tennis Quant">
          <span className="brandMark" aria-hidden="true">
            <i />
            <i />
          </span>
          <span>
            <strong>Tennis Quant</strong>
            <small>Decision Engine</small>
          </span>
        </a>

        <div className="desktopNavLinks">
          <a href="#picks">Picks</a>
          <a href="#performance">Performance</a>
          <a href="#audit">Audit</a>
          <a href="#shadow">Shadow</a>
          <a href="#research">Research</a>
        </div>

        <div className="topNavActions">
          <span className="liveDot">
            <i aria-hidden="true" />
            LIVE
          </span>
          <a className="courtVisionLink" href="/court-vision">
            Court Vision
          </a>
        </div>
      </nav>

      <header className="heroV2" id="top">
        <div className="heroV2Copy">
          <div className="heroSignal">
            <span>ATP + WTA</span>
            <b>FULL LOGIT</b>
            <span>VALUE ONLY</span>
          </div>

          <p className="eyebrow">QUANTITATIVE TENNIS RESEARCH</p>
          <h1>
            Le signal avant
            <span> le bruit.</span>
          </h1>
          <p className="subtitle">
            Probabilité, prix de marché, edge, qualité d&apos;exécution et validation
            économique réunis dans un seul cockpit.
          </p>

          <div className="heroActions">
            <a className="primaryAction" href="#picks">
              Voir les picks
              <span aria-hidden="true">↘</span>
            </a>
            <a className="secondaryAction" href="#audit">
              Ouvrir l&apos;audit
            </a>
          </div>

          <div className="heroMicroStats" aria-label="Résumé moteur">
            <div>
              <span>Univers</span>
              <strong>ATP / WTA</strong>
            </div>
            <div>
              <span>OOS</span>
              <strong>{totalOos.toLocaleString("fr-FR")}</strong>
            </div>
            <div>
              <span>Quality Gate</span>
              <strong>≥90</strong>
            </div>
          </div>
        </div>

        <div className="heroVisual" aria-label="Ambiance terrain de tennis">
          <div className="heroPhoto" />
          <div className="courtGrid" aria-hidden="true">
            <i className="courtLine courtLineA" />
            <i className="courtLine courtLineB" />
            <i className="courtLine courtLineC" />
            <i className="courtLine courtLineD" />
          </div>
          <div className="heroVisualCard heroVisualCardTop">
            <span>EXECUTION POLICY</span>
            <strong>VALUE · Q90+</strong>
            <small>PREMIUM reste en Shadow</small>
          </div>
          <div className="heroVisualCard heroVisualCardBottom">
            <span>MODEL STATUS</span>
            <strong>FULL LOGIT</strong>
            <small>validation temporelle active</small>
          </div>
          <a
            className="photoCredit"
            href="https://unsplash.com/photos/tennis-court-mKS80M2QS4E"
            target="_blank"
            rel="noreferrer"
          >
            Travis Jones · Unsplash
          </a>
        </div>
      </header>

      <section className="commandStrip" aria-label="Statut du moteur">
        <div>
          <span className="commandLabel">Champion</span>
          <strong>Full Logit</strong>
        </div>
        <div>
          <span className="commandLabel">Execution</span>
          <strong>VALUE uniquement</strong>
        </div>
        <div>
          <span className="commandLabel">Calibration Guard</span>
          <strong>Actif</strong>
        </div>
        <div>
          <span className="commandLabel">Market Anchor</span>
          <strong>Shadow challenger</strong>
        </div>
      </section>

      <div id="picks" className="anchorTarget">
        <QualityLiveSelections />
      </div>

      <div id="performance" className="anchorTarget">
        <EconomicValidation />
      </div>

      <div id="audit" className="anchorTarget">
        <LiveModelAudit />
      </div>

      <SystemHealth />

      <div id="shadow" className="anchorTarget">
        <ShadowLab />
      </div>

      <QualityEconomics />

      <section className="researchSection anchorTarget" id="research">
        <div className="sectionHeader sectionHeaderV2">
          <div>
            <p className="eyebrow">MODEL VALIDATION</p>
            <h2>Research desk</h2>
          </div>
          <p>
            Les métriques OOS servent à vérifier que les probabilités restent
            crédibles avant toute promotion de modèle.
          </p>
        </div>

        <div className="validationGrid">
          {tours.map((tour) => {
            const result = benchmark[tour];
            return (
              <article className="validationCard validationCardV2" key={tour}>
                <div className="validationTop">
                  <div>
                    <span className="pill">{tour}</span>
                    <span className="validationSample">
                      {result.test_n.toLocaleString("fr-FR")} matchs OOS
                    </span>
                  </div>
                  <span className="validationOk">CALIBRÉ</span>
                </div>

                <strong className="validationChampion">
                  {result.champion.replace("_", " ").toUpperCase()}
                </strong>

                <div className="researchMetrics">
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
                    <strong>{pct(result.ece_10)}</strong>
                  </div>
                </div>
              </article>
            );
          })}
        </div>

        <details className="researchDetails">
          <summary>Comment lire les métriques</summary>
          <div>
            <p>
              Accuracy mesure le nombre de vainqueurs trouvés. Brier et Log Loss
              évaluent surtout la qualité des probabilités annoncées. ECE mesure
              leur calibration.
            </p>
            <p>
              Pour l&apos;exécution, le moteur ajoute ensuite le prix bookmaker,
              l&apos;edge, l&apos;EV, le Quality Score, le mouvement de cote et les
              performances live.
            </p>
          </div>
        </details>
      </section>

      <section className="method governanceV2">
        <div>
          <p className="eyebrow">MODEL GOVERNANCE</p>
          <h2>Garde-fous actifs</h2>
        </div>
        <div className="methodGrid">
          <p><strong>01</strong> Validation temporelle sans mélange futur/passé.</p>
          <p><strong>02</strong> Calibration séparée avant chaque période de test.</p>
          <p><strong>03</strong> Retrait de marge avant calcul de l&apos;edge.</p>
          <p><strong>04</strong> NO BET reste une sortie de premier rang.</p>
          <p><strong>05</strong> ROI, drawdown et CLV suivis sur décisions uniques.</p>
          <p><strong>06</strong> Settlement et health checks automatisés.</p>
          <p><strong>07</strong> VALUE uniquement avec Quality ≥90.</p>
          <p><strong>08</strong> PREMIUM et Market Anchor restent en Shadow.</p>
        </div>
      </section>

      <footer className="footerV2">
        <div>
          <strong>Tennis Quant</strong>
          <span>Research engine · ATP/WTA</span>
        </div>
        <p>
          Outil de recherche quantitative. Les probabilités restent incertaines et
          aucun rendement n&apos;est garanti.
        </p>
      </footer>

      <nav className="mobileDock" aria-label="Navigation mobile">
        <a href="#picks">
          <span>●</span>
          Picks
        </a>
        <a href="#performance">
          <span>↗</span>
          Bilan
        </a>
        <a href="#audit">
          <span>◇</span>
          Audit
        </a>
        <a href="#shadow">
          <span>◌</span>
          Shadow
        </a>
      </nav>
    </main>
  );
}

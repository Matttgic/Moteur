import { demoMatches } from "@/lib/demo-data";
import { predictMatch } from "@/lib/model";

const pct = (value: number) => `${(value * 100).toFixed(1)}%`;
const signedPct = (value: number) =>
  `${value >= 0 ? "+" : ""}${(value * 100).toFixed(1)}%`;

export default function Home() {
  const rows = demoMatches.map((match) => ({
    match,
    prediction: predictMatch(match)
  }));

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
        <span className="status">BASELINE V0.1</span>
      </header>

      <section className="notice">
        <strong>Données de démonstration.</strong> Les cartes ci-dessous valident
        le moteur et l’interface. Elles ne représentent pas des matchs réels du jour.
      </section>

      <section className="kpis" aria-label="Principes du moteur">
        <article>
          <span>Univers</span>
          <strong>ATP / WTA</strong>
        </article>
        <article>
          <span>Décision</span>
          <strong>BET / NO BET</strong>
        </article>
        <article>
          <span>Risque max V1</span>
          <strong>0,75u</strong>
        </article>
        <article>
          <span>Validation cible</span>
          <strong>Walk-forward</strong>
        </article>
      </section>

      <section className="sectionHeader">
        <div>
          <p className="eyebrow">OPPORTUNITÉS</p>
          <h2>Lecture quantitative</h2>
        </div>
        <p>Le modèle peut refuser un pari même si un joueur est favori.</p>
      </section>

      <div className="cards">
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
            <article className="matchCard" key={`${match.playerA.name}-${match.playerB.name}`}>
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
          <h2>Ce que la V1 fait déjà correctement</h2>
        </div>
        <div className="methodGrid">
          <p><strong>1.</strong> Retire la marge du bookmaker avant de mesurer l’edge.</p>
          <p><strong>2.</strong> Réduit la confiance quand la qualité des données baisse.</p>
          <p><strong>3.</strong> Sépare probabilité brute, cote juste, edge et EV.</p>
          <p><strong>4.</strong> Autorise explicitement NO BET.</p>
          <p><strong>5.</strong> Plafonne la mise tant que le modèle n’est pas backtesté.</p>
          <p><strong>6.</strong> Prépare Brier, Log Loss, ROI et CLV pour la validation.</p>
        </div>
      </section>

      <footer>
        Outil de recherche quantitative. Aucun rendement n’est garanti.
      </footer>
    </main>
  );
}

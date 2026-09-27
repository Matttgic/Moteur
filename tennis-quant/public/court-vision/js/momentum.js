// Analyse du point par point + graphique de momentum (SVG).
import { winProbability, isBreakPoint } from './model.js';

const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

export function modelOptions(match) {
  return {
    bestOf: match?.format === 'BO5' ? 5 : 3,
    decidingTbTarget: match?.tier === 'grand_slam' ? 10 : 7,
    matchTiebreak: match?.draw === 'doubles' || !!match?.is_doubles,
  };
}

export function scoreLabel(row) {
  if (!row) return '';
  const idx = (row.sets?.[0] ?? 0) + (row.sets?.[1] ?? 0);
  const ga = row.games?.[0]?.[idx] ?? 0, gb = row.games?.[1]?.[idx] ?? 0;
  const pts = row.points ? `${row.points[0]}-${row.points[1]}` : '';
  const tb = row.is_tiebreak ? ' (TB)' : '';
  return `Sets ${row.sets?.[0] ?? 0}-${row.sets?.[1] ?? 0} · Jeux ${ga}-${gb} · ${pts}${tb}`;
}

/**
 * Enrichit chaque ligne : prob (API ou modèle), source, vainqueur du point, serveur du point.
 */
export function analyse(tape, match) {
  const opts = modelOptions(match);
  const rows = (tape || []).filter((r) => r && r.sets && r.games);
  let apiProbs = 0;
  const pts = rows.map((r, i) => {
    const api = typeof r.win_probability_p1 === 'number' ? r.win_probability_p1 : null;
    if (api != null) apiProbs++;
    const prev = rows[i - 1];
    return {
      row: r,
      i,
      prob: api ?? winProbability(r, opts),
      fromApi: api != null,
      winner: r.point_winner ?? r.winner ?? null,
      server: prev ? prev.server : null,
    };
  });

  const s = {
    points: [0, 0], servePts: [0, 0], serveWon: [0, 0],
    bpChances: [0, 0], breaks: [0, 0], bestStreak: [0, 0],
    swings: [], breakIdx: [],
  };
  let streak = [0, 0];
  for (let i = 1; i < pts.length; i++) {
    const p = pts[i], prev = pts[i - 1];
    const w = p.winner;
    if (w === 1 || w === 2) {
      s.points[w - 1]++;
      streak[w - 1]++; streak[2 - w] = 0;
      s.bestStreak[w - 1] = Math.max(s.bestStreak[w - 1], streak[w - 1]);
      if (p.server === 1 || p.server === 2) {
        s.servePts[p.server - 1]++;
        if (w === p.server) s.serveWon[p.server - 1]++;
      }
    }
    // Balle de break jouée sur ce point (état avant le point).
    if (isBreakPoint(prev.row) && (prev.row.server === 1 || prev.row.server === 2)) {
      s.bpChances[2 - prev.row.server]++;
    }
    // Break : un jeu hors tie-break gagné par le relanceur.
    const gameWinner = gameWonBy(prev.row, p.row);
    if (gameWinner && !prev.row.is_tiebreak && prev.row.server && gameWinner !== prev.row.server) {
      s.breaks[gameWinner - 1]++;
      s.breakIdx.push({ i, by: gameWinner });
    }
    s.swings.push({ i, delta: p.prob - prev.prob });
  }
  s.swings = s.swings
    .filter((x) => Math.abs(x.delta) > 0.001)
    .sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta))
    .slice(0, 5)
    .map((x) => ({ ...x, before: pts[x.i - 1].row, after: pts[x.i].row }));

  // Débuts de set (pour les repères verticaux).
  const setStarts = [];
  for (let i = 1; i < pts.length; i++) {
    const a = pts[i - 1].row.sets, b = pts[i].row.sets;
    if (a[0] + a[1] !== b[0] + b[1] && i < pts.length - 1) setStarts.push({ i, n: b[0] + b[1] + 1 });
  }

  return { pts, stats: s, setStarts, apiProbs, total: pts.length };
}

function gameWonBy(a, b) {
  const sa = (a.sets[0] ?? 0) + (a.sets[1] ?? 0);
  const g = (r, p) => r.games?.[p]?.[sa] ?? 0;
  if (g(b, 0) > g(a, 0)) return 1;
  if (g(b, 1) > g(a, 1)) return 2;
  return null;
}

/**
 * Graphique : probabilité de victoire de J1 au fil des points.
 * Zone au-dessus de 50 % = couleur J1, en dessous = couleur J2.
 */
export function renderMomentum(container, analysis, names) {
  if (!container) return;
  const { pts, setStarts, stats } = analysis;
  if (pts.length < 2) {
    container.innerHTML = '<p class="muted">Pas encore assez de points pour tracer le momentum.</p>';
    return;
  }
  // Dessiné à la largeur réelle du conteneur pour garder un texte lisible sur mobile.
  const W = Math.max(300, Math.round(container.clientWidth || 900));
  const H = W < 560 ? 220 : 280;
  const m = { t: 16, r: 12, b: 26, l: 12 };
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const x = (i) => m.l + (i / (pts.length - 1)) * iw;
  const y = (p) => m.t + (1 - p) * ih;
  const mid = y(0.5);

  const line = pts.map((p, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(p.prob).toFixed(1)}`).join('');
  const area = `${line}L${x(pts.length - 1).toFixed(1)},${mid}L${x(0)},${mid}Z`;

  const setLines = setStarts.map((s) => `
    <line x1="${x(s.i)}" x2="${x(s.i)}" y1="${m.t}" y2="${H - m.b}" class="mv-set"/>
    <text x="${x(s.i) + 4}" y="${H - m.b + 16}" class="mv-axis">Set ${s.n}</text>`).join('');

  const breaks = stats.breakIdx.map((b) => `
    <circle cx="${x(b.i)}" cy="${y(pts[b.i].prob)}" r="4.5" class="mv-break mv-p${b.by}"/>`).join('');

  container.innerHTML = `
    <div class="mv-wrap">
      <div class="mv-legend">
        <span><i class="sw sw-p1"></i>${esc(names[0])} favori</span>
        <span><i class="sw sw-p2"></i>${esc(names[1])} favori</span>
        <span><i class="sw sw-break"></i>Break</span>
      </div>
      <svg viewBox="0 0 ${W} ${H}" class="mv-svg" role="img" aria-label="Probabilité de victoire de ${esc(names[0])} point par point">
        <defs>
          <clipPath id="mv-top"><rect x="0" y="0" width="${W}" height="${mid}"/></clipPath>
          <clipPath id="mv-bot"><rect x="0" y="${mid}" width="${W}" height="${H - mid}"/></clipPath>
        </defs>
        <text x="${m.l + 2}" y="${m.t + 12}" class="mv-axis">${esc(names[0])} 100 %</text>
        <text x="${m.l + 2}" y="${H - m.b - 6}" class="mv-axis">${esc(names[1])} 100 %</text>
        <line x1="${m.l}" x2="${W - m.r}" y1="${mid}" y2="${mid}" class="mv-mid"/>
        ${setLines}
        <path d="${area}" class="mv-area-p1" clip-path="url(#mv-top)"/>
        <path d="${area}" class="mv-area-p2" clip-path="url(#mv-bot)"/>
        <path d="${line}" class="mv-line"/>
        ${breaks}
        <g class="mv-hover" style="display:none">
          <line class="mv-cross" y1="${m.t}" y2="${H - m.b}"/>
          <circle r="5" class="mv-dot"/>
        </g>
        <rect x="${m.l}" y="${m.t}" width="${iw}" height="${ih}" fill="transparent" class="mv-hit"/>
      </svg>
      <div class="mv-tip" hidden></div>
    </div>`;

  const svg = container.querySelector('svg');
  const hover = svg.querySelector('.mv-hover');
  const cross = hover.querySelector('.mv-cross');
  const dot = hover.querySelector('.mv-dot');
  const tip = container.querySelector('.mv-tip');
  const hit = svg.querySelector('.mv-hit');

  const show = (clientX) => {
    const rect = svg.getBoundingClientRect();
    const sx = ((clientX - rect.left) / rect.width) * W;
    const i = Math.max(0, Math.min(pts.length - 1, Math.round(((sx - m.l) / iw) * (pts.length - 1))));
    const p = pts[i];
    const px = x(i), py = y(p.prob);
    hover.style.display = '';
    cross.setAttribute('x1', px); cross.setAttribute('x2', px);
    dot.setAttribute('cx', px); dot.setAttribute('cy', py);
    const fav = p.prob >= 0.5 ? 0 : 1;
    const pct = Math.round((fav === 0 ? p.prob : 1 - p.prob) * 100);
    tip.hidden = false;
    tip.innerHTML = `
      <strong>Point ${i}</strong>
      <div>${esc(scoreLabel(p.row))}</div>
      <div><i class="sw sw-p${fav + 1}"></i>${esc(names[fav])} ${pct} %</div>
      <div class="muted">${p.fromApi ? 'Modèle Live Tennis API' : 'Modèle Court Vision (score)'}</div>`;
    const left = (px / W) * rect.width;
    tip.style.left = `${Math.min(Math.max(left, 90), rect.width - 90)}px`;
    tip.style.top = `${svg.offsetTop + 4}px`;
  };
  const hide = () => { hover.style.display = 'none'; tip.hidden = true; };
  hit.addEventListener('pointermove', (e) => show(e.clientX));
  hit.addEventListener('pointerdown', (e) => show(e.clientX));
  hit.addEventListener('pointerleave', hide);

  if (!container.dataset.resizeBound) {
    container.dataset.resizeBound = '1';
    let lastW = W;
    new ResizeObserver(() => {
      const w = Math.round(container.clientWidth);
      if (Math.abs(w - lastW) > 40 && container.__analysis) { lastW = w; renderMomentum(container, container.__analysis, container.__names); }
    }).observe(container);
  }
  container.__analysis = analysis;
  container.__names = names;
}

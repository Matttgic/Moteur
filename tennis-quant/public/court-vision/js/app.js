import { api, settings, localCallsToday, ApiError, initMode } from './api.js';
import { analyse, renderMomentum, scoreLabel, modelOptions } from './momentum.js';
import { winProbability } from './model.js';

const $ = (sel, root = document) => root.querySelector(sel);
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const view = $('#view');
// Écritures tolérantes : un chargement qui se termine après un changement de page ne plante pas.
const put = (sel, html) => { const el = $(sel); if (el) el.innerHTML = html; };
const putText = (sel, text) => { const el = $(sel); if (el) el.textContent = text; };

// ---------- Utilitaires ----------

const TOURS = [['', 'Tous'], ['atp', 'ATP'], ['wta', 'WTA'], ['challenger', 'Challenger'], ['itf', 'ITF']];
const SURFACES = { hard: 'Dur', clay: 'Terre battue', grass: 'Gazon' };
const TIERS = {
  grand_slam: 'Grand Chelem', atp_1000: 'Masters 1000', atp_500: 'ATP 500', atp_250: 'ATP 250',
  wta_1000: 'WTA 1000', wta_500: 'WTA 500', wta_250: 'WTA 250', wta_125: 'WTA 125', atp_finals: 'ATP Finals', wta_finals: 'WTA Finals',
};
const ROUNDS = { F: 'Finale', SF: 'Demi', QF: 'Quart', R16: '8e', R32: '16e', R64: '32e', R128: '64e', Q: 'Qualif', RR: 'Poules' };

const fmtTime = (iso) => iso ? new Date(iso).toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit' }) : '—';
const fmtDay = (iso) => new Date(iso).toLocaleDateString('fr-FR', { weekday: 'long', day: 'numeric', month: 'long' });
const pct = (x) => (x == null || Number.isNaN(x) ? '—' : `${Math.round(x * 100)} %`);
const ratio = (a, b) => (b ? a / b : null);
const round = (m) => ROUNDS[m.round_code] || m.round || '';
const surfaceChip = (s) => (s ? `<span class="chip chip-${esc(s)}">${esc(SURFACES[s] || s)}</span>` : '');
const tierLabel = (t) => TIERS[t] || '';
const today = () => new Date().toISOString().slice(0, 10);

function params() {
  const q = location.hash.split('?')[1] || '';
  return Object.fromEntries(new URLSearchParams(q));
}

function playerLine(p) {
  if (!p) return '<span class="muted">À déterminer</span>';
  const rank = p.ranking ? `<span class="rank">#${p.ranking}</span>` : '';
  const c = p.country ? `<span class="country">${esc(p.country.toUpperCase())}</span>` : '';
  return `<span class="pname">${esc(p.name)}</span>${rank}${c}`;
}

function errorBox(e) {
  const msg = e instanceof ApiError ? e.message : `Erreur : ${e.message || e}`;
  return `<div class="alert">${esc(msg)}</div>`;
}

function loading(label = 'Chargement…') {
  return `<div class="loading"><span class="spinner"></span>${esc(label)}</div>`;
}

function tourChips(current, base) {
  return `<div class="chips" role="tablist">${TOURS.map(([v, l]) =>
    `<a class="chip-btn ${v === (current || '') ? 'on' : ''}" href="${base}${v ? `?tour=${v}` : ''}">${l}</a>`).join('')}</div>`;
}

// Tableau de score compact : une colonne par set + points en cours.
function scoreGrid(m, { big = false } = {}) {
  const s = m.score;
  const live = m.status === 'live';
  const nSets = s?.games ? Math.max(s.games[0]?.length || 0, s.games[1]?.length || 0) : 0;
  const rowFor = (k) => {
    const p = m.players?.[`p${k}`];
    const won = m.winner === k;
    const serving = live && s?.server === k;
    const cells = [];
    for (let i = 0; i < nSets; i++) {
      const g = s.games[k - 1]?.[i] ?? 0;
      const o = s.games[2 - k]?.[i] ?? 0;
      const setDone = i < (s.sets[0] + s.sets[1]);
      cells.push(`<td class="g ${setDone && g > o ? 'won' : ''}">${g}</td>`);
    }
    const pts = live && s?.points ? `<td class="pt">${esc(s.points[k - 1])}</td>` : '';
    return `<tr class="${won ? 'winner' : ''}">
      <td class="srv">${serving ? '<i class="ball" title="Au service"></i>' : ''}</td>
      <td class="who">${playerLine(p)}</td>${cells.join('')}${pts}</tr>`;
  };
  return `<table class="score ${big ? 'big' : ''}"><tbody>${rowFor(1)}${rowFor(2)}</tbody></table>`;
}

// ---------- Rafraîchissement automatique ----------

let timers = [];
function every(seconds, fn) {
  const id = setInterval(() => { if (document.visibilityState === 'visible') fn(); }, seconds * 1000);
  timers.push(id);
}
function clearTimers() { timers.forEach(clearInterval); timers = []; }

// ---------- Vues ----------

async function liveView() {
  const { tour } = params();
  view.innerHTML = `
    <header class="page-head">
      <div><h1>En direct</h1><p class="muted" id="upd">&nbsp;</p></div>
      ${tourChips(tour, '#/live')}
    </header>
    <div id="list">${loading()}</div>`;
  const load = async () => {
    try {
      const res = await api.liveMatches(tour);
      const matches = (res.data || []).filter((m) => m.draw !== 'doubles');
      putText('#upd', `${matches.length} match(s) · mis à jour à ${fmtTime(new Date().toISOString())} · actualisation toutes les ${settings.refreshSeconds} s`);
      put('#list', matches.length ? groupByTournament(matches).map(([t, ms]) => `
        <section class="group">
          <h2>${esc(t)} ${surfaceChip(ms[0].surface)} <span class="muted small">${esc(tierLabel(ms[0].tier))}</span></h2>
          <div class="cards">${ms.map(matchCard).join('')}</div>
        </section>`).join('') : '<div class="empty">Aucun match en cours pour ce filtre.</div>');
    } catch (e) { put('#list', errorBox(e)); }
  };
  await load();
  every(settings.refreshSeconds, load);
}

function groupByTournament(matches) {
  const g = new Map();
  for (const m of matches) {
    const k = m.tournament || 'Tournoi';
    if (!g.has(k)) g.set(k, []);
    g.get(k).push(m);
  }
  return [...g.entries()];
}

function matchCard(m) {
  return `<a class="card match-card" href="#/match/${m.id}">
    <div class="card-top"><span>${esc(round(m))}</span>${m.status === 'live' ? '<span class="live-dot">LIVE</span>' : ''}</div>
    ${scoreGrid(m)}
  </a>`;
}

async function matchView(id) {
  view.innerHTML = loading('Chargement du match…');
  let m;
  try { m = await api.match(id); } catch (e) { view.innerHTML = errorBox(e); return; }
  if (!m) { view.innerHTML = '<div class="alert">Match introuvable.</div>'; return; }
  const p1 = m.players?.p1, p2 = m.players?.p2;
  const names = [p1?.name || 'Joueur 1', p2?.name || 'Joueur 2'];
  const live = m.status === 'live';

  view.innerHTML = `
    <a class="back" href="${live ? '#/live' : '#/results'}">← Retour</a>
    <header class="match-head card">
      <div class="meta">
        <strong>${esc(m.tournament)}</strong> · ${esc(round(m))} ${surfaceChip(m.surface)}
        <span class="muted small">${esc(tierLabel(m.tier))}${m.format ? ' · ' + (m.format === 'BO5' ? '3 sets gagnants' : '2 sets gagnants') : ''}</span>
        ${live ? '<span class="live-dot">LIVE</span>' : ''}
      </div>
      <div id="board">${scoreGrid(m, { big: true })}</div>
      <div id="prob-now"></div>
    </header>
    <section class="card">
      <div class="section-head"><h2>Momentum</h2><span class="muted small" id="tape-meta"></span></div>
      <div id="chart">${loading('Point par point…')}</div>
    </section>
    <div class="two-col">
      <section class="card"><h2>Stats du match</h2><div id="stats">${loading()}</div></section>
      <section class="card"><h2>Moments clés</h2><div id="swings">${loading()}</div></section>
    </div>
    <section class="card"><h2>Avant-match</h2><div id="pre">${loading()}</div></section>`;

  const loadTape = async () => {
    try {
      const t = await api.tape(id, live);
      const a = analyse(t?.tape, m);
      const meta = t?.meta || {};
      putText('#tape-meta', `${a.total} états · couverture : ${coverageLabel(meta.coverage)} · ` +
        (a.apiProbs ? `probabilités API sur ${a.apiProbs}/${a.total} points` : 'probabilités calculées par Court Vision (modèle de score)'));
      renderMomentum($('#chart'), a, names);
      put('#stats', statsTable(a.stats, names));
      put('#swings', swingsList(a.stats.swings, names));
      const last = a.pts[a.pts.length - 1];
      if (live && last) put('#prob-now', probBar(last.prob, names, last.fromApi));
    } catch (e) {
      const msg = e instanceof ApiError && e.status === 404 ? '<p class="muted">Pas de point par point disponible pour ce match.</p>' : errorBox(e);
      put('#chart', msg); put('#stats', ''); put('#swings', '');
    }
  };

  loadTape();
  if (p1 && p2) renderPreMatch($('#pre'), p1, p2, m.surface); else put('#pre', '');

  if (live) {
    // Score : à chaque intervalle. Point par point : toutes les 3 minutes (économie de quota).
    every(settings.refreshSeconds, async () => {
      try {
        const fresh = await api.match(id);
        put('#board', scoreGrid(fresh, { big: true }));
        if (fresh.status === 'live' && fresh.score) {
          put('#prob-now', probBar(winProbability(fresh.score, modelOptions(fresh)), names, false));
        }
      } catch { /* on garde l'affichage */ }
    });
    every(Math.max(180, settings.refreshSeconds), loadTape);
  }
}

function coverageLabel(c) {
  return ({ from_start: 'suivi depuis 0-0', partial: 'partielle', reconstructed: 'reconstituée', reconstructed_partial: 'reconstituée partielle', none: 'aucune' })[c] || c || '—';
}

function probBar(p, names, fromApi) {
  const a = Math.round(p * 100);
  return `<div class="prob">
    <div class="prob-labels"><span>${esc(names[0])} <b>${a} %</b></span><span class="muted small">Probabilité de victoire ${fromApi ? '(API)' : '(modèle de score)'}</span><span><b>${100 - a} %</b> ${esc(names[1])}</span></div>
    <div class="prob-bar"><i class="p1" style="width:${a}%"></i><i class="p2" style="width:${100 - a}%"></i></div>
  </div>`;
}

function statsTable(s, names) {
  const row = (label, a, b, fmt = (x) => x, higherIsBetter = true) => {
    const na = typeof a === 'number' ? a : -1, nb = typeof b === 'number' ? b : -1;
    const best = na === nb ? 0 : (na > nb) === higherIsBetter ? 1 : 2;
    return `<tr><td class="${best === 1 ? 'best' : ''}">${fmt(a)}</td><th>${label}</th><td class="${best === 2 ? 'best' : ''}">${fmt(b)}</td></tr>`;
  };
  const sw = [ratio(s.serveWon[0], s.servePts[0]), ratio(s.serveWon[1], s.servePts[1])];
  const rw = [ratio(s.servePts[1] - s.serveWon[1], s.servePts[1]), ratio(s.servePts[0] - s.serveWon[0], s.servePts[0])];
  return `<table class="stats">
    <thead><tr><td><i class="sw sw-p1"></i>${esc(names[0])}</td><th></th><td>${esc(names[1])}<i class="sw sw-p2"></i></td></tr></thead>
    <tbody>
      ${row('Points gagnés', s.points[0], s.points[1])}
      ${row('Points gagnés au service', sw[0], sw[1], pct)}
      ${row('Points gagnés en retour', rw[0], rw[1], pct)}
      ${row('Balles de break obtenues', s.bpChances[0], s.bpChances[1])}
      ${row('Breaks réalisés', s.breaks[0], s.breaks[1])}
      ${row('Balles de break converties', ratio(s.breaks[0], s.bpChances[0]), ratio(s.breaks[1], s.bpChances[1]), pct)}
      ${row('Plus longue série de points', s.bestStreak[0], s.bestStreak[1])}
    </tbody></table>
    <p class="muted small">Calculé à partir du point par point enregistré (peut être incomplet si la couverture est partielle).</p>`;
}

function swingsList(swings, names) {
  if (!swings.length) return '<p class="muted">Pas de variation notable.</p>';
  return `<ol class="swings">${swings.map((x) => {
    const who = x.delta > 0 ? 0 : 1;
    return `<li><span class="delta"><i class="sw sw-p${who + 1}"></i>${esc(names[who])} +${Math.round(Math.abs(x.delta) * 100)} % de probabilité</span>
      <span class="muted small">Point ${x.i} · ${esc(scoreLabel(x.before))} → ${esc(scoreLabel(x.after))}</span></li>`;
  }).join('')}</ol>`;
}

// ---------- Avant-match : H2H + forme + fiches ----------

async function renderPreMatch(el, p1, p2, surface) {
  el.innerHTML = loading('Face-à-face et forme récente…');
  const [pa, pb, h2h, fa, fb] = await Promise.allSettled([
    api.player(p1.id), api.player(p2.id),
    h2hLookup(p1.name, p2.name),
    api.recentForm(p1.id), api.recentForm(p2.id),
  ]);
  const P1 = pa.value || p1, P2 = pb.value || p2;
  const names = [P1.name, P2.name];
  el.innerHTML = `
    <div class="players">
      ${playerCard(P1, 1, fa)}
      ${playerCard(P2, 2, fb)}
    </div>
    <h3>Face-à-face</h3>
    ${h2h.status === 'fulfilled' ? h2hBlock(h2h.value, names, surface) : errorBox(h2h.reason)}
    <details class="career" data-n1="${esc(P1.name)}" data-n2="${esc(P2.name)}">
      <summary>Carrière dans les archives 1968–2022 (2 requêtes)</summary>
      <div class="career-body"></div>
    </details>`;
  const d = el.querySelector('details.career');
  d.addEventListener('toggle', () => { if (d.open && !d.dataset.loaded) { d.dataset.loaded = '1'; loadCareer(d, names); } });
}

// Le H2H prend des fragments de nom : on essaie le nom complet, puis le nom de famille.
async function h2hLookup(n1, n2) {
  const last = (n) => n.split(/[\s,]+/).filter((x) => x.length >= 3 && !x.endsWith('.')).sort((a, b) => b.length - a.length)[0] || n;
  const first = await api.h2h(n1, n2);
  if (first?.players) return first;
  return api.h2h(last(n1), last(n2));
}

function age(b) {
  if (!b) return null;
  const d = new Date(b);
  if (Number.isNaN(+d)) return null;
  return Math.floor((Date.now() - d) / (365.25 * 86400000));
}

function playerCard(p, k, formRes) {
  const a = age(p.birthday);
  const elo = p.stats?.ratings?.elo ?? p.stats?.ratings?.overall;
  const facts = [
    p.ranking ? `Classement <b>#${p.ranking}</b>` : null,
    p.ranking_points ? `${p.ranking_points} pts` : null,
    a ? `${a} ans` : null,
    p.hand ? (p.hand.startsWith('L') ? 'Gaucher' : 'Droitier') : null,
    p.country ? p.country.toUpperCase() : null,
    elo ? `Elo ${Math.round(elo)}` : null,
  ].filter(Boolean);
  return `<div class="pcard pcard-p${k}">
    <div class="pcard-name"><i class="sw sw-p${k}"></i>${esc(p.name)}</div>
    <div class="facts">${facts.map((f) => `<span>${f}</span>`).join('')}</div>
    <div class="form">${formRes?.status === 'fulfilled' ? formStrip(formRes.value, p.id) : '<span class="muted small">Forme indisponible</span>'}</div>
  </div>`;
}

function formStrip(res, pid) {
  const ms = (res?.data || []).filter((m) => m.winner === 1 || m.winner === 2);
  if (!ms.length) return '<span class="muted small">Aucun match terminé récent</span>';
  const items = ms.slice(0, 10).map((m) => {
    const me = m.players?.p1?.id === pid ? 1 : 2;
    const opp = m.players?.[`p${3 - me}`];
    const won = m.winner === me;
    return `<a class="form-dot ${won ? 'w' : 'l'}" href="#/match/${m.id}" title="${won ? 'Victoire' : 'Défaite'} vs ${esc(opp?.name || '?')} · ${esc(m.tournament)}">${won ? 'V' : 'D'}</a>`;
  });
  const w = ms.slice(0, 10).filter((m) => m.winner === (m.players?.p1?.id === pid ? 1 : 2)).length;
  return `<span class="muted small">Forme (${w}/${Math.min(10, ms.length)})</span> ${items.join('')}`;
}

function h2hBlock(h, names, surface) {
  if (!h?.players) return '<p class="muted">Aucune correspondance trouvée pour ces joueurs dans le face-à-face.</p>';
  const t = h.totals || {};
  const tot = (t.p1_wins || 0) + (t.p2_wins || 0);
  if (!tot) return '<p class="muted">Première confrontation.</p>';
  const w1 = Math.round((t.p1_wins / tot) * 100);
  const surf = Object.entries(h.by_surface || {}).filter(([k]) => k !== 'unknown');
  return `
    <div class="h2h-total">
      <span class="big-num">${t.p1_wins}</span>
      <div class="prob-bar"><i class="p1" style="width:${w1}%"></i><i class="p2" style="width:${100 - w1}%"></i></div>
      <span class="big-num">${t.p2_wins}</span>
    </div>
    ${surf.length ? `<div class="h2h-surf">${surf.map(([k, v]) => `
      <span class="${k === surface ? 'hl' : ''}">${esc(SURFACES[k] || k)} <b>${v.p1}-${v.p2}</b>${k === surface ? ' · surface du match' : ''}</span>`).join('')}</div>` : ''}
    <table class="meetings">
      <thead><tr><th>Date</th><th>Tournoi</th><th>Tour</th><th>Surface</th><th>Vainqueur</th><th>Score</th></tr></thead>
      <tbody>${(h.meetings || []).slice(0, 12).map((x) => `
        <tr>
          <td>${esc(x.date || '')}</td><td>${esc(x.tournament || '')}</td><td>${esc(x.round || '')}</td>
          <td>${esc(SURFACES[x.surface] || x.surface || '')}</td>
          <td>${x.winner ? `<i class="sw sw-p${x.winner}"></i>${esc(names[x.winner - 1])}` : '—'}</td>
          <td>${esc(x.score || (x.era === 'current' ? '' : '—'))}</td>
        </tr>`).join('')}</tbody>
    </table>`;
}

async function loadCareer(d, names) {
  const body = d.querySelector('.career-body');
  body.innerHTML = loading();
  const [a, b] = await Promise.allSettled([api.career(names[0]), api.career(names[1])]);
  const col = (r, k) => {
    if (r.status !== 'fulfilled' || !r.value?.record) return `<div class="muted small">${esc(names[k])} : pas de carrière dans l'archive (joueur récent ou nom ambigu).</div>`;
    const c = r.value, rec = c.record, sv = c.serve || {};
    const wl = (x) => (x ? `${x.wins}-${x.losses} (${pct(ratio(x.wins, x.wins + x.losses))})` : '—');
    return `<div class="pcard pcard-p${k + 1}">
      <div class="pcard-name"><i class="sw sw-p${k + 1}"></i>${esc(c.player?.name || names[k])}</div>
      <div class="muted small">${esc(c.span?.first?.slice(0, 4) || '')}–${esc(c.span?.last?.slice(0, 4) || '')}</div>
      <table class="kv"><tbody>
        <tr><th>Bilan</th><td>${wl(rec)}</td></tr>
        <tr><th>Titres</th><td>${rec.titles ?? '—'}</td></tr>
        ${Object.entries(rec.by_surface || {}).filter(([s]) => SURFACES[s]).map(([s, v]) => `<tr><th>${SURFACES[s]}</th><td>${wl(v)}</td></tr>`).join('')}
        <tr><th>1re balle</th><td>${pct(sv.first_in_pct)}</td></tr>
        <tr><th>Pts gagnés 1re</th><td>${pct(sv.first_won_pct)}</td></tr>
        <tr><th>Pts gagnés 2e</th><td>${pct(sv.second_won_pct)}</td></tr>
      </tbody></table>
    </div>`;
  };
  body.innerHTML = `<div class="players">${col(a, 0)}${col(b, 1)}</div>`;
}

// ---------- Programme ----------

async function fixturesView() {
  const { tour } = params();
  view.innerHTML = `
    <header class="page-head"><div><h1>Programme</h1><p class="muted">Matchs à venir — clique pour l'avant-match.</p></div>${tourChips(tour, '#/fixtures')}</header>
    <div id="list">${loading()}</div>`;
  try {
    const res = await api.fixtures(tour);
    const fx = (res.data || []).filter((f) => !/\//.test(`${f.player1_name}${f.player2_name}`));
    if (!fx.length) { put('#list', '<div class="empty">Aucun match programmé pour ce filtre.</div>'); return; }
    const days = new Map();
    for (const f of fx) {
      const d = f.event_date || (f.start_time || '').slice(0, 10) || 'Date à venir';
      if (!days.has(d)) days.set(d, new Map());
      const t = days.get(d);
      const k = f.tournament || 'Tournoi';
      if (!t.has(k)) t.set(k, []);
      t.get(k).push(f);
    }
    put('#list', [...days.entries()].map(([d, ts]) => `
      <section class="group">
        <h2 class="day">${esc(/\d{4}-/.test(d) ? fmtDay(d + 'T12:00:00Z') : d)}</h2>
        ${[...ts.entries()].map(([t, fs]) => `
          <div class="card fixture-group">
            <h3>${esc(t)} ${surfaceChip(fs[0].surface)}</h3>
            ${fs.map(fixtureRow).join('')}
          </div>`).join('')}
      </section>`).join(''));
  } catch (e) { put('#list', errorBox(e)); }
}

function fixtureRow(f) {
  const q = new URLSearchParams({ id1: f.player1_id ?? '', id2: f.player2_id ?? '', n1: f.player1_name ?? '', n2: f.player2_name ?? '', s: f.surface ?? '', t: f.tournament ?? '' });
  const canPreview = f.player1_id && f.player2_id;
  const inner = `<span class="time">${f.start_time ? fmtTime(f.start_time) : '—'}</span>
    <span class="vs"><b>${esc(f.player1_name || '?')}</b> <span class="muted">vs</span> <b>${esc(f.player2_name || '?')}</b></span>
    <span class="muted small">${esc(ROUNDS[f.round_code] || f.round || '')}</span>`;
  return canPreview ? `<a class="fixture" href="#/preview?${q}">${inner}</a>` : `<div class="fixture">${inner}</div>`;
}

// ---------- Résultats ----------

async function resultsView() {
  const p = params();
  const day = p.day || today();
  const tour = p.tour || '';
  const shift = (n) => { const d = new Date(day + 'T12:00:00Z'); d.setUTCDate(d.getUTCDate() + n); return d.toISOString().slice(0, 10); };
  const link = (d, t = tour) => `#/results?${new URLSearchParams({ day: d, ...(t ? { tour: t } : {}) })}`;
  view.innerHTML = `
    <header class="page-head">
      <div><h1>Résultats</h1><p class="muted">Matchs terminés — ouvre un match pour son replay point par point.</p></div>
      <div class="chips">${TOURS.map(([v, l]) => `<a class="chip-btn ${v === tour ? 'on' : ''}" href="${link(day, v)}">${l}</a>`).join('')}</div>
    </header>
    <div class="daynav">
      <a class="btn" href="${link(shift(-1))}">← Veille</a>
      <input type="date" id="day" value="${day}" max="${today()}">
      <a class="btn ${day >= today() ? 'disabled' : ''}" href="${link(shift(1))}">Lendemain →</a>
    </div>
    <div id="list">${loading()}</div>`;
  $('#day').addEventListener('change', (e) => { location.hash = link(e.target.value); });
  try {
    const res = await api.results({ from: day, to: day, tour, draw: 'singles', limit: 200 });
    const ms = res.data || [];
    put('#list', ms.length ? groupByTournament(ms).map(([t, list]) => `
      <section class="group">
        <h2>${esc(t)} ${surfaceChip(list[0].surface)}</h2>
        <div class="cards">${list.map((m) => `<a class="card match-card" href="#/match/${m.id}">
          <div class="card-top"><span>${esc(round(m))}</span><span class="muted small">${m.tape?.model_rows ? 'Momentum API' : m.tape?.rows || m.tape?.reconstructed_rows ? 'Point par point' : ''}</span></div>
          ${scoreGrid(m)}
        </a>`).join('')}</div>
      </section>`).join('') : '<div class="empty">Aucun résultat ce jour-là.</div>');
  } catch (e) { put('#list', errorBox(e)); }
}

// ---------- Face-à-face (recherche) et avant-match ----------

function h2hSearchView() {
  view.innerHTML = `
    <header class="page-head"><div><h1>Face-à-face</h1><p class="muted">Choisis deux joueurs pour comparer H2H, forme et fiches.</p></div></header>
    <form class="card h2h-form" id="h2hf">
      ${[1, 2].map((k) => `
        <label class="field"><span><i class="sw sw-p${k}"></i>Joueur ${k}</span>
          <input autocomplete="off" name="q${k}" placeholder="Nom (3 lettres min.)" required>
          <input type="hidden" name="id${k}">
          <ul class="suggest" data-k="${k}"></ul>
        </label>`).join('')}
      <button class="btn primary" type="submit">Comparer</button>
    </form>`;
  const f = $('#h2hf');
  for (const k of [1, 2]) {
    const input = f[`q${k}`], list = f.querySelector(`.suggest[data-k="${k}"]`);
    let t;
    input.addEventListener('input', () => {
      f[`id${k}`].value = '';
      clearTimeout(t);
      const q = input.value.trim();
      if (q.length < 3) { list.innerHTML = ''; return; }
      t = setTimeout(async () => {
        try {
          const res = await api.players(q);
          list.innerHTML = (res.data || []).filter((p) => !p.is_doubles_team).map((p) =>
            `<li><button type="button" data-id="${p.id}" data-name="${esc(p.name)}">${playerLine(p)}</button></li>`).join('') || '<li class="muted small">Aucun joueur</li>';
        } catch (e) { list.innerHTML = `<li>${errorBox(e)}</li>`; }
      }, 450);
    });
    list.addEventListener('click', (e) => {
      const b = e.target.closest('button[data-id]');
      if (!b) return;
      input.value = b.dataset.name; f[`id${k}`].value = b.dataset.id; list.innerHTML = '';
    });
  }
  f.addEventListener('submit', (e) => {
    e.preventDefault();
    if (!f.id1.value || !f.id2.value) { alert('Sélectionne chaque joueur dans la liste de suggestions.'); return; }
    location.hash = `#/preview?${new URLSearchParams({ id1: f.id1.value, id2: f.id2.value, n1: f.q1.value, n2: f.q2.value })}`;
  });
}

function previewView() {
  const p = params();
  view.innerHTML = `
    <a class="back" href="#/fixtures">← Programme</a>
    <header class="page-head"><div>
      <h1>${esc(p.n1)} <span class="muted">vs</span> ${esc(p.n2)}</h1>
      <p class="muted">${esc(p.t || '')} ${surfaceChip(p.s)}</p>
    </div></header>
    <section class="card" id="pre"></section>`;
  renderPreMatch($('#pre'), { id: +p.id1, name: p.n1 }, { id: +p.id2, name: p.n2 }, p.s);
}

// ---------- Réglages ----------

async function settingsView() {
  const mode = settings.mode;
  const srv = settings.server;
  const modeText = {
    proxy: '✅ Clé API trouvée côté serveur (variable d\'environnement Vercel). Rien à saisir.',
    direct: '✅ Clé API enregistrée dans ce navigateur.',
    demo: 'Mode démo : aucune clé trouvée. Ajoute la variable <code>LIVETENNISAPI_KEY</code> au projet Vercel, ou colle ta clé ci-dessous.',
  }[mode];
  view.innerHTML = `
    <header class="page-head"><div><h1>Réglages</h1></div></header>
    <section class="card"><h2>Connexion à l'API</h2><p>${modeText}</p></section>
    ${srv.needsCode ? `
    <form class="card" id="codef">
      <h2>Code d'accès</h2>
      <p class="muted small">Ce site est protégé par un code d'accès défini sur Vercel. Il est demandé une seule fois par appareil.</p>
      <label class="field"><span>Code</span>
        <input type="password" name="code" value="${esc(settings.accessCode)}" autocomplete="off"></label>
      <button class="btn primary" type="submit">Enregistrer</button>
    </form>` : ''}
    <form class="card" id="keyf">
      <h2>${mode === 'proxy' ? 'Autre clé (optionnel)' : 'Clé API'}</h2>
      <p class="muted small">${mode === 'proxy' ? 'Inutile ici : la clé du serveur est déjà utilisée. Une clé saisie ici la remplace pour ce navigateur uniquement.' : 'Ta clé reste dans ce navigateur (localStorage) et n\'est envoyée qu\'à api.livetennisapi.com.'}</p>
      <label class="field"><span>Clé Live Tennis API</span>
        <input type="password" name="key" value="${esc(settings.apiKey)}" placeholder="lta_…" autocomplete="off"></label>
      <div class="row">
        <button class="btn primary" type="submit">Enregistrer</button>
        <button class="btn" type="button" id="clear">Effacer</button>
      </div>
    </form>
    <form class="card" id="reff">
      <h2>Actualisation en direct</h2>
      <p class="muted small">Le plan Basic autorise 1 000 requêtes par jour. Un match suivi 2 h à 60 s consomme environ 160 requêtes.</p>
      <label class="field"><span>Intervalle</span>
        <select name="r">${[30, 60, 90, 120, 300].map((s) => `<option value="${s}" ${s === settings.refreshSeconds ? 'selected' : ''}>${s} s</option>`).join('')}</select></label>
    </form>
    <section class="card"><h2>Quota</h2><div id="usage">${loading()}</div></section>`;
  $('#keyf').addEventListener('submit', (e) => { e.preventDefault(); settings.apiKey = e.target.key.value; refreshChrome(); settingsView(); });
  $('#clear').addEventListener('click', () => { settings.apiKey = ''; refreshChrome(); settingsView(); });
  $('#codef')?.addEventListener('submit', (e) => { e.preventDefault(); settings.accessCode = e.target.code.value; settingsView(); });
  $('#reff').r.addEventListener('change', (e) => { settings.refreshSeconds = +e.target.value; });
  try {
    const u = await api.usage();
    const lim = u.limits || {};
    put('#usage', `<table class="kv"><tbody>
      <tr><th>Plan</th><td>${esc((u.tier || '').toUpperCase())}${settings.demo ? ' (démo)' : ''}</td></tr>
      <tr><th>Limites</th><td>${lim.per_minute ?? '—'}/min · ${lim.per_day ?? '—'}/jour</td></tr>
      <tr><th>Appels aujourd'hui (serveur)</th><td>${u.today?.calls ?? '—'}</td></tr>
      <tr><th>Restant aujourd'hui</th><td>${u.today?.remaining_day ?? '—'}</td></tr>
      <tr><th>Appels depuis ce navigateur</th><td>${localCallsToday()}</td></tr>
    </tbody></table>`);
  } catch (e) { put('#usage', errorBox(e)); }
}

// ---------- Routeur ----------

const routes = [
  [/^#\/live/, liveView],
  [/^#\/match\/(\d+)/, (m) => matchView(+m[1])],
  [/^#\/fixtures/, fixturesView],
  [/^#\/results/, resultsView],
  [/^#\/h2h/, h2hSearchView],
  [/^#\/preview/, previewView],
  [/^#\/settings/, settingsView],
];

function refreshChrome() {
  $('#demo-banner').hidden = !settings.demo;
  putText('#calls', settings.demo ? 'Démo' : `${localCallsToday()} req. aujourd'hui`);
  const h = location.hash || '#/live';
  document.querySelectorAll('.nav a').forEach((a) => a.classList.toggle('on', h.startsWith(a.getAttribute('href'))));
}

function render() {
  clearTimers();
  const h = location.hash || '#/live';
  refreshChrome();
  for (const [re, fn] of routes) {
    const m = h.match(re);
    if (m) { fn(m); window.scrollTo(0, 0); return; }
  }
  location.hash = '#/live';
}

window.addEventListener('hashchange', render);
window.addEventListener('cv:call', refreshChrome);
initMode().then(render);

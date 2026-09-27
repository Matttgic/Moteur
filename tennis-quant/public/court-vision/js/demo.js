// Données fictives (mode démo, sans clé API). Même forme que les réponses de l'API.
import { winProbability } from './model.js';

function rng(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const PLAYERS = [
  [101, 'Lucas Moreau', 'fra', 4, 'atp', 'R'], [102, 'Mateo Ruiz', 'esp', 2, 'atp', 'R'],
  [103, 'Henrik Lund', 'nor', 9, 'atp', 'R'], [104, 'Alessio Conti', 'ita', 1, 'atp', 'R'],
  [105, 'Daniil Orlov', 'kaz', 14, 'atp', 'R'], [106, 'Tom Whitaker', 'usa', 6, 'atp', 'L'],
  [107, 'Kenji Mori', 'jpn', 27, 'atp', 'R'], [108, 'Felix Brandt', 'ger', 11, 'atp', 'R'],
  [109, 'Nico Varga', 'hun', 45, 'atp', 'L'], [110, 'Rafael Costa', 'bra', 33, 'atp', 'R'],
  [201, 'Elena Petrova', 'bul', 3, 'wta', 'R'], [202, 'Chloé Martin', 'fra', 12, 'wta', 'R'],
  [203, 'Maya Johnson', 'usa', 1, 'wta', 'R'], [204, 'Sofia Rinaldi', 'ita', 7, 'wta', 'L'],
  [205, 'Aiko Tanaka', 'jpn', 19, 'wta', 'R'], [206, 'Lena Novak', 'cze', 5, 'wta', 'R'],
  [207, 'Iga Kowal', 'pol', 22, 'wta', 'R'], [208, 'Nora Svensson', 'swe', 38, 'wta', 'R'],
].map(([id, name, country, ranking, tour, hand]) => ({
  id, name, country, ranking, tour, hand,
  ranking_points: Math.round(9000 / Math.sqrt(ranking)),
  ranking_movement: null,
  birthday: `${1996 + (id % 9)}-0${1 + (id % 9)}-1${id % 10}`,
  backhand: 2,
  is_doubles_team: false,
}));
const byId = new Map(PLAYERS.map((p) => [p.id, p]));
const serveStrength = (p) => 0.6 + 0.08 * (1 - Math.log(p.ranking) / Math.log(60));

const TOURNAMENTS = [
  { name: 'Tokyo Open', tour: 'atp', tier: 'atp_500', surface: 'hard', indoor: false },
  { name: 'Beijing Masters', tour: 'atp', tier: 'atp_1000', surface: 'hard', indoor: false },
  { name: 'Wuhan Open', tour: 'wta', tier: 'wta_1000', surface: 'hard', indoor: false },
  { name: 'Lyon Challenger', tour: 'challenger', tier: 'challenger_100', surface: 'hard', indoor: true },
];

// ---------- Simulation point par point ----------

function pointStr(a, b) {
  if (a >= 3 && b >= 3) {
    if (a === b) return ['40', '40'];
    return a > b ? ['AD', '40'] : ['40', 'AD'];
  }
  const m = ['0', '15', '30', '40'];
  return [m[a], m[b]];
}

function tbServer(first, n) {
  if (n === 0) return first;
  return Math.floor((n - 1) / 2) % 2 === 0 ? 3 - first : first;
}

function simulate(seed, pA, pB, bestOf, startMs) {
  const r = rng(seed);
  const need = bestOf === 5 ? 3 : 2;
  const sets = [0, 0];
  const games = [[0], [0]];
  let pts = [0, 0];
  let server = r() < 0.5 ? 1 : 2;
  let tb = false, tbFirst = 1;
  const rows = [];
  let t = startMs;
  const snap = (winner) => {
    const row = {
      origin: 'observed',
      sets: [...sets],
      games: [games[0].slice(), games[1].slice()],
      points: tb ? pts.map(String) : pointStr(pts[0], pts[1]),
      server: tb ? tbServer(tbFirst, pts[0] + pts[1]) : server,
      is_tiebreak: tb,
      winner, point_winner: winner,
      danger: null,
      timestamp: new Date(t).toISOString(),
      sequence: rows.length + 1,
    };
    const exact = winProbability(row, { pA, pB, bestOf });
    row.win_probability_p1 = Math.min(0.999, Math.max(0.001, exact + (r() - 0.5) * 0.03));
    rows.push(row);
  };
  snap(null);
  while (sets[0] < need && sets[1] < need) {
    const srv = tb ? tbServer(tbFirst, pts[0] + pts[1]) : server;
    const p1Wins = r() < (srv === 1 ? pA : 1 - pB);
    const w = p1Wins ? 0 : 1;
    pts[w] += 1;
    t += 25000 + r() * 30000;
    const idx = sets[0] + sets[1];
    const target = tb ? 7 : 4;
    const gameOver = pts[w] >= target && pts[w] - pts[1 - w] >= 2;
    if (gameOver) {
      games[w][idx] += 1;
      pts = [0, 0];
      const g = [games[0][idx], games[1][idx]];
      const setOver = tb || (g[w] >= 6 && g[w] - g[1 - w] >= 2);
      if (tb) { server = 3 - tbFirst; tb = false; } else server = 3 - server;
      if (setOver) {
        sets[w] += 1;
        t += 90000;
        if (sets[0] < need && sets[1] < need) { games[0].push(0); games[1].push(0); }
      } else if (g[0] === 6 && g[1] === 6) {
        tb = true; tbFirst = server;
      }
      t += 20000;
    }
    snap(w + 1);
  }
  return rows;
}

// ---------- Construction des matchs ----------

function scoreFromRow(row) {
  return {
    sets: row.sets, games: row.games, points: row.points, server: row.server,
    is_tiebreak: row.is_tiebreak, win_probability_p1: null, danger: null,
    timestamp: row.timestamp, stale: false,
  };
}

function buildMatch(id, p1, p2, t, status, rows, scheduled, round) {
  const last = rows[rows.length - 1];
  const bestOf = t.tier === 'grand_slam' ? 5 : 3;
  const need = bestOf === 5 ? 3 : 2;
  const winner = status === 'completed' ? (last.sets[0] >= need ? 1 : 2) : null;
  return {
    id, tournament: t.name, tour: t.tour, tier: t.tier, surface: t.surface, indoor: t.indoor,
    format: bestOf === 5 ? 'BO5' : 'BO3', round, draw: 'singles', is_doubles: false,
    status, outcome: status === 'completed' ? 'completed' : null,
    scheduled_time: new Date(scheduled).toISOString(),
    players: { p1: byId.get(p1), p2: byId.get(p2) },
    score: status === 'upcoming' ? null : scoreFromRow(last),
    winner, has_analysis: false, has_market: false,
  };
}

const ROUNDS = ['Round of 32', 'Round of 16', 'Quarterfinals', 'Semifinals'];
const MATCH_POOL = new Map(); // id -> { match, rows }

function pairFor(r, tour, used = new Set()) {
  let pool = PLAYERS.filter((p) => (tour === 'wta' ? p.tour === 'wta' : p.tour === 'atp'));
  if (pool.filter((p) => !used.has(p.id)).length >= 2) pool = pool.filter((p) => !used.has(p.id));
  const a = pool[Math.floor(r() * pool.length)];
  let b;
  do { b = pool[Math.floor(r() * pool.length)]; } while (b.id === a.id);
  used.add(a.id); used.add(b.id);
  return [a, b];
}

function ensurePool() {
  if (MATCH_POOL.size) return;
  const now = Date.now();
  const day = Math.floor(now / 86400000);
  const r = rng(day);
  // Terminés : 24 matchs sur les 3 derniers jours.
  for (let i = 0; i < 24; i++) {
    const t = TOURNAMENTS[i % TOURNAMENTS.length];
    const [a, b] = pairFor(r, t.tour);
    const start = now - (3 + i * 3) * 3600000;
    const rows = simulate(day * 100 + i, serveStrength(a), serveStrength(b), 3, start);
    const id = 900000 + i;
    MATCH_POOL.set(id, { match: buildMatch(id, a.id, b.id, t, 'completed', rows, start, ROUNDS[i % 4]), rows });
  }
  // En direct : 5 matchs commencés il y a 10 à 70 min, joueurs tous différents.
  const busy = new Set();
  for (let i = 0; i < 5; i++) {
    const t = TOURNAMENTS[i % TOURNAMENTS.length];
    const [a, b] = pairFor(r, t.tour, busy);
    const start = now - (10 + i * 15) * 60000;
    const rows = simulate(day * 100 + 50 + i, serveStrength(a), serveStrength(b), 3, start);
    const id = 950000 + i;
    MATCH_POOL.set(id, { match: buildMatch(id, a.id, b.id, t, 'live', rows, start, ROUNDS[i % 4]), rows, live: true });
  }
}

// Un match en direct avance d'un point toutes les ~20 s de temps réel.
function liveView(entry) {
  const start = Date.parse(entry.match.scheduled_time);
  const played = Math.floor((Date.now() - start) / 20000);
  const n = Math.max(2, Math.min(entry.rows.length - 1, played));
  const rows = entry.rows.slice(0, n);
  return { match: { ...entry.match, score: scoreFromRow(rows[rows.length - 1]) }, rows };
}

function view(id) {
  const e = MATCH_POOL.get(id);
  if (!e) return null;
  return e.live ? liveView(e) : { match: e.match, rows: e.rows };
}

function list(data, params) {
  const limit = +params.limit || 50, offset = +params.offset || 0;
  const page = data.slice(offset, offset + limit);
  return { data: page, meta: { limit, offset, count: page.length, total: data.length, has_more: offset + limit < data.length } };
}

const tourOk = (m, tour) => !tour || m.tour === tour;
const tierOk = (m, tier) => !tier || String(tier).split(',').includes(m.tier);

function finalScoreString(match, p1Perspective = true) {
  const g = match.score.games;
  return g[0].map((x, i) => (p1Perspective ? `${x}-${g[1][i]}` : `${g[1][i]}-${x}`)).join(' ');
}

function h2hFor(n1, n2) {
  const a = PLAYERS.find((p) => p.name.toLowerCase().includes(n1.toLowerCase()));
  const b = PLAYERS.find((p) => p.name.toLowerCase().includes(n2.toLowerCase()));
  if (!a || !b || a.id === b.id) return { players: null, totals: { p1_wins: 0, p2_wins: 0, meetings: 0, undecided: 0 }, by_surface: {}, meetings: [] };
  const r = rng(a.id * 1000 + b.id);
  const n = 2 + Math.floor(r() * 8);
  const edge = serveStrength(a) - serveStrength(b);
  const meetings = [];
  const by = {};
  let w1 = 0, w2 = 0;
  for (let i = 0; i < n; i++) {
    const surface = ['hard', 'clay', 'grass'][Math.floor(r() * 3.3) % 3];
    const winner = r() < 0.5 + edge * 4 ? 1 : 2;
    winner === 1 ? w1++ : w2++;
    by[surface] = by[surface] || { p1: 0, p2: 0 };
    by[surface][winner === 1 ? 'p1' : 'p2'] += 1;
    const date = new Date(Date.now() - (30 + i * 150 + Math.floor(r() * 60)) * 86400000).toISOString().slice(0, 10);
    const year = +date.slice(0, 4);
    meetings.push({
      era: year >= 2023 ? 'current' : 'archive',
      date,
      tournament: ['Paris', 'Rome', 'Wimbledon', 'Miami', 'Cincinnati', 'Madrid', 'Vienna'][Math.floor(r() * 7)],
      level: ['M', 'G', 'A'][Math.floor(r() * 3)],
      round: ROUNDS[Math.floor(r() * 4)],
      surface,
      score: year < 2023 ? (winner === 1 ? '6-4 7-6(5)' : '6-3 3-6 6-4') : null,
      outcome: 'completed',
      winner,
    });
  }
  meetings.sort((x, y) => (x.date < y.date ? 1 : -1));
  return { players: { p1: { name: a.name }, p2: { name: b.name } }, totals: { p1_wins: w1, p2_wins: w2, meetings: n, undecided: 0 }, by_surface: by, meetings };
}

function demoFixtures() {
  const r = rng(Math.floor(Date.now() / 86400000) + 7);
  const out = [];
  for (let i = 0; i < 16; i++) {
    const t = TOURNAMENTS[i % TOURNAMENTS.length];
    const [a, b] = pairFor(r, t.tour);
    const start = new Date(Date.now() + (1 + i * 1.5) * 3600000);
    start.setMinutes(0, 0, 0);
    out.push({
      id: 980000 + i, event_date: start.toISOString().slice(0, 10), start_time: start.toISOString(),
      player1_id: a.id, player2_id: b.id, player1_name: a.name, player2_name: b.name,
      tour: t.tour, tournament: t.name, round: ROUNDS[i % 4], surface: t.surface, status: 'scheduled',
    });
  }
  return out;
}

// Faux instantané du moteur (même forme que /api/selections/quality-today).
export function demoPicks(tour) {
  const r = rng(Math.floor(Date.now() / 86400000) + 11);
  const margin = 1.06;
  const rows = demoFixtures().filter((f) => f.tour === tour).map((f, i) => {
    const a = byId.get(f.player1_id), b = byId.get(f.player2_id);
    const pa = Math.min(0.9, Math.max(0.1, 0.5 + (b.ranking - a.ranking) / 60 + (r() - 0.5) * 0.1));
    const oddsA = i % 4 === 1 ? +(1 / (pa * 0.86)).toFixed(2) : +(1 / ((pa - 0.04 + r() * 0.08) * margin)).toFixed(2);
    const oddsB = +(1 / ((1 - pa + 0.02) * margin)).toFixed(2);
    const side = pa * oddsA >= (1 - pa) * oddsB ? 'A' : 'B';
    const prob = side === 'A' ? pa : 1 - pa;
    const odds = side === 'A' ? oddsA : oddsB;
    const edge = prob - 1 / odds / margin;
    const ev = prob * odds - 1;
    const tier = edge >= 0.08 && ev >= 0.05 ? 'PREMIUM' : edge >= 0.06 && ev >= 0.03 ? 'VALUE' : edge >= 0.03 && ev > 0 ? 'LEAN' : 'NO_BET';
    const score = i % 4 === 1 ? 84 : Math.round(55 + r() * 40);
    return {
      matchId: f.id, tournament: f.tournament, surface: f.surface, scheduledTime: f.start_time,
      playerA: { name: a.name, ranking: a.ranking }, playerB: { name: b.name, ranking: b.ranking },
      model: {
        mode: 'full_logit', probabilityA: pa, probabilityB: 1 - pa, fairOddsA: 1 / pa, fairOddsB: 1 / (1 - pa), quality: 0.8,
        features: {
          elo_diff: (b.ranking - a.ranking) * 6, surface_elo_diff: (b.ranking - a.ranking) * 5,
          hold_diff: (r() - 0.5) * 0.08, break_diff: (r() - 0.5) * 0.06, form10_diff: (r() - 0.5) * 0.4,
          load14_diff: Math.round((r() - 0.5) * 4),
        },
      },
      market: { bookmaker: ['Winamax', 'Betclic', 'Unibet'][i % 3], oddsA, oddsB, sourceCount: 4 },
      decision: {
        side, player: side === 'A' ? a.name : b.name, odds, modelProbability: prob, edge, ev, fairOdds: 1 / prob, tier,
        bet: (tier === 'PREMIUM' || tier === 'VALUE') && score >= 70,
        guard: { blocked: false, reasons: i % 5 === 4 ? ['cross_book_price_outlier'] : [] },
        quality: { score, grade: score >= 90 ? 'A+' : score >= 80 ? 'A' : score >= 70 ? 'B' : score >= 60 ? 'C' : 'D', actionable: score >= 70 },
      },
    };
  });
  return { allAnalyzed: rows, snapshot: { generatedAt: new Date(Date.now() - 42 * 60000).toISOString(), ageMinutes: 42, stale: false } };
}

// Historique synthétique d'un joueur : 80 matchs terminés sur ~2 ans.
const HISTORY = new Map();
function playerHistory(pid) {
  if (HISTORY.has(pid)) return HISTORY.get(pid);
  const me = byId.get(pid);
  if (!me) return [];
  const r = rng(pid * 7919);
  const out = [];
  let t = Date.now() - (1 + Math.floor(r() * 4)) * 86400000;
  for (let i = 0; i < 80; i++) {
    const pool = PLAYERS.filter((p) => p.tour === me.tour && p.id !== pid);
    const opp = pool[Math.floor(r() * pool.length)];
    const surface = ['hard', 'hard', 'clay', 'grass'][Math.floor(r() * 4)];
    const rows = simulate(pid * 1000 + i, serveStrength(me), serveStrength(opp), 3, t);
    const flip = r() < 0.5;
    const tt = TOURNAMENTS[i % 3];
    const m = buildMatch(990000 + pid * 100 + i, flip ? opp.id : pid, flip ? pid : opp.id, { ...tt, surface }, 'completed', rows, t, ROUNDS[i % 4]);
    if (flip) {
      // La simulation est faite du point de vue du joueur : on inverse pour le mettre en p2.
      const g = m.score.games;
      m.score = { ...m.score, games: [g[1], g[0]], sets: [m.score.sets[1], m.score.sets[0]] };
      m.winner = 3 - m.winner;
    }
    out.push({ ...m, tape: { coverage: 'from_start', rows: rows.length, model_rows: rows.length } });
    t -= (i % 4 === 3 ? 9 : 2) * 86400000;
  }
  HISTORY.set(pid, out);
  return out;
}

export async function demoFetch(path, params = {}) {
  await new Promise((res) => setTimeout(res, 120));
  ensurePool();
  const all = [...MATCH_POOL.keys()].map(view);
  let m;

  if (path === '/matches') {
    const status = params.status || 'live';
    return list(all.filter((v) => v.match.status === status && tourOk(v.match, params.tour) && tierOk(v.match, params.tier)).map((v) => v.match), params);
  }
  if (path === '/fixtures') {
    return list(demoFixtures().filter((f) => !params.tour || f.tour === params.tour), params);
  }
  if ((m = path.match(/^\/matches\/(\d+)\/score$/))) {
    const v = view(+m[1]);
    return v?.match.score;
  }
  if ((m = path.match(/^\/matches\/(\d+)$/))) return view(+m[1])?.match;
  if (path === '/players') {
    const q = (params.search || '').toLowerCase();
    return list(PLAYERS.filter((p) => p.name.toLowerCase().includes(q)), params);
  }
  if ((m = path.match(/^\/players\/(\d+)$/))) {
    const p = byId.get(+m[1]);
    return p && { ...p, stats: { ratings: { elo: Math.round(2300 - p.ranking * 8) }, ratings_as_of: new Date().toISOString().slice(0, 10), season: null } };
  }
  if (path === '/h2h') return h2hFor(params.p1 || '', params.p2 || '');
  if (path === '/history/matches') {
    let rows = all.filter((v) => v.match.status === 'completed' && tourOk(v.match, params.tour) && tierOk(v.match, params.tier));
    const pl = [].concat(params.player || []).map(Number);
    if (pl.length) return list(pl.flatMap(playerHistory), params);
    const data = rows.map((v) => ({ ...v.match, tape: { coverage: 'from_start', rows: v.rows.length, reconstructed_rows: 0, model_rows: v.rows.length, points_complete: true } }));
    return list(data, params);
  }
  if ((m = path.match(/^\/history\/matches\/(\d+)$/))) {
    const v = view(+m[1]);
    if (!v) return null;
    return { match: v.match, tape: v.rows, tiebreaks: null, profiles: [], meta: { match_id: v.match.id, rows: v.rows.length, coverage: 'from_start', point_source: 'observed', sequence: 'clean' } };
  }
  if (path === '/usage') {
    return { tier: 'basic', base_tier: 'basic', limits: { per_minute: 60, per_day: 1000 }, today: { calls: 0, errors: 0, remaining_day: 1000 }, as_of: new Date().toISOString() };
  }
  return null;
}

export { finalScoreString };

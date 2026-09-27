// Client Live Tennis API — cache, suivi de quota, mode démo.
import { demoFetch, demoPicks } from './demo.js';

const BASE = 'https://api.livetennisapi.com/api/public/v1';
const LS_KEY = 'cv.apiKey';
const LS_CALLS = 'cv.calls';
const LS_CODE = 'cv.accessCode';

const memory = new Map();

// Mode de fonctionnement :
//  - 'proxy'  : la clé est côté serveur (variable d'environnement Vercel), appels via /api/lt
//  - 'direct' : la clé est saisie dans Réglages et stockée dans ce navigateur
//  - 'demo'   : aucune clé, données fictives
let serverStatus = { configured: false, needsCode: false };
// Adresse du relais serveur : <meta name="cv-proxy"> dans la page, sinon api/lt (déploiement autonome).
const PROXY = document.querySelector('meta[name="cv-proxy"]')?.content || 'api/lt';
// Picks du moteur (lecture de l'instantané déjà calculé, aucun appel Live Tennis API).
const PICKS = document.querySelector('meta[name="cv-picks"]')?.content || '';

export async function initMode() {
  try {
    const res = await fetch(`${PROXY}?path=__status`, { cache: 'no-store' });
    if (res.ok) serverStatus = await res.json();
  } catch { /* pas de fonction serveur (serveur local statique) */ }
}

function lsGet(k) { try { return localStorage.getItem(k); } catch { return null; } }
function lsSet(k, v) { try { localStorage.setItem(k, v); } catch { /* stockage indisponible */ } }
function lsDel(k) { try { localStorage.removeItem(k); } catch { /* idem */ } }

export const settings = {
  get apiKey() { return lsGet(LS_KEY) || ''; },
  set apiKey(v) { v ? lsSet(LS_KEY, v.trim()) : lsDel(LS_KEY); memory.clear(); },
  get accessCode() { return lsGet(LS_CODE) || ''; },
  set accessCode(v) { v ? lsSet(LS_CODE, v.trim()) : lsDel(LS_CODE); memory.clear(); },
  get server() { return serverStatus; },
  get mode() { return this.apiKey ? 'direct' : serverStatus.configured ? 'proxy' : 'demo'; },
  get demo() { return this.mode === 'demo'; },
  get refreshSeconds() { return parseInt(lsGet('cv.refresh') || '60', 10); },
  set refreshSeconds(v) { lsSet('cv.refresh', String(v)); },
};

export class ApiError extends Error {
  constructor(status, body) {
    const code = body?.error || `http_${status}`;
    super(messageFor(status, code, body));
    this.status = status;
    this.code = code;
  }
}

function messageFor(status, code, body) {
  if (code === 'upgrade_required') return 'Cette donnée demande un plan supérieur au Basic.';
  if (code === 'access_code') return 'Code d\'accès requis ou incorrect : saisis-le dans Réglages.';
  if (code === 'quota_reserved') return body?.message || 'Quota du jour presque épuisé : le reste est réservé au moteur.';
  if (code === 'not_configured') return 'Aucune clé API trouvée dans les variables d\'environnement Vercel du projet.';
  if (status === 401) return 'Clé API invalide ou absente. Vérifie-la dans Réglages.';
  if (status === 429) return 'Limite de requêtes atteinte (60/min ou 1 000/jour). Réessaie plus tard.';
  if (status === 404) return 'Introuvable.';
  return body?.message || body?.detail || `Erreur API (${status}).`;
}

// Compteur local des appels réseau du jour (complément de /usage).
function countCall() {
  const today = new Date().toISOString().slice(0, 10);
  let c = {};
  try { c = JSON.parse(lsGet(LS_CALLS) || '{}'); } catch { /* reset */ }
  if (c.day !== today) c = { day: today, n: 0 };
  c.n += 1;
  lsSet(LS_CALLS, JSON.stringify(c));
  window.dispatchEvent(new CustomEvent('cv:call', { detail: c.n }));
}

export function localCallsToday() {
  try {
    const c = JSON.parse(lsGet(LS_CALLS) || '{}');
    return c.day === new Date().toISOString().slice(0, 10) ? c.n : 0;
  } catch { return 0; }
}

function buildQuery(params = {}) {
  const q = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v == null || v === '') continue;
    if (Array.isArray(v)) v.forEach((x) => q.append(k, x));
    else q.append(k, v);
  }
  const s = q.toString();
  return s ? `?${s}` : '';
}

/**
 * GET avec cache. ttl en secondes ; persist=true garde la réponse dans localStorage
 * (données qui bougent peu : joueurs, H2H, matchs terminés).
 */
export async function get(path, params = {}, { ttl = 30, persist = false } = {}) {
  const url = path + buildQuery(params);
  const now = Date.now();
  const hit = memory.get(url);
  if (hit && hit.exp > now) return hit.data;
  if (persist) {
    try {
      const stored = JSON.parse(lsGet('cv.c:' + url) || 'null');
      if (stored && stored.exp > now) { memory.set(url, stored); return stored.data; }
    } catch { /* ignore */ }
  }

  let data;
  if (settings.demo) {
    data = await demoFetch(path, params);
  } else {
    countCall();
    const res = settings.mode === 'proxy'
      ? await fetch(`${PROXY}${buildQuery({ path, ...params })}`, { headers: { 'X-Access-Code': settings.accessCode } })
      : await fetch(BASE + url, { headers: { Authorization: `Bearer ${settings.apiKey}` } });
    let body = null;
    try { body = await res.json(); } catch { /* corps vide */ }
    if (!res.ok) throw new ApiError(res.status, body);
    data = body;
  }
  const entry = { exp: now + ttl * 1000, data };
  memory.set(url, entry);
  if (persist && !settings.demo) {
    const json = JSON.stringify(entry);
    if (json.length < 400_000) lsSet('cv.c:' + url, json);
  }
  return data;
}

// Circuits principaux uniquement (pas de Challenger, WTA 125, ITF, juniors, épreuves par équipes).
const ATP_TIERS = ['grand_slam', 'atp_finals', 'atp_1000', 'atp_500', 'atp_250', 'next_gen_finals'];
const WTA_TIERS = ['grand_slam', 'wta_finals', 'wta_elite_trophy', 'wta_1000', 'wta_500', 'wta_250'];
const TEAM_EVENTS = /davis cup|billie jean king|bjk cup|laver cup|hopman|united cup|\butr\b|exhibition|junior/i;

// Paramètres de filtre : un seul appel pour ATP + WTA grâce au filtre ?tier=.
export function mainTourParams(tour) {
  if (tour === 'atp') return { tour: 'atp', tier: ATP_TIERS.join(',') };
  if (tour === 'wta') return { tour: 'wta', tier: WTA_TIERS.join(',') };
  return { tier: [...new Set([...ATP_TIERS, ...WTA_TIERS])].join(',') };
}

// /fixtures n'a pas de filtre de niveau : on interroge atp et/ou wta (jamais challenger/itf).
async function mainTourFixtures(tour) {
  const tours = tour === 'atp' || tour === 'wta' ? [tour] : ['atp', 'wta'];
  const pages = await Promise.all(tours.map((t) => get('/fixtures', { tour: t, limit: 200 }, { ttl: 900 })));
  const data = pages.flatMap((p) => p?.data || [])
    .filter((f) => !TEAM_EVENTS.test(f.tournament || ''))
    .sort((a, b) => String(a.start_time || a.event_date).localeCompare(String(b.start_time || b.event_date)));
  return { data };
}

export const api = {
  liveMatches: (tour) => get('/matches', { status: 'live', ...mainTourParams(tour), limit: 200 }, { ttl: 25 }),
  fixtures: (tour) => mainTourFixtures(tour),
  match: (id) => get(`/matches/${id}`, {}, { ttl: 20 }),
  score: (id) => get(`/matches/${id}/score`, {}, { ttl: 15 }),
  players: (search) => get('/players', { search, limit: 8 }, { ttl: 86400, persist: true }),
  player: (id) => get(`/players/${id}`, {}, { ttl: 43200, persist: true }),
  h2h: (p1, p2) => get('/h2h', { p1, p2 }, { ttl: 43200, persist: true }),
  results: (params) => get('/history/matches', { limit: 50, ...params }, { ttl: 300 }),
  // 200 derniers matchs terminés (2023 → aujourd'hui) : forme, bilan, fatigue en un seul appel.
  playerHistory: (playerId) => get('/history/matches', { player: playerId, draw: 'singles', limit: 200 }, { ttl: 43200, persist: true }),
  tape: (id, live) => get(`/history/matches/${id}`, { sequence: 'clean' }, live ? { ttl: 60 } : { ttl: 604800, persist: true }),
  usage: () => get('/usage', {}, { ttl: 120 }),
};

// ---------- Picks du moteur ----------

const picksCache = new Map();

export const picksEnabled = () => Boolean(PICKS) || settings.demo;

async function loadPicks(tour) {
  const hit = picksCache.get(tour);
  if (hit && hit.exp > Date.now()) return hit.data;
  let data;
  if (settings.demo) data = demoPicks(tour);
  else {
    const res = await fetch(`${PICKS}?tour=${tour}`, { cache: 'no-store' });
    data = res.ok ? await res.json() : null;
  }
  picksCache.set(tour, { exp: Date.now() + 10 * 60000, data });
  return data;
}

const norm = (s) => String(s || '').normalize('NFD').replace(/[\u0300-\u036f]/g, '').toLowerCase();
const surname = (s) => norm(s).split(/[\s,.-]+/).filter((x) => x.length >= 3).sort((a, b) => b.length - a.length)[0] || norm(s);

/**
 * Cherche l'analyse du moteur pour un match (par id Live Tennis, sinon par noms).
 * Retourne { row, swapped, snapshot } ; swapped = true si playerA du moteur = notre joueur 2.
 */
export async function findPick({ matchId, n1, n2 }) {
  if (!picksEnabled()) return null;
  const payloads = await Promise.all(['atp', 'wta'].map((t) => loadPicks(t).catch(() => null)));
  for (const pl of payloads) {
    const rows = pl?.allAnalyzed || [];
    let row = matchId ? rows.find((r) => String(r.matchId) === String(matchId)) : null;
    if (!row) {
      const a = surname(n1), b = surname(n2);
      row = rows.find((r) => {
        const x = surname(r.playerA?.name), y = surname(r.playerB?.name);
        return (x === a && y === b) || (x === b && y === a);
      });
    }
    if (row) {
      const swapped = surname(row.playerA?.name) !== surname(n1) && surname(row.playerB?.name) === surname(n1);
      return { row, swapped, snapshot: pl.snapshot || null };
    }
  }
  return { row: null, snapshot: payloads.find(Boolean)?.snapshot || null };
}


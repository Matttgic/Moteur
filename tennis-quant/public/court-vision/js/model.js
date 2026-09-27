// Modèle de Markov : probabilité que le joueur 1 gagne le match depuis un état de score.
// pA / pB = probabilité que le joueur 1 / 2 gagne un point sur son propre service.

const POINTS = { '0': 0, '15': 1, '30': 2, '40': 3, 'AD': 4, 'A': 4 };

// Probabilité que le serveur gagne le jeu depuis (a, b) points.
function gameProb(p, a, b) {
  if (a >= 4 && a - b >= 2) return 1;
  if (b >= 4 && b - a >= 2) return 0;
  if (a >= 3 && b >= 3) {
    const deuce = (p * p) / (p * p + (1 - p) * (1 - p));
    if (a === b) return deuce;
    return a > b ? p + (1 - p) * deuce : p * deuce;
  }
  return p * gameProb(p, a + 1, b) + (1 - p) * gameProb(p, a, b + 1);
}

// Serveur du point n d'un tie-break (0-indexé) : 1er point au serveur initial, puis alternance par 2.
function tbServer(first, n) {
  if (n === 0) return first;
  return Math.floor((n - 1) / 2) % 2 === 0 ? 3 - first : first;
}

// Probabilité que le joueur 1 gagne un tie-break (premier à `target`, 2 d'écart).
function tiebreakProb(pA, pB, a, b, first, target, memo = new Map()) {
  if (a >= target && a - b >= 2) return 1;
  if (b >= target && b - a >= 2) return 0;
  const n = a + b;
  if (a === b && a >= target - 1 && n % 2 === 0 && n >= 2) {
    // Deux points suivants : un service chacun.
    const win2 = pA * (1 - pB), lose2 = (1 - pA) * pB;
    return win2 / (win2 + lose2);
  }
  const key = `${a},${b}`;
  if (memo.has(key)) return memo.get(key);
  const srv = tbServer(first, n);
  const pw = srv === 1 ? pA : 1 - pB; // proba que J1 gagne ce point
  const r = pw * tiebreakProb(pA, pB, a + 1, b, first, target, memo) +
            (1 - pw) * tiebreakProb(pA, pB, a, b + 1, first, target, memo);
  memo.set(key, r);
  return r;
}

// Probabilité que J1 gagne le set depuis (ga, gb) jeux, `server` servant le prochain jeu.
function setProb(pA, pB, ga, gb, server, tbTarget, memo = new Map()) {
  if ((ga >= 6 && ga - gb >= 2) || ga === 7) return 1;
  if ((gb >= 6 && gb - ga >= 2) || gb === 7) return 0;
  if (ga === 6 && gb === 6) return tiebreakProb(pA, pB, 0, 0, server, tbTarget);
  const key = `${ga},${gb},${server}`;
  if (memo.has(key)) return memo.get(key);
  const hold = server === 1 ? gameProb(pA, 0, 0) : 1 - gameProb(pB, 0, 0); // proba J1 gagne le jeu
  const next = 3 - server;
  const r = hold * setProb(pA, pB, ga + 1, gb, next, tbTarget, memo) +
            (1 - hold) * setProb(pA, pB, ga, gb + 1, next, tbTarget, memo);
  memo.set(key, r);
  return r;
}

// Probabilité que J1 gagne le match depuis (sa, sb) sets, au début d'un nouveau set.
function matchFromSets(pA, pB, sa, sb, opts) {
  const need = opts.bestOf === 5 ? 3 : 2;
  if (sa >= need) return 1;
  if (sb >= need) return 0;
  const deciding = sa === need - 1 && sb === need - 1;
  let ps;
  if (deciding && opts.matchTiebreak) {
    ps = (tiebreakProb(pA, pB, 0, 0, 1, 10) + tiebreakProb(pA, pB, 0, 0, 2, 10)) / 2;
  } else {
    const tb = deciding ? opts.decidingTbTarget : 7;
    ps = (setProb(pA, pB, 0, 0, 1, tb) + setProb(pA, pB, 0, 0, 2, tb)) / 2;
  }
  return ps * matchFromSets(pA, pB, sa + 1, sb, opts) + (1 - ps) * matchFromSets(pA, pB, sa, sb + 1, opts);
}

export function parsePoint(s, tiebreak) {
  if (s == null) return 0;
  if (tiebreak) return parseInt(s, 10) || 0;
  return POINTS[String(s).toUpperCase()] ?? 0;
}

/**
 * Probabilité de victoire de J1 depuis une ligne de score de l'API.
 * row: { sets:[s1,s2], games:[[..],[..]], points:[x,y], server:1|2, is_tiebreak }
 */
export function winProbability(row, opts = {}) {
  const pA = opts.pA ?? 0.63;
  const pB = opts.pB ?? 0.63;
  const o = {
    bestOf: opts.bestOf ?? 3,
    decidingTbTarget: opts.decidingTbTarget ?? 7,
    matchTiebreak: !!opts.matchTiebreak,
  };
  const need = o.bestOf === 5 ? 3 : 2;
  const [sa, sb] = row.sets || [0, 0];
  if (sa >= need) return 1;
  if (sb >= need) return 0;

  const idx = sa + sb;
  const ga = row.games?.[0]?.[idx] ?? 0;
  const gb = row.games?.[1]?.[idx] ?? 0;
  const server = row.server === 2 ? 2 : 1;
  const tb = !!row.is_tiebreak;
  const pa = parsePoint(row.points?.[0], tb);
  const pb = parsePoint(row.points?.[1], tb);
  const deciding = sa === need - 1 && sb === need - 1;

  let pSet;
  if (tb) {
    const target = !deciding ? 7 : o.matchTiebreak ? 10 : o.decidingTbTarget;
    // Retrouver le serveur initial du tie-break depuis le serveur actuel.
    const n = pa + pb;
    const first = tbServer(1, n) === server ? 1 : 2;
    pSet = tiebreakProb(pA, pB, pa, pb, first, target);
  } else {
    const tbTarget = deciding ? o.decidingTbTarget : 7;
    const serverPts = server === 1 ? [pa, pb] : [pb, pa];
    const pHoldSrv = gameProb(server === 1 ? pA : pB, serverPts[0], serverPts[1]);
    const p1Game = server === 1 ? pHoldSrv : 1 - pHoldSrv;
    const next = 3 - server;
    pSet = p1Game * setProb(pA, pB, ga + 1, gb, next, tbTarget) +
           (1 - p1Game) * setProb(pA, pB, ga, gb + 1, next, tbTarget);
  }
  return pSet * matchFromSets(pA, pB, sa + 1, sb, o) + (1 - pSet) * matchFromSets(pA, pB, sa, sb + 1, o);
}

// Balle de break : le relanceur est à un point du jeu (hors tie-break).
export function isBreakPoint(row) {
  if (row.is_tiebreak) return false;
  const a = parsePoint(row.points?.[0]);
  const b = parsePoint(row.points?.[1]);
  const [srv, ret] = row.server === 2 ? [b, a] : [a, b];
  return ret >= 3 && ret > srv;
}

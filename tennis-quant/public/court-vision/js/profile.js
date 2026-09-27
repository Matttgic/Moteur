// Profil d'un joueur calculé à partir de ses matchs terminés (2023 → aujourd'hui).
// Une seule requête /history/matches?player=ID&limit=200 alimente la forme, le bilan
// par surface, les tie-breaks, les sets décisifs, les abandons et la fatigue.

const DAY = 86400000;

function matchDate(m) {
  const t = Date.parse(m.live_at || m.scheduled_time || '');
  return Number.isNaN(t) ? null : t;
}

function setsOf(m) {
  const g = m.score?.games;
  if (!Array.isArray(g?.[0]) || !Array.isArray(g?.[1])) return [];
  return g[0].map((a, i) => [a, g[1][i] ?? 0]).filter(([a, b]) => a + b > 0);
}

const wl = () => ({ w: 0, l: 0 });

export function buildProfile(res, playerId) {
  const all = (res?.data || []).filter((m) => m.winner === 1 || m.winner === 2);
  const p = {
    matches: 0, record: wl(), bySurface: { hard: wl(), clay: wl(), grass: wl() },
    tiebreaks: wl(), deciders: wl(), straightWins: 0, retiredLast12m: 0,
    form: [], since: null, lastPlayed: null, restDays: null,
    sets7d: 0, matches14d: 0,
  };
  const now = Date.now();
  for (const m of all) {
    const me = m.players?.p1?.id === playerId ? 1 : m.players?.p2?.id === playerId ? 2 : 0;
    if (!me) continue;
    const won = m.winner === me;
    const date = matchDate(m);
    if (date) {
      p.since = p.since == null ? date : Math.min(p.since, date);
      if (p.lastPlayed == null || date > p.lastPlayed) p.lastPlayed = date;
    }
    if (m.outcome === 'retired' && m.withdrew === me && date && now - date < 365 * DAY) p.retiredLast12m++;
    if (m.outcome === 'walkover') continue; // pas joué : ne compte ni en bilan ni en fatigue

    const sets = setsOf(m);
    if (date && now - date < 7 * DAY) p.sets7d += sets.length;
    if (date && now - date < 14 * DAY) p.matches14d++;
    if (p.form.length < 10) p.form.push({ won, m });

    p.matches++;
    p.record[won ? 'w' : 'l']++;
    if (p.bySurface[m.surface]) p.bySurface[m.surface][won ? 'w' : 'l']++;

    for (const [a, b] of sets) {
      if (Math.max(a, b) === 7 && Math.min(a, b) === 6) {
        const mine = me === 1 ? a > b : b > a;
        p.tiebreaks[mine ? 'w' : 'l']++;
      }
    }
    const bestOf = m.format === 'BO5' ? 5 : 3;
    if (m.outcome !== 'retired' && sets.length === bestOf) p.deciders[won ? 'w' : 'l']++;
    if (won && m.outcome !== 'retired' && sets.length === Math.ceil(bestOf / 2)) p.straightWins++;
  }
  if (p.lastPlayed) p.restDays = Math.floor((now - p.lastPlayed) / DAY);
  return p;
}

export const rate = (x) => (x && x.w + x.l ? x.w / (x.w + x.l) : null);

/**
 * Signaux de contexte (pas des conseils de pari) comparant deux profils.
 * Retourne [{ level: 'warn'|'info'|'good', text, side }] ; side = 1|2|null (joueur concerné).
 */
export function contextSignals(pa, pb, names, { surface, h2h, handA, handB } = {}) {
  const out = [];
  const pctTxt = (x) => `${Math.round(x * 100)} %`;
  const surf = { hard: 'dur', clay: 'terre battue', grass: 'gazon' }[surface];

  if (surf && pa && pb) {
    const ra = rate(pa.bySurface[surface]), rb = rate(pb.bySurface[surface]);
    const na = pa.bySurface[surface].w + pa.bySurface[surface].l;
    const nb = pb.bySurface[surface].w + pb.bySurface[surface].l;
    if (ra != null && rb != null && na >= 8 && nb >= 8 && Math.abs(ra - rb) >= 0.15) {
      const best = ra > rb ? 1 : 2;
      out.push({ level: 'info', side: best, text: `Sur ${surf} depuis 2023 : ${names[0]} ${pctTxt(ra)} (${na} matchs), ${names[1]} ${pctTxt(rb)} (${nb} matchs).` });
    } else if ((na < 8 || nb < 8) && (na || nb)) {
      const low = na < nb ? 0 : 1;
      out.push({ level: 'warn', side: low + 1, text: `${names[low]} a peu joué sur ${surf} depuis 2023 (${Math.min(na, nb)} matchs) : moins de repères.` });
    }
  }

  for (const [k, p] of [[0, pa], [1, pb]]) {
    if (!p) continue;
    const other = k === 0 ? pb : pa;
    if (p.restDays != null && p.restDays <= 1 && p.sets7d >= 6) {
      out.push({ level: 'warn', side: k + 1, text: `Fatigue possible : ${names[k]} a joué ${p.sets7d} sets ces 7 derniers jours et a joué ${p.restDays === 0 ? "aujourd'hui" : 'hier'}.` });
    } else if (other && p.sets7d - other.sets7d >= 4) {
      out.push({ level: 'warn', side: k + 1, text: `${names[k]} a joué ${p.sets7d} sets en 7 jours, contre ${other.sets7d} pour ${names[1 - k]}.` });
    }
    if (p.restDays != null && p.restDays >= 30) {
      out.push({ level: 'warn', side: k + 1, text: `${names[k]} n'a pas joué depuis ${p.restDays} jours (reprise, blessure ?).` });
    }
    if (p.retiredLast12m >= 2) {
      out.push({ level: 'warn', side: k + 1, text: `${names[k]} a abandonné ${p.retiredLast12m} matchs ces 12 derniers mois.` });
    }
  }

  if (pa && pb) {
    const ta = rate(pa.tiebreaks), tb = rate(pb.tiebreaks);
    if (ta != null && tb != null && pa.tiebreaks.w + pa.tiebreaks.l >= 10 && pb.tiebreaks.w + pb.tiebreaks.l >= 10 && Math.abs(ta - tb) >= 0.15) {
      const best = ta > tb ? 0 : 1;
      out.push({ level: 'info', side: best + 1, text: `Tie-breaks depuis 2023 : ${names[best]} en gagne ${pctTxt(Math.max(ta, tb))}, ${names[1 - best]} ${pctTxt(Math.min(ta, tb))}. Utile pour les paris sur les jeux ou le nombre de sets.` });
    }
  }

  if (handA && handB && handA[0] !== handB[0] && /^[LR]/i.test(handA) && /^[LR]/i.test(handB)) {
    const lefty = /^L/i.test(handA) ? 0 : 1;
    out.push({ level: 'info', side: lefty + 1, text: `${names[lefty]} est gaucher face à un droitier.` });
  }

  if (h2h?.players && surface && h2h.by_surface?.[surface]) {
    const s = h2h.by_surface[surface];
    if (s.p1 + s.p2 >= 3 && Math.abs(s.p1 - s.p2) >= 3) {
      const lead = s.p1 > s.p2 ? 0 : 1;
      out.push({ level: 'info', side: lead + 1, text: `Face-à-face sur ${surf} : ${names[lead]} mène ${Math.max(s.p1, s.p2)}-${Math.min(s.p1, s.p2)}.` });
    }
  }
  return out;
}

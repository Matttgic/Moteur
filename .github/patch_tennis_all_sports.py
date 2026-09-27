from pathlib import Path

p = Path("tennis-quant/lib/oddspapi.ts")
s = p.read_text(encoding="utf-8")

old_score = '''function sportHintScore(sport: TheOddsSport, tournamentHints: string[]) {
  if (!tournamentHints.length) return 0;

  const sportText = normalizeTournamentHint(
    [sport.key, sport.title].filter(Boolean).join(" "),
  );

  let best = 0;
  for (const rawHint of tournamentHints) {
    const tokens = normalizeTournamentHint(rawHint)
      .split(" ")
      .filter(
        (token) =>
          token.length >= 3 &&
          !["wta", "atp", "open", "tennis", "women", "men"].includes(token),
      );

    const score = tokens.reduce(
      (sum, token) => sum + (sportText.includes(token) ? 1 : 0),
      0,
    );
    best = Math.max(best, score);
  }

  return best;
}
'''

new_score = '''function tournamentHintTokens(rawHint: string) {
  const ignored = new Set(["wta", "atp", "open", "tennis", "women", "men"]);
  const aliases: Record<string, string[]> = {
    beijing: ["china"],
    tokyo: ["japan"],
  };

  const base = normalizeTournamentHint(rawHint)
    .split(" ")
    .filter((token) => token.length >= 3 && !ignored.has(token));

  return Array.from(
    new Set(base.flatMap((token) => [token, ...(aliases[token] ?? [])])),
  );
}

function sportHintScore(sport: TheOddsSport, tournamentHints: string[]) {
  if (!tournamentHints.length) return 0;

  const sportText = normalizeTournamentHint(
    [sport.key, sport.title].filter(Boolean).join(" "),
  );

  let best = 0;
  for (const rawHint of tournamentHints) {
    const tokens = tournamentHintTokens(rawHint);
    const score = tokens.reduce(
      (sum, token) => sum + (sportText.includes(token) ? 1 : 0),
      0,
    );
    best = Math.max(best, score);
  }

  return best;
}
'''

if s.count(old_score) != 1:
    raise SystemExit(f"sportHintScore block count={s.count(old_score)}")
s = s.replace(old_score, new_score, 1)

old_discovery = '''  const prefix = tour === "atp" ? "tennis_atp_" : "tennis_wta_";
  const activeSports = (Array.isArray(sportsResponse.payload)
    ? sportsResponse.payload
    : []
  ).filter(
    (sport) =>
      sport.active !== false &&
      sport.group?.toLowerCase() === "tennis" &&
      typeof sport.key === "string" &&
      sport.key.startsWith(prefix),
  );
  const sports = [...activeSports]
    .sort(
      (a, b) =>
        sportHintScore(b, tournamentHints) - sportHintScore(a, tournamentHints),
    )
    .slice(0, 8);

  if (!sports.length) {
'''

new_discovery = '''  const prefix = tour === "atp" ? "tennis_atp_" : "tennis_wta_";
  const activeSports = (Array.isArray(sportsResponse.payload)
    ? sportsResponse.payload
    : []
  ).filter(
    (sport) =>
      sport.active !== false &&
      sport.group?.toLowerCase() === "tennis" &&
      typeof sport.key === "string" &&
      sport.key.startsWith(prefix),
  );

  let sportsPool = activeSports;
  let allSportsAvailable = 0;
  let discoveryMode = "the_odds_api";
  let discoveryQuota = sportsResponse.quota;

  // /sports only returns in-season competitions by default. Around tournament
  // transitions The Odds API can temporarily omit a supported event (for
  // example Beijing / China Open). /sports?all=true is quota-free, so when the
  // active list is empty we use it only to recover sport keys matching the
  // tournaments already seen by Live Tennis.
  if (!sportsPool.length && tournamentHints.length) {
    const allSportsResponse = await theOddsApi<TheOddsSport[]>(
      apiKey,
      "sports",
      { all: "true" },
      3600,
    );
    discoveryQuota = allSportsResponse.quota;

    const allTourSports = (Array.isArray(allSportsResponse.payload)
      ? allSportsResponse.payload
      : []
    ).filter(
      (sport) =>
        sport.group?.toLowerCase() === "tennis" &&
        typeof sport.key === "string" &&
        sport.key.startsWith(prefix),
    );

    allSportsAvailable = allTourSports.length;
    const hintedSports = allTourSports.filter(
      (sport) => sportHintScore(sport, tournamentHints) > 0,
    );

    if (hintedSports.length) {
      sportsPool = hintedSports;
      discoveryMode = "the_odds_api_all_sports_fallback";
    }
  }

  const sports = [...sportsPool]
    .sort(
      (a, b) =>
        sportHintScore(b, tournamentHints) - sportHintScore(a, tournamentHints),
    )
    .slice(0, 8);

  if (!sports.length) {
'''

if s.count(old_discovery) != 1:
    raise SystemExit(f"discovery block count={s.count(old_discovery)}")
s = s.replace(old_discovery, new_discovery, 1)

s = s.replace('''        activeSportsAvailable: 0,
        oddsRequests: 0,
        quota: sportsResponse.quota,''','''        activeSportsAvailable: activeSports.length,
        allSportsAvailable,
        oddsRequests: 0,
        quota: discoveryQuota,''',1)

s = s.replace('''        discoveryMode: "the_odds_api",
        fallbackConfigured: true,
        activeSports: sports.length,
        activeSportsAvailable: activeSports.length,
        oddsRequests: 0,''','''        discoveryMode,
        fallbackConfigured: true,
        activeSports: sports.length,
        activeSportsAvailable: activeSports.length,
        allSportsAvailable,
        oddsRequests: 0,''',1)

s = s.replace('''  let quota = sportsResponse.quota;''','''  let quota = discoveryQuota;''',1)

s = s.replace('''      discoveryMode: "the_odds_api",
      fallbackConfigured: true,
      activeSports: sports.length,
      activeSportsAvailable: activeSports.length,
      oddsRequests: requests,''','''      discoveryMode,
      fallbackConfigured: true,
      activeSports: sports.length,
      activeSportsAvailable: activeSports.length,
      allSportsAvailable,
      oddsRequests: requests,''',1)

p.write_text(s, encoding="utf-8")
print("Patched all-sports tennis discovery fallback")

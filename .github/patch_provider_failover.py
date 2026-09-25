from pathlib import Path

path = Path("tennis-quant/lib/oddspapi.ts")
text = path.read_text(encoding="utf-8")

old_fixture = '''  const fixturePayload = await oddsApi<unknown>(
    apiKey,
    "fixtures",
    {
      sportId: String(TENNIS_SPORT_ID),
      from: window.from,
      to: window.to,
      statusId: "0",
      language: "en",
    },
    ODDSPAPI_DISCOVERY_CACHE_SECONDS,
  );'''

new_fixture = '''  let fixturePayload: unknown;
  try {
    fixturePayload = await oddsApi<unknown>(
      apiKey,
      "fixtures",
      {
        sportId: String(TENNIS_SPORT_ID),
        from: window.from,
        to: window.to,
        statusId: "0",
        language: "en",
      },
      ODDSPAPI_DISCOVERY_CACHE_SECONDS,
    );
  } catch (error) {
    if (isHttpStatus(error, 429)) {
      const fallback = await maybeTheOddsApiFallback(
        tour,
        bookmakers,
        window,
        tournamentHints,
      );
      if (fallback) return fallback;
    }
    throw error;
  }'''

old_tournaments = '''    const tournamentPayload = await oddsApi<unknown>(
      apiKey,
      "tournaments",
      {
        sportId: String(TENNIS_SPORT_ID),
        language: "en",
      },
      ODDSPAPI_DISCOVERY_CACHE_SECONDS,
    );'''

new_tournaments = '''    let tournamentPayload: unknown;
    try {
      tournamentPayload = await oddsApi<unknown>(
        apiKey,
        "tournaments",
        {
          sportId: String(TENNIS_SPORT_ID),
          language: "en",
        },
        ODDSPAPI_DISCOVERY_CACHE_SECONDS,
      );
    } catch (error) {
      if (isHttpStatus(error, 429)) {
        const fallback = await maybeTheOddsApiFallback(
          tour,
          bookmakers,
          window,
          tournamentHints,
        );
        if (fallback) return fallback;
      }
      throw error;
    }'''

old_catch = '''        // OddsPapi returns FIXTURE_NOT_FOUND when a bookmaker has no board for
        // the requested tournament set. That is an empty result, not a provider
        // outage, so continue with the other bookmaker.
        if (
          isHttpStatus(error, 400) ||
          isHttpStatus(error, 403) ||
          isHttpStatus(error, 404)
        ) {
          emptyBookmakerQueries += 1;
          continue;
        }

        throw error;'''

new_catch = '''        if (isHttpStatus(error, 429)) {
          const fallback = await maybeTheOddsApiFallback(
            tour,
            bookmakers,
            window,
            tournamentHints,
          );
          if (fallback) return fallback;
          throw error;
        }

        // OddsPapi returns FIXTURE_NOT_FOUND when a bookmaker has no board for
        // the requested tournament set. That is an empty result, not a provider
        // outage, so continue with the other bookmaker.
        if (
          isHttpStatus(error, 400) ||
          isHttpStatus(error, 403) ||
          isHttpStatus(error, 404)
        ) {
          emptyBookmakerQueries += 1;
          continue;
        }

        throw error;'''

for old, new, label in [
    (old_fixture, new_fixture, "fixtures discovery"),
    (old_tournaments, new_tournaments, "tournament discovery"),
    (old_catch, new_catch, "odds-by-tournaments"),
]:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"Expected exactly one {label} block, found {count}")
    text = text.replace(old, new, 1)

path.write_text(text, encoding="utf-8")
print("Applied OddsPapi 429 -> The Odds API failover patch")

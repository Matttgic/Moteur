const ODDS_BASE_URL = "https://api.oddspapi.io/v4";
const TENNIS_SPORT_ID = 12;
const TENNIS_WINNER_MARKET_ID = 121;
const TENNIS_WINNER_OUTCOME_1 = "121";
const TENNIS_WINNER_OUTCOME_2 = "122";
const THE_ODDS_API_BASE = "https://api.the-odds-api.com/v4";
const ODDSPAPI_DISCOVERY_CACHE_SECONDS = 900;
const ODDSPAPI_PRICE_CACHE_SECONDS = 300;
const THE_ODDS_API_PRICE_CACHE_SECONDS = 7_200;

const THE_ODDS_API_BOOKMAKERS: Record<string, string> = {
  "winamax.fr": "winamax_fr",
  "unibet.fr": "unibet_fr",
  "betclic.fr": "betclic_fr",
  "pmu": "pmu_fr",
  "netbet.fr": "netbet_fr",
  "pinnacle": "pinnacle",
};

export type TennisTour = "atp" | "wta";

export const FRENCH_EXECUTION_BOOKMAKERS = [
  "bet365.fr",
  "winamax.fr",
  "unibet.fr",
  "betclic.fr",
  "pmu",
  "netbet.fr",
  "bwin.fr",
  "zebet.fr",
] as const;

type PriceNode = {
  active?: boolean;
  price?: number;
  changedAt?: string;
  bookmakerChangedAt?: string | null;
};

type OddsOutcome = {
  players?: Record<string, PriceNode>;
};

type OddsMarket = {
  marketActive?: boolean;
  outcomes?: Record<string, OddsOutcome>;
};

type BookmakerNode = {
  bookmakerIsActive?: boolean;
  suspended?: boolean;
  markets?: Record<string, OddsMarket>;
};

type OddsFixture = {
  fixtureId?: string;
  participant1Id?: number;
  participant2Id?: number;
  participant1Name?: string;
  participant2Name?: string;
  tournamentId?: number;
  tournamentSlug?: string;
  tournamentName?: string;
  categorySlug?: string;
  categoryName?: string;
  sportId?: number;
  statusId?: number;
  statusName?: string;
  startTime?: string;
  hasOdds?: boolean;
  bookmakerOdds?: Record<string, BookmakerNode>;
};

type OddsTournament = {
  tournamentId?: number;
  tournamentSlug?: string;
  tournamentName?: string;
  categorySlug?: string;
  categoryName?: string;
  futureFixtures?: number;
  upcomingFixtures?: number;
  liveFixtures?: number;
};

type TheOddsSport = {
  key?: string;
  group?: string;
  title?: string;
  active?: boolean;
};

type TheOddsOutcome = {
  name?: string;
  price?: number;
};

type TheOddsMarket = {
  key?: string;
  last_update?: string;
  outcomes?: TheOddsOutcome[];
};

type TheOddsBookmaker = {
  key?: string;
  title?: string;
  last_update?: string;
  markets?: TheOddsMarket[];
};

type TheOddsEvent = {
  id?: string;
  sport_key?: string;
  sport_title?: string;
  commence_time?: string;
  home_team?: string;
  away_team?: string;
  bookmakers?: TheOddsBookmaker[];
};

const EXCLUDED_TENNIS =
  /challenger|itf|utr|junior|doubles?|davis cup|billie jean king|bjk cup|laver cup|hopman cup|united cup|exhibition/i;

const GRAND_SLAM =
  /australian open|roland garros|french open|wimbledon|us open|grand slam/i;

function fixtureText(fixture: OddsFixture) {
  return [
    fixture.tournamentName,
    fixture.tournamentSlug,
    fixture.categoryName,
    fixture.categorySlug,
  ]
    .filter(Boolean)
    .join(" ")
    .toLowerCase();
}

export function isTargetTourFixture(
  fixture: OddsFixture,
  tour: TennisTour,
) {
  const text = fixtureText(fixture);
  if (EXCLUDED_TENNIS.test(text)) return false;

  if (tour === "atp") {
    if (/\bwta\b|women singles|women's singles/.test(text)) return false;
    return (
      /\batp\b|men singles|men's singles/.test(text) ||
      GRAND_SLAM.test(text)
    );
  }

  if (/\batp\b|men singles|men's singles/.test(text)) return false;
  return (
    /\bwta\b|women singles|women's singles/.test(text) ||
    GRAND_SLAM.test(text)
  );
}

async function oddsApi<T>(
  apiKey: string,
  path: string,
  params: Record<string, string>,
  revalidate: number,
): Promise<T> {
  const url = new URL(`${ODDS_BASE_URL}/${path}`);
  url.searchParams.set("apiKey", apiKey);

  for (const [key, value] of Object.entries(params)) {
    url.searchParams.set(key, value);
  }

  const response = await fetch(url, {
    headers: { Accept: "application/json" },
    next: { revalidate },
  });

  const payload = await response.json().catch(() => null);

  if (!response.ok) {
    const detail =
      payload && typeof payload === "object"
        ? JSON.stringify(payload).slice(0, 800)
        : String(payload ?? "");

    const error = new Error(
      `OddsPapi ${path} failed with HTTP ${response.status}${detail ? `: ${detail}` : ""}`,
    );
    Object.assign(error, { status: response.status, payload });
    throw error;
  }

  return payload as T;
}

function normalizeFixturePayload(payload: unknown): OddsFixture[] {
  if (Array.isArray(payload)) return payload as OddsFixture[];

  if (payload && typeof payload === "object") {
    const row = payload as Record<string, unknown>;
    if (Array.isArray(row.data)) return row.data as OddsFixture[];
    if (typeof row.fixtureId === "string") return [row as OddsFixture];
  }

  return [];
}

function normalizeTournamentPayload(payload: unknown): OddsTournament[] {
  if (Array.isArray(payload)) return payload as OddsTournament[];

  if (payload && typeof payload === "object") {
    const row = payload as Record<string, unknown>;
    if (Array.isArray(row.data)) return row.data as OddsTournament[];
  }

  return [];
}

function isTargetTourTournament(
  tournament: OddsTournament,
  tour: TennisTour,
) {
  const text = [
    tournament.tournamentName,
    tournament.tournamentSlug,
    tournament.categoryName,
    tournament.categorySlug,
  ]
    .filter(Boolean)
    .join(" ")
    .toLowerCase();

  if (EXCLUDED_TENNIS.test(text)) return false;

  if (tour === "atp") {
    if (/\bwta\b|women singles|women's singles/.test(text)) return false;
    return (
      /\batp\b|men singles|men's singles/.test(text) ||
      GRAND_SLAM.test(text)
    );
  }

  if (/\batp\b|men singles|men's singles/.test(text)) return false;
  return (
    /\bwta\b|women singles|women's singles/.test(text) ||
    GRAND_SLAM.test(text)
  );
}

function activePrice(outcome: OddsOutcome | undefined) {
  const players = outcome?.players;
  if (!players) return null;

  const price = Object.values(players).find(
    (candidate) =>
      candidate &&
      candidate.active !== false &&
      typeof candidate.price === "number" &&
      Number.isFinite(candidate.price) &&
      candidate.price > 1,
  );

  if (!price || typeof price.price !== "number") return null;

  return {
    price: price.price,
    changedAt: price.changedAt ?? null,
    bookmakerChangedAt: price.bookmakerChangedAt ?? null,
  };
}

function extractWinner(
  fixture: OddsFixture,
  bookmaker: string,
) {
  const book = fixture.bookmakerOdds?.[bookmaker];
  if (!book || book.bookmakerIsActive === false || book.suspended) return null;

  const market = book.markets?.[String(TENNIS_WINNER_MARKET_ID)];
  if (!market || market.marketActive === false) return null;

  const side1 = activePrice(market.outcomes?.[TENNIS_WINNER_OUTCOME_1]);
  const side2 = activePrice(market.outcomes?.[TENNIS_WINNER_OUTCOME_2]);
  if (!side1 || !side2) return null;

  const inv1 = 1 / side1.price;
  const inv2 = 1 / side2.price;
  const total = inv1 + inv2;

  return {
    marketId: TENNIS_WINNER_MARKET_ID,
    marketName: "Winner",
    outcome1Id: Number(TENNIS_WINNER_OUTCOME_1),
    outcome2Id: Number(TENNIS_WINNER_OUTCOME_2),
    odds1: side1.price,
    odds2: side2.price,
    noVigProbability1: inv1 / total,
    noVigProbability2: inv2 / total,
    overround: total - 1,
    changedAt1: side1.changedAt,
    changedAt2: side2.changedAt,
  };
}

function boardWindow() {
  const now = new Date();
  const from = new Date(
    Date.UTC(
      now.getUTCFullYear(),
      now.getUTCMonth(),
      now.getUTCDate(),
      0,
      0,
      0,
    ),
  );
  const to = new Date(from.getTime() + 71 * 60 * 60 * 1000 + 59 * 60 * 1000);

  return {
    from: from.toISOString(),
    to: to.toISOString(),
    fromMs: from.getTime(),
    toMs: to.getTime(),
  };
}

function withinWindow(
  fixture: OddsFixture,
  window: ReturnType<typeof boardWindow>,
) {
  if (!fixture.startTime) return false;
  const timestamp = Date.parse(fixture.startTime);
  return (
    Number.isFinite(timestamp) &&
    timestamp >= window.fromMs &&
    timestamp <= window.toMs
  );
}

function chunks<T>(values: T[], size: number) {
  const output: T[][] = [];
  for (let index = 0; index < values.length; index += size) {
    output.push(values.slice(index, index + size));
  }
  return output;
}

function isHttpStatus(error: unknown, status: number) {
  return (
    typeof error === "object" &&
    error !== null &&
    "status" in error &&
    (error as { status?: unknown }).status === status
  );
}

async function theOddsApi<T>(
  apiKey: string,
  path: string,
  params: Record<string, string>,
  revalidate = 900,
): Promise<{ payload: T; quota: { remaining: string | null; used: string | null; last: string | null } }> {
  const url = new URL(`${THE_ODDS_API_BASE}/${path}`);
  url.searchParams.set("apiKey", apiKey);

  for (const [key, value] of Object.entries(params)) {
    url.searchParams.set(key, value);
  }

  const response = await fetch(url, {
    headers: { Accept: "application/json" },
    next: { revalidate },
  });

  const payload = await response.json().catch(() => null);

  if (!response.ok) {
    const detail =
      payload && typeof payload === "object"
        ? JSON.stringify(payload).slice(0, 800)
        : String(payload ?? "");

    const error = new Error(
      `The Odds API ${path} failed with HTTP ${response.status}${detail ? `: ${detail}` : ""}`,
    );
    Object.assign(error, { status: response.status, payload });
    throw error;
  }

  return {
    payload: payload as T,
    quota: {
      remaining: response.headers.get("x-requests-remaining"),
      used: response.headers.get("x-requests-used"),
      last: response.headers.get("x-requests-last"),
    },
  };
}

function theOddsWinner(
  event: TheOddsEvent,
  canonicalBookmaker: string,
) {
  const providerBookmaker = THE_ODDS_API_BOOKMAKERS[canonicalBookmaker];
  if (!providerBookmaker) return null;

  const bookmaker = event.bookmakers?.find(
    (candidate) => candidate.key === providerBookmaker,
  );
  const market = bookmaker?.markets?.find((candidate) => candidate.key === "h2h");
  if (!market || !event.home_team || !event.away_team) return null;

  const outcome1 = market.outcomes?.find(
    (outcome) => outcome.name === event.home_team,
  );
  const outcome2 = market.outcomes?.find(
    (outcome) => outcome.name === event.away_team,
  );
  const odds1 = Number(outcome1?.price);
  const odds2 = Number(outcome2?.price);

  if (
    !Number.isFinite(odds1) ||
    !Number.isFinite(odds2) ||
    odds1 <= 1 ||
    odds2 <= 1
  ) {
    return null;
  }

  const inv1 = 1 / odds1;
  const inv2 = 1 / odds2;
  const total = inv1 + inv2;
  const changedAt = market.last_update ?? bookmaker?.last_update ?? null;

  return {
    marketId: TENNIS_WINNER_MARKET_ID,
    marketName: "Winner",
    outcome1Id: Number(TENNIS_WINNER_OUTCOME_1),
    outcome2Id: Number(TENNIS_WINNER_OUTCOME_2),
    odds1,
    odds2,
    noVigProbability1: inv1 / total,
    noVigProbability2: inv2 / total,
    overround: total - 1,
    changedAt1: changedAt,
    changedAt2: changedAt,
  };
}

function normalizeTournamentHint(value: string) {
  return value
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .trim()
    .replace(/\s+/g, " ");
}

function sportHintScore(sport: TheOddsSport, tournamentHints: string[]) {
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

async function getTheOddsApiTennisOdds(
  apiKey: string,
  tour: TennisTour,
  bookmakers: string[],
  window: ReturnType<typeof boardWindow>,
  tournamentHints: string[],
) {
  const sportsResponse = await theOddsApi<TheOddsSport[]>(
    apiKey,
    "sports",
    {},
    3600,
  );

  const prefix = tour === "atp" ? "tennis_atp_" : "tennis_wta_";
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
    return {
      provider: "The Odds API" as const,
      tour: tour.toUpperCase(),
      tournamentCount: 0,
      discoveredFixtures: 0,
      tournaments: [],
      bookmakers,
      moneylineMarketCandidates: [
        { marketId: TENNIS_WINNER_MARKET_ID, marketName: "Winner" },
      ],
      requestWindow: { from: window.from, to: window.to },
      providerDiagnostics: {
        discoveryMode: "the_odds_api_no_active_sports",
        fallbackConfigured: true,
        activeSports: 0,
        activeSportsAvailable: 0,
        oddsRequests: 0,
        quota: sportsResponse.quota,
      },
      fixtures: [],
    };
  }

  const supportedCanonicalBooks = bookmakers.filter(
    (bookmaker) => Boolean(THE_ODDS_API_BOOKMAKERS[bookmaker]),
  );
  const providerBookmakers = Array.from(
    new Set(
      supportedCanonicalBooks
        .map((bookmaker) => THE_ODDS_API_BOOKMAKERS[bookmaker])
        .filter(Boolean),
    ),
  );

  if (!providerBookmakers.length) {
    return {
      provider: "The Odds API" as const,
      tour: tour.toUpperCase(),
      tournamentCount: sports.length,
      discoveredFixtures: 0,
      tournaments: sports.map((sport) => ({
        tournamentId: null,
        tournamentName: sport.title ?? sport.key ?? null,
        categoryName: tour.toUpperCase(),
      })),
      bookmakers,
      moneylineMarketCandidates: [
        { marketId: TENNIS_WINNER_MARKET_ID, marketName: "Winner" },
      ],
      requestWindow: { from: window.from, to: window.to },
      providerDiagnostics: {
        discoveryMode: "the_odds_api",
        fallbackConfigured: true,
        activeSports: sports.length,
        activeSportsAvailable: activeSports.length,
        oddsRequests: 0,
        supportedBookmakers: 0,
        quota: sportsResponse.quota,
      },
      fixtures: [],
    };
  }

  const events: TheOddsEvent[] = [];
  let quota = sportsResponse.quota;
  let requests = 0;

  for (const sport of sports) {
    if (!sport.key) continue;

    const response = await theOddsApi<TheOddsEvent[]>(
      apiKey,
      `sports/${sport.key}/odds`,
      {
        bookmakers: providerBookmakers.join(","),
        markets: "h2h",
        oddsFormat: "decimal",
        dateFormat: "iso",
        commenceTimeFrom: window.from.replace(/\.\d{3}Z$/, "Z"),
        commenceTimeTo: window.to.replace(/\.\d{3}Z$/, "Z"),
      },
      THE_ODDS_API_PRICE_CACHE_SECONDS,
    );

    requests += 1;
    quota = response.quota;

    if (Array.isArray(response.payload)) {
      events.push(...response.payload);
    }
  }

  const fixtures = events
    .filter(
      (event) =>
        typeof event.id === "string" &&
        typeof event.home_team === "string" &&
        typeof event.away_team === "string" &&
        typeof event.commence_time === "string" &&
        Number.isFinite(Date.parse(event.commence_time)) &&
        Date.parse(event.commence_time) >= window.fromMs &&
        Date.parse(event.commence_time) <= window.toMs,
    )
    .map((event) => {
      const prices = Object.fromEntries(
        bookmakers.map((bookmaker) => [
          bookmaker,
          theOddsWinner(event, bookmaker),
        ]),
      );

      return {
        fixtureId: `toa:${event.id}`,
        tournamentId: null,
        tournamentName: event.sport_title ?? event.sport_key ?? null,
        categoryName: tour.toUpperCase(),
        startTime: event.commence_time ?? null,
        participant1Id: null,
        participant2Id: null,
        participant1Name: event.home_team ?? null,
        participant2Name: event.away_team ?? null,
        hasOdds: true,
        prices,
      };
    })
    .filter((fixture) => Object.values(fixture.prices).some(Boolean));

  return {
    provider: "The Odds API" as const,
    tour: tour.toUpperCase(),
    tournamentCount: sports.length,
    discoveredFixtures: events.length,
    tournaments: sports.map((sport) => ({
      tournamentId: null,
      tournamentName: sport.title ?? sport.key ?? null,
      categoryName: tour.toUpperCase(),
    })),
    bookmakers,
    moneylineMarketCandidates: [
      { marketId: TENNIS_WINNER_MARKET_ID, marketName: "Winner" },
    ],
    requestWindow: { from: window.from, to: window.to },
    providerDiagnostics: {
      discoveryMode: "the_odds_api",
      fallbackConfigured: true,
      activeSports: sports.length,
      activeSportsAvailable: activeSports.length,
      oddsRequests: requests,
      quota,
    },
    fixtures,
  };
}

async function maybeTheOddsApiFallback(
  tour: TennisTour,
  bookmakers: string[],
  window: ReturnType<typeof boardWindow>,
  tournamentHints: string[],
) {
  const apiKey = process.env.THE_ODDS_API_KEY;
  if (!apiKey) return null;

  try {
    return await getTheOddsApiTennisOdds(
      apiKey,
      tour,
      bookmakers,
      window,
      tournamentHints,
    );
  } catch (error) {
    return {
      provider: "The Odds API" as const,
      tour: tour.toUpperCase(),
      tournamentCount: 0,
      discoveredFixtures: 0,
      tournaments: [],
      bookmakers,
      moneylineMarketCandidates: [
        { marketId: TENNIS_WINNER_MARKET_ID, marketName: "Winner" },
      ],
      requestWindow: { from: window.from, to: window.to },
      providerDiagnostics: {
        discoveryMode: "the_odds_api_error",
        fallbackConfigured: true,
        fallbackError:
          error instanceof Error ? error.message : "unknown_fallback_error",
        oddsRequests: 0,
      },
      fixtures: [],
    };
  }
}

export async function getTennisOdds(
  apiKey: string,
  tour: TennisTour,
  bookmakers = [...FRENCH_EXECUTION_BOOKMAKERS, "pinnacle"],
  tournamentHints: string[] = [],
) {
  const window = boardWindow();

  // Discover the current tennis board without bookmaker/hasOdds filters.
  // Those filters can remove a fixture when one requested bookmaker has not
  // opened its market yet, even when another bookmaker has.
  let fixturePayload: unknown;
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
  }

  const boardFixtures = normalizeFixturePayload(fixturePayload)
    .filter(
      (fixture) =>
        fixture.sportId === TENNIS_SPORT_ID &&
        fixture.statusId === 0 &&
        withinWindow(fixture, window) &&
        isTargetTourFixture(fixture, tour),
    );

  const tournamentMap = new Map<
    number,
    { tournamentId: number; tournamentName: string | null; categoryName: string | null }
  >();

  for (const fixture of boardFixtures) {
    if (typeof fixture.tournamentId !== "number") continue;

    if (!tournamentMap.has(fixture.tournamentId)) {
      tournamentMap.set(fixture.tournamentId, {
        tournamentId: fixture.tournamentId,
        tournamentName: fixture.tournamentName ?? null,
        categoryName: fixture.categoryName ?? null,
      });
    }
  }

  const fixtureIds = new Set(
    boardFixtures
      .map((fixture) => fixture.fixtureId)
      .filter((value): value is string => typeof value === "string"),
  );

  let discoveryMode: "fixtures" | "tournaments_fallback" | "none" =
    tournamentMap.size ? "fixtures" : "none";
  let fallbackTournamentCount = 0;

  // Some tennis tours can be missing from the generic fixtures discovery
  // while still being present in the provider tournament catalog. The
  // provider docs recommend tournament discovery before odds-by-tournaments,
  // so use it as a conservative fallback instead of treating the tour as empty.
  if (!tournamentMap.size) {
    let tournamentPayload: unknown;
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
    }

    const fallbackTournaments = normalizeTournamentPayload(tournamentPayload)
      .filter((tournament) => {
        const activeFixtures =
          Number(tournament.futureFixtures ?? 0) +
          Number(tournament.upcomingFixtures ?? 0) +
          Number(tournament.liveFixtures ?? 0);

        return (
          activeFixtures > 0 &&
          isTargetTourTournament(tournament, tour)
        );
      });

    for (const tournament of fallbackTournaments) {
      if (typeof tournament.tournamentId !== "number") continue;

      tournamentMap.set(tournament.tournamentId, {
        tournamentId: tournament.tournamentId,
        tournamentName: tournament.tournamentName ?? null,
        categoryName: tournament.categoryName ?? null,
      });
    }

    fallbackTournamentCount = tournamentMap.size;
    if (tournamentMap.size) {
      discoveryMode = "tournaments_fallback";
    }
  }

  if (!tournamentMap.size) {
    const fallback = await maybeTheOddsApiFallback(tour, bookmakers, window, tournamentHints);
    if (fallback?.fixtures.length) return fallback;

    return {
      provider: "OddsPapi" as const,
      tour: tour.toUpperCase(),
      tournamentCount: 0,
      discoveredFixtures: 0,
      tournaments: [],
      bookmakers,
      moneylineMarketCandidates: [
        { marketId: TENNIS_WINNER_MARKET_ID, marketName: "Winner" },
      ],
      requestWindow: { from: window.from, to: window.to },
      providerDiagnostics: {
        discoveryMode,
        fallbackTournamentCount,
        fallbackConfigured: Boolean(process.env.THE_ODDS_API_KEY),
        fallbackProvider: fallback?.provider ?? null,
        fallbackDiagnostics: fallback?.providerDiagnostics ?? null,
        oddsRequests: 0,
        emptyBookmakerQueries: 0,
      },
      fixtures: [],
    };
  }

  const merged = new Map<string, OddsFixture>();
  let requests = 0;
  let emptyBookmakerQueries = 0;

  for (const batch of chunks(Array.from(tournamentMap.keys()), 8)) {
    for (const bookmaker of bookmakers) {
      if (requests > 0) {
        await new Promise((resolve) => setTimeout(resolve, 1_050));
      }

      try {
        const payload = await oddsApi<unknown>(
          apiKey,
          "odds-by-tournaments",
          {
            tournamentIds: batch.join(","),
            bookmaker,
            language: "en",
            verbosity: "3",
            oddsFormat: "decimal",
          },
          ODDSPAPI_PRICE_CACHE_SECONDS,
        );

        requests += 1;

        for (const fixture of normalizeFixturePayload(payload)) {
          if (typeof fixture.fixtureId !== "string") continue;

          const existing = merged.get(fixture.fixtureId);
          merged.set(
            fixture.fixtureId,
            existing
              ? {
                  ...existing,
                  ...fixture,
                  bookmakerOdds: {
                    ...(existing.bookmakerOdds ?? {}),
                    ...(fixture.bookmakerOdds ?? {}),
                  },
                }
              : fixture,
          );
        }
      } catch (error) {
        requests += 1;

        if (isHttpStatus(error, 429)) {
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

        throw error;
      }
    }
  }

  const fixtures = Array.from(merged.values())
    .filter(
      (fixture) =>
        typeof fixture.fixtureId === "string" &&
        typeof fixture.tournamentId === "number" &&
        tournamentMap.has(fixture.tournamentId) &&
        (fixtureIds.size === 0 || fixtureIds.has(fixture.fixtureId)) &&
        (fixture.statusId == null || fixture.statusId === 0) &&
        withinWindow(fixture, window),
    )
    .map((fixture) => {
      const prices = Object.fromEntries(
        bookmakers.map((bookmaker) => [
          bookmaker,
          extractWinner(fixture, bookmaker),
        ]),
      );

      return {
        fixtureId: fixture.fixtureId ?? null,
        tournamentId: fixture.tournamentId ?? null,
        tournamentName: fixture.tournamentName ?? null,
        categoryName: fixture.categoryName ?? null,
        startTime: fixture.startTime ?? null,
        participant1Id: fixture.participant1Id ?? null,
        participant2Id: fixture.participant2Id ?? null,
        participant1Name: fixture.participant1Name ?? null,
        participant2Name: fixture.participant2Name ?? null,
        hasOdds: fixture.hasOdds ?? false,
        prices,
      };
    })
    .filter((fixture) => Object.values(fixture.prices).some(Boolean));

  const primaryResult = {
    provider: "OddsPapi" as const,
    tour: tour.toUpperCase(),
    tournamentCount: tournamentMap.size,
    discoveredFixtures: boardFixtures.length || fixtures.length,
    tournaments: Array.from(tournamentMap.values()),
    bookmakers,
    moneylineMarketCandidates: [
      { marketId: TENNIS_WINNER_MARKET_ID, marketName: "Winner" },
    ],
    requestWindow: { from: window.from, to: window.to },
    providerDiagnostics: {
      discoveryMode,
      fallbackTournamentCount,
      boardDiscoveredFixtures: boardFixtures.length,
      fallbackConfigured: Boolean(process.env.THE_ODDS_API_KEY),
      oddsRequests: requests,
      emptyBookmakerQueries,
    },
    fixtures,
  };

  if (fixtures.length) return primaryResult;

  const fallback = await maybeTheOddsApiFallback(tour, bookmakers, window, tournamentHints);
  if (fallback?.fixtures.length) return fallback;

  return {
    ...primaryResult,
    providerDiagnostics: {
      ...primaryResult.providerDiagnostics,
      fallbackProvider: fallback?.provider ?? null,
      fallbackDiagnostics: fallback?.providerDiagnostics ?? null,
    },
  };
}

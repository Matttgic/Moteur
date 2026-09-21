const ODDS_BASE_URL = "https://api.oddspapi.io/v4";
const TENNIS_SPORT_ID = 12;

export type TennisTour = "atp" | "wta";

type MarketCatalogItem = {
  marketId: number;
  marketLength?: number;
  marketName?: string;
  playerProp?: boolean;
  sportId?: number;
  handicap?: number;
  period?: string;
  marketType?: string;
  outcomes?: Array<{
    outcomeId: number;
    outcomeName?: string;
  }>;
};

type PriceNode = {
  active?: boolean;
  price?: number;
  changedAt?: string;
  bookmakerChangedAt?: string | null;
  mainLine?: boolean;
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

const EXCLUDED_TENNIS =
  /challenger|itf|utr|junior|davis cup|billie jean king|bjk cup|laver cup|hopman cup|united cup|exhibition/i;
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

function moneylineCandidates(catalog: MarketCatalogItem[]) {
  return catalog
    .filter((market) => {
      if (market.sportId !== TENNIS_SPORT_ID) return false;
      if (market.playerProp) return false;
      if ((market.marketLength ?? market.outcomes?.length) !== 2) return false;
      if (market.handicap != null && market.handicap !== 0) return false;

      const name = (market.marketName ?? "").toLowerCase();
      const type = (market.marketType ?? "").toLowerCase();
      const period = (market.period ?? "").toLowerCase();

      const winnerLike =
        /winner|moneyline|match winner/.test(name) ||
        /moneyline|winner/.test(type);
      const matchPeriod =
        !period || /fulltime|full time|match|game/.test(period);
      return winnerLike && matchPeriod;
    })
    .sort((a, b) => {
      const score = (m: MarketCatalogItem) => {
        const name = (m.marketName ?? "").toLowerCase();
        if (name === "winner" || name === "match winner") return 0;
        if (name.includes("moneyline")) return 1;
        return 2;
      };
      return score(a) - score(b);
    });
}

function orderedOutcomeIds(market: MarketCatalogItem) {
  const outcomes = market.outcomes ?? [];
  if (outcomes.length !== 2) return [];

  const one = outcomes.find((o) =>
    /^(1|home|player 1|p1)$/i.test((o.outcomeName ?? "").trim()),
  );
  const two = outcomes.find((o) =>
    /^(2|away|player 2|p2)$/i.test((o.outcomeName ?? "").trim()),
  );

  if (one && two) return [String(one.outcomeId), String(two.outcomeId)];
  return outcomes.map((o) => String(o.outcomeId));
}

function activePrice(outcome: OddsOutcome | undefined) {
  const prices = outcome?.players;
  if (!prices) return null;

  const price = Object.values(prices).find(
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

function extractMoneyline(
  fixture: OddsFixture,
  bookmaker: string,
  candidates: MarketCatalogItem[],
) {
  const book = fixture.bookmakerOdds?.[bookmaker];
  if (!book || book.bookmakerIsActive === false || book.suspended) return null;

  for (const candidate of candidates) {
    const market = book.markets?.[String(candidate.marketId)];
    if (!market || market.marketActive === false) continue;

    const [outcome1, outcome2] = orderedOutcomeIds(candidate);
    if (!outcome1 || !outcome2) continue;

    const side1 = activePrice(market.outcomes?.[outcome1]);
    const side2 = activePrice(market.outcomes?.[outcome2]);
    if (!side1 || !side2) continue;

    const inv1 = 1 / side1.price;
    const inv2 = 1 / side2.price;
    const overround = inv1 + inv2;

    return {
      marketId: candidate.marketId,
      marketName: candidate.marketName ?? "Winner",
      outcome1Id: Number(outcome1),
      outcome2Id: Number(outcome2),
      odds1: side1.price,
      odds2: side2.price,
      noVigProbability1: inv1 / overround,
      noVigProbability2: inv2 / overround,
      overround: overround - 1,
      changedAt1: side1.changedAt,
      changedAt2: side2.changedAt,
    };
  }

  return null;
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
  const to = new Date(from.getTime() + 47 * 60 * 60 * 1000 + 59 * 60 * 1000);
  return {
    from: from.toISOString(),
    to: to.toISOString(),
  };
}

function chunks<T>(values: T[], size: number) {
  const result: T[][] = [];
  for (let index = 0; index < values.length; index += size) {
    result.push(values.slice(index, index + size));
  }
  return result;
}

export async function getTennisOdds(
  apiKey: string,
  tour: TennisTour,
  bookmakers = ["winamax.fr", "pinnacle"],
) {
  const window = boardWindow();

  const [fixturePayload, catalog] = await Promise.all([
    oddsApi<unknown>(
      apiKey,
      "fixtures",
      {
        sportId: String(TENNIS_SPORT_ID),
        from: window.from,
        to: window.to,
        statusId: "0",
        hasOdds: "true",
        bookmakers: bookmakers.join(","),
        language: "en",
      },
      86_400,
    ),
    oddsApi<MarketCatalogItem[]>(
      apiKey,
      "markets",
      { language: "en" },
      86_400,
    ),
  ]);

  const boardFixtures = normalizeFixturePayload(fixturePayload)
    .filter(
      (fixture) =>
        fixture.sportId === TENNIS_SPORT_ID &&
        fixture.statusId === 0 &&
        fixture.hasOdds !== false,
    )
    .filter((fixture) => isTargetTourFixture(fixture, tour));

  if (!boardFixtures.length) {
    return {
      tour: tour.toUpperCase(),
      tournamentCount: 0,
      tournaments: [],
      bookmakers,
      moneylineMarketCandidates: [],
      requestWindow: window,
      fixtures: [],
    };
  }

  const fixtureIds = new Set(
    boardFixtures
      .map((fixture) => fixture.fixtureId)
      .filter((id): id is string => typeof id === "string"),
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

  const tournamentIds = Array.from(tournamentMap.keys());
  const mergedOddsFixtures = new Map<string, OddsFixture>();
  let oddsRequestCount = 0;

  // OddsPapi's live endpoint currently accepts exactly one bookmaker per
  // odds-by-tournaments request, using the singular "bookmaker" parameter.
  // We merge Winamax/Pinnacle payloads by fixtureId after retrieval.
  for (const batch of chunks(tournamentIds, 10)) {
    for (const bookmaker of bookmakers) {
      if (oddsRequestCount > 0) {
        // Respect the documented 1000 ms endpoint cooldown.
        await new Promise((resolve) => setTimeout(resolve, 1_050));
      }

      const payload = await oddsApi<unknown>(
        apiKey,
        "odds-by-tournaments",
        {
          tournamentIds: batch.join(","),
          bookmaker,
          language: "en",
          verbosity: "3",
        },
        86_400,
      );
      oddsRequestCount += 1;

      for (const fixture of normalizeFixturePayload(payload)) {
        if (typeof fixture.fixtureId !== "string") continue;

        const existing = mergedOddsFixtures.get(fixture.fixtureId);
        if (!existing) {
          mergedOddsFixtures.set(fixture.fixtureId, fixture);
          continue;
        }

        mergedOddsFixtures.set(fixture.fixtureId, {
          ...existing,
          ...fixture,
          bookmakerOdds: {
            ...(existing.bookmakerOdds ?? {}),
            ...(fixture.bookmakerOdds ?? {}),
          },
        });
      }
    }
  }

  const candidates = moneylineCandidates(catalog);
  const fixtures = Array.from(mergedOddsFixtures.values())
    .filter(
      (fixture) =>
        typeof fixture.fixtureId === "string" &&
        fixtureIds.has(fixture.fixtureId) &&
        fixture.sportId === TENNIS_SPORT_ID &&
        fixture.statusId !== 3,
    )
    .map((fixture) => {
      const prices = Object.fromEntries(
        bookmakers.map((book) => [
          book,
          extractMoneyline(fixture, book, candidates),
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

  return {
    tour: tour.toUpperCase(),
    tournamentCount: tournamentMap.size,
    tournaments: Array.from(tournamentMap.values()),
    bookmakers,
    moneylineMarketCandidates: candidates.slice(0, 10).map((market) => ({
      marketId: market.marketId,
      marketName: market.marketName ?? null,
    })),
    requestWindow: window,
    fixtures,
  };
}

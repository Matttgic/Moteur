import "jsr:@supabase/functions-js/edge-runtime.d.ts";
import { createClient } from "npm:@supabase/supabase-js@2.116.0";

function json(data: unknown, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { "content-type": "application/json" },
  });
}

async function sha256(value: string) {
  const bytes = new TextEncoder().encode(value);
  const hash = await crypto.subtle.digest("SHA-256", bytes);
  return Array.from(new Uint8Array(hash))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

function normalizeName(value: string) {
  return value
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .trim()
    .replace(/\s+/g, " ");
}

async function createAdmin() {
  const supabaseUrl = Deno.env.get("SUPABASE_URL");
  const legacyKey = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY");
  let serviceKey = legacyKey ?? null;

  if (!serviceKey) {
    const raw = Deno.env.get("SUPABASE_SECRET_KEYS");
    if (raw) {
      const parsed = JSON.parse(raw);
      serviceKey = parsed.default ?? Object.values(parsed)[0] ?? null;
    }
  }

  if (!supabaseUrl || !serviceKey) {
    throw new Error("supabase_admin_not_available");
  }

  return createClient(supabaseUrl, String(serviceKey), {
    auth: { persistSession: false, autoRefreshToken: false },
  });
}

async function cronSecretValid(supabase: any, cronSecret: string) {
  if (!cronSecret) return false;

  const { data, error } = await supabase
    .from("tennis_ingestion_runs")
    .select("details,finished_at")
    .eq("job_name", "cron-auth-binding")
    .eq("status", "success")
    .order("finished_at", { ascending: false })
    .limit(1);

  if (error) throw error;
  const expectedHash = data?.[0]?.details?.sha256;
  if (typeof expectedHash !== "string") return false;

  return expectedHash === (await sha256(cronSecret));
}

function numberOrNull(value: unknown) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function decisionKey(row: any) {
  const provider = String(row.provider_match_id ?? "");
  const selected = normalizeName(String(row.selected_player_name ?? ""));
  if (provider && selected) {
    return [String(row.tour ?? ""), provider, selected].join("|");
  }

  const pair = [
    normalizeName(String(row.player_a_name ?? "")),
    normalizeName(String(row.player_b_name ?? "")),
  ].sort().join("|");

  return [
    String(row.tour ?? ""),
    String(row.scheduled_at ?? ""),
    pair,
    selected,
  ].join("|");
}

function dedupeBestOdds(rows: any[]) {
  const groups = new Map<string, any[]>();

  for (const row of rows) {
    const key = decisionKey(row);
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key)!.push(row);
  }

  return Array.from(groups.values()).map((group) => {
    const best = [...group].sort(
      (a, b) => Number(b.odds ?? 0) - Number(a.odds ?? 0),
    )[0];
    const settled = group.find(
      (row) =>
        row.result === "WIN" ||
        row.result === "LOSS" ||
        row.result === "VOID",
    );
    const result = settled?.result ?? best.result ?? null;
    const stake = Number(best.stake_units ?? 0);
    const odds = Number(best.odds ?? 0);
    const profitUnits =
      result === "WIN"
        ? stake * (odds - 1)
        : result === "LOSS"
          ? -stake
          : result === "VOID"
            ? 0
            : null;

    return {
      ...best,
      result,
      profit_units: profitUnits,
      settled_at: settled?.settled_at ?? best.settled_at ?? null,
    };
  });
}

async function settlePending(supabase: any) {
  const { data: pending, error: pendingError } = await supabase
    .from("tennis_bet_log")
    .select("*")
    .is("result", null)
    .lt("scheduled_at", new Date().toISOString())
    .order("scheduled_at", { ascending: true })
    .limit(1000);

  if (pendingError) throw pendingError;
  if (!pending?.length) return { settled: 0 };

  const providerIds = Array.from(
    new Set(
      pending
        .map((row: any) => row.provider_match_id)
        .filter((value: unknown): value is string => typeof value === "string"),
    ),
  );

  if (!providerIds.length) return { settled: 0 };

  const { data: matches, error: matchError } = await supabase
    .from("tennis_matches")
    .select("provider_match_id,winner_id,completed_at,status")
    .in("provider_match_id", providerIds);

  if (matchError) throw matchError;

  const winnerIds = Array.from(
    new Set(
      (matches ?? [])
        .map((match: any) => match.winner_id)
        .filter((value: unknown): value is string => typeof value === "string"),
    ),
  );

  const winnerNameById = new Map<string, string>();
  if (winnerIds.length) {
    const { data: winners, error: winnerError } = await supabase
      .from("tennis_players")
      .select("id,name")
      .in("id", winnerIds);

    if (winnerError) throw winnerError;
    for (const row of winners ?? []) {
      winnerNameById.set(row.id, normalizeName(row.name));
    }
  }

  const matchByProvider = new Map(
    (matches ?? []).map((match: any) => [match.provider_match_id, match]),
  );

  let settled = 0;
  for (const bet of pending) {
    const match = matchByProvider.get(bet.provider_match_id);
    if (!match || !match.winner_id || match.status !== "completed") continue;

    const winnerName = winnerNameById.get(match.winner_id);
    if (!winnerName) continue;

    const selectedName = normalizeName(String(bet.selected_player_name ?? ""));
    const stake = Number(bet.stake_units ?? 0);
    const odds = Number(bet.odds ?? 0);
    if (!selectedName || !Number.isFinite(stake) || !Number.isFinite(odds)) continue;

    const won = selectedName === winnerName;
    const profitUnits = won ? stake * (odds - 1) : -stake;

    const { error } = await supabase
      .from("tennis_bet_log")
      .update({
        result: won ? "WIN" : "LOSS",
        profit_units: profitUnits,
        settled_at: match.completed_at ?? new Date().toISOString(),
        updated_at: new Date().toISOString(),
      })
      .eq("id", bet.id);

    if (error) throw error;
    settled += 1;
  }

  return { settled };
}

async function recordBets(supabase: any, betsInput: unknown) {
  const bets = Array.isArray(betsInput) ? betsInput : [];
  let inserted = 0;
  let updatedClosing = 0;

  for (const bet of bets) {
    const providerMatchId = String(bet?.providerMatchId ?? "");
    const bookmaker = String(bet?.bookmaker ?? "");
    const selectedSide = String(bet?.selectedSide ?? "");
    const scheduledAt = String(bet?.scheduledAt ?? "");

    if (!providerMatchId || !bookmaker || (selectedSide !== "A" && selectedSide !== "B")) {
      continue;
    }

    const { data: existing, error: existingError } = await supabase
      .from("tennis_bet_log")
      .select("id,odds,scheduled_at,closing_odds")
      .eq("provider_match_id", providerMatchId)
      .eq("bookmaker", bookmaker)
      .eq("selected_side", selectedSide)
      .maybeSingle();

    if (existingError) throw existingError;
    const currentOdds = numberOrNull(bet?.odds);

    if (existing) {
      const scheduledTs = Date.parse(existing.scheduled_at ?? scheduledAt);
      const minutesToStart = Number.isFinite(scheduledTs)
        ? (scheduledTs - Date.now()) / 60000
        : null;

      if (
        currentOdds &&
        minutesToStart !== null &&
        minutesToStart <= 180 &&
        minutesToStart >= -30
      ) {
        const openingOdds = Number(existing.odds);
        const clv =
          Number.isFinite(openingOdds) && currentOdds > 1
            ? openingOdds / currentOdds - 1
            : null;

        const { error } = await supabase
          .from("tennis_bet_log")
          .update({
            closing_odds: currentOdds,
            clv,
            updated_at: new Date().toISOString(),
          })
          .eq("id", existing.id);

        if (error) throw error;
        updatedClosing += 1;
      }
      continue;
    }

    const qualityScore = numberOrNull(bet?.qualityScore);
    const qualityGrade = bet?.qualityGrade == null ? null : String(bet.qualityGrade);
    const qualitySignals = Array.isArray(bet?.qualitySignals)
      ? bet.qualitySignals.map((value: unknown) => String(value))
      : [];
    const modelProbability = numberOrNull(bet?.modelProbability);
    const edge = numberOrNull(bet?.edge);
    const tier = String(bet?.tier ?? "NO_BET");

    const v2PolicySignal = qualitySignals.includes("quality_v2_live_evidence");
    const calibrationRedZone =
      modelProbability !== null &&
      ((modelProbability >= 0.4 && modelProbability < 0.5) ||
        (modelProbability >= 0.6 && modelProbability < 0.7));
    const extremeEdge = edge !== null && edge >= 0.2;

    // Defense in depth: until callers emit Quality v2, nothing new can enter
    // the real economic log. PREMIUM remains research-only.
    if (
      !v2PolicySignal ||
      tier !== "VALUE" ||
      qualityScore === null ||
      qualityScore < 90 ||
      calibrationRedZone ||
      extremeEdge
    ) {
      continue;
    }

    const row = {
      provider_match_id: providerMatchId,
      tour: String(bet?.tour ?? "").toUpperCase(),
      scheduled_at: scheduledAt || null,
      tournament: bet?.tournament ?? null,
      player_a_name: String(bet?.playerAName ?? ""),
      player_b_name: String(bet?.playerBName ?? ""),
      selected_side: selectedSide,
      selected_player_name: String(bet?.selectedPlayerName ?? ""),
      bookmaker,
      odds: currentOdds,
      model_probability: modelProbability,
      market_probability: numberOrNull(bet?.marketProbability),
      edge,
      ev: numberOrNull(bet?.ev),
      fair_odds: numberOrNull(bet?.fairOdds),
      tier,
      stake_units: numberOrNull(bet?.stakeUnits) ?? 0,
      model_mode: String(bet?.modelMode ?? "unknown"),
      model_quality: numberOrNull(bet?.modelQuality),
      source_count: numberOrNull(bet?.sourceCount),
      quality_score: qualityScore,
      quality_grade: qualityGrade,
      quality_components:
        bet?.qualityComponents && typeof bet.qualityComponents === "object"
          ? bet.qualityComponents
          : null,
      quality_signals: qualitySignals,
      updated_at: new Date().toISOString(),
    };

    if (
      !row.player_a_name ||
      !row.player_b_name ||
      !row.selected_player_name ||
      !row.odds ||
      !row.model_probability ||
      row.stake_units <= 0
    ) continue;

    if (
      row.quality_score !== null &&
      (row.quality_score < 0 || row.quality_score > 100)
    ) continue;

    const { error } = await supabase.from("tennis_bet_log").insert(row);
    if (error) {
      if (String(error.code) === "23505") continue;
      throw error;
    }
    inserted += 1;
  }

  return { inserted, updatedClosing };
}

function aggregate(rows: any[]) {
  const settled = rows.filter(
    (row) => row.result === "WIN" || row.result === "LOSS" || row.result === "VOID",
  );
  const graded = settled.filter(
    (row) => row.result === "WIN" || row.result === "LOSS",
  );
  const pending = rows.length - settled.length;
  const wins = graded.filter((row) => row.result === "WIN").length;
  const losses = graded.filter((row) => row.result === "LOSS").length;
  const voids = settled.filter((row) => row.result === "VOID").length;
  const stake = graded.reduce(
    (sum, row) => sum + Number(row.stake_units ?? 0),
    0,
  );
  const profit = graded.reduce(
    (sum, row) => sum + Number(row.profit_units ?? 0),
    0,
  );
  const roi = stake > 0 ? profit / stake : null;
  const hitRate = graded.length ? wins / graded.length : null;
  const avgOdds = rows.length
    ? rows.reduce((sum, row) => sum + Number(row.odds ?? 0), 0) / rows.length
    : null;
  const avgEdge = rows.length
    ? rows.reduce((sum, row) => sum + Number(row.edge ?? 0), 0) / rows.length
    : null;
  const avgEv = rows.length
    ? rows.reduce((sum, row) => sum + Number(row.ev ?? 0), 0) / rows.length
    : null;

  const qualityRows = rows.filter(
    (row) => row.quality_score !== null && Number.isFinite(Number(row.quality_score)),
  );
  const avgQualityScore = qualityRows.length
    ? qualityRows.reduce((sum, row) => sum + Number(row.quality_score), 0) /
      qualityRows.length
    : null;

  const clvRows = rows.filter(
    (row) => row.clv !== null && Number.isFinite(Number(row.clv)),
  );
  const avgClv = clvRows.length
    ? clvRows.reduce((sum, row) => sum + Number(row.clv), 0) / clvRows.length
    : null;

  const equityRows = [...graded].sort((a, b) => {
    const left = Date.parse(a.settled_at ?? a.scheduled_at ?? a.created_at);
    const right = Date.parse(b.settled_at ?? b.scheduled_at ?? b.created_at);
    return left - right;
  });

  let equity = 0;
  let peak = 0;
  let maxDrawdown = 0;
  for (const row of equityRows) {
    equity += Number(row.profit_units ?? 0);
    peak = Math.max(peak, equity);
    maxDrawdown = Math.max(maxDrawdown, peak - equity);
  }

  return {
    total: rows.length,
    settled: settled.length,
    graded: graded.length,
    pending,
    wins,
    losses,
    voids,
    stakeUnits: stake,
    profitUnits: profit,
    roi,
    hitRate,
    avgOdds,
    avgEdge,
    avgEv,
    avgQualityScore,
    maxDrawdownUnits: maxDrawdown,
    avgClv,
    clvSamples: clvRows.length,
  };
}

function groupSummary(rows: any[], key: string) {
  const groups = new Map<string, any[]>();
  for (const row of rows) {
    const value = String(row?.[key] ?? "UNKNOWN");
    if (!groups.has(value)) groups.set(value, []);
    groups.get(value)!.push(row);
  }
  return Array.from(groups.entries()).map(([name, values]) => ({
    name,
    ...aggregate(values),
  }));
}

async function economicSummary(supabase: any) {
  await settlePending(supabase);

  const { data, error } = await supabase
    .from("tennis_bet_log")
    .select("*")
    .order("created_at", { ascending: true })
    .limit(10000);

  if (error) throw error;
  const rawRows = data ?? [];
  const rows = dedupeBestOdds(rawRows);
  const overall = {
    ...aggregate(rows),
    rawTotal: rawRows.length,
    uniqueDecisions: rows.length,
    duplicateRows: Math.max(0, rawRows.length - rows.length),
  };
  const sampleStatus =
    overall.graded >= 300
      ? "ROBUST_SAMPLE"
      : overall.graded >= 100
        ? "VALIDATING"
        : overall.graded >= 30
          ? "EARLY_SAMPLE"
          : "COLLECTING";

  const last10 = [...rows]
    .filter((row) => row.result === "WIN" || row.result === "LOSS")
    .sort((a, b) => {
      const left = Date.parse(a.settled_at ?? a.scheduled_at ?? a.created_at);
      const right = Date.parse(b.settled_at ?? b.scheduled_at ?? b.created_at);
      return right - left;
    })
    .slice(0, 10)
    .map((row) => ({
      result: row.result,
      tour: row.tour,
      player: row.selected_player_name,
      bookmaker: row.bookmaker,
      odds: row.odds,
      stakeUnits: row.stake_units,
      profitUnits: row.profit_units,
      modelMode: row.model_mode,
      tier: row.tier,
      qualityScore: row.quality_score,
      qualityGrade: row.quality_grade,
      scheduledAt: row.scheduled_at,
    }));

  const history = [...rawRows]
    .sort((a, b) => Date.parse(b.created_at) - Date.parse(a.created_at))
    .slice(0, 500)
    .map((row) => ({
      id: row.id,
      providerMatchId: row.provider_match_id,
      tour: row.tour,
      scheduledAt: row.scheduled_at,
      tournament: row.tournament,
      playerA: row.player_a_name,
      playerB: row.player_b_name,
      selectedPlayer: row.selected_player_name,
      selectedSide: row.selected_side,
      bookmaker: row.bookmaker,
      odds: row.odds,
      modelProbability: row.model_probability,
      marketProbability: row.market_probability,
      edge: row.edge,
      ev: row.ev,
      fairOdds: row.fair_odds,
      tier: row.tier,
      stakeUnits: row.stake_units,
      modelMode: row.model_mode,
      modelQuality: row.model_quality,
      qualityScore: row.quality_score,
      qualityGrade: row.quality_grade,
      qualityComponents: row.quality_components,
      qualitySignals: row.quality_signals,
      closingOdds: row.closing_odds,
      clv: row.clv,
      result: row.result,
      profitUnits: row.profit_units,
      settledAt: row.settled_at,
      createdAt: row.created_at,
    }));

  return {
    generatedAt: new Date().toISOString(),
    sampleStatus,
    overall,
    byTour: groupSummary(rows, "tour"),
    byModel: groupSummary(rows, "model_mode"),
    byTier: groupSummary(rows, "tier"),
    byQuality: groupSummary(rows, "quality_grade"),
    last10,
    history,
    decisionPolicy: {
      aggregation: "unique_match_selected_player_best_odds",
      rawRows: rawRows.length,
      uniqueDecisions: rows.length,
      duplicateRows: Math.max(0, rawRows.length - rows.length),
    },
  };
}

Deno.serve(async (req: Request) => {
  if (req.method !== "POST") return json({ error: "method_not_allowed" }, 405);

  try {
    const body = await req.json().catch(() => null);
    const cronSecret = String(body?.cronSecret ?? "");
    const action = String(body?.action ?? "summary");
    const supabase = await createAdmin();

    if (!(await cronSecretValid(supabase, cronSecret))) {
      return json({ error: "unauthorized" }, 401);
    }

    if (action === "record_bets") {
      return json({ ok: true, result: await recordBets(supabase, body?.bets) });
    }
    if (action === "summary") {
      return json({ ok: true, summary: await economicSummary(supabase) });
    }
    if (action === "settle") {
      return json({ ok: true, result: await settlePending(supabase) });
    }

    return json({ error: "unsupported_action" }, 400);
  } catch (error) {
    return json(
      { ok: false, error: error instanceof Error ? error.message : String(error) },
      500,
    );
  }
});

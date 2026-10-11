
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

async function admin() {
  const url = Deno.env.get("SUPABASE_URL");
  const legacy = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY");
  let key = legacy ?? null;

  if (!key) {
    const raw = Deno.env.get("SUPABASE_SECRET_KEYS");
    if (raw) {
      const parsed = JSON.parse(raw);
      key = parsed.default ?? Object.values(parsed)[0] ?? null;
    }
  }

  if (!url || !key) throw new Error("supabase_admin_not_available");

  return createClient(url, String(key), {
    auth: { persistSession: false, autoRefreshToken: false },
  });
}

async function authorized(supabase: any, cronSecret: string) {
  if (!cronSecret) return false;

  const { data, error } = await supabase
    .from("tennis_ingestion_runs")
    .select("details")
    .eq("job_name", "cron-auth-binding")
    .eq("status", "success")
    .order("finished_at", { ascending: false })
    .limit(1);

  if (error) throw error;

  const expected = data?.[0]?.details?.sha256;
  return typeof expected === "string" && expected === (await sha256(cronSecret));
}

function validTour(value: unknown): value is "ATP" | "WTA" {
  return value === "ATP" || value === "WTA";
}

function num(value: unknown, fallback = 0) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function marketDepthScore(sourceCount: number) {
  if (sourceCount >= 4) return 15;
  if (sourceCount === 3) return 13;
  if (sourceCount === 2) return 10;
  if (sourceCount === 1) return 6;
  return 0;
}

function sharpScore(input: {
  sharpProbability: number | null;
  marketProbability: number;
}) {
  if (input.sharpProbability == null) return 5;
  const sharpEdge = input.sharpProbability - input.marketProbability;
  if (sharpEdge >= 0.03) return 15;
  if (sharpEdge >= 0.01) return 13;
  if (sharpEdge >= 0) return 10;
  if (sharpEdge > -0.02) return 6;
  return 2;
}

function edgeScore(edge: number) {
  if (edge <= 0) return 0;
  if (edge < 0.03) return 3;
  if (edge < 0.06) return 6;
  if (edge < 0.08) return 10;
  if (edge < 0.10) return 8;
  if (edge < 0.12) return 6;
  if (edge < 0.16) return 4;
  if (edge < 0.20) return 2;
  return 0;
}

function evScore(ev: number) {
  if (ev <= 0) return 0;
  if (ev < 0.03) return 3;
  if (ev < 0.20) return 10;
  if (ev < 0.40) return 8;
  if (ev < 0.80) return 6;
  return 0;
}

function priceStabilityScore(value: number | null) {
  if (value == null) return 5;
  if (value <= 1.05) return 10;
  if (value <= 1.10) return 9;
  if (value <= 1.20) return 6;
  if (value <= 1.35) return 3;
  return 0;
}

function oddsBandScore(odds: number) {
  if (!Number.isFinite(odds) || odds <= 1 || odds > 15) return 0;
  if (odds <= 1.5) return 5;
  if (odds <= 2.5) return 1;
  if (odds <= 4) return 5;
  if (odds <= 6) return 2;
  return 0;
}

function calibrationScore(probability: number) {
  if (probability < 0.4) return 8;
  if (probability < 0.5) return 0;
  if (probability < 0.6) return 10;
  if (probability < 0.7) return 0;
  if (probability < 0.8) return 8;
  return 10;
}

function lineMovementScore(value: number | null) {
  if (value == null || !Number.isFinite(value)) return 3;
  if (value >= 0.03) return 5;
  if (value >= 0) return 4;
  if (value > -0.03) return 2;
  return 0;
}

function qualityV2(row: any) {
  const model = row?.model ?? {};
  const market = row?.market ?? {};
  const decision = row?.decision ?? {};
  const sharp =
    decision.sharpProbability == null ? null : num(decision.sharpProbability);
  const marketProbability = num(decision.marketProbability);
  const modelProbability = num(decision.modelProbability);
  const edge = num(decision.edge);
  const ev = num(decision.ev);
  const odds = num(decision.odds);
  const spread =
    market.priceSpreadRatio == null ? null : num(market.priceSpreadRatio);
  const movement =
    market.lineMovementPct == null ? null : num(market.lineMovementPct);

  const score = Math.max(
    0,
    Math.min(
      100,
      Math.round(
        Math.max(0, Math.min(1, num(model.quality))) * 20 +
          marketDepthScore(Math.max(0, Math.floor(num(market.sourceCount)))) +
          sharpScore({ sharpProbability: sharp, marketProbability }) +
          edgeScore(edge) +
          evScore(ev) +
          priceStabilityScore(spread) +
          oddsBandScore(odds) +
          calibrationScore(modelProbability) +
          lineMovementScore(movement),
      ),
    ),
  );

  const redZone =
    (modelProbability >= 0.4 && modelProbability < 0.5) ||
    (modelProbability >= 0.6 && modelProbability < 0.7);
  const rawTier = String(decision.rawTier ?? decision.tier ?? "");

  return {
    score,
    eligible:
      model.mode === "full_logit" &&
      rawTier === "VALUE" &&
      score >= 90 &&
      !redZone &&
      edge < 0.2,
    rawTier,
    redZone,
    edge,
  };
}

function applyPolicyV2(payloadInput: unknown) {
  if (!payloadInput || typeof payloadInput !== "object" || Array.isArray(payloadInput)) {
    return payloadInput;
  }

  const payload = { ...(payloadInput as Record<string, any>) };
  if (!Array.isArray(payload.allAnalyzed)) return payload;

  const allAnalyzed = payload.allAnalyzed.map((row: any) => {
    const policy = qualityV2(row);
    const decision = row?.decision ?? {};
    const guard = decision?.guard ?? {};
    const reasons = new Set<string>(
      Array.isArray(guard.reasons) ? guard.reasons.map(String) : [],
    );

    if (policy.rawTier === "PREMIUM") reasons.add("premium_shadow_policy_v2");
    if (policy.score < 90) reasons.add("quality_v2_below_90");
    if (policy.redZone) reasons.add("live_calibration_red_zone");
    if (policy.edge >= 0.2) reasons.add("extreme_live_edge_red_zone");

    const blockedByV2 = !policy.eligible;

    return {
      ...row,
      decision: {
        ...decision,
        bet: Boolean(decision.bet) && !blockedByV2,
        guard: {
          ...guard,
          blocked: Boolean(guard.blocked) || blockedByV2,
          reasons: Array.from(reasons),
        },
        policyV2: {
          score: policy.score,
          threshold: 90,
          eligible: policy.eligible,
          mode: "VALUE_ONLY",
        },
      },
    };
  });

  const bets = allAnalyzed
    .filter((row: any) => Boolean(row?.decision?.policyV2?.eligible))
    .sort(
      (a: any, b: any) =>
        num(b?.decision?.policyV2?.score) - num(a?.decision?.policyV2?.score) ||
        num(b?.decision?.ev) - num(a?.decision?.ev),
    )
    .slice(0, 5);

  const dataUnavailable = payload.status === "DATA_UNAVAILABLE";

  return {
    ...payload,
    allAnalyzed,
    bets,
    bestBet: bets[0] ?? null,
    status: dataUnavailable
      ? "DATA_UNAVAILABLE"
      : bets.length
        ? "BET_OPPORTUNITIES_FOUND"
        : "NO_BET_TODAY",
    policyOverlay: {
      version: "v2_live_evidence",
      execution: "VALUE_ONLY",
      qualityThreshold: 90,
      premiumMode: "SHADOW",
    },
  };
}

Deno.serve(async (req: Request) => {
  if (req.method !== "POST") return json({ error: "method_not_allowed" }, 405);

  try {
    const body = await req.json().catch(() => null);
    const action = String(body?.action ?? "");
    const cronSecret = String(body?.cronSecret ?? "");
    const tour = String(body?.tour ?? "").toUpperCase();

    const supabase = await admin();

    if (!(await authorized(supabase, cronSecret))) {
      return json({ error: "unauthorized" }, 401);
    }

    if (!validTour(tour)) {
      return json({ error: "unsupported_tour", allowed: ["ATP", "WTA"] }, 400);
    }

    if (action === "get") {
      const { data, error } = await supabase
        .from("tennis_selection_snapshots")
        .select("tour,payload,generated_at,updated_at")
        .eq("tour", tour)
        .maybeSingle();

      if (error) throw error;

      if (!data) return json({ ok: true, snapshot: null });

      return json({
        ok: true,
        snapshot: {
          tour: data.tour,
          payload: applyPolicyV2(data.payload ?? {}),
          generatedAt: data.generated_at,
          updatedAt: data.updated_at,
        },
      });
    }

    if (action === "save") {
      const payload =
        body?.payload && typeof body.payload === "object" && !Array.isArray(body.payload)
          ? body.payload
          : null;

      if (!payload) {
        return json({ error: "invalid_payload" }, 400);
      }

      const generatedAt =
        typeof payload.generatedAt === "string"
          ? payload.generatedAt
          : new Date().toISOString();
      const updatedAt = new Date().toISOString();

      const { error } = await supabase
        .from("tennis_selection_snapshots")
        .upsert(
          {
            tour,
            payload,
            generated_at: generatedAt,
            updated_at: updatedAt,
          },
          { onConflict: "tour" },
        );

      if (error) throw error;

      return json({
        ok: true,
        snapshot: { tour, generatedAt, updatedAt },
      });
    }

    return json({ error: "unsupported_action" }, 400);
  } catch (error) {
    return json(
      {
        ok: false,
        error: error instanceof Error ? error.message : String(error),
      },
      500,
    );
  }
});

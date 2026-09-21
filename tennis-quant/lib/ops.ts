const OPS_EDGE_URL =
  "https://uciolnhvddbindxajzti.supabase.co/functions/v1/tennis-ops-runtime";

async function opsRequest(
  action:
    | "heartbeat"
    | "settle_all"
    | "pending_clv"
    | "update_clv"
    | "record_shadow"
    | "shadow_summary"
    | "health_summary",
  payload: Record<string, unknown> = {},
) {
  const cronSecret = process.env.CRON_SECRET;
  if (!cronSecret) return null;

  const response = await fetch(OPS_EDGE_URL, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      accept: "application/json",
    },
    cache: "no-store",
    body: JSON.stringify({
      action,
      cronSecret,
      ...payload,
    }),
  });

  const body = await response.json().catch(() => null);

  if (!response.ok) {
    throw new Error(
      `Ops runtime ${action} failed with HTTP ${response.status}`,
    );
  }

  return body;
}

export async function heartbeat(input: {
  jobName: string;
  tour?: "ATP" | "WTA" | null;
  status: "success" | "failed";
  details?: Record<string, unknown>;
}) {
  return opsRequest("heartbeat", input);
}

export async function settleAll() {
  return opsRequest("settle_all");
}

export async function getPendingClv() {
  return opsRequest("pending_clv");
}

export async function updateClvQuotes(
  updates: Array<{
    kind: "bet" | "shadow";
    id: string;
    closingOdds: number;
  }>,
) {
  if (!updates.length) return null;
  return opsRequest("update_clv", { updates });
}

export async function recordShadowPicks(
  tour: "ATP" | "WTA",
  picks: Array<{
    variant: string;
    providerMatchId: string;
    scheduledAt: string | null;
    tournament: string | null;
    playerAName: string;
    playerBName: string;
    selectedSide: "A" | "B";
    selectedPlayerName: string;
    bookmaker: string;
    odds: number;
    modelProbability: number;
    marketProbability: number;
    edge: number;
    ev: number;
    fairOdds: number;
  }>,
) {
  if (!picks.length) return null;
  return opsRequest("record_shadow", {
    picks: picks.map((pick) => ({ ...pick, tour })),
  });
}

export async function getShadowSummary() {
  return opsRequest("shadow_summary");
}

export async function getHealthSummary() {
  return opsRequest("health_summary");
}

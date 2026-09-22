const SNAPSHOT_EDGE_URL =
  "https://uciolnhvddbindxajzti.supabase.co/functions/v1/tennis-selection-snapshots";

export type SelectionTour = "ATP" | "WTA";

export type SelectionSnapshot = {
  tour: SelectionTour;
  payload: Record<string, unknown>;
  generatedAt: string;
  updatedAt: string;
};

async function snapshotRequest(
  action: "get" | "save",
  tour: SelectionTour,
  payload?: Record<string, unknown>,
) {
  const cronSecret = process.env.CRON_SECRET;
  if (!cronSecret) {
    throw new Error("selection_snapshot_store_not_configured");
  }

  const response = await fetch(SNAPSHOT_EDGE_URL, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      accept: "application/json",
    },
    cache: "no-store",
    body: JSON.stringify({
      action,
      tour,
      cronSecret,
      ...(payload ? { payload } : {}),
    }),
  });

  const body = await response.json().catch(() => null);

  if (!response.ok) {
    const detail =
      body && typeof body === "object" && "error" in body
        ? String(body.error)
        : `http_${response.status}`;
    throw new Error(`selection_snapshot_runtime_failed:${detail}`);
  }

  return body;
}

export async function getSelectionSnapshot(
  tour: SelectionTour,
): Promise<SelectionSnapshot | null> {
  const body = await snapshotRequest("get", tour);
  const snapshot = body?.snapshot;

  if (!snapshot) return null;

  return {
    tour: snapshot.tour as SelectionTour,
    payload:
      snapshot.payload && typeof snapshot.payload === "object"
        ? (snapshot.payload as Record<string, unknown>)
        : {},
    generatedAt: String(snapshot.generatedAt),
    updatedAt: String(snapshot.updatedAt),
  };
}

export async function saveSelectionSnapshot(
  tour: SelectionTour,
  payload: Record<string, unknown>,
) {
  const body = await snapshotRequest("save", tour, payload);
  const snapshot = body?.snapshot;

  if (!snapshot) {
    throw new Error("selection_snapshot_runtime_invalid_response");
  }

  return {
    generatedAt: String(snapshot.generatedAt),
    updatedAt: String(snapshot.updatedAt),
  };
}

import { getSupabaseServerClient } from "./supabase";

export type SelectionTour = "ATP" | "WTA";

export type SelectionSnapshot = {
  tour: SelectionTour;
  payload: Record<string, unknown>;
  generatedAt: string;
  updatedAt: string;
};

export async function getSelectionSnapshot(
  tour: SelectionTour,
): Promise<SelectionSnapshot | null> {
  const supabase = getSupabaseServerClient();
  if (!supabase) {
    throw new Error("selection_snapshot_store_not_configured");
  }

  const { data, error } = await (supabase as any)
    .from("tennis_selection_snapshots")
    .select("tour,payload,generated_at,updated_at")
    .eq("tour", tour)
    .maybeSingle();

  if (error) throw error;
  if (!data) return null;

  return {
    tour: data.tour as SelectionTour,
    payload:
      data.payload && typeof data.payload === "object"
        ? (data.payload as Record<string, unknown>)
        : {},
    generatedAt: String(data.generated_at),
    updatedAt: String(data.updated_at),
  };
}

export async function saveSelectionSnapshot(
  tour: SelectionTour,
  payload: Record<string, unknown>,
) {
  const supabase = getSupabaseServerClient();
  if (!supabase) {
    throw new Error("selection_snapshot_store_not_configured");
  }

  const generatedAt =
    typeof payload.generatedAt === "string"
      ? payload.generatedAt
      : new Date().toISOString();
  const updatedAt = new Date().toISOString();

  const { error } = await (supabase as any)
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

  return { generatedAt, updatedAt };
}

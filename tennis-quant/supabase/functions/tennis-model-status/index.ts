import "jsr:@supabase/functions-js/edge-runtime.d.ts";
import { createClient } from "npm:@supabase/supabase-js@2";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
  "Access-Control-Allow-Methods": "GET, OPTIONS",
};

Deno.serve(async (req: Request) => {
  if (req.method === "OPTIONS") {
    return new Response("ok", { headers: corsHeaders });
  }

  if (req.method !== "GET") {
    return new Response(JSON.stringify({ error: "method_not_allowed" }), {
      status: 405,
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  }

  try {
    const url = Deno.env.get("SUPABASE_URL");
    const secretKeysRaw = Deno.env.get("SUPABASE_SECRET_KEYS");

    if (!url || !secretKeysRaw) {
      throw new Error("supabase_env_missing");
    }

    const secretKeys = JSON.parse(secretKeysRaw);
    const secret = secretKeys["default"];

    if (!secret) {
      throw new Error("supabase_secret_key_missing");
    }

    const supabase = createClient(url, secret, {
      auth: {
        persistSession: false,
        autoRefreshToken: false,
      },
    });

    const { data, error } = await supabase
      .from("tennis_model_runs")
      .select(
        "tour,model_name,model_version,trained_through,validation_method,brier_score,log_loss,roi,avg_clv,sample_size,created_at",
      )
      .order("tour", { ascending: true })
      .order("created_at", { ascending: false });

    if (error) throw error;

    const latestByTour: Record<string, unknown> = {};
    for (const row of data ?? []) {
      if (!latestByTour[row.tour]) {
        latestByTour[row.tour] = row;
      }
    }

    return new Response(
      JSON.stringify({
        service: "tennis-quant",
        status: "ok",
        economic_validation: {
          roi_validated: Object.values(latestByTour).some(
            (row: any) => row?.roi !== null,
          ),
          clv_validated: Object.values(latestByTour).some(
            (row: any) => row?.avg_clv !== null,
          ),
        },
        models: latestByTour,
      }),
      {
        status: 200,
        headers: {
          ...corsHeaders,
          "Content-Type": "application/json",
          "Cache-Control": "public, max-age=60, s-maxage=300",
        },
      },
    );
  } catch (error) {
    return new Response(
      JSON.stringify({
        service: "tennis-quant",
        status: "error",
        error: error instanceof Error ? error.message : "unknown_error",
      }),
      {
        status: 500,
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      },
    );
  }
});

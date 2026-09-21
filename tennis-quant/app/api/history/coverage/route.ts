import { NextResponse } from "next/server";
import { getHistoryCoverage } from "@/lib/live-tennis";

export const runtime = "nodejs";

export async function GET() {
  const apiKey = process.env.LIVE_TENNIS_API_KEY;

  if (!apiKey) {
    return NextResponse.json(
      {
        error: "provider_not_configured",
        required_env: "LIVE_TENNIS_API_KEY",
      },
      { status: 503 },
    );
  }

  try {
    const payload = await getHistoryCoverage(apiKey);
    return NextResponse.json({
      provider: "Live Tennis API",
      required_tier: "BASIC",
      entitlement_ok: true,
      payload,
    });
  } catch (error) {
    const status =
      typeof error === "object" &&
      error &&
      "status" in error &&
      typeof (error as { status?: unknown }).status === "number"
        ? (error as { status: number }).status
        : 502;

    return NextResponse.json(
      {
        provider: "Live Tennis API",
        required_tier: "BASIC",
        entitlement_ok: false,
        error: error instanceof Error ? error.message : "Unknown error",
      },
      { status: status === 403 ? 403 : 502 },
    );
  }
}

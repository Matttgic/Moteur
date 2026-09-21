import { NextResponse } from "next/server";
import { getEconomicSummary } from "@/lib/economics";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const payload = await getEconomicSummary();

    if (!payload?.summary) {
      return NextResponse.json(
        { error: "economic_tracking_not_configured" },
        { status: 503 },
      );
    }

    return NextResponse.json(payload);
  } catch (error) {
    return NextResponse.json(
      {
        error:
          error instanceof Error
            ? error.message
            : "economic_tracking_error",
      },
      { status: 502 },
    );
  }
}

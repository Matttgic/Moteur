import { NextResponse } from "next/server";
import { getHealthSummary } from "@/lib/ops";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const payload = await getHealthSummary();

    if (!payload?.health) {
      return NextResponse.json(
        { error: "health_tracking_not_configured" },
        { status: 503 },
      );
    }

    return NextResponse.json(payload);
  } catch (error) {
    return NextResponse.json(
      {
        error:
          error instanceof Error ? error.message : "health_tracking_error",
      },
      { status: 502 },
    );
  }
}

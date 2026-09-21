import { NextResponse } from "next/server";
import { getShadowSummary } from "@/lib/ops";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const payload = await getShadowSummary();

    if (!payload?.variants) {
      return NextResponse.json(
        { error: "shadow_tracking_not_configured" },
        { status: 503 },
      );
    }

    return NextResponse.json(payload);
  } catch (error) {
    return NextResponse.json(
      {
        error:
          error instanceof Error ? error.message : "shadow_tracking_error",
      },
      { status: 502 },
    );
  }
}

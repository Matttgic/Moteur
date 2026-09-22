import { NextRequest, NextResponse } from "next/server";
import { isCronAuthorized } from "@/lib/cron-auth";\nimport { heartbeat, settleAll } from "@/lib/ops";

export const runtime = "nodejs";
export const maxDuration = 120;

export async function GET(request: NextRequest) {
  if (!isCronAuthorized(request)) {
    return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  }

  try {
    const payload = await settleAll();

    await heartbeat({
      jobName: "economic-settlement",
      status: "success",
      details: payload?.result ?? {},
    });

    return NextResponse.json({
      ok: true,
      result: payload?.result ?? payload,
      completedAt: new Date().toISOString(),
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "unknown_error";

    try {
      await heartbeat({
        jobName: "economic-settlement",
        status: "failed",
        details: { message },
      });
    } catch {}

    return NextResponse.json({ ok: false, error: message }, { status: 500 });
  }
}

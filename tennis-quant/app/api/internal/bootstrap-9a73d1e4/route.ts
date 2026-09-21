import { createHash, timingSafeEqual } from "node:crypto";
import { NextRequest, NextResponse } from "next/server";
import { syncCompletedHistory } from "@/lib/history-ingest";
import type { LiveTour } from "@/lib/live-tennis";

export const runtime = "nodejs";
export const maxDuration = 300;

const TOKEN_SHA256 =
  "278f1bd5ea9787a0c217d63cbf5dd69c53d801fcbbbdcc149393b196e6f17658";
const EXPIRES_AT = Date.parse("2026-09-22T00:00:00.000Z");

const WINDOWS = [
  ["2026-05-26", "2026-06-15"],
  ["2026-06-16", "2026-06-30"],
  ["2026-07-01", "2026-07-15"],
  ["2026-07-16", "2026-07-31"],
  ["2026-08-01", "2026-08-15"],
  ["2026-08-16", "2026-08-31"],
  ["2026-09-01", "2026-09-10"],
  ["2026-09-11", "2026-09-21"],
] as const;

function validToken(token: string) {
  const actual = createHash("sha256").update(token).digest();
  const expected = Buffer.from(TOKEN_SHA256, "hex");
  return actual.length === expected.length && timingSafeEqual(actual, expected);
}

export async function GET(request: NextRequest) {
  if (process.env.VERCEL_ENV !== "production") {
    return NextResponse.json({ error: "not_found" }, { status: 404 });
  }
  if (Date.now() > EXPIRES_AT) {
    return NextResponse.json({ error: "expired" }, { status: 410 });
  }

  const token = request.nextUrl.searchParams.get("token") ?? "";
  if (!validToken(token)) {
    return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  }

  const rawTour = request.nextUrl.searchParams.get("tour")?.toLowerCase();
  const part = Number(request.nextUrl.searchParams.get("part"));

  if ((rawTour !== "atp" && rawTour !== "wta") || !Number.isInteger(part) || part < 1 || part > WINDOWS.length) {
    return NextResponse.json(
      { error: "invalid_params", allowedParts: [1, 2, 3, 4, 5, 6, 7, 8] },
      { status: 400 },
    );
  }

  const apiKey = process.env.LIVE_TENNIS_API_KEY;
  if (!apiKey) {
    return NextResponse.json({ error: "LIVE_TENNIS_API_KEY_missing" }, { status: 503 });
  }

  const [from, to] = WINDOWS[part - 1];
  try {
    const result = await syncCompletedHistory({
      apiKey,
      tour: rawTour as LiveTour,
      from: new Date(`${from}T00:00:00.000Z`).toISOString(),
      to: new Date(`${to}T23:59:59.999Z`).toISOString(),
    });
    return NextResponse.json({
      ok: true,
      tour: rawTour.toUpperCase(),
      part,
      from,
      to,
      result,
    });
  } catch (error) {
    return NextResponse.json(
      {
        ok: false,
        tour: rawTour.toUpperCase(),
        part,
        from,
        to,
        error: error instanceof Error ? error.message : "unknown_error",
      },
      { status: 500 },
    );
  }
}

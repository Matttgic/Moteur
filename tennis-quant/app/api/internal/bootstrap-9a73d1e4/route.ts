import { createHash, timingSafeEqual } from "node:crypto";
import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";
export const maxDuration = 300;

const TOKEN_SHA256 =
  "f82766732773229469533ca34c6f2ff5b1726fa18363f32c98b2918c1173cca0";
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
  const cronSecret = process.env.CRON_SECRET;

  if (!apiKey) {
    return NextResponse.json(
      { error: "LIVE_TENNIS_API_KEY_missing" },
      { status: 503 },
    );
  }

  if (!cronSecret) {
    return NextResponse.json(
      { error: "CRON_SECRET_missing" },
      { status: 503 },
    );
  }

  const [from, to] = WINDOWS[part - 1];
  try {
    const edgeResponse = await fetch(
      "https://uciolnhvddbindxajzti.supabase.co/functions/v1/tennis-basic-backfill",
      {
        method: "POST",
        headers: {
          "content-type": "application/json",
          "accept": "application/json",
        },
        body: JSON.stringify({
          action: "sync",
          token,
          cronSecret,
          liveApiKey: apiKey,
          tour: rawTour,
          from: new Date(`${from}T00:00:00.000Z`).toISOString(),
          to: new Date(`${to}T23:59:59.999Z`).toISOString(),
        }),
      },
    );

    const payload = await edgeResponse.json().catch(() => null);

    if (!edgeResponse.ok) {
      return NextResponse.json(
        {
          ok: false,
          tour: rawTour.toUpperCase(),
          part,
          from,
          to,
          error:
            payload && typeof payload === "object"
              ? payload
              : `edge_http_${edgeResponse.status}`,
        },
        { status: 500 },
      );
    }

    return NextResponse.json({
      ok: true,
      tour: rawTour.toUpperCase(),
      part,
      from,
      to,
      result: payload?.result ?? payload,
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

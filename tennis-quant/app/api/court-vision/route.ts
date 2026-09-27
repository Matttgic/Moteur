import { NextRequest, NextResponse } from "next/server";

// Relais lecture seule vers Live Tennis API pour la page /court-vision.
// La clé LIVE_TENNIS_API_KEY reste côté serveur. Le quota journalier est partagé
// avec les crons du moteur : une réserve (COURT_VISION_RESERVE, 300 par défaut)
// leur est toujours laissée.

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const BASE_URL = "https://api.livetennisapi.com/api/public/v1";

const ALLOWED_PATHS = [
  /^\/matches$/,
  /^\/matches\/\d+$/,
  /^\/matches\/\d+\/score$/,
  /^\/fixtures$/,
  /^\/tournaments$/,
  /^\/players$/,
  /^\/players\/\d+$/,
  /^\/h2h$/,
  /^\/history\/matches$/,
  /^\/history\/matches\/\d+$/,
  /^\/history\/archive\/career$/,
  /^\/usage$/,
];

// Durée de cache CDN par type de donnée (secondes) : plusieurs visiteurs = un seul appel.
function cdnTtl(path: string, params: URLSearchParams) {
  if (path === "/matches" && (params.get("status") ?? "live") === "live") return 20;
  if (/^\/matches\/\d+/.test(path)) return 15;
  if (/^\/history\/matches\/\d+$/.test(path)) return 60;
  if (path === "/usage") return 30;
  return 300;
}

let usageCache: { at: number; remaining: number | null } | null = null;

// /usage ne consomme pas de quota (documentation Live Tennis API).
async function remainingToday(apiKey: string): Promise<number | null> {
  if (usageCache && Date.now() - usageCache.at < 60_000) return usageCache.remaining;
  try {
    const res = await fetch(`${BASE_URL}/usage`, {
      headers: { Authorization: `Bearer ${apiKey}` },
      cache: "no-store",
    });
    const body = res.ok ? await res.json() : null;
    const remaining =
      typeof body?.today?.remaining_day === "number" ? body.today.remaining_day : null;
    usageCache = { at: Date.now(), remaining };
    return remaining;
  } catch {
    return null;
  }
}

function json(body: unknown, status: number) {
  return NextResponse.json(body, {
    status,
    headers: { "Cache-Control": "no-store" },
  });
}

export async function GET(request: NextRequest) {
  const apiKey = process.env.LIVE_TENNIS_API_KEY;
  const accessCode = process.env.COURT_VISION_ACCESS_CODE ?? "";
  const reserve = Number(process.env.COURT_VISION_RESERVE ?? 300);
  const params = new URLSearchParams(request.nextUrl.searchParams);
  const path = params.get("path") ?? "";
  params.delete("path");

  if (path === "__status") {
    return json({ configured: Boolean(apiKey), needsCode: Boolean(accessCode) }, 200);
  }
  if (!apiKey) {
    return json(
      { error: "not_configured", message: "LIVE_TENNIS_API_KEY manquante." },
      503,
    );
  }
  if (accessCode && request.headers.get("x-access-code") !== accessCode) {
    return json({ error: "access_code", message: "Code d'accès requis." }, 401);
  }
  if (!ALLOWED_PATHS.some((pattern) => pattern.test(path))) {
    return json({ error: "path_not_allowed" }, 400);
  }

  if (path !== "/usage") {
    const remaining = await remainingToday(apiKey);
    if (remaining !== null && remaining <= reserve) {
      return json(
        {
          error: "quota_reserved",
          message: `Quota du jour presque épuisé : les ${reserve} dernières requêtes sont réservées au moteur.`,
        },
        429,
      );
    }
  }

  const query = params.toString();
  try {
    const upstream = await fetch(`${BASE_URL}${path}${query ? `?${query}` : ""}`, {
      headers: { Authorization: `Bearer ${apiKey}`, Accept: "application/json" },
      cache: "no-store",
    });
    const body = await upstream.text();
    const ttl = cdnTtl(path, params);
    const cacheable = upstream.ok && !accessCode;
    return new NextResponse(body, {
      status: upstream.status,
      headers: {
        "Content-Type": upstream.headers.get("content-type") ?? "application/json",
        "Cache-Control": cacheable
          ? `public, s-maxage=${ttl}, stale-while-revalidate=${ttl}`
          : "no-store",
      },
    });
  } catch (error) {
    return json(
      {
        error: "upstream_unreachable",
        message: error instanceof Error ? error.message : "Unknown error",
      },
      502,
    );
  }
}

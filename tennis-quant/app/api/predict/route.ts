import { NextResponse } from "next/server";
import { predictMatch } from "@/lib/model";
import type { MatchInput } from "@/lib/types";

export const runtime = "nodejs";

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function isValidInput(input: unknown): input is MatchInput {
  if (!input || typeof input !== "object") return false;
  const row = input as Partial<MatchInput>;

  if (!row.playerA || !row.playerB) return false;
  if (!isFiniteNumber(row.oddsA) || !isFiniteNumber(row.oddsB)) return false;
  if (row.oddsA <= 1 || row.oddsB <= 1) return false;
  if (!isFiniteNumber(row.dataQuality)) return false;

  const players = [row.playerA, row.playerB];
  return players.every((p) =>
    Boolean(
      p &&
        typeof p.name === "string" &&
        isFiniteNumber(p.elo) &&
        isFiniteNumber(p.surfaceElo) &&
        isFiniteNumber(p.holdPct) &&
        isFiniteNumber(p.breakPct) &&
        isFiniteNumber(p.formScore) &&
        isFiniteNumber(p.fatigueScore) &&
        isFiniteNumber(p.physicalRisk)
    )
  );
}

export async function POST(request: Request) {
  let body: unknown;

  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON." }, { status: 400 });
  }

  if (!isValidInput(body)) {
    return NextResponse.json(
      { error: "Invalid match payload." },
      { status: 422 }
    );
  }

  try {
    return NextResponse.json({ prediction: predictMatch(body) });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Prediction failed." },
      { status: 400 }
    );
  }
}

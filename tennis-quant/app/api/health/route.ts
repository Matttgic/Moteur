import { NextResponse } from "next/server";

export const runtime = "nodejs";

export function GET() {
  return NextResponse.json({
    ok: true,
    service: "tennis-quant-engine",
    version: "0.1.0"
  });
}

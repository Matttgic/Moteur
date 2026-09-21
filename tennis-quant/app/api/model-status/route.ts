import benchmark from "@/ml/benchmark_reference.json";
import { NextResponse } from "next/server";

export const runtime = "nodejs";

export function GET() {
  return NextResponse.json({
    ...benchmark,
    economic_validation: {
      status: "pending_authorized_historical_odds",
      roi_validated: false,
      clv_validated: false
    }
  });
}
